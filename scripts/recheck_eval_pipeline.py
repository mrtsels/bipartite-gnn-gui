#!/usr/bin/env python3
"""Re-evaluate the GNN completion pipeline with correctly-loaded checkpoints.

The original eval_real_vlm_pipeline.py loaded violation_detection/best_model.pt
(hidden_dim=16) into a hidden_dim=128 model with strict=False, silently
dropping 89% of weights (random init). This script loads checkpoints with
shape-filtering so only matched weights are used, and compares which
checkpoint produces genuine improvements.
"""

import json, math, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import (
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
    _normalize_label,
)

VLM_DIR = ROOT / "data/vlm_predictions/rico_qwen_flash"
RICO_DIR = ROOT / "data/rico_local/combined"

CHECKPOINTS = [
    "checkpoints/violation_detection_joint/best_model.pt",
    "checkpoints/violation_detection_violation_only/best_model.pt",
    "checkpoints/violation_detection/visual_fusion_model.pt",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def center_distance(box_a, box_b):
    ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    return math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)


def hungarian_match(pred, gt, threshold=0.1):
    from scipy.optimize import linear_sum_assignment
    M, N = len(pred), len(gt)
    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))
    INF = 1e9
    cost = torch.full((M, N), INF, dtype=torch.float32)
    for i, pe in enumerate(pred):
        for j, ge in enumerate(gt):
            d = center_distance(pe.bbox, ge.bbox)
            if d <= threshold:
                cost[i, j] = d
    if not torch.isfinite(cost).any():
        return [], list(range(M)), list(range(N))
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    matched, matched_rows, matched_cols = [], set(), set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF / 2:
            matched.append((int(i), int(j)))
            matched_rows.add(int(i))
            matched_cols.add(int(j))
    fp = [i for i in range(M) if i not in matched_rows]
    fn = [j for j in range(N) if j not in matched_cols]
    return matched, fp, fn


def compute_iou(box1, box2, eps=1e-8):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (a1 + a2 - inter + eps)


def nms(bboxes, scores, iou_threshold=0.5):
    if not bboxes:
        return []
    indices = list(range(len(bboxes)))
    indices.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(i)
        to_remove = [j for j in indices if compute_iou(bboxes[i], bboxes[j]) > iou_threshold]
        for j in to_remove:
            indices.remove(j)
    return keep


def load_vlm_elements(vlm_path):
    try:
        vlm_data = json.loads(vlm_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    raw = vlm_data.get("elements", [])
    img_w = vlm_data.get("image_width", 1)
    img_h = vlm_data.get("image_height", 1)
    if img_w <= 0 or img_h <= 0:
        return None
    elems = []
    for item in raw:
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        x1, x2 = x1 / img_w, x2 / img_w
        y1, y2 = y1 / img_h, y2 / img_h
        if x2 <= x1 or y2 <= y1:
            continue
        label = _normalize_label(item.get("label", "other"))
        elems.append(ElementNode(bbox=[x1, y1, x2, y2], label=label, confidence=1.0))
    return elems


def load_gt_elements(gt_path, min_elems=1):
    if not gt_path.exists():
        return None
    parsed = parse_rico_vh(gt_path)
    if parsed is None:
        return None
    rw, rh = parsed["width"], parsed["height"]
    gt_raw = extract_elements(parsed["root"])
    gt_elems = [normalize_bbox(e, rw, rh) for e in gt_raw]
    gt_elems = [e for e in gt_elems if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]]
    if len(gt_elems) < min_elems:
        return None
    return gt_elems


@torch.no_grad()
def run_gnn(model, builder, vlm_elems, threshold=0.5):
    if len(vlm_elems) < 3:
        return list(vlm_elems), []
    constraints = extract_all_constraints(vlm_elems)
    if not constraints:
        return list(vlm_elems), []
    data = builder.build(vlm_elems, constraints)
    if data is None:
        return list(vlm_elems), []
    data = data.to(DEVICE)
    pred = model(data)
    violation = pred.get("violation", torch.zeros(len(constraints), 1, device=DEVICE)).cpu()
    proposals_raw = pred.get("proposal")
    proposed = []
    if proposals_raw is not None:
        violated_idx = (violation.view(-1) > threshold).nonzero(as_tuple=False).view(-1).tolist()
        pboxes, pscores = [], []
        for vi in violated_idx:
            bbox = proposals_raw[vi].cpu().tolist()
            x1, y1, x2, y2 = bbox
            x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            pboxes.append([x1, y1, x2, y2])
            pscores.append(float(violation[vi].item()))
        keep = nms(pboxes, pscores)
        for ki in keep:
            proposed.append(ElementNode(bbox=pboxes[ki], label="other", confidence=pscores[ki]))
    return list(vlm_elems) + proposed, proposed


def main():
    vlm_files = sorted(VLM_DIR.glob("*.json"))
    print(f"VLM predictions: {len(vlm_files)} files\n")

    for ckpt_rel in CHECKPOINTS:
        ckpt = ROOT / ckpt_rel
        if not ckpt.exists():
            print(f"=== {ckpt_rel}: NOT FOUND ===\n")
            continue

        # Try hidden_dim=128 first
        model = BipartiteGNNCorrector(hidden_dim=128, dropout=0.0)
        state = torch.load(str(ckpt), map_location="cpu")
        ms = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
        n_matched = len(filtered)
        n_total = len(state)
        if n_matched < 0.5 * n_total:
            # Try hidden_dim=16
            model = BipartiteGNNCorrector(hidden_dim=16, dropout=0.0)
            ms = model.state_dict()
            filtered = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
            n_matched = len(filtered)
        model.load_state_dict(filtered, strict=False)
        model.eval()
        model.to(DEVICE)

        builder = BipartiteGraphBuilder()
        hid = "16" if "e_to_c_convs.0.lin_l.weight" in state and state["e_to_c_convs.0.lin_l.weight"].shape[0] == 16 else "128"
        print(f"=== {ckpt_rel} (checkpoint hd={hid}, loaded {n_matched}/{n_total} keys) ===")

        before_tp = before_fp = before_fn = 0
        after_tp = after_fp = after_fn = 0
        n_pred_b = n_gt = n_pred_a = 0
        n_images = n_skipped = 0
        total_proposals = 0
        per_image = []

        t0 = time.time()
        for idx, vlm_path in enumerate(vlm_files):
            vlm_elems = load_vlm_elements(vlm_path)
            if vlm_elems is None or len(vlm_elems) < 1:
                n_skipped += 1
                continue
            gt_elems = load_gt_elements(RICO_DIR / f"{vlm_path.stem}.json")
            if gt_elems is None or len(gt_elems) < 1:
                n_skipped += 1
                continue

            m_b, fp_b, fn_b = hungarian_match(vlm_elems, gt_elems, 0.1)
            before_tp += len(m_b); before_fp += len(fp_b); before_fn += len(fn_b)
            n_pred_b += len(vlm_elems); n_gt += len(gt_elems)

            corrected, proposals = run_gnn(model, builder, vlm_elems)
            total_proposals += len(proposals)

            m_a, fp_a, fn_a = hungarian_match(corrected, gt_elems, 0.1)
            after_tp += len(m_a); after_fp += len(fp_a); after_fn += len(fn_a)
            n_pred_a += len(corrected)

            tp_delta = len(m_a) - len(m_b)
            fp_delta = len(fp_a) - len(fp_b)
            per_image.append({
                "image_id": vlm_path.stem,
                "before_tp": len(m_b), "after_tp": len(m_a),
                "tp_delta": tp_delta, "fp_delta": fp_delta,
                "n_proposals": len(proposals),
                "n_gt": len(gt_elems), "n_vlm": len(vlm_elems),
            })
            n_images += 1

        dt = time.time() - t0

        prec_b = before_tp / max(n_pred_b, 1)
        rec_b = before_tp / max(n_gt, 1)
        f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-8)
        prec_a = after_tp / max(n_pred_a, 1)
        rec_a = after_tp / max(n_gt, 1)
        f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-8)

        print(f"  Images: {n_images} ({n_skipped} skipped) in {dt:.1f}s")
        print(f"  Proposals total: {total_proposals}")
        print(f"  Before: TP={before_tp} FP={before_fp} FN={before_fn} P={prec_b:.3f} R={rec_b:.3f} F1={f1_b:.3f}")
        print(f"  After:  TP={after_tp} FP={after_fp} FN={after_fn} P={prec_a:.3f} R={rec_a:.3f} F1={f1_a:.3f}")
        print(f"  Delta:  TP={after_tp-before_tp:+d} FP={after_fp-before_fp:+d} F1={f1_a-f1_b:+.4f}")

        # Best images
        per_image.sort(key=lambda x: x["tp_delta"], reverse=True)
        print("  Top 5 by TP delta:")
        for r in per_image[:5]:
            print(f"    img={r['image_id']}: TP {r['before_tp']}->{r['after_tp']} (+{r['tp_delta']}) FP+{r['fp_delta']} proposals={r['n_proposals']} gt={r['n_gt']}")

        # Count images with improvement
        n_improved = sum(1 for r in per_image if r["tp_delta"] > 0)
        n_worse = sum(1 for r in per_image if r["tp_delta"] < 0)
        print(f"  Images with TP gain: {n_improved}, TP loss: {n_worse}")

        # Save per-image results
        out = ROOT / "experiments/vlm_completion" / f"recheck_{Path(ckpt_rel).stem}.json"
        out.write_text(json.dumps({
            "checkpoint": str(ckpt), "loaded_keys": f"{n_matched}/{n_total}",
            "before": {"tp": before_tp, "fp": before_fp, "fn": before_fn,
                       "precision": prec_b, "recall": rec_b, "f1": f1_b},
            "after": {"tp": after_tp, "fp": after_fp, "fn": after_fn,
                      "precision": prec_a, "recall": rec_a, "f1": f1_a},
            "delta": {"tp": after_tp - before_tp, "fp": after_fp - before_fp,
                      "f1": f1_a - f1_b},
            "n_images": n_images, "total_proposals": total_proposals,
            "per_image": per_image,
        }, indent=2))
        print(f"  Saved: {out}\n")


if __name__ == "__main__":
    main()
