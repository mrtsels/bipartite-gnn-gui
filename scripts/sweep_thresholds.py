#!/usr/bin/env python3
"""Sweep violation thresholds for the joint model on real VLM data.

Goal: find a threshold where the GNN adds TPs without exploding FPs.
Also computes the actual existence-score discrimination (TP vs FP).
"""

import json, math, sys
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
DEVICE = torch.device("cpu")


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


def main():
    ckpt = ROOT / "checkpoints/violation_detection_joint/best_model.pt"
    model = BipartiteGNNCorrector(hidden_dim=128, dropout=0.0)
    state = torch.load(str(ckpt), map_location="cpu")
    ms = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    print(f"Loaded {len(filtered)}/{len(state)} keys from {ckpt}")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    builder = BipartiteGraphBuilder()

    vlm_files = sorted(VLM_DIR.glob("*.json"))

    # Preload all data
    samples = []
    for vlm_path in vlm_files:
        vlm_elems = load_vlm_elements(vlm_path)
        if vlm_elems is None or len(vlm_elems) < 1:
            continue
        gt_elems = load_gt_elements(RICO_DIR / f"{vlm_path.stem}.json")
        if gt_elems is None or len(gt_elems) < 1:
            continue
        samples.append((vlm_path.stem, vlm_elems, gt_elems))
    print(f"Samples: {len(samples)}")

    # Baseline
    b_tp = b_fp = b_fn = b_np = b_ng = 0
    for _, vlm_elems, gt_elems in samples:
        m, fp, fn = hungarian_match(vlm_elems, gt_elems, 0.1)
        b_tp += len(m); b_fp += len(fp); b_fn += len(fn)
        b_np += len(vlm_elems); b_ng += len(gt_elems)

    print(f"\nBaseline: TP={b_tp} FP={b_fp} FN={b_fn} P={b_tp/b_np:.3f} R={b_tp/b_ng:.3f}")

    # Sweep thresholds
    print(f"\n{'thresh':>8s} {'TP':>5s} {'FP':>5s} {'P':>6s} {'R':>6s} {'F1':>6s} {'dF1':>7s} {'dTP':>5s} {'dFP':>5s} {'props':>6s}")
    for thresh in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        a_tp = a_fp = a_fn = a_np = 0
        n_props = 0
        for _, vlm_elems, gt_elems in samples:
            constraints = extract_all_constraints(vlm_elems)
            if not constraints or len(vlm_elems) < 3:
                corrected = list(vlm_elems)
            else:
                data = builder.build(vlm_elems, constraints)
                if data is None:
                    corrected = list(vlm_elems)
                else:
                    with torch.no_grad():
                        pred = model(data.to(DEVICE))
                    violation = pred.get("violation", torch.zeros(len(constraints), 1)).cpu()
                    proposals_raw = pred.get("proposal")
                    proposed = []
                    if proposals_raw is not None:
                        violated_idx = (violation.view(-1) > thresh).nonzero(as_tuple=False).view(-1).tolist()
                        pboxes, pscores = [], []
                        for vi in violated_idx:
                            bb = proposals_raw[vi].cpu().tolist()
                            x1, y1, x2, y2 = bb
                            x1 = max(0.0, min(1.0, x1)); y1 = max(0.0, min(1.0, y1))
                            x2 = max(0.0, min(1.0, x2)); y2 = max(0.0, min(1.0, y2))
                            if x2 <= x1 or y2 <= y1:
                                continue
                            pboxes.append([x1, y1, x2, y2])
                            pscores.append(float(violation[vi].item()))
                        keep = nms(pboxes, pscores)
                        for ki in keep:
                            proposed.append(ElementNode(bbox=pboxes[ki], label="other", confidence=pscores[ki]))
                    n_props += len(proposed)
                    corrected = list(vlm_elems) + proposed
            m, fp, fn = hungarian_match(corrected, gt_elems, 0.1)
            a_tp += len(m); a_fp += len(fp); a_fn += len(fn)
            a_np += len(corrected)

        p = a_tp / max(a_np, 1); r = a_tp / max(b_ng, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        f1_b = 2 * (b_tp / b_np) * (b_tp / b_ng) / max(b_tp / b_np + b_tp / b_ng, 1e-8)
        print(f"{thresh:8.2f} {a_tp:5d} {a_fp:5d} {p:6.3f} {r:6.3f} {f1:6.3f} {f1 - f1_b:+7.4f} {a_tp - b_tp:+5d} {a_fp - b_fp:+5d} {n_props:6d}")


if __name__ == "__main__":
    main()
