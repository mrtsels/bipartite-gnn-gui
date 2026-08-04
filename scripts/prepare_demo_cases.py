#!/usr/bin/env python3
"""Prepare hero demo cases for the web demo.

Runs the joint GNN model (threshold=0.6) over all 200 RICO VLM predictions,
filters cases with ΔTP >= 4 and ΔFP <= 5, keeps the top 12 by ΔTP, and writes:

    demo_data/cases.json       full case data (boxes + before/after metrics)
    demo_data/summary.json     aggregate stats
    demo_data/screenshots/     copied screenshots for each selected case

Self-contained: copies the core evaluation helpers (hungarian_match, nms,
load helpers, run_gnn) from scripts/recheck_eval_pipeline.py so it can be
run standalone.
"""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector

VLM_DIR = ROOT / "data/vlm_predictions/rico_qwen_flash"
RICO_DIR = ROOT / "data/rico_local/combined"
CKPT = ROOT / "checkpoints/violation_detection_joint/best_model.pt"
OUT_DIR = ROOT / "demo_data"
SHOT_DIR = OUT_DIR / "screenshots"

VIOLATION_THRESHOLD = 0.6
MIN_DTP = 4          # GNN must recover at least 4 true elements
MAX_DFP = 5          # at most 5 extra false positives
TOP_N = 12           # keep at most this many cases

NAMES = {
    "10027": "Weather App",
    "10043": "Settings Page",
    "10068": "Music Player",
    "10005": "Travel Booking",
    "10059": "News Feed",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Matching / NMS helpers (copied from scripts/recheck_eval_pipeline.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading helpers (copied from scripts/recheck_eval_pipeline.py and
# scripts/run_experiment.py)
# ---------------------------------------------------------------------------

_LABEL_ALIASES = {
    "btn": "button", "img": "image", "glyph": "icon",
    "textbox": "input", "search": "input", "textarea": "input", "textfield": "input",
    "div": "container", "section": "container", "frame": "container", "panel": "container",
    "check": "checkbox", "radiobutton": "radio", "range": "slider",
    "toggle": "switch", "dropdown": "menu", "nav": "menu",
    "separator": "divider", "hr": "divider",
    "dialog": "modal", "overlay": "modal",
    "snackbar": "toast", "notification": "toast",
    "announcement": "banner", "alertbar": "banner",
}


def _normalize_label(label: str) -> str:
    key = label.strip().lower()
    return _LABEL_ALIASES.get(key, key)


def rico_class_to_label(cls: str) -> str:
    short = cls.rsplit(".", 1)[-1]
    mapping = {
        "Button": "button", "ImageButton": "icon", "ImageView": "image",
        "TextView": "text", "EditText": "input", "CheckBox": "checkbox",
        "Switch": "switch", "Spinner": "icon", "ProgressBar": "icon",
        "WebView": "container", "ListView": "list", "ScrollView": "container",
        "TabWidget": "tab", "RadioButton": "radio", "SeekBar": "slider",
    }
    for suffix, label in mapping.items():
        if short.endswith(suffix):
            return label
    return "other"


def parse_rico_vh(path: str | Path) -> dict | None:
    """Load and validate a RICO View Hierarchy JSON -> {root, width, height}."""
    try:
        with open(path) as f:
            raw = json.load(f)
        activity = raw.get("activity", {})
        root = activity.get("root")
        if not root:
            root = raw.get("root")
        if not root:
            return None
        bounds = root.get("bounds", [0, 0, 0, 0])
        if len(bounds) != 4 or bounds[2] <= 0 or bounds[3] <= 0:
            return None
        return {"root": root, "width": bounds[2], "height": bounds[3]}
    except Exception:
        return None


def extract_elements(root: dict) -> list[ElementNode]:
    """Extract visible leaf element nodes from a RICO View Hierarchy tree."""
    elements: list[ElementNode] = []

    def walk(node: dict, depth: int = 0):
        if depth > 50:
            return
        children = node.get("children")
        is_leaf = not isinstance(children, list) or len(children) == 0
        if is_leaf:
            vis = node.get("visibility", "visible")
            if vis != "visible":
                return
            v2u = node.get("visible-to-user", True)
            if v2u is False:
                return
            bounds = node.get("bounds", [0, 0, 0, 0])
            if len(bounds) != 4:
                return
            x1, y1, x2, y2 = bounds
            if x2 <= x1 or y2 <= y1:
                return
            cls = node.get("class", "")
            if not cls:
                return
            label = rico_class_to_label(cls)
            text = node.get("text") or ""
            if not text:
                cd = node.get("content-desc", [None])
                if isinstance(cd, list) and cd[0] is not None:
                    text = str(cd[0])
            elements.append(
                ElementNode(
                    bbox=[x1, y1, x2, y2],
                    label=label,
                    confidence=1.0,
                    element_id=f"elem_{len(elements)}",
                    features={"text_len": len(str(text))},
                )
            )
        else:
            for child in children:
                if isinstance(child, dict):
                    walk(child, depth + 1)

    walk(root)
    return elements


def normalize_bbox(elem: ElementNode, img_w: int, img_h: int) -> ElementNode:
    x1, y1, x2, y2 = elem.bbox
    return ElementNode(
        bbox=[x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
        label=elem.label,
        confidence=elem.confidence,
        element_id=elem.element_id,
        features=elem.features,
    )


# qwen3-vl-flash returns bbox coords in a fixed 1080x960 internal frame,
# regardless of the input image size. image_width/image_height in the JSON
# record the ORIGINAL image size, not the VLM coordinate baseline — using
# them to normalize shifts all boxes (y is halved for 1080x1920 inputs).
# Verified: all 200 RICO VLM files have coords within 1080x960.
VLM_COORD_W = 1080
VLM_COORD_H = 960


def load_vlm_elements(vlm_path: Path):
    """Return (elements, img_w, img_h) with normalized bboxes, or None.

    bboxes are normalized against the VLM coordinate baseline (1080x960),
    NOT the original image size stored in the JSON.
    """
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
        x1, x2 = x1 / VLM_COORD_W, x2 / VLM_COORD_W
        y1, y2 = y1 / VLM_COORD_H, y2 / VLM_COORD_H
        if x2 <= x1 or y2 <= y1:
            continue
        label = _normalize_label(item.get("label", "other"))
        elems.append(ElementNode(bbox=[x1, y1, x2, y2], label=label, confidence=1.0))
    return elems, int(img_w), int(img_h)


def load_gt_elements(gt_path: Path, min_elems=1):
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


def verify_screenshot(path: Path):
    """PIL verify; return (width, height) or None if corrupt."""
    try:
        from PIL import Image
        img = Image.open(path)
        img.verify()
        with Image.open(path) as img2:
            return img2.size
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GNN inference + metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_gnn(model, builder, vlm_elems, threshold=VIOLATION_THRESHOLD):
    """Run the GNN over VLM elements; return (corrected_elements, proposals).

    proposals: list of ElementNode with confidence = violation score.
    """
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


def metrics_for(pred_elems, gt_elems):
    """Per-image detection metrics via Hungarian matching (center dist <= 0.1)."""
    matched, fp, fn = hungarian_match(pred_elems, gt_elems, 0.1)
    tp = len(matched)
    detections = len(pred_elems)
    precision = tp / max(detections, 1)
    recall = tp / max(tp + len(fn), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "detections": detections,
        "tp": tp,
        "fp": len(fp),
        "fn": len(fn),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def r4(x):
    return round(float(x), 4)


def main():
    vlm_files = sorted(VLM_DIR.glob("*.json"))
    total = len(vlm_files)
    print(f"VLM predictions: {total} files\n")

    # --- load model (shape-filter, verified 44/44 for joint checkpoint) ---
    model = BipartiteGNNCorrector(hidden_dim=128, dropout=0.0)
    state = torch.load(str(CKPT), map_location="cpu")
    ms = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in ms and v.shape == ms[k].shape}
    if len(filtered) < 30:
        raise RuntimeError(
            f"Checkpoint mismatch: only {len(filtered)}/{len(state)} keys matched "
            f"(expected >=30). Check hidden_dim / checkpoint path."
        )
    model.load_state_dict(filtered, strict=False)
    model.eval()
    model.to(DEVICE)
    builder = BipartiteGraphBuilder()
    print(f"Model loaded: {len(filtered)}/{len(state)} keys matched "
          f"(joint hd=128), threshold={VIOLATION_THRESHOLD}, device={DEVICE}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHOT_DIR.mkdir(parents=True, exist_ok=True)

    skipped = 0
    evaluated = 0
    results = []  # candidate records
    t0 = time.time()

    for vlm_path in vlm_files:
        img_id = vlm_path.stem
        loaded = load_vlm_elements(vlm_path)
        if loaded is None or len(loaded[0]) < 1:
            skipped += 1
            continue
        vlm_elems, img_w, img_h = loaded

        gt_elems = load_gt_elements(RICO_DIR / f"{img_id}.json")
        if gt_elems is None or len(gt_elems) < 1:
            skipped += 1
            continue

        shot_size = verify_screenshot(RICO_DIR / f"{img_id}.jpg")
        if shot_size is None:
            print(f"  skip {img_id}: corrupted/missing screenshot")
            skipped += 1
            continue

        if img_w <= 0 or img_h <= 0:
            img_w, img_h = shot_size

        # GT 与截图宽高比必须一致, 否则 GT 归一化坐标映射到截图会错位
        # (实测 10005: RICO 1440x2392 vs 截图 1080x1920 比例不一致 → 绿框偏移)
        gt_parsed = parse_rico_vh(RICO_DIR / f"{img_id}.json")
        if gt_parsed is not None:
            rw, rh = gt_parsed["width"], gt_parsed["height"]
            if abs(rw / rh - shot_size[0] / shot_size[1]) > 1e-3:
                print(f"  skip {img_id}: GT ratio {rw}x{rh} != screenshot {shot_size[0]}x{shot_size[1]}")
                skipped += 1
                continue

        m_b, fp_b, fn_b = hungarian_match(vlm_elems, gt_elems, 0.1)
        corrected, proposals = run_gnn(model, builder, vlm_elems, VIOLATION_THRESHOLD)
        m_a, fp_a, fn_a = hungarian_match(corrected, gt_elems, 0.1)

        dtp = len(m_a) - len(m_b)
        dfp = len(fp_a) - len(fp_b)
        before = metrics_for(vlm_elems, gt_elems)
        after = metrics_for(corrected, gt_elems)
        delta_f1 = after["f1"] - before["f1"]

        # 标记每个 proposal 是否匹配到 GT（前端据此区分绿=补对 / 蓝=误提议）
        matched_gt_indices = {j for _, j in m_a}
        proposal_list = []
        for p in proposals:
            # 中心距离匹配（与 hungarian_match 阈值一致）
            matched = any(
                center_distance(p.bbox, ge.bbox) <= 0.1
                for ge in gt_elems
            )
            proposal_list.append({
                "bbox": [r4(v) for v in p.bbox],
                "violation_score": r4(p.confidence),
                "matched": matched,
            })

        # 漏检标记：VLM 未匹配到的 GT bbox（前端画红 X）
        missed = [
            [r4(v) for v in gt_elems[j].bbox] for j in fn_b
        ]
        # 绿色 GT 匹配框：用 GT 元素的真实 bbox（不是 proposal 预测的 bbox）。
        # 对每个 matched proposal 找中心距离最近的 GT 元素。
        # 语义: 绿 = GNN 找回的元素（框在 GT 真实位置）, 蓝虚线 = GNN 误提议。
        gt_matches = []
        for p in proposals:
            if not any(center_distance(p.bbox, ge.bbox) <= 0.1 for ge in gt_elems):
                continue
            best = min(gt_elems, key=lambda ge: center_distance(p.bbox, ge.bbox))
            gt_matches.append([r4(v) for v in best.bbox])

        results.append({
            "id": img_id,
            "name": NAMES.get(img_id, f"Case {img_id}"),
            "screenshot": f"{img_id}.jpg",
            "img_w": int(img_w),
            "img_h": int(img_h),
            "vlm_elements": [
                {"bbox": [r4(v) for v in e.bbox], "label": e.label} for e in vlm_elems
            ],
            "proposals": proposal_list,
            "missed": missed,
            "gt_matches": gt_matches,
            "metrics": {"before": before, "after": after},
            "_dtp": dtp,
            "_dfp": dfp,
            "_delta_f1": round(delta_f1, 4),
        })
        evaluated += 1

    dt = time.time() - t0

    # --- filter + rank ---
    candidates = [r for r in results if r["_dtp"] >= MIN_DTP and r["_dfp"] <= MAX_DFP]
    candidates.sort(key=lambda r: (-r["_dtp"], r["_dfp"]))
    selected = candidates[:TOP_N]

    # --- copy screenshots ---
    copied = 0
    for case in selected:
        src = RICO_DIR / f"{case['id']}.jpg"
        dst = SHOT_DIR / case["screenshot"]
        shutil.copy(src, dst)
        copied += 1

    # --- strip internal keys, emit ---
    out_cases = []
    for case in selected:
        clean = {k: v for k, v in case.items() if not k.startswith("_")}
        out_cases.append(clean)

    cases_path = OUT_DIR / "cases.json"
    cases_path.write_text(json.dumps(out_cases, indent=2))
    print(f"Saved {len(out_cases)} cases to {cases_path} ({copied} screenshots)")

    avg_delta_f1 = (
        sum(c["_delta_f1"] for c in selected) / len(selected) if selected else 0.0
    )
    summary = {
        "n_cases": len(out_cases),
        "avg_delta_f1": round(avg_delta_f1, 4),
        "n_total_evaluated": evaluated,
        "n_skipped": skipped,
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved {summary_path}: {summary}")

    # --- stats printout ---
    print(f"\n=== Summary ===")
    print(f"Total images: {total}, evaluated: {evaluated}, skipped: {skipped}, "
          f"time: {dt:.1f}s")
    print(f"Selected cases: {len(selected)} "
          f"(filter: ΔTP>={MIN_DTP} & ΔFP<={MAX_DFP}, top {TOP_N} by ΔTP)")
    print("Top 5 cases by ΔTP:")
    for r in selected[:5]:
        b, a = r["metrics"]["before"], r["metrics"]["after"]
        print(f"  {r['id']} ({r['name']}): ΔTP {r['_dtp']:+d} ΔFP {r['_dfp']:+d} "
              f"F1 {b['f1']} -> {a['f1']} (Δ{r['_delta_f1']:+.4f}) "
              f"proposals={len(r['proposals'])}")


if __name__ == "__main__":
    main()
