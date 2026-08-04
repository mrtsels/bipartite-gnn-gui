#!/usr/bin/env python3
"""Visualize GNN correction on real VLM data — see what actually changes."""

import json, sys, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode

# Paths
ROOT = Path(__file__).resolve().parent.parent
VLM_DIR = ROOT / "data/vlm_predictions/rico_qwen_flash"
RICO_DIR = ROOT / "data/rico_local/combined"
CKPT = ROOT / "checkpoints/violation_detection_joint/best_model.pt"
VIOLATION_THRESHOLD = 0.60
OUT_DIR = ROOT / "demo_vis"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ------------------------ Load model ------------------------
model = BipartiteGNNCorrector(hidden_dim=128, dropout=0.0).to(DEVICE)
state = torch.load(str(CKPT), map_location="cpu")
model_state = model.state_dict()
filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
model.load_state_dict(filtered, strict=False)
model.eval()
print(f"Model loaded from {CKPT}")

builder = BipartiteGraphBuilder()

# ------------------------ Helpers ------------------------
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
    cost = torch.full((M, N), INF)
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

def load_vlm(img_id):
    path = VLM_DIR / f"{img_id}.json"
    if not path.exists():
        return None, None
    data = json.loads(path.read_text())
    raw = data.get("elements", [])
    # qwen3-vl-flash returns coords in a fixed 1080x960 frame, not the
    # original image size recorded in image_width/image_height.
    w, h = 1080, 960
    elems = []
    for item in raw:
        bbox = item.get("bbox_xyxy") or item.get("bbox") or item.get("bbox_2d")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        x1, x2 = x1 / w, x2 / w
        y1, y2 = y1 / h, y2 / h
        if x2 <= x1 or y2 <= y1:
            continue
        label = str(item.get("label", "other")).lower()
        elems.append(ElementNode(bbox=[x1, y1, x2, y2], label=label, confidence=1.0))
    return elems, (w, h)

def load_gt(img_id):
    path = RICO_DIR / f"{img_id}.json"
    if not path.exists():
        return None
    from scripts.run_experiment import extract_elements, normalize_bbox, parse_rico_vh, _normalize_label
    parsed = parse_rico_vh(path)
    if parsed is None:
        return None
    rw, rh = parsed["width"], parsed["height"]
    raw = extract_elements(parsed["root"])
    elems = [normalize_bbox(e, rw, rh) for e in raw]
    elems = [e for e in elems if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]]
    return elems

def run_gnn(vlm_elems, violation_threshold=0.5):
    if len(vlm_elems) < 3:
        return list(vlm_elems), []
    constraints = extract_all_constraints(vlm_elems)
    if not constraints:
        return list(vlm_elems), []
    data = builder.build(vlm_elems, constraints)
    if data is None:
        return list(vlm_elems), []
    data_gpu = data.to(DEVICE)
    pred = model(data_gpu)
    
    violation = pred.get("violation", torch.zeros(len(constraints), 1, device=DEVICE)).cpu()
    proposals_raw = pred.get("proposal")
    
    proposed = []
    if proposals_raw is not None:
        violated_idx = (violation.view(-1) > violation_threshold).nonzero(as_tuple=False).view(-1).tolist()
        pboxes, pscores = [], []
        for vi in violated_idx:
            bbox = proposals_raw[vi].cpu().tolist()
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(1, x1)); y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2)); y2 = max(0, min(1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            pboxes.append([x1, y1, x2, y2])
            pscores.append(float(violation[vi].item()))
        keep = nms(pboxes, pscores)
        for ki in keep:
            proposed.append(ElementNode(bbox=pboxes[ki], label="proposal", confidence=pscores[ki]))
    return list(vlm_elems) + proposed, proposed

# Colors
VLM_COLOR = (255, 80, 80, 180)      # red
GT_COLOR = (0, 180, 0, 180)         # green
PROPOSAL_COLOR = (80, 130, 255, 200)  # blue
MATCH_LINE = (255, 200, 0, 120)     # yellow line

def draw_boxes(draw, elems, img_w, img_h, color, label=False, width=2):
    """Draw bboxes on PIL draw."""
    r, g, b, a = color
    for e in elems:
        x1, y1, x2, y2 = [int(v * img_w) if i % 2 == 0 else int(v * img_h) for i, v in enumerate(e.bbox)]
        # Draw semi-transparent fill
        draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, 30), outline=(r, g, b), width=width)
        if label:
            lbl = getattr(e, 'label', '') or ''
            draw.text((x1 + 2, y1 + 2), lbl[:12], fill=(255, 255, 255))

def visualize(img_id):
    print(f"\n--- Visualizing img={img_id} ---")
    
    # Load data
    vlm_elems, (vw, vh) = load_vlm(img_id)
    gt_elems = load_gt(img_id)
    if vlm_elems is None or gt_elems is None:
        print(f"  SKIP: missing data")
        return
    
    # Load screenshot
    img_path = RICO_DIR / f"{img_id}.jpg"
    if not img_path.exists():
        print(f"  SKIP: no screenshot")
        return
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    
    # Run GNN
    corrected, proposals = run_gnn(vlm_elems, violation_threshold=VIOLATION_THRESHOLD)
    
    # Match before and after
    matched_before, fp_before, fn_before = hungarian_match(vlm_elems, gt_elems)
    matched_after, fp_after, fn_after = hungarian_match(corrected, gt_elems)
    
    tp_b = len(matched_before)
    tp_a = len(matched_after)
    prec_b = tp_b / max(len(vlm_elems), 1)
    rec_b = tp_b / max(len(gt_elems), 1)
    f1_b = 2 * prec_b * rec_b / max(prec_b + rec_b, 1e-8)
    prec_a = tp_a / max(len(corrected), 1)
    rec_a = tp_a / max(len(gt_elems), 1)
    f1_a = 2 * prec_a * rec_a / max(prec_a + rec_a, 1e-8)
    
    print(f"  VLM: {len(vlm_elems)} elems, GT: {len(gt_elems)} elems, Proposals: {len(proposals)}")
    print(f"  Before: TP={tp_b} FP={len(fp_before)} FN={len(fn_before)} P={prec_b:.3f} R={rec_b:.3f} F1={f1_b:.3f}")
    print(f"  After:  TP={tp_a} FP={len(fp_after)} FN={len(fn_after)} P={prec_a:.3f} R={rec_a:.3f} F1={f1_a:.3f}")
    print(f"  Delta:  TP={tp_a-tp_b:+d} FP={len(fp_after)-len(fp_before):+d} F1={f1_a-f1_b:+.3f}")
    
    # ---- Create side-by-side visualization ----
    # Canvas: two images side by side (Before | After)
    canvas_w = img_w * 2 + 40
    canvas_h = img_h + 120  # extra space for stats bar
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 40))
    
    # Left: Before (VLM only)
    left = img.copy()
    draw_l = ImageDraw.Draw(left, "RGBA")
    draw_boxes(draw_l, vlm_elems, img_w, img_h, VLM_COLOR, label=True, width=2)
    # Draw GT in green outline (for matched)
    matched_gt_indices = {j for i, j in matched_before}
    for idx, ge in enumerate(gt_elems):
        if idx in matched_gt_indices:
            x1, y1, x2, y2 = [int(v * img_w) if i % 2 == 0 else int(v * img_h) for i, v in enumerate(ge.bbox)]
            draw_l.rectangle([x1, y1, x2, y2], outline=GT_COLOR[:3], width=2)
    # Red X for FNs (unmatched GT)
    for idx in fn_before:
        ge = gt_elems[idx]
        cx = int((ge.bbox[0] + ge.bbox[2]) / 2 * img_w)
        cy = int((ge.bbox[1] + ge.bbox[3]) / 2 * img_h)
        s = 6
        draw_l.line([cx - s, cy - s, cx + s, cy + s], fill=(255, 50, 50), width=3)
        draw_l.line([cx + s, cy - s, cx - s, cy + s], fill=(255, 50, 50), width=3)
    
    canvas.paste(left, (0, 100))
    
    # Right: After (VLM + GNN proposals)
    right = img.copy()
    draw_r = ImageDraw.Draw(right, "RGBA")
    # VLM elements (original)
    draw_boxes(draw_r, vlm_elems, img_w, img_h, VLM_COLOR, label=False, width=1)
    # Proposals in blue
    draw_boxes(draw_r, proposals, img_w, img_h, PROPOSAL_COLOR, label=True, width=3)
    # GT in green for matched
    matched_gt_indices_after = {j for i, j in matched_after}
    for idx, ge in enumerate(gt_elems):
        if idx in matched_gt_indices_after:
            x1, y1, x2, y2 = [int(v * img_w) if i % 2 == 0 else int(v * img_h) for i, v in enumerate(ge.bbox)]
            draw_r.rectangle([x1, y1, x2, y2], outline=GT_COLOR[:3], width=2)
    # X for remaining FNs
    for idx in fn_after:
        ge = gt_elems[idx]
        cx = int((ge.bbox[0] + ge.bbox[2]) / 2 * img_w)
        cy = int((ge.bbox[1] + ge.bbox[3]) / 2 * img_h)
        s = 6
        draw_r.line([cx - s, cy - s, cx + s, cy + s], fill=(255, 50, 50), width=3)
        draw_r.line([cx + s, cy - s, cx - s, cy + s], fill=(255, 50, 50), width=3)
    
    canvas.paste(right, (img_w + 40, 100))
    
    # ---- Stats bar ----
    draw_c = ImageDraw.Draw(canvas)
    title = f"GUI Correction: img_{img_id}"
    draw_c.text((20, 10), title, fill=(220, 220, 220))
    
    stats_before = f"BEFORE (VLM only): {len(vlm_elems)} detections | TP={tp_b} FP={len(fp_before)} FN={len(fn_before)}"
    stats_after = f"AFTER (VLM + GNN): {len(corrected)} detections (+{len(proposals)} proposals) | TP={tp_a} FP={len(fp_after)} FN={len(fn_after)}"
    delta_str = f"DELTA: TP {tp_b}->{tp_a} ({tp_a-tp_b:+d}) | F1 {f1_b:.3f}->{f1_a:.3f} ({f1_a-f1_b:+.3f})"
    
    draw_c.text((20, 35), stats_before, fill=(255, 150, 150))
    draw_c.text((20, 55), stats_after, fill=(150, 200, 255))
    draw_c.text((20, 75), delta_str, fill=(255, 220, 100))
    
    # Legend
    leg_x = canvas_w - 250
    draw_c.rectangle([leg_x, 10, leg_x + 15, 25], fill=VLM_COLOR[:3])
    draw_c.text((leg_x + 20, 10), "VLM detection", fill=(200, 200, 200))
    draw_c.rectangle([leg_x, 30, leg_x + 15, 45], fill=GT_COLOR[:3])
    draw_c.text((leg_x + 20, 30), "GT match", fill=(200, 200, 200))
    draw_c.rectangle([leg_x, 50, leg_x + 15, 65], fill=PROPOSAL_COLOR[:3])
    draw_c.text((leg_x + 20, 50), "GNN proposal", fill=(200, 200, 200))
    
    out_path = OUT_DIR / f"demo_{img_id}.png"
    canvas.save(out_path)
    print(f"  Saved: {out_path}")

# ---- Main ----
# 最终 hero cases（来自 demo_data/cases.json，joint 模型 @ threshold=0.6）
# 坐标基准修正(1080x960) + GT/截图比例一致筛选后为 5 个
best_ids = [
    "10067", "10027", "10179", "10033", "1013",
]

for img_id in best_ids:
    visualize(img_id)

print("\nDone! Check demo_vis/ for output images.")
