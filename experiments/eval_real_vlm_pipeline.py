#!/usr/bin/env python3
"""Evaluate the full GNN completion pipeline on real VLM data.

Compares detection quality Before (raw VLM) vs After (VLM + GNN correction).
The GNN pipeline:
  1. Build constraint graph from VLM-detected elements
  2. Model inference → violation scores + element proposals
  3. For violated constraints (violation > 0.5), propose missing elements
  4. NMS-deduplicate proposals
  5. Corrected set = all VLM elements + proposed elements

Key finding from ablation: only the violation_detection model (trained on simulated
dropping) produces meaningful proposals on real VLM data. The "joint" model's
existence head collapses to ~0.48, making confidence filtering useless.
The "completion" model's violation head outputs ~0.0 on real data, so it
generates zero proposals.

Usage:
  python experiments/eval_real_vlm_pipeline.py

Output: experiments/vlm_completion/pipeline_comparison.json + printed table
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import (
    DEVICE,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
    _normalize_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VLM_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/vlm_predictions/rico_qwen_flash")
RICO_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/combined")

# The violation_detection model is the only one producing useful proposals
# on real VLM data. hidden_dim=128, all heads trained on simulated dropping.
CHECKPOINT_PATH = Path(
    "/Users/minimx/bipartite-gnn-gui/checkpoints/violation_detection/best_model.pt"
)

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def center_distance(box_a, box_b):
    """Euclidean distance between box centers (normalised coords)."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def hungarian_match(
    pred_elems: list[ElementNode],
    gt_elems: list[ElementNode],
    threshold: float = 0.1,
):
    """Match predicted elements to GT using center-distance Hungarian.

    Returns:
        matched: list of (pred_idx, gt_idx) pairs.
        fp_indices: list of unmatched pred indices.
        fn_indices: list of unmatched gt indices.
    """
    M, N = len(pred_elems), len(gt_elems)
    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))
    INF = 1e9
    cost = torch.full((M, N), INF, dtype=torch.float32)
    for i, pe in enumerate(pred_elems):
        for j, ge in enumerate(gt_elems):
            d = center_distance(pe.bbox, ge.bbox)
            if d <= threshold:
                cost[i, j] = d
    has_finite = torch.isfinite(cost).any()
    if not has_finite:
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


def compute_metrics(matched, fp, fn, total_pred, total_gt):
    """Compute Precision, Recall, F1."""
    tp = len(matched)
    precision = tp / max(total_pred, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp,
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": total_pred,
        "n_gt": total_gt,
    }


def compute_iou(box1, box2, eps=1e-8):
    """IoU of two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter + eps
    return inter / union


def nms(bboxes, scores, iou_threshold=0.5):
    """Greedy NMS on a list of xyxy bboxes."""
    if len(bboxes) == 0:
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
# GNN correction pipeline
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_gnn_pipeline(
    model: BipartiteGNNCorrector,
    vlm_elems: list[ElementNode],
    builder: BipartiteGraphBuilder,
    violation_threshold: float = 0.5,
) -> list[ElementNode]:
    """Run GNN correction on VLM-detected elements.

    1. Build constraint graph from VLM elements
    2. Model inference → violation scores + proposals
    3. For violated constraints, propose new elements at predicted bbox
    4. NMS-deduplicate proposals
    5. Return all VLM elements + proposals (=corrected set)

    NOTE: Existence head is NOT used for filtering — experiments show it
    does not discriminate well on real VLM data. The pipeline focuses on
    adding missed elements via proposal rather than removing FPs.

    If graph has < 3 elements or no constraints, return original elements.
    """
    if len(vlm_elems) < 3:
        return list(vlm_elems)

    # 1. Extract constraints and build graph
    constraints = extract_all_constraints(vlm_elems)
    if len(constraints) == 0:
        return list(vlm_elems)

    data = builder.build(vlm_elems, constraints)
    if data is None:
        return list(vlm_elems)

    # 2. Model inference
    data_gpu = data.to(DEVICE)
    pred = model(data_gpu)

    n_con = len(constraints)

    # 3. Violation scores → proposals
    violation = pred.get("violation", torch.zeros(n_con, 1, device=DEVICE)).cpu()
    proposals_raw = pred.get("proposal")  # (N_con, 4) bbox in [0,1] via sigmoid

    proposed_elems: list[ElementNode] = []
    if proposals_raw is not None and violation is not None:
        violated_mask = violation.view(-1) > violation_threshold
        violated_indices = violated_mask.nonzero(as_tuple=False).view(-1).tolist()

        proposal_bboxes: list[list[float]] = []
        proposal_scores: list[float] = []
        for vi in violated_indices:
            bbox = proposals_raw[vi].cpu().tolist()
            x1, y1, x2, y2 = bbox
            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            proposal_bboxes.append([x1, y1, x2, y2])
            proposal_scores.append(float(violation[vi].item()))

        # 4. NMS deduplication
        keep_indices = nms(proposal_bboxes, proposal_scores, iou_threshold=0.5)
        for ki in keep_indices:
            bbox = proposal_bboxes[ki]
            proposed_elems.append(ElementNode(
                bbox=bbox,
                label="other",
                confidence=proposal_scores[ki],
            ))

    # 5. Corrected set = all VLM elements + proposals (no filtering)
    corrected = list(vlm_elems) + proposed_elems
    return corrected


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------


def load_vlm_elements(vlm_path: Path) -> list[ElementNode] | None:
    """Load VLM predictions from JSON and return normalised ElementNodes."""
    try:
        vlm_data = json.loads(vlm_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    raw_elems = vlm_data.get("elements", [])
    img_w = vlm_data.get("image_width", 1)
    img_h = vlm_data.get("image_height", 1)
    if img_w <= 0 or img_h <= 0:
        return None

    elements: list[ElementNode] = []
    for item in raw_elems:
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        x1, x2 = x1 / img_w, x2 / img_w
        y1, y2 = y1 / img_h, y2 / img_h
        if x2 <= x1 or y2 <= y1:
            continue
        label = _normalize_label(item.get("label", "other"))
        elements.append(ElementNode(
            bbox=[x1, y1, x2, y2],
            label=label,
            confidence=1.0,
        ))
    return elements


def load_gt_elements(gt_path: Path, min_elems: int = 1) -> list[ElementNode] | None:
    """Load RICO GT and return normalised ElementNodes."""
    if not gt_path.exists():
        return None
    parsed = parse_rico_vh(gt_path)
    if parsed is None:
        return None
    rico_w, rico_h = parsed["width"], parsed["height"]
    gt_raw = extract_elements(parsed["root"])
    gt_elems = [normalize_bbox(e, rico_w, rico_h) for e in gt_raw]
    gt_elems = [e for e in gt_elems if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]]
    if len(gt_elems) < min_elems:
        return None
    return gt_elems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    logger.info("=" * 70)
    logger.info("REAL VLM PIPELINE — Before vs After GNN Correction")
    logger.info("=" * 70)

    vlm_files = sorted(VLM_DIR.glob("*.json"))
    logger.info("VLM predictions: %d files", len(vlm_files))

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        logger.error("Checkpoint not found: %s", CHECKPOINT_PATH)
        return

    logger.info("Loading model from %s", CHECKPOINT_PATH)
    model = BipartiteGNNCorrector(
        hidden_dim=128, dropout=0.0,
    ).to(DEVICE)

    state = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    logger.info("Model loaded (strict=False)")

    builder = BipartiteGraphBuilder()

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    # Before (raw VLM)
    before_tp, before_fp, before_fn = 0, 0, 0
    before_n_pred, before_n_gt = 0, 0

    # After (GNN-corrected)
    after_tp, after_fp, after_fn = 0, 0, 0
    after_n_pred, after_n_gt = 0, 0

    # Per-image tracking
    per_image_before: list[dict] = []
    per_image_after: list[dict] = []

    n_images = 0
    n_skipped = 0
    n_no_graph = 0
    n_total_vlm_elems = 0
    n_proposals_total = 0

    t0 = time.time()

    for idx, vlm_path in enumerate(vlm_files):
        if idx % 50 == 0 and idx > 0:
            dt = time.time() - t0
            logger.info("  Processed %d/%d (%.1f img/s)", idx, len(vlm_files), idx / max(dt, 0.01))

        # Load VLM
        vlm_elems = load_vlm_elements(vlm_path)
        if vlm_elems is None or len(vlm_elems) < 1:
            n_skipped += 1
            continue

        # Load GT
        gt_elems = load_gt_elements(RICO_DIR / f"{vlm_path.stem}.json")
        if gt_elems is None or len(gt_elems) < 1:
            n_skipped += 1
            continue

        # ------------------------------------------------------------------
        # BEFORE: match raw VLM → GT
        # ------------------------------------------------------------------
        matched_before, fp_idx, fn_idx = hungarian_match(
            vlm_elems, gt_elems, threshold=0.1
        )
        met_before = compute_metrics(
            matched_before, fp_idx, fn_idx, len(vlm_elems), len(gt_elems)
        )

        before_tp += met_before["tp"]
        before_fp += met_before["fp"]
        before_fn += met_before["fn"]
        before_n_pred += met_before["n_pred"]
        before_n_gt += met_before["n_gt"]

        per_image_before.append({
            "image_id": vlm_path.stem,
            **met_before,
        })

        # ------------------------------------------------------------------
        # GNN CORRECTION
        # ------------------------------------------------------------------
        n_total_vlm_elems += len(vlm_elems)

        corrected = run_gnn_pipeline(
            model, vlm_elems, builder,
            violation_threshold=0.5,
        )

        n_proposals_this = len(corrected) - len(vlm_elems)
        n_proposals_total += n_proposals_this

        if len(corrected) == 0:
            corrected = list(vlm_elems)

        # ------------------------------------------------------------------
        # AFTER: match corrected → GT
        # ------------------------------------------------------------------
        matched_after, fp_after_idx, fn_after_idx = hungarian_match(
            corrected, gt_elems, threshold=0.1
        )
        met_after = compute_metrics(
            matched_after, fp_after_idx, fn_after_idx,
            len(corrected), len(gt_elems),
        )

        after_tp += met_after["tp"]
        after_fp += met_after["fp"]
        after_fn += met_after["fn"]
        after_n_pred += met_after["n_pred"]
        after_n_gt += met_after["n_gt"]

        per_image_after.append({
            "image_id": vlm_path.stem,
            **met_after,
            "n_proposals": n_proposals_this,
        })

        n_images += 1

    dt = time.time() - t0
    logger.info(
        "Processed %d images (%d skipped, %d no-graph) in %.1fs",
        n_images, n_skipped, n_no_graph, dt,
    )

    if n_images == 0:
        logger.error("No images processed!")
        return

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    before_agg = compute_metrics(
        [(0, 0)] * before_tp,
        [0] * before_fp,
        [0] * before_fn,
        before_n_pred,
        before_n_gt,
    )
    before_agg["tp"] = before_tp
    before_agg["fp"] = before_fp
    before_agg["fn"] = before_fn

    after_agg = compute_metrics(
        [(0, 0)] * after_tp,
        [0] * after_fp,
        [0] * after_fn,
        after_n_pred,
        after_n_gt,
    )
    after_agg["tp"] = after_tp
    after_agg["fp"] = after_fp
    after_agg["fn"] = after_fn

    # Per-image averages
    avg_prec_before = sum(m["precision"] for m in per_image_before) / n_images
    avg_rec_before = sum(m["recall"] for m in per_image_before) / n_images
    avg_f1_before = sum(m["f1"] for m in per_image_before) / n_images

    avg_prec_after = sum(m["precision"] for m in per_image_after) / n_images
    avg_rec_after = sum(m["recall"] for m in per_image_after) / n_images
    avg_f1_after = sum(m["f1"] for m in per_image_after) / n_images

    # Delta
    d_prec = after_agg["precision"] - before_agg["precision"]
    d_rec = after_agg["recall"] - before_agg["recall"]
    d_f1 = after_agg["f1"] - before_agg["f1"]

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("  REAL VLM COMPLETION PIPELINE — BEFORE vs AFTER GNN CORRECTION")
    print("=" * 75)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    print(f"  Images: {n_images} processed, {n_skipped} skipped")
    print(f"  GT elements: {before_n_gt}  |  VLM elements: {before_n_pred}")
    print()

    table_header = f"  {'':30s} {'Before (VLM only)':>18s} {'After (VLM+GNN)':>18s} {'Δ':>10s}"
    table_sep = f"  {'─' * 30:30s} {'─' * 18:>18s} {'─' * 18:>18s} {'─' * 10:>10s}"

    print(f"  Aggregated (pooled over all images):")
    print(table_header)
    print(table_sep)
    print(f"  {'Precision':30s} {before_agg['precision']:>10.4f}   {after_agg['precision']:>10.4f}   {d_prec:>+9.4f}")
    print(f"  {'Recall':30s} {before_agg['recall']:>10.4f}   {after_agg['recall']:>10.4f}   {d_rec:>+9.4f}")
    print(f"  {'F1':30s} {before_agg['f1']:>10.4f}   {after_agg['f1']:>10.4f}   {d_f1:>+9.4f}")
    print(f"  {'TP count':30s} {before_agg['tp']:>10d}   {after_agg['tp']:>10d}")
    print(f"  {'FP count':30s} {before_agg['fp']:>10d}   {after_agg['fp']:>10d}")
    print(f"  {'FN count':30s} {before_agg['fn']:>10d}   {after_agg['fn']:>10d}")

    print()
    print(f"  Correction mechanics:")
    print(f"  {'Total VLM elements':35s} {n_total_vlm_elems}")
    print(f"  {'Proposals added (after NMS)':35s} {n_proposals_total}")
    print(f"  {'Corrected element count':35s} {after_n_pred}")
    print(f"  {'GT elements total':35s} {before_n_gt}")

    # Per-image average table
    print()
    print(f"  Per-image averages ({n_images} images):")
    print(table_header)
    print(table_sep)
    print(f"  {'Precision (avg)':30s} {avg_prec_before:>10.4f}   {avg_prec_after:>10.4f}   {avg_prec_after - avg_prec_before:>+9.4f}")
    print(f"  {'Recall (avg)':30s} {avg_rec_before:>10.4f}   {avg_rec_after:>10.4f}   {avg_rec_after - avg_rec_before:>+9.4f}")
    print(f"  {'F1 (avg)':30s} {avg_f1_before:>10.4f}   {avg_f1_after:>10.4f}   {avg_f1_after - avg_f1_before:>+9.4f}")

    print()
    print("=" * 75)
    print(f"  Summary: Precision {before_agg['precision']:.4f} → {after_agg['precision']:.4f} ({d_prec:+.4f})")
    print(f"           Recall    {before_agg['recall']:.4f} → {after_agg['recall']:.4f} ({d_rec:+.4f})")
    print(f"           F1        {before_agg['f1']:.4f} → {after_agg['f1']:.4f} ({d_f1:+.4f})")
    print("=" * 75)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    summary = {
        "checkpoint": str(CHECKPOINT_PATH),
        "n_images": n_images,
        "n_skipped": n_skipped,
        "model_config": {"hidden_dim": 128, "dropout": 0.0},
        "matching": {"metric": "center_distance", "threshold": 0.1},
        "before": {
            "precision": before_agg["precision"],
            "recall": before_agg["recall"],
            "f1": before_agg["f1"],
            "precision_per_img_avg": avg_prec_before,
            "recall_per_img_avg": avg_rec_before,
            "f1_per_img_avg": avg_f1_before,
            "tp": before_agg["tp"],
            "fp": before_agg["fp"],
            "fn": before_agg["fn"],
            "n_pred": before_agg["n_pred"],
            "n_gt": before_agg["n_gt"],
        },
        "after": {
            "precision": after_agg["precision"],
            "recall": after_agg["recall"],
            "f1": after_agg["f1"],
            "precision_per_img_avg": avg_prec_after,
            "recall_per_img_avg": avg_rec_after,
            "f1_per_img_avg": avg_f1_after,
            "tp": after_agg["tp"],
            "fp": after_agg["fp"],
            "fn": after_agg["fn"],
            "n_pred": after_agg["n_pred"],
            "n_gt": after_agg["n_gt"],
        },
        "delta": {
            "precision": d_prec,
            "recall": d_rec,
            "f1": d_f1,
        },
        "correction_mechanics": {
            "total_vlm_elements": n_total_vlm_elems,
            "proposals_added_nms": n_proposals_total,
            "corrected_total": after_n_pred,
        },
    }
    out_dir = Path("experiments/vlm_completion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pipeline_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved comparison results to %s", out_path)

    per_img_path = out_dir / "pipeline_per_image.json"
    with open(per_img_path, "w") as f:
        json.dump({
            "before": per_image_before,
            "after": per_image_after,
        }, f, indent=2)
    logger.info("Saved per-image details to %s", per_img_path)


if __name__ == "__main__":
    main()
