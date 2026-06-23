#!/usr/bin/env python3
"""Evaluate VLM completion: test how many missed elements the GNN can recover.

Process:
  1. Load VLM predictions and corresponding RICO GT
  2. Match VLM→GT via Hungarian (IoU ≥ 0.3) → identify missed elements
  3. Build constraint graph from detected (matched) elements using
     the same logic as build_violation_graph: GT constraints with
     < 2 detected participants are "violated"
  4. Run trained model → get violation scores + proposals
  5. Compare against baselines (center, nearest-neighbor)

Usage:
  python scripts/evaluate_vlm_completion.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode, ConstraintNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from bipartite_gnn_gui.model.heads import N_TYPES
from bipartite_gnn_gui.utils.bbox import compute_iou
from scripts.run_experiment import (
    DEVICE,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
    _normalize_label,
)

logger = logging.getLogger(__name__)

# Same type mapping as train_violation.py
TYPE_TO_IDX: dict[str, int] = {
    "button": 0, "text": 1, "icon": 2, "image": 3,
    "input": 4, "container": 5, "list": 6,
}
TYPE_UNKNOWN_IDX: int = 7

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _type_idx(label: str) -> int:
    return TYPE_TO_IDX.get(label.lower(), TYPE_UNKNOWN_IDX)


def bbox_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy → cxcywh for (N, 4) tensor."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)


def batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for two sets of boxes in xyxy format (same length)."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-8)


def match_vlm_to_gt(
    vlm_elements: list[ElementNode],
    gt_elements: list[ElementNode],
    iou_thresh: float = 0.3,
) -> Tuple[list[int], list[int]]:
    """Hungarian match VLM→GT elements.

    Returns:
        gt_matched: List of GT indices that were detected (len = num matched).
        gt_missed: List of GT indices that VLM missed.
    """
    if not vlm_elements or not gt_elements:
        return [], list(range(len(gt_elements)))

    vlm_boxes = torch.tensor([e.bbox for e in vlm_elements], dtype=torch.float32)
    gt_boxes = torch.tensor([e.bbox for e in gt_elements], dtype=torch.float32)

    iou_matrix = compute_iou(vlm_boxes, gt_boxes)
    LARGE = 1e9
    cost = 1.0 - iou_matrix
    cost[iou_matrix < iou_thresh] = LARGE

    row_ind, col_ind = linear_sum_assignment(cost.numpy())

    matched_rows = set()
    matched_cols = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < LARGE / 2:
            matched_rows.add(int(i))
            matched_cols.add(int(j))

    gt_matched = sorted(matched_cols)
    gt_missed = [j for j in range(len(gt_elements)) if j not in matched_cols]
    return gt_matched, gt_missed


# ---------------------------------------------------------------------------
# VLM-specific graph builder (mirrors build_violation_graph from train_violation)
# ---------------------------------------------------------------------------


def build_vlm_violation_graph(
    gt_elements: list[ElementNode],
    vlm_elements: list[ElementNode],
    gt_matched_indices: list[int],  # GT indices VLM detected
    builder: BipartiteGraphBuilder,
) -> Tuple[Any, Dict[str, torch.Tensor], list[int], list[int]] | None:
    """Build a graph from VLM-detected elements with violation labels.

    Mirrors train_violation.build_violation_graph:
      - Full constraint extraction from ALL GT elements
      - Keep constraints with ≥ 1 VLM-detected participant
      - Label violation = 1 if < 2 VLM-detected participants
      - Proposal target = avg bbox of missed GT elements for violated constraints

    Returns:
        (hetero_data, targets, gt_matched_indices, gt_missed_indices)
        or None if degenerate.
    """
    N = len(gt_elements)
    if N < 3:
        return None

    # Full constraint set from GT
    full_constraints = extract_all_constraints(gt_elements)
    if len(full_constraints) == 0:
        return None

    # Build old→new index mapping for survivors (matched GT indices)
    survivor_set = set(gt_matched_indices)
    missed_set = set(range(N)) - survivor_set
    survivor_list = sorted(survivor_set)

    if len(survivor_list) < 2:
        return None

    old_to_new = {old: new for new, old in enumerate(survivor_list)}
    N_surv = len(survivor_list)

    # Build VLM element list for survivors (use VLM bboxes for matched)
    # Map: matched GT index → VLM element (we need to find which VLM element matched)
    gt_to_vlm: dict[int, int] = {}
    vlm_boxes = torch.tensor([e.bbox for e in vlm_elements], dtype=torch.float32)
    gt_boxes = torch.tensor([e.bbox for e in gt_elements], dtype=torch.float32)
    iou_matrix = compute_iou(vlm_boxes, gt_boxes)
    LARGE = 1e9
    cost = 1.0 - iou_matrix
    cost[iou_matrix < 0.3] = LARGE
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < LARGE / 2:
            gt_to_vlm[j] = i

    # Build survivor elements: use VLM element if available, else GT element
    surv_elements: list[ElementNode] = []
    for gt_idx in survivor_list:
        if gt_idx in gt_to_vlm:
            surv_elements.append(vlm_elements[gt_to_vlm[gt_idx]])
        else:
            surv_elements.append(gt_elements[gt_idx])

    # Filter constraints + compute violation labels and proposal targets
    kept_constraints: list[ConstraintNode] = []
    violation_labels: list[float] = []
    proposal_targets: list[list[float]] = []  # (N_con, 5) — bbox xyxy + type_idx

    for con in full_constraints:
        all_participants = set(con.source_indices + con.target_indices)
        surviving_participants = all_participants & survivor_set
        missing_participants = all_participants - survivor_set

        n_surviving = len(surviving_participants)
        if n_surviving < 1:
            continue  # Constraint has no surviving participants — drop it

        # Violation: constraint is incomplete (< 2 participants)
        is_violated = n_surviving < 2
        violation_labels.append(1.0 if is_violated else 0.0)

        # Proposal target: avg bbox of missed participants
        if is_violated and missing_participants:
            missed_list = list(missing_participants)
            x1s = [gt_elements[i].bbox[0] for i in missed_list]
            y1s = [gt_elements[i].bbox[1] for i in missed_list]
            x2s = [gt_elements[i].bbox[2] for i in missed_list]
            y2s = [gt_elements[i].bbox[3] for i in missed_list]
            type_idx = _type_idx(gt_elements[missed_list[0]].label)
            proposal_targets.append([
                sum(x1s) / len(x1s), sum(y1s) / len(y1s),
                sum(x2s) / len(x2s), sum(y2s) / len(y2s),
                float(type_idx),
            ])
        else:
            proposal_targets.append([0.0, 0.0, 0.0, 0.0, 0.0])

        # Remap constraint indices to survivor space
        con.source_indices = sorted([old_to_new[i] for i in surviving_participants])
        con.target_indices = sorted([old_to_new[i] for i in surviving_participants])
        kept_constraints.append(con)

    if len(kept_constraints) == 0:
        return None

    # Build graph
    hetero_data = builder.build(surv_elements, kept_constraints)

    N_con = len(kept_constraints)
    gt_boxes_surv = torch.tensor(
        [[e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]] for e in surv_elements],
        dtype=torch.float32,
    )

    targets = {
        "coord": torch.zeros((N_surv, 4), dtype=torch.float32),
        "violation": torch.tensor(violation_labels, dtype=torch.float32).view(-1, 1),
        "existence": torch.ones((N_surv, 1), dtype=torch.float32),
        "gt_boxes": bbox_xyxy_to_xywh(gt_boxes_surv),
        "proposal_target": torch.tensor(proposal_targets, dtype=torch.float32),  # (N_con, 5)
        "proposal_violation_mask": torch.tensor(
            [v > 0.5 for v in violation_labels], dtype=torch.bool
        ),
        "proposal_type_target": torch.tensor(
            [int(t[4]) for t in proposal_targets], dtype=torch.long
        ),
    }
    return hetero_data, targets, gt_matched_indices, list(missed_set)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def baseline_nearest_neighbor(
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Copy the closest surviving element's bbox as the proposal."""
    if "gt_boxes" not in targets or "proposal_target" not in targets:
        return {"mse": 0.0, "iou": 0.0}
    gt_boxes = targets["gt_boxes"]  # (N_surv, 4) xywh
    mask = targets.get("proposal_violation_mask", torch.zeros(0, dtype=torch.bool))
    prop_tgt = targets["proposal_target"]  # (N_con, 5) xyxy + type
    if mask.sum() == 0 or gt_boxes.shape[0] < 2:
        return {"mse": 0.0, "iou": 0.0}

    p_masked = prop_tgt[mask][:, :4]  # (N_violated, 4) xyxy
    gt_xyxy = torch.stack([
        gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
        gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
        gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
        gt_boxes[:, 1] + gt_boxes[:, 3] / 2,
    ], dim=1)  # (N_surv, 4) xyxy

    p_compat = p_masked[:, None, :]  # (N_v, 1, 4)
    gt_compat = gt_xyxy[None, :, :]  # (1, N_s, 4)
    dists = (p_compat - gt_compat).abs().sum(dim=2)  # (N_v, N_s)
    nearest_idx = dists.argmin(dim=1)
    nearest_boxes = gt_xyxy[nearest_idx]

    mse = F.mse_loss(nearest_boxes, p_masked).item()
    iou_val = batch_iou(nearest_boxes, p_masked).mean().item()
    return {"mse": mse, "iou": iou_val}


def baseline_center(targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Always predict layout center as the missing element."""
    if "proposal_target" not in targets:
        return {"mse": 0.0, "iou": 0.0}
    prop_tgt = targets.get("proposal_target", torch.zeros(0, 5))
    mask = targets.get("proposal_violation_mask", torch.zeros(0, dtype=torch.bool))
    if mask.sum() == 0:
        return {"mse": 0.0, "iou": 0.0}

    p_masked = prop_tgt[mask][:, :4]  # (N_violated, 4) xyxy
    cx, cy = 0.5, 0.5
    w, h = 0.05, 0.05
    center_box = torch.tensor(
        [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]],
        dtype=torch.float32,
    ).expand(mask.sum(), -1)

    mse = F.mse_loss(center_box, p_masked).item()
    iou_val = batch_iou(center_box, p_masked).mean().item()
    return {"mse": mse, "iou": iou_val}


# ---------------------------------------------------------------------------
# GNN evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_gnn(
    model: BipartiteGNNCorrector,
    graphs_cache: List[Tuple[Any, Dict[str, torch.Tensor]]],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate violation accuracy, proposal MSE, and proposal IoU."""
    model.eval()
    v_acc_all: List[torch.Tensor] = []
    mse_all: List[torch.Tensor] = []
    iou_all: List[torch.Tensor] = []
    iou_recall_count = 0
    iou_recall_total = 0

    for data, targets in graphs_cache:
        data = data.to(device)
        t = {k: v.to(device) for k, v in targets.items()}
        preds = model(data)

        # Violation accuracy
        if "violation" in preds and "violation" in t:
            v_acc_all.append(
                ((preds["violation"].view(-1) > 0.5)
                 == (t["violation"].view(-1) > 0.5)).float()
            )

        # Proposal
        mask = t.get("proposal_violation_mask", torch.zeros(0, dtype=torch.bool))
        if "proposal" in preds and mask.sum() > 0:
            p_masked = preds["proposal"][mask]  # (N_v, 12) → first 4 = bbox xyxy
            p_bbox = p_masked[:, :4]
            tgt_masked = t["proposal_target"][mask, :4]  # (N_v, 4) xyxy
            mse_all.append(F.mse_loss(p_bbox, tgt_masked))
            iou_vals = batch_iou(p_bbox, tgt_masked)
            iou_all.append(iou_vals.mean())
            iou_recall_count += (iou_vals > 0.3).sum().item()
            iou_recall_total += iou_vals.numel()

    result = {
        "violation_acc": float(torch.cat(v_acc_all).mean().cpu()) if v_acc_all else 0.0,
        "proposal_mse": float(torch.stack(mse_all).mean().cpu()) if mse_all else 0.0,
        "proposal_iou": float(torch.stack(iou_all).mean().cpu()) if iou_all else 0.0,
        "proposal_recall@0.3": float(iou_recall_count / max(iou_recall_total, 1)),
        "n_violated": iou_recall_total,
    }
    return result


@torch.no_grad()
def evaluate_baselines(
    graphs_cache: List[Tuple[Any, Dict[str, torch.Tensor]]],
) -> Dict[str, float]:
    """Evaluate baselines over all graphs."""
    nn_mse_all: List[float] = []
    nn_iou_all: List[float] = []
    center_mse_all: List[float] = []
    center_iou_all: List[float] = []
    nn_recall_count = 0
    nn_recall_total = 0
    center_recall_count = 0
    center_recall_total = 0

    for data, targets in graphs_cache:
        nn_res = baseline_nearest_neighbor(targets)
        nn_mse_all.append(nn_res["mse"])
        nn_iou_all.append(nn_res["iou"])

        center_res = baseline_center(targets)
        center_mse_all.append(center_res["mse"])
        center_iou_all.append(center_res["iou"])

    return {
        "nn_mse": float(torch.tensor(nn_mse_all).mean()),
        "nn_iou": float(torch.tensor(nn_iou_all).mean()),
        "center_mse": float(torch.tensor(center_mse_all).mean()),
        "center_iou": float(torch.tensor(center_iou_all).mean()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Paths
    vlm_dir = Path("data/vlm_predictions/rico_qwen_flash")
    rico_dir = Path("data/rico_local/combined")
    checkpoint_path = Path("checkpoints/violation_detection/best_model.pt")
    output_dir = Path("experiments/vlm_completion")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("VLM COMPLETION EVALUATION — Qwen3-VL Flash on RICO")
    logger.info("=" * 60)
    logger.info("VLM predictions: %s", vlm_dir)
    logger.info("RICO GT: %s", rico_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Device: %s", DEVICE)

    # Discover VLM prediction files
    vlm_files = sorted(vlm_dir.glob("*.json"))
    logger.info("Found %d VLM prediction files", len(vlm_files))

    # Load model
    logger.info("Loading model...")
    model = BipartiteGNNCorrector(
        hidden_dim=16, dropout=0.1,
        coord_weight=0.0, existence_weight=0.0,
    ).to(DEVICE)
    model.loss_fn.violation_weight = 1.0
    model.loss_fn.coord_weight = 0.0
    model.loss_fn.existence_weight = 0.0
    model.loss_fn.alignment_weight = 0.0
    model.proposal_weight = 1.0
    model.proposal_type_weight = 0.5

    state = torch.load(str(checkpoint_path), map_location="cpu")
    # Checkpoint is plain state_dict (OrderedDict), not wrapped
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded (hidden_dim=16 from checkpoint)")

    # Process each VLM file
    builder = BipartiteGraphBuilder()
    graphs_cache: List[Tuple[Any, Dict[str, torch.Tensor]]] = []
    per_image_results: List[Dict[str, Any]] = []

    n_total = 0
    n_skipped = 0
    n_gt_elems = 0
    n_vlm_elems = 0
    n_missed_elems = 0
    t0 = time.time()

    for idx, vlm_path in enumerate(vlm_files):
        if idx % 50 == 0 and idx > 0:
            dt = time.time() - t0
            logger.info("  Processed %d/%d (%.1f files/s)", idx, len(vlm_files), idx / max(dt, 0.01))

        image_id = vlm_path.stem  # e.g. "0", "1", etc.

        # 1. Load VLM predictions
        try:
            vlm_data = json.loads(vlm_path.read_text())
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue

        vlm_raw_elements = vlm_data.get("elements", [])
        img_w = vlm_data.get("image_width", 1)
        img_h = vlm_data.get("image_height", 1)

        # Convert VLM to ElementNode (normalize pixel coords to [0,1])
        vlm_elements: list[ElementNode] = []
        for item in vlm_raw_elements:
            bbox = item.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = map(float, bbox)
            x1, x2 = x1 / max(img_w, 1), x2 / max(img_w, 1)
            y1, y2 = y1 / max(img_h, 1), y2 / max(img_h, 1)
            label = _normalize_label(item.get("label", "other"))
            vlm_elements.append(ElementNode(
                bbox=[x1, y1, x2, y2],
                label=label,
                confidence=1.0,
            ))

        # 2. Load RICO GT
        rico_path = rico_dir / f"{image_id}.json"
        if not rico_path.exists():
            n_skipped += 1
            continue

        parsed = parse_rico_vh(rico_path)
        if parsed is None:
            n_skipped += 1
            continue

        rico_w, rico_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, rico_w, rico_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        if len(gt_elements) < 3 or len(vlm_elements) < 1:
            n_skipped += 1
            continue

        # 3. Match VLM→GT
        gt_matched, gt_missed = match_vlm_to_gt(vlm_elements, gt_elements, iou_thresh=0.3)

        n_total += 1
        n_gt_elems += len(gt_elements)
        n_vlm_elems += len(vlm_elements)
        n_missed_elems += len(gt_missed)

        # 4. Build violation graph from VLM detections
        result = build_vlm_violation_graph(
            gt_elements, vlm_elements, gt_matched, builder
        )
        if result is None:
            n_skipped += 1
            continue

        hetero_data, targets, _, _ = result
        graphs_cache.append((hetero_data, targets))

        per_image_results.append({
            "image_id": image_id,
            "n_gt": len(gt_elements),
            "n_vlm": len(vlm_elements),
            "n_matched": len(gt_matched),
            "n_missed": len(gt_missed),
            "n_constraints": hetero_data["constraint"].x.shape[0],
            "n_violated": int(targets["violation"].sum().item()),
        })

    dt = time.time() - t0
    logger.info("Processed %d valid graphs (%d skipped) in %.1fs",
                len(graphs_cache), n_skipped, dt)

    if len(graphs_cache) == 0:
        logger.error("No valid graphs to evaluate!")
        return

    # Summary stats
    avg_gt = n_gt_elems / max(n_total, 1)
    avg_vlm = n_vlm_elems / max(n_total, 1)
    avg_missed = n_missed_elems / max(n_total, 1)

    # 5. Run GNN evaluation
    logger.info("Running GNN evaluation on %d graphs...", len(graphs_cache))
    gnn_metrics = evaluate_gnn(model, graphs_cache, DEVICE)

    # 6. Baselines
    logger.info("Evaluating baselines...")
    baseline_metrics = evaluate_baselines(graphs_cache)

    # 7. Print results
    print()
    print("=" * 60)
    print("=== VLM Completion Evaluation ===")
    print("=" * 60)
    print(f"Images:              {n_total}")
    print(f"VLM elements:        avg {avg_vlm:.1f}/img (vs GT {avg_gt:.1f}/img)")
    print(f"Missed elements:     avg {avg_missed:.1f}/img")
    print(f"Violated constraints: avg {sum(r['n_violated'] for r in per_image_results) / max(len(per_image_results), 1):.1f}/img")
    print()
    print(f"Violation detection accuracy: {gnn_metrics['violation_acc']*100:.1f}%")
    print(f"Proposal recall@IoU=0.3:      {gnn_metrics['proposal_recall@0.3']*100:.1f}%")
    print(f"Proposal Avg IoU:             {gnn_metrics['proposal_iou']:.4f}")
    print(f"Proposal MSE:                 {gnn_metrics['proposal_mse']:.6f}")
    print()
    print("--- Baselines ---")
    print(f"NearestNeighbor MSE: {baseline_metrics['nn_mse']:.6f}")
    print(f"NearestNeighbor IoU: {baseline_metrics['nn_iou']:.4f}")
    print(f"Center MSE:          {baseline_metrics['center_mse']:.6f}")
    print(f"Center IoU:          {baseline_metrics['center_iou']:.4f}")
    print("=" * 60)

    # Save per-image results
    output_path = output_dir / "per_image_results.json"
    with open(output_path, "w") as f:
        json.dump(per_image_results, f, indent=2)
    logger.info("Saved per-image results to %s", output_path)

    # Save summary
    summary = {
        "n_images": n_total,
        "avg_gt_elements": avg_gt,
        "avg_vlm_elements": avg_vlm,
        "avg_missed_elements": avg_missed,
        **gnn_metrics,
        **baseline_metrics,
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary to %s", summary_path)


if __name__ == "__main__":
    main()
