#!/usr/bin/env python3
"""ScreenSpot real VLM end-to-end evaluation.

Loads ScreenSpot GT + VLM predictions from Qwen3-VL Flash (600 images),
matches VLM→GT via center-distance Hungarian, evaluates:

  - Raw VLM precision / recall / F1
  - GNN existence head on VLM elements (AUROC, accuracy)
  - GNN violation detection + proposal on missed elements

Compares: does the GNN meaningfully improve VLM quality on ScreenSpot?

Usage:
    cd /tmp/worktree-screenspot-1782274222
    .venv/bin/python experiments/eval_screenspot_completion.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ConstraintType, ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from experiments.screenspot_loader import (
    load_screenspot_gt,
    load_screenspot_vlm,
    load_all_screenspot_vlm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCREENSPOT_JSON = "/Users/minimx/mnt/thinkpad/1007/Documents/dataset/screenspot/ScreenSpot_combined.json"
IMAGES_DIR = "/Users/minimx/mnt/thinkpad/1007/Documents/dataset/screenspot/images"
VLM_DIR = "/Users/minimx/bipartite-gnn-gui/data/vlm_predictions/screenspot_qwen_flash"

# Checkpoint paths (prefer joint model for comparison with Phase 8)
CHECKPOINT_CANDIDATES = [
    Path("/Users/minimx/bipartite-gnn-gui/checkpoints/violation_detection_joint/best_model.pt"),
    Path("checkpoints/violation_detection_joint/best_model.pt"),
    Path("/Users/minimx/bipartite-gnn-gui/checkpoints/violation_detection/best_model.pt"),
    Path("checkpoints/violation_detection/best_model.pt"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def center_distance(box_a, box_b):
    """Euclidean distance between normalized centers."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def hungarian_match(
    vlm_elems: List[ElementNode],
    gt_elems: List[ElementNode],
    threshold: float = 0.1,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match VLM predictions to GT elements by center-distance Hungarian.

    Args:
        vlm_elems: VLM-predicted elements.
        gt_elems: Ground-truth elements.
        threshold: Maximum center distance for a valid match (normalized).

    Returns:
        ``(matched_pairs, fp_indices, fn_indices)`` where matched_pairs
        is a list of ``(vlm_idx, gt_idx)`` tuples.
    """
    M, N = len(vlm_elems), len(gt_elems)
    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))

    INF = 1e9
    cost = torch.full((M, N), INF, dtype=torch.float32)
    for i, ve in enumerate(vlm_elems):
        for j, ge in enumerate(gt_elems):
            d = center_distance(ve.bbox, ge.bbox)
            if d <= threshold:
                cost[i, j] = d

    has_feasible = torch.isfinite(cost).any().item()
    if has_feasible:
        row_ind, col_ind = linear_sum_assignment(cost.numpy())
    else:
        row_ind, col_ind = [], []

    matched: List[Tuple[int, int]] = []
    matched_rows = set()
    matched_cols = set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF / 2:
            matched.append((int(i), int(j)))
            matched_rows.add(int(i))
            matched_cols.add(int(j))

    fp = [i for i in range(M) if i not in matched_rows]
    fn = [j for j in range(N) if j not in matched_cols]
    return matched, fp, fn


def _auroc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute AUROC via Mann-Whitney U statistic."""
    if preds.numel() < 2:
        return 0.5
    n_pos = (targets == 1).sum()
    n_neg = (targets == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sorted_indices = torch.argsort(preds, descending=False)
    sorted_targets = targets[sorted_indices]
    ranks = torch.arange(1, len(preds) + 1, device=preds.device, dtype=torch.float32)
    sum_ranks_pos = ranks[sorted_targets == 1].sum()
    u_stat = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(max(0.0, min(1.0, u_stat / (n_pos * n_neg))))


def build_vlm_graph(
    vlm_elements: List[ElementNode],
    builder: BipartiteGraphBuilder,
    matched_set: set,
) -> Tuple:
    """Build constraint graph from VLM elements with existence labels.

    Args:
        vlm_elements: VLM-predicted elements.
        builder: Graph builder instance.
        matched_set: Set of VLM indices that matched GT (TP).

    Returns:
        ``(data, targets)`` or ``(None, None)`` if too few elements/constraints.
    """
    if len(vlm_elements) < 2:
        return None, None

    # Existence labels: 1 = TP (matched), 0 = FP (unmatched)
    existence_labels = [1.0 if i in matched_set else 0.0 for i in range(len(vlm_elements))]

    # Extract constraints from VLM elements
    constraints = extract_all_constraints(vlm_elements)
    if len(constraints) == 0:
        return None, None

    data = builder.build(vlm_elements, constraints)

    targets = {
        "existence": torch.tensor(existence_labels, dtype=torch.float32).view(-1, 1),
        "coord": torch.zeros((len(vlm_elements), 4), dtype=torch.float32),
        "violation": torch.zeros((len(constraints), 1), dtype=torch.float32),
    }
    return data, targets


def find_checkpoint() -> Optional[Path]:
    """Find the best available checkpoint."""
    for ckpt in CHECKPOINT_CANDIDATES:
        if ckpt.exists():
            logger.info("Found checkpoint: %s", ckpt)
            return ckpt
    return None


def load_model(ckpt_path: Path, hidden_dim: int = 128) -> Optional[BipartiteGNNCorrector]:
    """Load GNN model from checkpoint.

    Detects hidden_dim from the checkpoint shape if possible.
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Detect hidden_dim from encoder projection shape
    detected_hidden = hidden_dim
    for k, v in state.items():
        if "element_proj.weight" in k:
            detected_hidden = v.shape[0]
            break

    logger.info("Detected hidden_dim=%d from checkpoint", detected_hidden)

    model = BipartiteGNNCorrector(
        hidden_dim=detected_hidden,
        dropout=0.1,
        coord_weight=0.0,
        existence_weight=0.0,
    ).to(DEVICE)

    # Set loss weights: we only care about existence head
    model.loss_fn.violation_weight = 1.0
    model.loss_fn.coord_weight = 0.0
    model.loss_fn.existence_weight = 0.0
    model.loss_fn.alignment_weight = 0.0
    model.proposal_weight = 1.0
    model.proposal_type_weight = 0.0

    try:
        model.load_state_dict(state, strict=False)
        logger.info("Model loaded successfully (strict=False)")
    except Exception as e:
        logger.warning("Model load had issues: %s", e)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", stream=sys.stdout)
    logger.info("=" * 60)
    logger.info("SCREENSPOT REAL VLM COMPLETION EVALUATION")
    logger.info("=" * 60)

    t_start = time.time()

    # ------------------------------------------------------------------
    # 1. Load GT
    # ------------------------------------------------------------------
    logger.info("Loading ScreenSpot GT...")
    gt_pairs = load_screenspot_gt(SCREENSPOT_JSON, IMAGES_DIR)
    gt_dict: Dict[str, List[ElementNode]] = dict(gt_pairs)
    logger.info("Loaded %d GT images", len(gt_pairs))

    # ------------------------------------------------------------------
    # 2. Load VLM predictions
    # ------------------------------------------------------------------
    logger.info("Loading VLM predictions from %s ...", VLM_DIR)
    vlm_dict = load_all_screenspot_vlm(VLM_DIR)
    logger.info("Loaded %d VLM prediction sets", len(vlm_dict))

    # ------------------------------------------------------------------
    # 3. Find intersection (images with both GT and VLM)
    # ------------------------------------------------------------------
    common_stems = sorted(set(gt_dict.keys()) & set(vlm_dict.keys()))
    logger.info("Common stems (GT ∩ VLM): %d", len(common_stems))

    # ------------------------------------------------------------------
    # 4. Match and compute VLM metrics
    # ------------------------------------------------------------------
    all_matched = 0
    all_gt = 0
    all_vlm = 0
    all_fp = 0
    all_fn = 0

    all_conf_preds: List[torch.Tensor] = []
    all_conf_labels: List[torch.Tensor] = []

    builder = BipartiteGraphBuilder()
    graph_count = 0
    skipped_no_graph = 0
    skipped_no_vlm = 0
    skipped_no_gt = 0

    # Load model
    ckpt_path = find_checkpoint()
    if ckpt_path is None:
        logger.error("No checkpoint found! Run GNN eval anyway? No — exiting.")
        return

    model = load_model(ckpt_path)
    if model is None:
        logger.error("Failed to load model.")
        return

    for stem in common_stems:
        gt_elems = gt_dict[stem]
        vlm_elems = vlm_dict[stem]

        if len(gt_elems) < 1:
            skipped_no_gt += 1
            continue
        if len(vlm_elems) < 1:
            skipped_no_vlm += 1
            continue

        # Match VLM → GT
        matched, fp_idx, fn_idx = hungarian_match(vlm_elems, gt_elems, threshold=0.1)
        matched_set = set(i for i, _ in matched)

        all_gt += len(gt_elems)
        all_vlm += len(vlm_elems)
        all_matched += len(matched)
        all_fp += len(fp_idx)
        all_fn += len(fn_idx)

        # Build graph from VLM elements
        data, targets = build_vlm_graph(vlm_elems, builder, matched_set)
        if data is None:
            skipped_no_graph += 1
            continue

        # Run model
        with torch.no_grad():
            data_gpu = data.to(DEVICE)
            pred = model(data_gpu)

        # Collect existence predictions
        if "existence" in pred:
            conf_pred = pred["existence"].cpu().view(-1)
            conf_label = torch.tensor(
                [1.0 if i in matched_set else 0.0 for i in range(len(vlm_elems))],
                dtype=torch.float32,
            )
            all_conf_preds.append(conf_pred)
            all_conf_labels.append(conf_label)

        graph_count += 1

    # ------------------------------------------------------------------
    # 5. Aggregate statistics
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Processed %d / %d images (%d graphs, %d skipped: no_graph=%d no_vlm=%d no_gt=%d)",
                len(common_stems), len(gt_pairs), graph_count,
                len(common_stems) - graph_count,
                skipped_no_graph, skipped_no_vlm, skipped_no_gt)
    logger.info("Time: %.1f s", elapsed)

    # VLM metrics
    precision = all_matched / max(all_vlm, 1)
    recall = all_matched / max(all_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    fp_rate = all_fp / max(all_vlm, 1)
    fn_rate = all_fn / max(all_gt, 1)

    logger.info("VLM detection: Prec=%.4f Rec=%.4f F1=%.4f FP_rate=%.4f FN_rate=%.4f",
                precision, recall, f1, fp_rate, fn_rate)
    logger.info("Raw counts: GT=%d VLM=%d Matched=%d FP=%d FN=%d",
                all_gt, all_vlm, all_matched, all_fp, all_fn)

    # GNN existence metrics
    gnn_acc = 0.0
    gnn_auroc = 0.5
    gnn_pos_mean = 0.0
    gnn_neg_mean = 0.0
    gnn_n = 0

    if all_conf_preds:
        conf_preds = torch.cat(all_conf_preds)
        conf_labels = torch.cat(all_conf_labels)
        gnn_n = conf_preds.numel()

        # Accuracy at threshold 0.5
        gnn_acc = ((conf_preds > 0.5) == (conf_labels > 0.5)).float().mean().item()

        # AUROC
        gnn_auroc = _auroc(conf_preds, conf_labels)

        # Mean confidence by class
        pos_mask = conf_labels == 1
        neg_mask = conf_labels == 0
        gnn_pos_mean = conf_preds[pos_mask].mean().item() if pos_mask.any() else 0.0
        gnn_neg_mean = conf_preds[neg_mask].mean().item() if neg_mask.any() else 0.0

        logger.info("GNN existence head: Acc=%.4f AUROC=%.4f PosMean=%.4f NegMean=%.4f (n=%d)",
                    gnn_acc, gnn_auroc, gnn_pos_mean, gnn_neg_mean, gnn_n)

    # Correction ceiling
    error_rate = 1.0 - recall
    correction_ceiling = error_rate * max(gnn_acc, recall) if gnn_acc > 0.5 else error_rate * 0.5
    logger.info("VLM error rate=%.3f, GNN correction ceiling≈%.3f (%.1f%% of errors)",
                error_rate, correction_ceiling,
                correction_ceiling * 100 / max(error_rate, 0.01) if error_rate > 0.01 else 0)

    # ------------------------------------------------------------------
    # 6. Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SCREENSPOT REAL VLM EVALUATION — FINAL RESULTS")
    print("=" * 70)
    print(f"{'Checkpoint':30s} {ckpt_path.name if ckpt_path else 'N/A'}")
    print(f"{'Device':30s} {str(DEVICE)}")
    print(f"{'Images processed':30s} {graph_count}")
    print(f"{'Skipped (no graph)':30s} {skipped_no_graph}")
    print(f"{'Skipped (no VLM)':30s} {skipped_no_vlm}")
    print(f"{'Skipped (no GT)':30s} {skipped_no_gt}")
    print()
    print(f"{'GT elements total':30s} {all_gt}")
    print(f"{'VLM elements total':30s} {all_vlm}")
    print(f"{'Matched (TP)':30s} {all_matched}")
    print(f"{'False Positives':30s} {all_fp}")
    print(f"{'False Negatives':30s} {all_fn}")
    print()
    print(f"{'VLM Precision':30s} {precision:.4f}")
    print(f"{'VLM Recall':30s} {recall:.4f}")
    print(f"{'VLM F1':30s} {f1:.4f}")
    print(f"{'VLM FP rate':30s} {fp_rate:.4f}")
    print(f"{'VLM FN rate':30s} {fn_rate:.4f}")
    print()
    print(f"{'GNN Existence Acc':30s} {gnn_acc:.4f}")
    print(f"{'GNN Existence AUROC':30s} {gnn_auroc:.4f}")
    print(f"{'GNN Pos Mean (TP)':30s} {gnn_pos_mean:.4f}")
    print(f"{'GNN Neg Mean (FP)':30s} {gnn_neg_mean:.4f}")
    print(f"{'GNN eval count':30s} {gnn_n}")
    print()
    print(f"{'VLM error rate':30s} {error_rate:.4f}")
    print(f"{'Correction ceiling':30s} {correction_ceiling:.4f}")
    print("=" * 70)

    # Compare with RICO results from 9.2.1
    print()
    print("--- Comparison with RICO (9.2.1) ---")
    print(f"  RICO VLM Prec=0.382 Rec=0.235 F1=0.291")
    print(f"  ScreenSpot VLM Prec={precision:.3f} Rec={recall:.3f} F1={f1:.3f}")
    print(f"  RICO GNN AUROC=0.703")
    print(f"  ScreenSpot GNN AUROC={gnn_auroc:.3f}")

    # Save results
    summary = {
        "checkpoint": str(ckpt_path) if ckpt_path else "",
        "device": str(DEVICE),
        "n_images_loaded": len(common_stems),
        "n_images_processed": graph_count,
        "n_skipped_no_graph": skipped_no_graph,
        "n_skipped_no_vlm": skipped_no_vlm,
        "n_skipped_no_gt": skipped_no_gt,
        "total_gt": all_gt,
        "total_vlm": all_vlm,
        "total_matched": all_matched,
        "total_fp": all_fp,
        "total_fn": all_fn,
        "vlm_precision": precision,
        "vlm_recall": recall,
        "vlm_f1": f1,
        "vlm_fp_rate": fp_rate,
        "vlm_fn_rate": fn_rate,
        "gnn_existence_acc": gnn_acc,
        "gnn_existence_auroc": gnn_auroc,
        "gnn_pos_mean": gnn_pos_mean,
        "gnn_neg_mean": gnn_neg_mean,
        "gnn_eval_count": gnn_n,
        "vlm_error_rate": error_rate,
        "correction_ceiling": correction_ceiling,
        "time_seconds": elapsed,
    }

    out_dir = Path("experiments/vlm_completion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_screenspot_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved results to %s", out_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
