#!/usr/bin/env python3
"""ScreenSpot cross-domain confidence evaluation.

Loads ScreenSpot GT + VLM predictions, computes confidence scores using the
real-data-trained model from checkpoints/confidence_scoring/best_model.pt,
and reports AUROC, accuracy.

Compares with RICO performance (AUROC = 0.703 from 9.2.1).

Usage:
    cd /tmp/worktree-screenspot-1782274222
    .venv/bin/python experiments/eval_screenspot_confidence.py
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
from bipartite_gnn_gui.graph.schema import ElementNode
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

# Confidence model checkpoint (real-data trained)
CONFIDENCE_CKPT = Path("/Users/minimx/bipartite-gnn-gui/checkpoints/confidence_scoring/best_model.pt")

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
    """Match VLM predictions to GT by center-distance Hungarian."""
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


def compute_metrics_at_threshold(
    preds: torch.Tensor,
    labels: torch.Tensor,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1 at various thresholds.

    Args:
        preds: Confidence predictions in [0, 1].
        labels: Binary ground-truth labels (0 or 1).
        thresholds: Thresholds to evaluate. Defaults to [0.3, 0.5, 0.7].

    Returns:
        Dict mapping metric name to value.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    results = {}
    for t in thresholds:
        bin_pred = (preds > t).float()
        tp = ((bin_pred == 1) & (labels == 1)).sum().item()
        fp = ((bin_pred == 1) & (labels == 0)).sum().item()
        fn = ((bin_pred == 0) & (labels == 1)).sum().item()
        tn = ((bin_pred == 0) & (labels == 0)).sum().item()

        acc = (tp + tn) / max(tp + fp + fn + tn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        results[f"acc@{t:.1f}"] = acc
        results[f"prec@{t:.1f}"] = prec
        results[f"rec@{t:.1f}"] = rec
        results[f"f1@{t:.1f}"] = f1

    return results


def build_confidence_graph(
    vlm_elements: List[ElementNode],
    builder: BipartiteGraphBuilder,
) -> Tuple:
    """Build a constraint graph from VLM elements (no targets needed).

    Returns:
        ``(data, None)`` or ``(None, None)`` if too few elements/constraints.
    """
    if len(vlm_elements) < 2:
        return None, None

    constraints = extract_all_constraints(vlm_elements)
    if len(constraints) == 0:
        return None, None

    data = builder.build(vlm_elements, constraints)
    return data, None


def load_confidence_model(ckpt_path: Path) -> Optional[BipartiteGNNCorrector]:
    """Load the confidence model from checkpoint.

    The confidence scoring model uses a smaller hidden_dim=16 (detected from
    the checkpoint).
    """
    if not ckpt_path.exists():
        logger.error("Confidence checkpoint not found: %s", ckpt_path)
        return None

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Detect hidden_dim from encoder projection shape
    detected_hidden = 16  # default for confidence model
    for k, v in state.items():
        if "element_proj.weight" in k:
            detected_hidden = v.shape[0]
            break

    logger.info("Confidence model: detected hidden_dim=%d", detected_hidden)

    model = BipartiteGNNCorrector(
        hidden_dim=detected_hidden,
        dropout=0.1,
        coord_weight=0.0,
        existence_weight=0.0,
    ).to(DEVICE)

    try:
        model.load_state_dict(state, strict=False)
        logger.info("Confidence model loaded successfully")
    except Exception as e:
        logger.warning("Confidence model load had issues: %s", e)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", stream=sys.stdout)
    logger.info("=" * 60)
    logger.info("SCREENSPOT CROSS-DOMAIN CONFIDENCE EVALUATION")
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
    logger.info("Loading VLM predictions...")
    vlm_dict = load_all_screenspot_vlm(VLM_DIR)
    logger.info("Loaded %d VLM prediction sets", len(vlm_dict))

    # ------------------------------------------------------------------
    # 3. Intersection
    # ------------------------------------------------------------------
    common_stems = sorted(set(gt_dict.keys()) & set(vlm_dict.keys()))
    logger.info("Common stems: %d", len(common_stems))

    # ------------------------------------------------------------------
    # 4. Load confidence model
    # ------------------------------------------------------------------
    model = load_confidence_model(CONFIDENCE_CKPT)
    if model is None:
        logger.error("Cannot proceed without confidence model.")
        return

    # ------------------------------------------------------------------
    # 5. Match and collect confidence predictions
    # ------------------------------------------------------------------
    builder = BipartiteGraphBuilder()

    all_conf_preds: List[torch.Tensor] = []
    all_conf_labels: List[torch.Tensor] = []

    # Overall matching statistics
    all_matched = 0
    all_gt = 0
    all_vlm = 0

    n_images_with_graph = 0
    n_images_no_graph = 0

    for stem in common_stems:
        gt_elems = gt_dict[stem]
        vlm_elems = vlm_dict[stem]

        if len(gt_elems) < 1 or len(vlm_elems) < 1:
            continue

        # Match VLM → GT
        matched, _, _ = hungarian_match(vlm_elems, gt_elems, threshold=0.1)
        matched_set = set(i for i, _ in matched)

        all_gt += len(gt_elems)
        all_vlm += len(vlm_elems)
        all_matched += len(matched)

        # Build graph
        data, _ = build_confidence_graph(vlm_elems, builder)
        if data is None:
            n_images_no_graph += 1
            continue

        # Run model
        with torch.no_grad():
            data_gpu = data.to(DEVICE)
            pred = model(data_gpu)

        # Existence head gives confidence scores
        if "existence" in pred:
            conf_pred = pred["existence"].cpu().view(-1)
            conf_label = torch.tensor(
                [1.0 if i in matched_set else 0.0 for i in range(len(vlm_elems))],
                dtype=torch.float32,
            )
            all_conf_preds.append(conf_pred)
            all_conf_labels.append(conf_label)

        n_images_with_graph += 1

    # ------------------------------------------------------------------
    # 6. Compute metrics
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("Processed %d stems: %d graphs built, %d no-graph skipped",
                len(common_stems), n_images_with_graph, n_images_no_graph)
    logger.info("Time: %.1f s", elapsed)

    logger.info("Matching: GT=%d, VLM=%d, Matched=%d",
                all_gt, all_vlm, all_matched)
    logger.info("VLM Prec=%.4f Rec=%.4f F1=%.4f",
                all_matched / max(all_vlm, 1),
                all_matched / max(all_gt, 1),
                2 * all_matched / max(all_vlm + all_gt, 1))

    # Aggregate confidence metrics
    auroc_val = 0.5
    acc_val = 0.0
    pos_mean = 0.0
    neg_mean = 0.0
    n_total = 0

    if all_conf_preds:
        conf_preds = torch.cat(all_conf_preds)
        conf_labels = torch.cat(all_conf_labels)
        n_total = conf_preds.numel()

        # Accuracy at 0.5 threshold
        acc_val = ((conf_preds > 0.5) == (conf_labels > 0.5)).float().mean().item()

        # AUROC
        auroc_val = _auroc(conf_preds, conf_labels)

        # Means by class
        pos_mean = conf_preds[conf_labels == 1].mean().item() if (conf_labels == 1).any() else 0.0
        neg_mean = conf_preds[conf_labels == 0].mean().item() if (conf_labels == 0).any() else 0.0

        # Metrics at thresholds
        threshold_metrics = compute_metrics_at_threshold(conf_preds, conf_labels)

        logger.info("Confidence model evaluation:")
        logger.info("  AUROC:        %.4f", auroc_val)
        logger.info("  Accuracy@0.5: %.4f", acc_val)
        logger.info("  Pos mean:     %.4f (n=%d)", pos_mean, (conf_labels == 1).sum().item())
        logger.info("  Neg mean:     %.4f (n=%d)", neg_mean, (conf_labels == 0).sum().item())
        logger.info("  Total samples: %d", n_total)
        for t in [0.3, 0.5, 0.7]:
            logger.info("  F1@%.1f:       %.4f", t, threshold_metrics.get(f"f1@{t:.1f}", 0.0))
    else:
        logger.warning("No confidence predictions collected!")

    # ------------------------------------------------------------------
    # 7. Comparison with RICO
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("SCREENSPOT CONFIDENCE — FINAL RESULTS")
    print("=" * 70)
    print(f"{'Model checkpoint':30s} {CONFIDENCE_CKPT.name}")
    print(f"{'Images with graph':30s} {n_images_with_graph}")
    print(f"{'Skipped (no graph)':30s} {n_images_no_graph}")
    print()
    print(f"{'GT elements total':30s} {all_gt}")
    print(f"{'VLM elements total':30s} {all_vlm}")
    print(f"{'Matched (TP)':30s} {all_matched}")
    print()
    print(f"{'VLM Precision':30s} {all_matched / max(all_vlm, 1):.4f}")
    print(f"{'VLM Recall':30s} {all_matched / max(all_gt, 1):.4f}")
    print()
    print(f"{'Confidence AUROC':30s} {auroc_val:.4f}")
    print(f"{'Confidence Acc@0.5':30s} {acc_val:.4f}")
    print(f"{'Pos Mean (TP)':30s} {pos_mean:.4f}")
    print(f"{'Neg Mean (FP)':30s} {neg_mean:.4f}")
    print(f"{'Total scored':30s} {n_total}")
    print()
    print("--- Cross-Domain Comparison ---")
    print(f"  RICO (9.2.1): AUROC = 0.703")
    print(f"  ScreenSpot:   AUROC = {auroc_val:.3f}")
    print()
    if auroc_val >= 0.6:
        print("  ✓ Confidence model generalizes to ScreenSpot domain!")
    else:
        print("  ⚠ Confidence model shows limited cross-domain transfer.")
    print("=" * 70)

    # Save results
    summary = {
        "checkpoint": str(CONFIDENCE_CKPT),
        "n_images_total": len(common_stems),
        "n_images_with_graph": n_images_with_graph,
        "n_images_no_graph": n_images_no_graph,
        "total_gt": all_gt,
        "total_vlm": all_vlm,
        "total_matched": all_matched,
        "vlm_precision": all_matched / max(all_vlm, 1),
        "vlm_recall": all_matched / max(all_gt, 1),
        "confidence_auroc": auroc_val,
        "confidence_acc_at_05": acc_val,
        "confidence_pos_mean": pos_mean,
        "confidence_neg_mean": neg_mean,
        "confidence_n": n_total,
        "rico_comparison_auroc": 0.703,
        "time_seconds": elapsed,
    }

    if all_conf_preds:
        conf_preds = torch.cat(all_conf_preds)
        conf_labels = torch.cat(all_conf_labels)
        tm = compute_metrics_at_threshold(conf_preds, conf_labels)
        summary["threshold_metrics"] = tm

    out_dir = Path("experiments/vlm_completion")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_screenspot_confidence.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved results to %s", out_path)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
