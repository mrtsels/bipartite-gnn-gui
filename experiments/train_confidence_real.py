#!/usr/bin/env python3
"""Train GNN confidence model using real VLM FP/FN data.

Compares against the old synthetic-imposter-based model.

Usage:
    python experiments/train_confidence_real.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import (
    DEVICE,
    GraphListDataset,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CANONICAL_TYPES = [
    "button", "text", "icon", "image",
    "input", "container", "list", "other",
]


def _random_imposter(img_size=(1.0, 1.0), types=None):
    """Generate a random imposter element with random bbox and type."""
    import random
    w = random.uniform(0.03, 0.3)
    h = random.uniform(0.02, 0.2)
    x1 = random.uniform(0, 1 - w)
    y1 = random.uniform(0, 1 - h)
    return ElementNode(
        bbox=[x1, y1, x1 + w, y1 + h],
        confidence=random.uniform(0.5, 1.0),
        label=random.choice(types or ["button"]),
    )


def _center_distance_bbox(box_a, box_b):
    """Euclidean distance between centers of two bboxes ([x1,y1,x2,y2])."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def _match_vlm_to_gt_centers(
    vlm_elements: list[ElementNode],
    gt_elements: list[ElementNode],
    distance_threshold: float = 0.1,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match VLM elements to GT elements using center-distance Hungarian matching.

    Returns (matched_pairs, fp_indices, fn_indices) where matched_pairs
    is list of (vlm_idx, gt_idx).
    """
    M = len(vlm_elements)
    N = len(gt_elements)

    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))

    # Build cost matrix: center distance, with INF for pairs above threshold
    INF = 1e9
    cost = torch.full((M, N), INF, dtype=torch.float32)

    for i, vlm_e in enumerate(vlm_elements):
        for j, gt_e in enumerate(gt_elements):
            d = _center_distance_bbox(vlm_e.bbox, gt_e.bbox)
            if d <= distance_threshold:
                cost[i, j] = d

    has_feasible = torch.isfinite(cost).any().item()
    if has_feasible:
        row_indices, col_indices = linear_sum_assignment(cost.numpy())
    else:
        row_indices, col_indices = [], []

    matched_pairs = []
    matched_rows = set()
    matched_cols = set()

    for i, j in zip(row_indices, col_indices):
        if cost[i, j] < INF / 2:
            matched_pairs.append((int(i), int(j)))
            matched_rows.add(int(i))
            matched_cols.add(int(j))

    fp_indices = [i for i in range(M) if i not in matched_rows]
    fn_indices = [j for j in range(N) if j not in matched_cols]

    return matched_pairs, fp_indices, fn_indices


def _load_vlm_prediction(path: Path) -> dict | None:
    """Load a VLM prediction JSON.

    Returns dict with keys: elements, image_width, image_height.
    VLM bboxes are in raw pixel coords.
    """
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    elements_raw = data.get("elements", [])
    if not isinstance(elements_raw, list):
        return None

    img_w = data.get("image_width", 0)
    img_h = data.get("image_height", 0)
    if img_w <= 0 or img_h <= 0:
        return None

    return {"elements": elements_raw, "image_width": img_w, "image_height": img_h}


def _vlm_raw_to_element_nodes(elements_raw: list[dict], img_w: int, img_h: int) -> list[ElementNode]:
    """Convert raw VLM prediction elements (pixel bbox_xyxy) to normalized ElementNodes."""
    nodes = []
    for item in elements_raw:
        bbox_raw = item.get("bbox_xyxy") or item.get("bbox")
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, bbox_raw)
        except (ValueError, TypeError):
            continue
        # Normalize to [0, 1]
        x1_n = x1 / img_w
        y1_n = y1 / img_h
        x2_n = x2 / img_w
        y2_n = y2 / img_h
        # Clamp
        x1_n = max(0.0, min(1.0, x1_n))
        y1_n = max(0.0, min(1.0, y1_n))
        x2_n = max(0.0, min(1.0, x2_n))
        y2_n = max(0.0, min(1.0, y2_n))
        if x2_n <= x1_n or y2_n <= y1_n:
            continue
        label = item.get("label", "other")
        nodes.append(ElementNode(
            bbox=[x1_n, y1_n, x2_n, y2_n],
            label=label,
            confidence=1.0,
        ))
    return nodes


# ---------------------------------------------------------------------------
# Graph construction with real VLM FP/FN + imposter negatives
# ---------------------------------------------------------------------------


def build_confidence_graph_real(
    vlm_elements: list[ElementNode],
    gt_elements: list[ElementNode],
    builder: BipartiteGraphBuilder,
    imposter_ratio: float = 0.5,
    distance_threshold: float = 0.1,
    seed: int | None = None,
    constraint_types: list[str] | None = None,
) -> tuple[Any, Dict[str, torch.Tensor]] | None:
    """Build a graph with VLM elements and imposters.

    Labels:
        - 1 (positive): VLM elements that matched a GT element
        - 0 (negative): unmatched VLM elements (FPs) + random imposters

    Args:
        vlm_elements: Normalised VLM-predicted element nodes.
        gt_elements: Normalised ground-truth element nodes.
        builder: Graph builder.
        imposter_ratio: Fraction of VLM elements to add as imposters.
        distance_threshold: Max center distance for a valid match.
        seed: RNG seed.
        constraint_types: If provided, only these constraint types are
            extracted (e.g. ``["containment"]``).  Passed directly
            to :func:`extract_all_constraints`.

    Returns:
        (hetero_data, targets) or None if degenerate.
    """
    if len(vlm_elements) < 1 or len(gt_elements) < 1:
        return None

    N_vlm = len(vlm_elements)

    # Match VLM → GT using center-distance Hungarian
    matched_pairs, fp_indices, fn_indices = _match_vlm_to_gt_centers(
        vlm_elements, gt_elements, distance_threshold=distance_threshold
    )

    # Build existence labels for VLM elements
    # 1 = matched (positive), 0 = FP (negative)
    matched_indices = set()
    for vlm_i, gt_j in matched_pairs:
        matched_indices.add(vlm_i)

    vlm_labels = []
    for i in range(N_vlm):
        vlm_labels.append(1.0 if i in matched_indices else 0.0)

    # Add random imposters as additional negatives
    N_imposter = max(1, int(N_vlm * imposter_ratio))
    if seed is not None:
        torch.manual_seed(seed)
    imposters = [_random_imposter(types=_CANONICAL_TYPES) for _ in range(N_imposter)]

    # Combine: VLM elements first, then imposters
    all_elements = vlm_elements + imposters
    existence_labels = vlm_labels + [0.0] * N_imposter

    # Extract constraints from ALL elements (including imposters)
    # This mirrors the existing build_confidence_graph approach
    constraints = extract_all_constraints(all_elements, constraint_types=constraint_types)
    if len(constraints) == 0:
        return None

    # Build graph
    hetero_data = builder.build(all_elements, constraints)

    N_elem = len(all_elements)
    N_con = len(constraints)

    targets = {
        "existence": torch.tensor(existence_labels, dtype=torch.float32).view(-1, 1),
        "coord": torch.zeros((N_elem, 4), dtype=torch.float32),
        "violation": torch.zeros((N_con, 1), dtype=torch.float32),
        "gt_boxes": torch.zeros((N_elem, 4), dtype=torch.float32),
    }

    return hetero_data, targets


# ---------------------------------------------------------------------------
# AUROC (manual, no sklearn dependency)
# ---------------------------------------------------------------------------


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
    auroc_val = u_stat / (n_pos * n_neg)
    return max(0.0, min(1.0, auroc_val.item()))


# ---------------------------------------------------------------------------
# Evaluation: existence prediction
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_confidence(
    model: BipartiteGNNCorrector,
    dataset: Dataset,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate existence prediction.

    Returns: acc, precision, recall, auroc, pos_mean, neg_mean, n
    """
    model.eval()
    preds_all = []
    labels_all = []

    for data, targets in dataset:
        data = data.to(device)
        predictions = model(data)
        if "existence" not in predictions:
            continue
        preds_all.append(predictions["existence"].cpu())
        labels_all.append(targets["existence"].cpu())

    if not preds_all:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0,
                "auroc": 0.5, "pos_mean": 0.0, "neg_mean": 0.0, "n": 0.0}

    preds = torch.cat(preds_all).view(-1)
    labels = torch.cat(labels_all).view(-1).float()

    n = labels.numel()
    if n == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0,
                "auroc": 0.5, "pos_mean": 0.0, "neg_mean": 0.0, "n": 0.0}

    # Threshold at 0.5
    binary_preds = (preds > 0.5).float()
    correct = (binary_preds == labels).float().mean().item()

    true_pos = ((binary_preds == 1) & (labels == 1)).sum().item()
    false_pos = ((binary_preds == 1) & (labels == 0)).sum().item()
    false_neg = ((binary_preds == 0) & (labels == 1)).sum().item()

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)

    auroc_val = _auroc(preds, labels)

    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_mean = preds[pos_mask].mean().item() if pos_mask.any() else 0.0
    neg_mean = preds[neg_mask].mean().item() if neg_mask.any() else 0.0

    return {
        "acc": correct,
        "precision": precision,
        "recall": recall,
        "auroc": auroc_val,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# Load OLD model and evaluate on same data
# ---------------------------------------------------------------------------


def evaluate_old_model(
    old_model_path: Path,
    dataset: Dataset,
    device: torch.device,
    hidden_dim: int = 128,
) -> dict[str, float]:
    """Load old synthetic-based model and evaluate on dataset."""
    old_model = BipartiteGNNCorrector(
        hidden_dim=hidden_dim,
        dropout=0.1,
        coord_weight=0.0,
        existence_weight=1.0,
    ).to(device)

    try:
        old_model.load_state_dict(torch.load(old_model_path, map_location=device))
        logger.info("Loaded old model from %s", old_model_path)
    except Exception as e:
        logger.warning("Failed to load old model: %s", e)
        return {}

    return evaluate_confidence(old_model, dataset, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train GNN confidence model on real VLM FP/FN data"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--imposter-ratio", type=float, default=0.5,
                        help="Additional random imposter negatives as fraction of VLM count")
    parser.add_argument("--distance-threshold", type=float, default=0.1,
                        help="Center-distance threshold for Hungarian matching")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vlm-dir", type=str, default="data/vlm_predictions/rico_qwen_flash")
    parser.add_argument("--rico-dir", type=str, default="data/rico_local/combined")
    parser.add_argument("--old-checkpoint", type=str,
                        default="checkpoints/confidence_scoring/best_model.pt")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--constraint-types", type=str, nargs="+", default=None,
                        help="Constraint types to use (e.g. containment). "
                             "Default: all types.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 60)
    logger.info("REAL VLM FP/FN CONFIDENCE TRAINING — Existence Head")
    logger.info("=" * 60)
    logger.info("Device: %s | epochs=%d | hidden=%d | lr=%.1e",
                DEVICE, args.epochs, args.hidden, args.lr)
    logger.info("VLM dir: %s | RICO dir: %s", args.vlm_dir, args.rico_dir)
    logger.info("Imposter ratio: %.2f | Distance threshold: %.3f",
                args.imposter_ratio, args.distance_threshold)
    logger.info("Constraint types: %s", args.constraint_types if args.constraint_types else "all")

    vlm_dir = Path(args.vlm_dir)
    rico_dir = Path(args.rico_dir)

    # Discover VLM prediction files (200 images, 0..199)
    vlm_paths = sorted(vlm_dir.glob("*.json"))
    logger.info("Found %d VLM prediction files", len(vlm_paths))

    builder = BipartiteGraphBuilder()
    all_graphs = []
    n_skipped = 0
    n_matched_total = 0
    n_vlm_total = 0
    n_imposter_total = 0
    t0 = time.time()

    for idx, vlm_path in enumerate(vlm_paths):
        # Load VLM prediction
        vlm_data = _load_vlm_prediction(vlm_path)
        if vlm_data is None:
            n_skipped += 1
            continue

        vlm_elements_raw = vlm_data["elements"]
        img_w = vlm_data["image_width"]
        img_h = vlm_data["image_height"]
        vlm_elements = _vlm_raw_to_element_nodes(vlm_elements_raw, img_w, img_h)
        if len(vlm_elements) < 1:
            n_skipped += 1
            continue

        # Load corresponding RICO GT
        gt_path = rico_dir / f"{vlm_path.stem}.json"
        if not gt_path.exists():
            n_skipped += 1
            continue

        parsed = parse_rico_vh(gt_path)
        if parsed is None:
            n_skipped += 1
            continue

        gt_img_w, gt_img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, gt_img_w, gt_img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]
        if len(gt_elements) < 1:
            n_skipped += 1
            continue

        # Build graph with real matching + imposters
        result = build_confidence_graph_real(
            vlm_elements,
            gt_elements,
            builder,
            imposter_ratio=args.imposter_ratio,
            distance_threshold=args.distance_threshold,
            seed=args.seed + idx if args.seed is not None else None,
            constraint_types=args.constraint_types,
        )
        if result is None:
            n_skipped += 1
            continue

        all_graphs.append(result)
        n_vlm_total += len(vlm_elements)
        n_imposter_total += max(1, int(len(vlm_elements) * args.imposter_ratio))

        # Count matches
        _, fp_indices, _ = _match_vlm_to_gt_centers(
            vlm_elements, gt_elements, distance_threshold=args.distance_threshold
        )
        n_matched_total += len(vlm_elements) - len(fp_indices)

    dt = time.time() - t0
    logger.info("Built %d graphs (%d skipped) in %.1fs", len(all_graphs), n_skipped, dt)

    if len(all_graphs) < 2:
        logger.error("Need ≥2 graphs, got %d", len(all_graphs))
        return

    # Stats
    n_elems = [g[0]["element"].x.shape[0] for g in all_graphs]
    n_cons = [g[0]["constraint"].x.shape[0] for g in all_graphs]
    import statistics as stat
    logger.info("Graphs: %.1f±%.1f elem, %.1f±%.1f con",
                stat.mean(n_elems), stat.stdev(n_elems) if len(n_elems) > 1 else 0,
                stat.mean(n_cons), stat.stdev(n_cons) if len(n_cons) > 1 else 0)

    # Positive ratio
    pos_ratios = []
    for _, targets in all_graphs:
        ex = targets["existence"]
        pos_ratios.append(ex.float().mean().item())
    logger.info("Positive ratio: %.3f ± %.3f (matched=%d, vlm=%d, imposter=%d)",
                stat.mean(pos_ratios),
                stat.stdev(pos_ratios) if len(pos_ratios) > 1 else 0,
                n_matched_total, n_vlm_total, n_imposter_total)

    # Split
    split_idx = int(len(all_graphs) * (1.0 - args.val_split))
    train_dataset = GraphListDataset(all_graphs[:split_idx])
    val_dataset = GraphListDataset(all_graphs[split_idx:])
    logger.info("Split: %d train / %d val", len(train_dataset), len(val_dataset))

    # Model
    model = BipartiteGNNCorrector(
        hidden_dim=args.hidden,
        dropout=0.1,
        coord_weight=0.0,
        existence_weight=1.0,
    ).to(DEVICE)

    model.loss_fn.existence_weight = 1.0
    model.loss_fn.violation_weight = 0.0
    model.loss_fn.coord_weight = 0.0
    model.loss_fn.alignment_weight = 0.0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for data, targets in train_dataset:
            data = data.to(DEVICE)
            targets_gpu = {k: v.to(DEVICE) for k, v in targets.items()}
            optimizer.zero_grad()
            predictions = model(data)
            loss = model.compute_loss(predictions, targets_gpu)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for data, targets in val_dataset:
                data = data.to(DEVICE)
                targets_gpu = {k: v.to(DEVICE) for k, v in targets.items()}
                predictions = model(data)
                loss = model.compute_loss(predictions, targets_gpu)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)

        metrics = evaluate_confidence(model, val_dataset, DEVICE)
        logger.info(
            "Epoch %2d/%d — train: %.4f | val: %.4f | acc: %.3f | prec: %.3f | "
            "rec: %.3f | auroc: %.3f | pos: %.3f | neg: %.3f",
            epoch, args.epochs, avg_train_loss, avg_val_loss,
            metrics["acc"], metrics["precision"], metrics["recall"],
            metrics["auroc"], metrics["pos_mean"], metrics["neg_mean"],
        )

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Final evaluation: real model
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    real_metrics = evaluate_confidence(model, val_dataset, DEVICE)
    logger.info("Real FP/FN model — AUROC: %.4f | Acc: %.3f | Prec: %.3f | Rec: %.3f",
                real_metrics["auroc"], real_metrics["acc"],
                real_metrics["precision"], real_metrics["recall"])
    logger.info("  Pos mean: %.4f | Neg mean: %.4f",
                real_metrics["pos_mean"], real_metrics["neg_mean"])

    # Compare against old model
    old_checkpoint_path = Path(args.old_checkpoint)
    old_metrics = {}
    if old_checkpoint_path.exists():
        old_metrics = evaluate_old_model(old_checkpoint_path, val_dataset, DEVICE)
        if old_metrics:
            logger.info("Old synthetic model — AUROC: %.4f | Acc: %.3f | Prec: %.3f | Rec: %.3f",
                        old_metrics["auroc"], old_metrics["acc"],
                        old_metrics["precision"], old_metrics["recall"])
            logger.info("  Pos mean: %.4f | Neg mean: %.4f",
                        old_metrics["pos_mean"], old_metrics["neg_mean"])
    else:
        logger.warning("Old checkpoint not found at %s", old_checkpoint_path)

    # Print comparison table
    print()
    print("=" * 85)
    print(f"{'Method':<35s} | {'AUROC':>6s} | {'Acc':>5s} | {'Prec':>5s} | "
          f"{'Rec':>5s} | {'Pos Mean':>8s} | {'Neg Mean':>8s}")
    print("-" * 85)

    if old_metrics:
        print(f"{'Synthetic imposter (old)':<35s} | {old_metrics['auroc']:>6.3f} | "
              f"{old_metrics['acc']:>5.3f} | {old_metrics['precision']:>5.3f} | "
              f"{old_metrics['recall']:>5.3f} | {old_metrics['pos_mean']:>8.4f} | "
              f"{old_metrics['neg_mean']:>8.4f}")

    print(f"{'Real FP/FN (this experiment)':<35s} | {real_metrics['auroc']:>6.3f} | "
          f"{real_metrics['acc']:>5.3f} | {real_metrics['precision']:>5.3f} | "
          f"{real_metrics['recall']:>5.3f} | {real_metrics['pos_mean']:>8.4f} | "
          f"{real_metrics['neg_mean']:>8.4f}")
    print("=" * 85)

    # Interpretation
    if old_metrics:
        auroc_improvement = real_metrics["auroc"] - old_metrics["auroc"]
        if auroc_improvement > 0.05:
            interpretation = (
                f"Significant improvement: AUROC increased by {auroc_improvement:.3f}. "
                "Real VLM FP/FN data provides harder, more realistic negatives "
                "that better train the existence head to distinguish genuine "
                "elements from VLM false positives."
            )
        elif auroc_improvement > 0.01:
            interpretation = (
                f"Moderate improvement: AUROC increased by {auroc_improvement:.3f}. "
                "Real VLM data provides somewhat better training signal."
            )
        elif auroc_improvement > -0.01:
            interpretation = (
                f"Comparable performance: AUROC difference of {auroc_improvement:.3f}. "
                "Real and synthetic data perform similarly for this task."
            )
        else:
            interpretation = (
                f"Synthetic model performs better: AUROC decreased by {-auroc_improvement:.3f}. "
                "The random imposter approach may actually provide a more "
                "discriminative training signal than structurally valid VLM FPs."
            )
        logger.info("Interpretation: %s", interpretation)
        print(f"\nInterpretation: {interpretation}")

    logger.info("Done.")

    # Save checkpoint
    checkpoint_dir = Path("checkpoints/confidence_scoring")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
    logger.info("Checkpoint saved to %s", checkpoint_dir / "best_model.pt")


if __name__ == "__main__":
    main()
