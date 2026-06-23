#!/usr/bin/env python3
"""Phase 4.8 — Constraint-aware confidence scoring.

Trains the GNN to predict reliability scores for GUI elements.
Uses GT elements (positive) + random imposters (negative).

Key insight: the GNN can detect imposters through structural analysis —
real elements form consistent constraints (alignments, containment),
while random boxes don't.

Usage:
  python scripts/train_confidence.py --n 500 --epochs 30 --hidden 128 --imposter-ratio 0.5
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
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
# Imposter generation helpers
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


# ---------------------------------------------------------------------------
# Graph construction with imposters
# ---------------------------------------------------------------------------


def build_confidence_graph(
    gt_elements: list[ElementNode],
    builder: BipartiteGraphBuilder,
    imposter_ratio: float = 0.5,
    seed: int | None = None,
) -> tuple[Any, Dict[str, torch.Tensor]] | None:
    """Build a graph with GT elements (positive) + random imposters (negative).

    The training target is binary: 1 for real GT element, 0 for imposter.
    The GNN encodes ALL elements through the constraint graph and predicts
    existence confidence for each.

    Args:
        gt_elements: Normalised ground-truth elements.
        builder: Graph builder.
        imposter_ratio: Fraction of real elements to add as imposters.
        seed: RNG seed.

    Returns:
        ``(hetero_data, targets)`` or ``None`` if degenerate.
    """
    if len(gt_elements) < 2:
        return None

    N_real = len(gt_elements)
    N_imposter = max(1, int(N_real * imposter_ratio))

    # Create imposter elements.
    rng_state = None if seed is None else torch.Generator()
    if rng_state is not None:
        torch.manual_seed(seed)

    imposters = [_random_imposter(types=_CANONICAL_TYPES) for _ in range(N_imposter)]

    # Combine and shuffle so real/imposter order is not learnable.
    all_elements = gt_elements + imposters
    existence_labels = (
        [1.0] * N_real + [0.0] * N_imposter
    )  # 1 = real, 0 = imposter

    # Shuffle.
    import random as _random

    indices = list(range(len(all_elements)))
    _random.shuffle(indices)
    all_elements = [all_elements[i] for i in indices]
    existence_labels = [existence_labels[i] for i in indices]

    # Extract constraints on ALL elements (real + imposter).
    constraints = extract_all_constraints(all_elements)
    if len(constraints) == 0:
        return None

    # Build the graph.
    hetero_data = builder.build(all_elements, constraints)

    N_elem = len(all_elements)
    N_con = len(constraints)

    # Targets: existence (primary) + violation (auxiliary, all valid = 0).
    # GT bboxes for coordinate targets — zeros for imposters (no GT).
    gt_boxes_xyxy = torch.tensor(
        [[e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]] for e in gt_elements],
        dtype=torch.float32,
    )

    # Build coord targets mapping shuffled indices back to GT.
    coord_targets = torch.zeros((N_elem, 4), dtype=torch.float32)
    # For real elements, compute delta from their current bbox to their GT bbox.
    # Since we shuffled, we need to track which indices are real.
    # For imposters there's no GT, so coord target stays 0.
    # For real elements, the "delta" is 0 since they already have GT positions.
    # (They haven't been moved — they are exact GT elements.)
    # So coord targets are all zeros, which is fine.

    targets = {
        "existence": torch.tensor(existence_labels, dtype=torch.float32).view(-1, 1),
        "coord": torch.zeros((N_elem, 4), dtype=torch.float32),
        "violation": torch.zeros((N_con, 1), dtype=torch.float32),
        "gt_boxes": _bbox_xyxy_to_xywh(
            torch.tensor(
                [
                    [e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]]
                    for e in all_elements
                ],
                dtype=torch.float32,
            )
        ),
        # Remembers which elements are real (1) vs imposter (0) for evaluation.
        "existence_binary": torch.tensor(existence_labels, dtype=torch.float32),
    }

    return hetero_data, targets


def _bbox_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert xyxy to xywh format."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)


# ---------------------------------------------------------------------------
# Evaluation: AUROC (manual implementation, no sklearn)
# ---------------------------------------------------------------------------


def _auroc(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute AUROC manually using the Mann-Whitney U statistic.

    Args:
        preds: ``(N,)`` predicted scores in ``[0, 1]``.
        targets: ``(N,)`` binary labels (0 or 1).

    Returns:
        AUROC score in ``[0, 1]``.
    """
    if preds.numel() < 2:
        return 0.5

    n_pos = (targets == 1).sum()
    n_neg = (targets == 0).sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by prediction ASCENDING (lower score → lower rank).
    sorted_indices = torch.argsort(preds, descending=False)
    sorted_targets = targets[sorted_indices]

    # Mann-Whitney U: sum of ranks of positive examples.
    # Rank is 1-indexed position in ascending sorted order.
    ranks = torch.arange(1, len(preds) + 1, device=preds.device, dtype=torch.float32)
    sum_ranks_pos = ranks[sorted_targets == 1].sum()

    # U statistic: U = R_pos - n_pos * (n_pos + 1) / 2
    u_stat = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    auroc_val = u_stat / (n_pos * n_neg)

    # Clamp to [0, 1].
    return max(0.0, min(1.0, auroc_val.item()))


@torch.no_grad()
def evaluate_confidence(
    model: BipartiteGNNCorrector,
    dataset: Dataset,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate confidence (existence) prediction accuracy.

    Computes:
      - Accuracy at 0.5 threshold
      - Precision at 0.5 threshold
      - Recall at 0.5 threshold
      - AUROC

    Args:
        model: Trained GNN model.
        dataset: Dataset with ``(data, targets)`` pairs.
        device: Torch device.

    Returns:
        Dict with keys: ``acc``, ``precision``, ``recall``, ``auroc``, ``n``.
    """
    model.eval()
    preds_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    for data, targets in dataset:
        data = data.to(device)
        predictions = model(data)
        if "existence" not in predictions:
            continue
        preds_all.append(predictions["existence"].cpu())
        labels_all.append(targets["existence"].cpu())

    if not preds_all:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "auroc": 0.5, "n": 0.0}

    preds = torch.cat(preds_all).view(-1)
    labels = torch.cat(labels_all).view(-1).float()

    n = labels.numel()
    if n == 0:
        return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "auroc": 0.5, "n": 0.0}

    # Threshold at 0.5.
    binary_preds = (preds > 0.5).float()
    correct = (binary_preds == labels).float().mean().item()

    true_pos = ((binary_preds == 1) & (labels == 1)).sum().item()
    false_pos = ((binary_preds == 1) & (labels == 0)).sum().item()
    false_neg = ((binary_preds == 0) & (labels == 1)).sum().item()

    precision = true_pos / max(true_pos + false_pos, 1)
    recall = true_pos / max(true_pos + false_neg, 1)

    auroc_val = _auroc(preds, labels)

    return {
        "acc": correct,
        "precision": precision,
        "recall": recall,
        "auroc": auroc_val,
        "n": float(n),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Constraint-aware confidence scoring (Phase 4.8)"
    )
    parser.add_argument("--n", type=int, default=500, help="RICO samples")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--imposter-ratio",
        type=float,
        default=0.5,
        help="Fraction of real elements to add as imposters",
    )
    parser.add_argument(
        "--rico-dir",
        type=str,
        default="data/rico_local/combined",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/confidence_scoring",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 55)
    logger.info("CONFIDENCE SCORING — Constraint-aware VLM reliability")
    logger.info("=" * 55)
    logger.info(
        "Device: %s | n=%d | epochs=%d | hidden=%d | imposter_ratio=%.2f",
        DEVICE, args.n, args.epochs, args.hidden, args.imposter_ratio,
    )

    rico_dir = Path(args.rico_dir)
    all_jsons = sorted(rico_dir.glob("*.json"))[: args.n]
    logger.info("RICO JSONs: %d", len(all_jsons))

    builder = BipartiteGraphBuilder()
    all_graphs: List[Tuple[Any, Dict[str, torch.Tensor]]] = []
    n_skipped = 0
    t0 = time.time()

    for idx, path in enumerate(all_jsons):
        parsed = parse_rico_vh(path)
        if parsed is None:
            n_skipped += 1
            continue
        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        result = build_confidence_graph(
            gt_elements,
            builder,
            imposter_ratio=args.imposter_ratio,
            seed=args.seed + idx if args.seed is not None else None,
        )
        if result is None:
            n_skipped += 1
            continue
        all_graphs.append(result)

    dt = time.time() - t0
    logger.info(
        "Built %d graphs (%d skipped) in %.1fs", len(all_graphs), n_skipped, dt
    )

    if len(all_graphs) < 2:
        logger.error("Need ≥2 graphs")
        return

    # Statistics.
    n_elems = [g[0]["element"].x.shape[0] for g in all_graphs]
    n_cons = [g[0]["constraint"].x.shape[0] for g in all_graphs]
    n_edges = [
        (
            g[0]["element", "to", "constraint"].edge_index.shape[1]
            if "element" in g[0].node_types
            else 0
        )
        for g in all_graphs
    ]
    import statistics as stat

    logger.info(
        "Graphs: %.1f±%.1f elem, %.1f±%.1f con, %.1f±%.1f edges",
        stat.mean(n_elems),
        stat.stdev(n_elems) if len(n_elems) > 1 else 0,
        stat.mean(n_cons),
        stat.stdev(n_cons) if len(n_cons) > 1 else 0,
        stat.mean(n_edges),
        stat.stdev(n_edges) if len(n_edges) > 1 else 0,
    )

    # Positive (real) vs negative (imposter) ratio.
    pos_ratios = []
    for _, targets in all_graphs:
        ex = targets["existence"]
        pos_ratios.append(ex.float().mean().item())
    logger.info(
        "Positive ratio: %.3f ± %.3f",
        stat.mean(pos_ratios),
        stat.stdev(pos_ratios) if len(pos_ratios) > 1 else 0,
    )

    split_idx = int(len(all_graphs) * (1.0 - args.val_split))
    train_dataset = GraphListDataset(all_graphs[:split_idx])
    val_dataset = GraphListDataset(all_graphs[split_idx:])
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)
    logger.info(
        "Split: %d train / %d val", len(train_dataset), len(val_dataset)
    )

    # Model — use existence head (primary) + violation head (auxiliary).
    # coord_weight = 0.0 (no coordinate correction needed).
    model = BipartiteGNNCorrector(
        hidden_dim=args.hidden,
        dropout=0.1,
        coord_weight=0.0,
        existence_weight=1.0,
    ).to(DEVICE)

    # Configure weights: existence (primary) + violation (auxiliary).
    model.loss_fn.existence_weight = 1.0
    model.loss_fn.violation_weight = 1.0
    model.loss_fn.coord_weight = 0.0
    model.loss_fn.alignment_weight = 0.0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for data, targets in train_loader:
            data = data.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}
            optimizer.zero_grad()
            predictions = model(data)
            loss = model.compute_loss(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(DEVICE)
                targets = {k: v.to(DEVICE) for k, v in targets.items()}
                predictions = model(data)
                loss = model.compute_loss(predictions, targets)
                val_loss += loss.item()
                val_batches += 1
        avg_val_loss = val_loss / max(val_batches, 1)

        metrics = evaluate_confidence(model, val_dataset, DEVICE)
        logger.info(
            "Epoch %2d/%d — train: %.4f | val: %.4f | acc: %.3f | prec: %.3f | "
            "rec: %.3f | auroc: %.3f | n: %.0f",
            epoch,
            args.epochs,
            avg_train_loss,
            avg_val_loss,
            metrics["acc"],
            metrics["precision"],
            metrics["recall"],
            metrics["auroc"],
            metrics["n"],
        )

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("=" * 55)
    logger.info("DONE ✅ — Best val loss: %.6f", best_val_loss)
    final_metrics = evaluate_confidence(model, val_dataset, DEVICE)
    logger.info(
        "Final confidence — acc: %.4f | prec: %.4f | rec: %.4f | auroc: %.4f",
        final_metrics["acc"],
        final_metrics["precision"],
        final_metrics["recall"],
        final_metrics["auroc"],
    )


if __name__ == "__main__":
    main()
