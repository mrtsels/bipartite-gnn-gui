#!/usr/bin/env python3
"""Self-supervised structural completion pretraining.

Trains the GNN as a **masked graph autoencoder**: element node features
are randomly masked, and the model must recover the original
``[x1, y1, x2, y2, confidence]`` from the biparti te constraint graph
context alone.

Usage:
  python scripts/train_completion.py --n 500 --epochs 50 --hidden 128
  python scripts/train_completion.py --n 2000 --epochs 100 --hidden 256 --mask-ratio 0.7
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# Add project root to path.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.data.masking import compute_mask_loss, random_mask
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector

# Reuse helpers from run_experiment.
from scripts.run_experiment import (
    DEVICE,
    GraphListDataset,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
)

logger = logging.getLogger(__name__)


def build_completion_graph(
    gt_elements: list[ElementNode],
    builder: BipartiteGraphBuilder,
    mask_ratio: float = 0.6,
    seed: int | None = None,
) -> Tuple[Any, Dict[str, torch.Tensor]] | None:
    """Build a (masked HeteroData, targets) pair for completion pretraining.

    Uses GT elements directly (no VLM noise), then masks a fraction of
    element features.  The targets include ``"mask_completion_target"``
    (original 5-d features) and ``"mask_completion_mask"`` (bool mask)
    so the model's ``compute_loss`` can compute the self-supervised loss.

    Args:
        gt_elements: Normalised ground-truth element nodes.
        builder: Graph builder instance.
        mask_ratio: Fraction of elements to mask (default 0.6).
        seed: Optional RNG seed.

    Returns:
        ``(hetero_data, targets)`` tuple, or ``None`` if degenerate.
    """
    if len(gt_elements) < 2:
        return None

    # Extract constraints from the *full* layout.
    constraints = extract_all_constraints(gt_elements)
    if len(constraints) == 0:
        return None

    # Build HeteroData graph from full layout.
    # IMPORTANT: element nodes = gt_elements (no VLM noise).
    # Constraints are built from all elements.
    data = builder.build(gt_elements, constraints)

    # Apply random masking.
    data, mask_info = random_mask(data, mask_ratio=mask_ratio, seed=seed)

    N = len(gt_elements)
    N_con = len(constraints)

    # Build targets (same structure as run_experiment for compatibility).
    # For completion we only need mask targets — coord/violation/existence
    # are not used in this pretraining stage.
    gt_xywh = _bbox_xyxy_to_xywh(
        torch.tensor(
            [[e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]] for e in gt_elements],
            dtype=torch.float32,
        )
    )
    targets: Dict[str, torch.Tensor] = {
        "coord": torch.zeros((N, 4), dtype=torch.float32),
        "violation": torch.zeros((N_con, 1), dtype=torch.float32),
        "existence": torch.ones((N, 1), dtype=torch.float32),
        "gt_boxes": gt_xywh,
        # Mask completion targets.
        "mask_completion_target": mask_info["target"],
        "mask_completion_mask": mask_info["mask"],
    }
    return data, targets


def _bbox_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, 4)`` from ``[x1, y1, x2, y2]`` to ``[cx, cy, w, h]``."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=1)


@torch.no_grad()
def evaluate_completion(
    model: BipartiteGNNCorrector,
    dataset: Dataset,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate mask completion accuracy on a dataset.

    Metrics:
        - ``mask_mse``: overall MSE on masked element features.
        - ``mask_pos_err``: centre-distance error for masked elements.
        - ``mask_recall``: fraction of masked elements recovered within
          a 5% tolerance (IoU > 0.5 equivalent for bbox).

    Returns:
        Dict of metric name → scalar value.
    """
    model.eval()
    total_mse = 0.0
    total_pos_err = 0.0
    total_recalled = 0
    total_masked = 0

    for data, targets in dataset:
        data = data.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        predictions = model(data)
        mask = targets["mask_completion_mask"]
        gt_feats = targets["mask_completion_target"]

        if mask.sum() == 0:
            continue

        pred_feats = predictions["mask_completion"]

        # MSE on masked elements.
        total_mse += compute_mask_loss(pred_feats, gt_feats, mask).item() * mask.sum().item()
        total_masked += mask.sum().item()

        # Position error: centre distance.
        pred_xyxy = pred_feats[mask][:, :4]  # (M, 4) in xyxy
        gt_xyxy = gt_feats[mask][:, :4]
        pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2.0
        pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2.0
        gt_cx = (gt_xyxy[:, 0] + gt_xyxy[:, 2]) / 2.0
        gt_cy = (gt_xyxy[:, 1] + gt_xyxy[:, 3]) / 2.0
        dist = torch.sqrt((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2)
        total_pos_err += dist.sum().item()

        # Recall: predicted position within 0.05 (normalised coords).
        total_recalled += (dist < 0.05).sum().item()

    n = max(total_masked, 1)
    return {
        "mask_mse": total_mse / n,
        "mask_pos_err": total_pos_err / n,
        "mask_recall": total_recalled / n,
        "mask_count": float(total_masked),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Self-supervised structural completion pretraining"
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of RICO samples")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs")
    parser.add_argument("--hidden", type=int, default=128,
                        help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (batch_size=None = single graphs)")
    parser.add_argument("--mask-ratio", type=float, default=0.6,
                        help="Fraction of elements to mask during training")
    parser.add_argument("--mask-val-ratio", type=float, default=0.3,
                        help="Mask ratio for validation (usually lower)")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/completion",
                        help="Checkpoint directory")
    parser.add_argument("--rico-dir", type=str,
                        default="data/rico_local/combined",
                        help="RICO data directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                        help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 55)
    logger.info("STRUCTURAL COMPLETION PRETRAINING")
    logger.info("=" * 55)
    logger.info("Device: %s", DEVICE)
    logger.info("Config: n=%d, epochs=%d, hidden=%d, lr=%.1e, mask=%.2f",
                args.n, args.epochs, args.hidden, args.lr, args.mask_ratio)

    rico_dir = Path(args.rico_dir)
    if not rico_dir.is_dir():
        logger.error("Not found: %s", rico_dir)
        return

    all_jsons = sorted(rico_dir.glob("*.json"))
    n_use = min(args.n, len(all_jsons))
    jsons = all_jsons[:n_use]
    logger.info("RICO JSONs: %d, using %d", len(all_jsons), n_use)

    # ── Build masked graphs ──
    builder = BipartiteGraphBuilder()
    all_graphs: List[Tuple[Any, Dict[str, torch.Tensor]]] = []

    t0 = time.time()
    n_skipped = 0
    for idx, path in enumerate(jsons):
        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - t0
            logger.info("  %d/%d (%d skipped, %.1f/s)",
                        idx, n_use, n_skipped, idx / max(elapsed, 0.01))

        parsed = parse_rico_vh(path)
        if parsed is None:
            n_skipped += 1
            continue

        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        result = build_completion_graph(
            gt_elements, builder, mask_ratio=args.mask_ratio, seed=args.seed
        )
        if result is None:
            n_skipped += 1
            continue
        all_graphs.append(result)

    dt = time.time() - t0
    logger.info("Built %d graphs (%d skipped) in %.1fs (%.1f/s)",
                len(all_graphs), n_skipped, dt, len(all_graphs) / max(dt, 0.01))

    if len(all_graphs) < 2:
        logger.error("Need ≥2 graphs, got %d", len(all_graphs))
        return

    # Statistics.
    n_elems = [g[0]["element"].x.shape[0] for g in all_graphs]
    masked_frac = [
        g[1]["mask_completion_mask"].float().mean().item() for g in all_graphs
    ]
    logger.info("Graphs: %.1f±%.1f elem, mask=%.2f±%.2f",
                sum(n_elems) / len(n_elems),
                __import__("statistics").stdev(n_elems) if len(n_elems) > 1 else 0.0,
                sum(masked_frac) / len(masked_frac),
                __import__("statistics").stdev(masked_frac) if len(masked_frac) > 1 else 0.0)

    # ── Split ──
    split_idx = int(len(all_graphs) * (1.0 - args.val_split))
    train_items = all_graphs[:split_idx]
    val_items = all_graphs[split_idx:]

    # Rebuild validation with a different (lower) mask ratio.
    val_graphs: List[Tuple[Any, Dict[str, torch.Tensor]]] = []
    for data, targets in val_items:
        # Re-mask with lower ratio for evaluation.
        data, mask_info = random_mask(data, mask_ratio=args.mask_val_ratio, seed=args.seed)
        targets["mask_completion_target"] = mask_info["target"]
        targets["mask_completion_mask"] = mask_info["mask"]
        val_graphs.append((data, targets))

    train_dataset = GraphListDataset(train_items)
    val_dataset = GraphListDataset(val_graphs)

    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    logger.info("Split: %d train / %d val", len(train_dataset), len(val_dataset))

    # ── Model ──
    model = BipartiteGNNCorrector(
        hidden_dim=args.hidden,
        dropout=0.1,
        coord_weight=0.0,         # disable coordinate loss
        existence_weight=0.0,     # disable existence loss
    ).to(DEVICE)
    # Enable mask completion.
    model.mask_weight = 1.0

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params, hidden=%d", n_params, args.hidden)

    # ── Optimiser ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # ── Training loop ──
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0

    logger.info("=" * 55)
    logger.info("Pre-training — mask_ratio=%.2f, mask_weight=%.1f",
                args.mask_ratio, model.mask_weight)
    logger.info("=" * 55)

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

        # Validation.
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

        # Evaluation metrics.
        metrics = evaluate_completion(model, val_dataset, DEVICE)

        logger.info(
            "Epoch %2d/%d — train_loss: %.6f | val_loss: %.6f | MSE: %.6f | pos_err: %.4f | recall: %.3f",
            epoch, args.epochs, avg_train_loss, avg_val_loss,
            metrics["mask_mse"], metrics["mask_pos_err"], metrics["mask_recall"],
        )

        # Checkpoint.
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": {
                    "hidden_dim": args.hidden,
                    "mask_ratio": args.mask_ratio,
                },
            }, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("=" * 55)
    logger.info("PRETRAINING COMPLETE ✅")
    logger.info("Best val_loss: %.6f", best_val_loss)
    logger.info("Checkpoint: %s", checkpoint_dir / "best_model.pt")
    logger.info("=" * 55)

    # Final evaluation.
    final_metrics = evaluate_completion(model, val_dataset, DEVICE)
    logger.info("Final metrics:")
    for k, v in final_metrics.items():
        logger.info("  %s: %.6f", k, v)


if __name__ == "__main__":
    main()
