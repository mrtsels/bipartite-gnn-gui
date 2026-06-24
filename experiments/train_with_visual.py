#!/usr/bin/env python3
"""Visual feature fusion experiment: compare GNN accuracy with ViT vs DINOv2 visual features.

Architecture:
  - Element node features: 5-d spatial [x1,y1,x2,y2,confidence]
    → with visual: + visual_dim-d embedding = (5 + visual_dim)-d total
  - Constraint node features: 11-d (one-hot type + first param)
  - BipartiteGraphSAGE encoder (element_dim adjusts accordingly)
  - Violation + Proposal + Type prediction heads

Comparison:
  - "Without visual": existing checkpoint evaluated on test set
  - "With visual (vit_tiny)": newly trained model with vit_tiny features (192-dim)
  - "With visual (DINOv2)": newly trained model with DINOv2 features (768-dim)

Usage:
    python experiments/train_with_visual.py                         # vit_tiny (192)
    python experiments/train_with_visual.py --visual-dim 768 --feat-dir data/rico_local/visual_features_dinov2  # DINOv2
"""

from __future__ import annotations

import json
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
from bipartite_gnn_gui.graph.schema import ConstraintType, ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from bipartite_gnn_gui.model.heads import N_TYPES
from scripts.run_experiment import (
    DEVICE,
    GraphListDataset,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RICO_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/combined")
_VISUAL_FEAT_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/visual_features")
_DINOV2_FEAT_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/visual_features_dinov2")
_CHECKPOINT_DIR = Path("/Users/minimx/bipartite-gnn-gui/checkpoints/violation_detection")
_BEST_MODEL_PATH = _CHECKPOINT_DIR / "best_model.pt"
_VISUAL_CONCAT_PATH = _CHECKPOINT_DIR / "visual_fusion_model.pt"

# RICO semantic type → index (matching train_violation.py).
TYPE_TO_IDX: dict[str, int] = {
    "button": 0, "text": 1, "icon": 2, "image": 3,
    "input": 4, "container": 5, "list": 6,
}
TYPE_UNKNOWN_IDX: int = 7

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------


def _type_idx(label: str) -> int:
    return TYPE_TO_IDX.get(label.lower(), TYPE_UNKNOWN_IDX)


# ---------------------------------------------------------------------------
# Visual feature loading
# ---------------------------------------------------------------------------


def _load_visual_features(uid: str, n_elements: int, feat_dir: Path,
                          visual_dim: int) -> torch.Tensor | None:
    """Load precomputed visual features for a single RICO image.

    Args:
        uid: Image filename stem (e.g. ``"0"``).
        n_elements: Expected number of elements (for validation).
        feat_dir: Directory containing ``.pt`` feature files.
        visual_dim: Expected feature dimension (192 for vit_tiny, 768 for DINOv2).

    Returns:
        ``(N_elements, visual_dim)`` tensor, or ``None`` if not found or
        dimension mismatch.
    """
    path = feat_dir / f"{uid}.pt"
    if not path.exists():
        return None
    try:
        feats = torch.load(path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(feats, torch.Tensor) or feats.dim() != 2 or feats.shape[1] != visual_dim:
        return None
    if feats.shape[0] != n_elements:
        logger.debug(
            "Visual feature count mismatch for %s: expected %d, got %d",
            uid, n_elements, feats.shape[0],
        )
        return None
    return feats


# ---------------------------------------------------------------------------
# Graph building helpers
# ---------------------------------------------------------------------------


def _bbox_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (N, 4) xyxy → cxcywh."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)


def build_violation_graph(
    gt_elements: list[ElementNode],
    builder: BipartiteGraphBuilder,
    drop_ratio: float = 0.4,
    seed: int | None = None,
    allowed_constraint_types: set[ConstraintType] | None = None,
    single_element_removal: bool = False,
    visual_features: torch.Tensor | None = None,
) -> Tuple[Any, Dict[str, torch.Tensor]] | None:
    """Build a violation graph, optionally with visual features.

    Mirrors ``train_violation.build_violation_graph`` but accepts
    a ``visual_features`` tensor passed through to the builder.

    Args:
        gt_elements: Normalised ground-truth element nodes.
        builder: Graph builder instance.
        drop_ratio: Fraction of elements to remove.
        seed: RNG seed.
        allowed_constraint_types: Optional constraint type filter.
        single_element_removal: If True, only graphs with ≤1 removed
            element are kept.
        visual_features: Optional ``(N_gt, visual_dim)`` tensor.  The builder
            will concatenate it to element spatial features.

    Returns:
        ``(hetero_data, targets)`` or ``None`` if degenerate.
    """
    if len(gt_elements) < 3:
        return None

    N = len(gt_elements)

    # Full constraint extraction.
    full_constraints = extract_all_constraints(gt_elements)
    if len(full_constraints) == 0:
        return None

    if allowed_constraint_types is not None:
        full_constraints = [
            c for c in full_constraints
            if c.constraint_type in allowed_constraint_types
        ]
        if len(full_constraints) == 0:
            return None

    # Random survivor mask.
    rng = torch.Generator() if seed is not None else None
    if rng is not None:
        rng.manual_seed(seed)
    survivor_mask = torch.rand(N, generator=rng) >= drop_ratio

    if survivor_mask.sum() < 2:
        return None

    removed_indices_all = torch.where(~survivor_mask)[0].tolist()
    if single_element_removal and len(removed_indices_all) > 1:
        return None

    survivor_indices_old = torch.where(survivor_mask)[0].tolist()
    old_to_new = {old: new for new, old in enumerate(survivor_indices_old)}
    removed_indices = torch.where(~survivor_mask)[0].tolist()

    # Filter visual features to survivors if provided.
    survivor_visual = None
    if visual_features is not None:
        survivor_visual = visual_features[survivor_indices_old]

    kept_constraints = []
    violation_labels = []
    proposal_targets = []

    for con in full_constraints:
        src_set = set(con.source_indices) & set(survivor_indices_old)
        tgt_set = set(con.target_indices) & set(survivor_indices_old)
        all_surviving = src_set | tgt_set

        if len(all_surviving) < 1:
            continue

        all_original = set(con.source_indices + con.target_indices)
        missing_idx = [i for i in all_original if i in removed_indices]

        is_violated = len(all_surviving) < 2
        violation_labels.append(1.0 if is_violated else 0.0)

        if is_violated and missing_idx:
            x1s = [gt_elements[i].bbox[0] for i in missing_idx]
            y1s = [gt_elements[i].bbox[1] for i in missing_idx]
            x2s = [gt_elements[i].bbox[2] for i in missing_idx]
            y2s = [gt_elements[i].bbox[3] for i in missing_idx]
            type_idx = _type_idx(gt_elements[missing_idx[0]].label)
            proposal_targets.append([
                sum(x1s) / len(x1s), sum(y1s) / len(y1s),
                sum(x2s) / len(x2s), sum(y2s) / len(y2s),
                float(type_idx),
            ])
        else:
            proposal_targets.append([0.0, 0.0, 0.0, 0.0, 0.0])

        con.source_indices = [old_to_new[i] for i in src_set]
        con.target_indices = [old_to_new[i] for i in tgt_set]
        kept_constraints.append(con)

    if len(kept_constraints) == 0:
        return None

    survivors = [gt_elements[i] for i in survivor_indices_old]

    # Build graph — pass visual features for survivors.
    hetero_data = builder.build(survivors, kept_constraints,
                                visual_features=survivor_visual)

    N_con = len(kept_constraints)
    N_surv = len(survivors)
    gt_xywh = _bbox_xyxy_to_xywh(
        torch.tensor(
            [[e.bbox[0], e.bbox[1], e.bbox[2], e.bbox[3]] for e in survivors],
            dtype=torch.float32,
        )
    )
    targets = {
        "coord": torch.zeros((N_surv, 4), dtype=torch.float32),
        "violation": torch.tensor(violation_labels, dtype=torch.float32).view(-1, 1),
        "existence": torch.ones((N_surv, 1), dtype=torch.float32),
        "gt_boxes": gt_xywh,
        "proposal_target": torch.tensor(proposal_targets, dtype=torch.float32),
        "proposal_violation_mask": torch.tensor(
            [v > 0.5 for v in violation_labels], dtype=torch.bool
        ),
        "proposal_type_target": torch.tensor(
            [int(t[4]) for t in proposal_targets], dtype=torch.long
        ),
    }
    return hetero_data, targets


# ---------------------------------------------------------------------------
# Evaluation (mirrors train_violation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_violation(
    model: BipartiteGNNCorrector,
    dataset: Dataset,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate violation detection accuracy."""
    model.eval()
    preds_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []

    for data, targets in dataset:
        data = data.to(device)
        predictions = model(data)
        if "violation" not in predictions:
            continue
        preds_all.append(predictions["violation"].cpu())
        labels_all.append(targets["violation"].cpu())

    if not preds_all:
        return {"acc": 0.0, "pos_frac": 0.0, "n": 0.0}

    preds = torch.cat(preds_all).view(-1)
    labels = torch.cat(labels_all).view(-1).float()
    n = labels.numel()
    acc = ((preds > 0.5) == (labels > 0.5)).float().mean().item()
    pos_frac = labels.mean().item()
    return {"acc": acc, "pos_frac": pos_frac, "n": float(n)}


@torch.no_grad()
def evaluate_proposal(
    model: BipartiteGNNCorrector,
    dataset: Dataset,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate element proposal MSE and type accuracy."""
    model.eval()
    mse_total = 0.0
    mse_count = 0
    type_correct = 0
    type_total = 0

    for data, targets in dataset:
        data = data.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        predictions = model(data)

        if "proposal" not in predictions or "proposal_violation_mask" not in targets:
            continue

        mask = targets["proposal_violation_mask"]
        if mask.sum() == 0:
            continue

        pred = predictions["proposal"]
        tgt = targets["proposal_target"]
        mse_total += torch.nn.functional.mse_loss(
            pred[mask], tgt[mask, :4]
        ).item() * mask.sum().item()
        mse_count += mask.sum().item()

        if "proposal_type" in predictions and "proposal_type_target" in targets:
            type_logits = predictions["proposal_type"]
            type_target = targets["proposal_type_target"]
            type_pred = type_logits.argmax(dim=1)
            type_correct += (type_pred[mask] == type_target[mask]).sum().item()
            type_total += mask.sum().item()

    n = max(mse_count, 1)
    return {
        "proposal_mse": mse_total / n,
        "type_acc": type_correct / max(type_total, 1),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rico_data(
    n_samples: int,
    drop_ratio: float,
    seed: int,
    use_visual: bool,
    builder: BipartiteGraphBuilder,
    feat_dir: Path = _VISUAL_FEAT_DIR,
    visual_dim: int = 192,
    allowed_constraint_types: set[ConstraintType] | None = None,
) -> list[Tuple[Any, Dict[str, torch.Tensor]]]:
    """Load RICO graphs with or without visual features.

    Args:
        n_samples: Number of JSONs to process.
        drop_ratio: Fraction of elements to drop per graph.
        seed: RNG seed for element dropping.
        use_visual: Whether to load and attach visual features.
        builder: Graph builder instance.
        feat_dir: Directory containing precomputed ``.pt`` visual features.
        visual_dim: Dimension of visual features (192 for vit_tiny, 768 for DINOv2).
        allowed_constraint_types: Optional constraint type filter.

    Returns:
        List of ``(hetero_data, targets)`` pairs.
    """
    all_jsons = sorted(_RICO_DIR.glob("*.json"))[:n_samples]
    logger.info("Loading %d RICO graphs (use_visual=%s, visual_dim=%d) ...",
                len(all_jsons), use_visual, visual_dim)

    all_graphs: list[Tuple[Any, Dict[str, torch.Tensor]]] = []
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
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        # Load visual features if requested.
        visual_feats = None
        if use_visual:
            uid = path.stem
            visual_feats = _load_visual_features(uid, len(gt_elements),
                                                  feat_dir, visual_dim)
            if visual_feats is None:
                n_skipped += 1
                continue

        result = build_violation_graph(
            gt_elements, builder,
            drop_ratio=drop_ratio,
            seed=seed + idx,
            allowed_constraint_types=allowed_constraint_types,
            visual_features=visual_feats,
        )
        if result is None:
            n_skipped += 1
            continue

        all_graphs.append(result)

    dt = time.time() - t0
    logger.info(
        "Loaded %d graphs (%d skipped) in %.1fs", len(all_graphs), n_skipped, dt
    )
    return all_graphs


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    train_dataset: Dataset,
    val_dataset: Dataset,
    element_dim: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
    checkpoint_path: Path,
    coord_weight: float = 0.0,
    violation_weight: float = 1.0,
    existence_weight: float = 0.0,
    proposal_weight: float = 1.0,
    proposal_type_weight: float = 0.5,
    fusion_dim: int | None = None,
) -> BipartiteGNNCorrector:
    """Train a BipartiteGNNCorrector on violation + proposal tasks.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        element_dim: Element feature dimension (5 without visual, 5+visual_dim with).
        hidden_dim: Hidden dimension for encoder and heads.
        lr: Learning rate.
        epochs: Number of training epochs.
        checkpoint_path: Where to save the best model.
        coord_weight: Coordinate regression loss weight.
        violation_weight: Violation prediction loss weight.
        existence_weight: Existence prediction loss weight.
        proposal_weight: Proposal bbox loss weight.
        proposal_type_weight: Proposal type prediction loss weight.
        fusion_dim: If set, enables cross-attention fusion (SplitAndFuse)
            before the GNN encoder.  The encoder receives fusion_dim-d
            vectors instead of element_dim-d vectors.

    Returns:
        Trained model (best checkpoint loaded).
    """
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)

    model = BipartiteGNNCorrector(
        element_dim=element_dim,
        constraint_dim=11,
        hidden_dim=hidden_dim,
        dropout=0.1,
        coord_weight=coord_weight,
        existence_weight=existence_weight,
        fusion_dim=fusion_dim,
    ).to(DEVICE)

    model.loss_fn.violation_weight = violation_weight
    model.loss_fn.coord_weight = coord_weight
    model.loss_fn.existence_weight = existence_weight
    model.loss_fn.alignment_weight = 0.0
    model.proposal_weight = proposal_weight
    model.proposal_type_weight = proposal_type_weight

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params (element_dim=%d, hidden_dim=%d)",
                n_params, element_dim, hidden_dim)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
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

        metrics = evaluate_violation(model, val_dataset, DEVICE)
        prop_metrics = evaluate_proposal(model, val_dataset, DEVICE)
        logger.info(
            "Epoch %2d/%d — train: %.4f | val: %.4f | acc: %.3f | prop_mse: %.4f | type_acc: %.3f",
            epoch, epochs, avg_train_loss, avg_val_loss,
            metrics["acc"], prop_metrics["proposal_mse"], prop_metrics["type_acc"],
        )

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best checkpoint.
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    logger.info("Best val loss: %.6f (loaded checkpoint)", best_val_loss)
    return model


# ---------------------------------------------------------------------------
# Evaluation wrappers
# ---------------------------------------------------------------------------


def evaluate_model(
    model: BipartiteGNNCorrector,
    dataset: Dataset,
) -> dict[str, float]:
    """Run both violation and proposal evaluation, return flat dict."""
    viol = evaluate_violation(model, dataset, DEVICE)
    prop = evaluate_proposal(model, dataset, DEVICE)
    return {
        "violation_acc": viol["acc"],
        "violation_n": viol["n"],
        "proposal_mse": prop["proposal_mse"],
        "type_acc": prop["type_acc"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Visual feature fusion experiment (ViT-tiny vs DINOv2)"
    )
    parser.add_argument("--n", type=int, default=500,
                        help="Number of RICO images to use (default 500)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--drop-ratio", type=float, default=0.4)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-dim", type=int, default=192,
                        help="Visual feature dimension (192 for vit_tiny, 768 for DINOv2)")
    parser.add_argument("--feat-dir", type=str, default=None,
                        help="Directory with precomputed .pt visual features. "
                             "Default: data/rico_local/visual_features for 192-dim, "
                             "data/rico_local/visual_features_dinov2 for 768-dim.")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    # Determine feat dir based on visual_dim if not explicitly set.
    if args.feat_dir is not None:
        feat_dir = Path(args.feat_dir)
    elif args.visual_dim == 768:
        feat_dir = _DINOV2_FEAT_DIR
    else:
        feat_dir = _VISUAL_FEAT_DIR

    visual_dim = args.visual_dim
    element_dim = 5 + visual_dim

    logger.info("=" * 60)
    logger.info("VISUAL FEATURE EXPERIMENT")
    logger.info("=" * 60)
    logger.info("Device: %s | n=%d | epochs=%d | hidden=%d | drop=%.2f",
                DEVICE, args.n, args.epochs, args.hidden, args.drop_ratio)
    logger.info("Visual dim: %d | element_dim: %d", visual_dim, element_dim)
    logger.info("RICO dir: %s", _RICO_DIR)
    logger.info("Feat dir: %s", feat_dir)

    # ------------------------------------------------------------------
    # 1. Load data with visual features
    # ------------------------------------------------------------------
    builder = BipartiteGraphBuilder()

    all_graphs_vis = load_rico_data(
        n_samples=args.n,
        drop_ratio=args.drop_ratio,
        seed=args.seed,
        use_visual=True,
        builder=builder,
        feat_dir=feat_dir,
        visual_dim=visual_dim,
    )
    if len(all_graphs_vis) < 10:
        logger.error("Not enough visual graphs: %d", len(all_graphs_vis))
        return

    n_val = max(1, int(len(all_graphs_vis) * args.val_split))
    split = len(all_graphs_vis) - n_val
    train_vis = GraphListDataset(all_graphs_vis[:split])
    val_vis = GraphListDataset(all_graphs_vis[split:])

    logger.info("Visual data: %d train / %d val", len(train_vis), len(val_vis))

    # ------------------------------------------------------------------
    # 2. Train with visual features (simple concat)
    # ------------------------------------------------------------------
    logger.info("-" * 60)
    logger.info("TRAINING: Simple Concat — %d-d visual features", visual_dim)
    logger.info("-" * 60)

    checkpoint_path = _CHECKPOINT_DIR / f"visual_fusion_model_dim{visual_dim}.pt"
    model = train_model(
        train_dataset=train_vis,
        val_dataset=val_vis,
        element_dim=element_dim,
        hidden_dim=args.hidden,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )

    metrics = evaluate_model(model, val_vis)
    logger.info("Results: %s", metrics)

    # ------------------------------------------------------------------
    # 3. Report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"{'Metric':<25s} | {'Value':>14s}")
    print("-" * 60)
    for k, v in metrics.items():
        print(f"{k:<25s} | {v:>14.4f}")
    print("=" * 60)

    print()
    logger.info("Experiment complete.")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
