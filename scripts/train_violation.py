#!/usr/bin/env python3
"""Constraint violation detection as a proxy for structural completion.

Experiment: given a partial GUI layout (elements randomly removed), can the
GNN detect which constraints are violated because their participating elements
are missing?

Rationale: when an element is removed, all constraints that involved it become
incomplete.  The violation prediction head should learn to detect this
signal — constraints with fewer-than-expected connected elements get a high
violation score.

This uses the *existing* model heads and loss.  Only the data pipeline is new.

Usage:
  python scripts/train_violation.py --n 500 --epochs 50 --hidden 128
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
from bipartite_gnn_gui.model.heads import N_TYPES
from scripts.run_experiment import (
    DEVICE,
    GraphListDataset,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
)

logger = logging.getLogger(__name__)
MASK_TOKEN = -1.0

# RICO semantic type → index mapping.
TYPE_TO_IDX: dict[str, int] = {
    "button": 0, "text": 1, "icon": 2, "image": 3,
    "input": 4, "container": 5, "list": 6,
}
# Any label not in the map → index 7 ("other")
TYPE_UNKNOWN_IDX: int = 7


def _type_idx(label: str) -> int:
    return TYPE_TO_IDX.get(label.lower(), TYPE_UNKNOWN_IDX)


def build_violation_graph(
    gt_elements: list[ElementNode],
    builder: BipartiteGraphBuilder,
    drop_ratio: float = 0.3,
    seed: int | None = None,
) -> Tuple[Any, Dict[str, torch.Tensor]] | None:
    """Build a graph from a *partial* layout and label violated constraints.

    Also computes **proposal targets**: for each violated constraint, the
    GT bbox of the missing element(s) that should complete it.

    Args:
        gt_elements: Normalised ground-truth elements.
        builder: Graph builder.
        drop_ratio: Fraction of elements to remove (default 0.3).
        seed: RNG seed.

    Returns:
        ``(hetero_data, targets)`` or ``None`` if degenerate.
    """
    if len(gt_elements) < 3:
        return None

    N = len(gt_elements)

    # Full constraint extraction (on all elements before removal).
    full_constraints = extract_all_constraints(gt_elements)
    if len(full_constraints) == 0:
        return None

    # Randomly select survivors.
    rng = torch.Generator() if seed is not None else None
    if rng is not None:
        rng.manual_seed(seed)
    survivor_mask = torch.rand(N, generator=rng) >= drop_ratio

    if survivor_mask.sum() < 2:
        return None

    survivor_indices_old = torch.where(survivor_mask)[0].tolist()
    old_to_new = {old: new for new, old in enumerate(survivor_indices_old)}
    removed_indices = torch.where(~survivor_mask)[0].tolist()

    kept_constraints = []
    violation_labels = []
    proposal_targets = []  # (N_con, 5) — bbox + type_idx of missing element

    for con in full_constraints:
        src_set = set(con.source_indices) & set(survivor_indices_old)
        tgt_set = set(con.target_indices) & set(survivor_indices_old)
        all_surviving = src_set | tgt_set

        if len(all_surviving) < 1:
            continue

        # All original participants.
        all_original = set(con.source_indices + con.target_indices)
        # Which participants were removed?
        missing_idx = [i for i in all_original if i in removed_indices]

        is_violated = len(all_surviving) < 2
        violation_labels.append(1.0 if is_violated else 0.0)

        # Proposal target: average bbox of missing GT elements + type.
        if is_violated and missing_idx:
            x1s = [gt_elements[i].bbox[0] for i in missing_idx]
            y1s = [gt_elements[i].bbox[1] for i in missing_idx]
            x2s = [gt_elements[i].bbox[2] for i in missing_idx]
            y2s = [gt_elements[i].bbox[3] for i in missing_idx]
            # Use the type of the first missing element as target.
            type_idx = _type_idx(gt_elements[missing_idx[0]].label)
            proposal_targets.append([
                sum(x1s) / len(x1s), sum(y1s) / len(y1s),
                sum(x2s) / len(x2s), sum(y2s) / len(y2s),
                float(type_idx),
            ])
        else:
            proposal_targets.append([0.0, 0.0, 0.0, 0.0, 0.0])

        # Remap indices.
        con.source_indices = [old_to_new[i] for i in src_set]
        con.target_indices = [old_to_new[i] for i in tgt_set]
        kept_constraints.append(con)

    if len(kept_constraints) == 0:
        return None

    survivors = [gt_elements[i] for i in survivor_indices_old]
    hetero_data = builder.build(survivors, kept_constraints)

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
        # Proposal targets.
        "proposal_target": torch.tensor(proposal_targets, dtype=torch.float32),  # (N_con, 5)
        "proposal_violation_mask": torch.tensor(
            [v > 0.5 for v in violation_labels], dtype=torch.bool
        ),
        "proposal_type_target": torch.tensor(
            [int(t[4]) for t in proposal_targets], dtype=torch.long
        ),
    }
    return hetero_data, targets


def _bbox_xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1)


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
    """Evaluate element proposal accuracy on violated constraints."""
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
            pred[mask], tgt[mask, :4]  # tgt is (N_con, 5), only first 4 are bbox
        ).item() * mask.sum().item()
        mse_count += mask.sum().item()

        # Type accuracy.
        if "proposal_type" in predictions and "proposal_type_target" in targets:
            type_logits = predictions["proposal_type"]  # (N_con, N_TYPES)
            type_target = targets["proposal_type_target"]  # (N_con,)
            type_pred = type_logits.argmax(dim=1)  # (N_con,)
            type_correct += (type_pred[mask] == type_target[mask]).sum().item()
            type_total += mask.sum().item()

    n = max(mse_count, 1)
    return {
        "proposal_mse": mse_total / n,
        "type_acc": type_correct / max(type_total, 1),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Constraint violation detection experiment"
    )
    parser.add_argument("--n", type=int, default=500, help="RICO samples")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--drop-ratio", type=float, default=0.4,
                        help="Fraction of elements to remove")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str,
                        default="checkpoints/violation_detection")
    parser.add_argument("--rico-dir", type=str,
                        default="data/rico_local/combined")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s", stream=sys.stdout,
    )

    logger.info("=" * 55)
    logger.info("VIOLATION DETECTION — Structural Completeness Proxy")
    logger.info("=" * 55)
    logger.info("Device: %s | n=%d | epochs=%d | hidden=%d | drop=%.2f",
                DEVICE, args.n, args.epochs, args.hidden, args.drop_ratio)

    rico_dir = Path(args.rico_dir)
    all_jsons = sorted(rico_dir.glob("*.json"))[:args.n]
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
        gt_elements = [e for e in gt_elements
                       if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]]

        result = build_violation_graph(
            gt_elements, builder, drop_ratio=args.drop_ratio, seed=args.seed
        )
        if result is None:
            n_skipped += 1
            continue
        all_graphs.append(result)

    dt = time.time() - t0
    logger.info("Built %d graphs (%d skipped) in %.1fs",
                len(all_graphs), n_skipped, dt)

    if len(all_graphs) < 2:
        logger.error("Need ≥2 graphs")
        return

    # Statistics.
    n_elems = [g[0]["element"].x.shape[0] for g in all_graphs]
    n_cons = [g[0]["constraint"].x.shape[0] for g in all_graphs]
    n_edges = [(g[0]["element", "to", "constraint"].edge_index.shape[1]
                if "element" in g[0].node_types else 0)
               for g in all_graphs]
    import statistics as stat
    logger.info("Graphs: %.1f±%.1f elem, %.1f±%.1f con, %.1f±%.1f edges",
                stat.mean(n_elems), stat.stdev(n_elems) if len(n_elems) > 1 else 0,
                stat.mean(n_cons), stat.stdev(n_cons) if len(n_cons) > 1 else 0,
                stat.mean(n_edges), stat.stdev(n_edges) if len(n_edges) > 1 else 0)

    split_idx = int(len(all_graphs) * (1.0 - args.val_split))
    train_dataset = GraphListDataset(all_graphs[:split_idx])
    val_dataset = GraphListDataset(all_graphs[split_idx:])
    train_loader = DataLoader(train_dataset, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=None, shuffle=False)
    logger.info("Split: %d train / %d val", len(train_dataset), len(val_dataset))

    # Model — enable violation + proposal, zero out other heads.
    model = BipartiteGNNCorrector(
        hidden_dim=args.hidden, dropout=0.1,
        coord_weight=0.0, existence_weight=0.0,
    ).to(DEVICE)
    model.loss_fn.violation_weight = 1.0
    model.loss_fn.coord_weight = 0.0
    model.loss_fn.existence_weight = 0.0
    model.loss_fn.alignment_weight = 0.0
    model.proposal_weight = 1.0  # enable proposal loss
    model.proposal_type_weight = 0.5  # enable type prediction loss

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

        metrics = evaluate_violation(model, val_dataset, DEVICE)
        prop_metrics = evaluate_proposal(model, val_dataset, DEVICE)
        logger.info(
            "Epoch %2d/%d — train: %.4f | val: %.4f | acc: %.3f | prop_mse: %.4f | n_con: %.0f",
            epoch, args.epochs, avg_train_loss, avg_val_loss,
            metrics["acc"], prop_metrics["proposal_mse"], metrics["n"],
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
    logger.info("Final violation detection accuracy: %.4f",
                evaluate_violation(model, val_dataset, DEVICE)["acc"])


if __name__ == "__main__":
    main()
