#!/usr/bin/env python3
"""Phase 7.2 — Constraint type ablation experiment.

Measures which constraint types contribute to violation detection accuracy
by sequentially removing groups of constraint types and measuring the impact
on violation accuracy and proposal MSE.

Usage:
    python experiments/ablation_constraints.py --n 500 --epochs 50 --hidden 128 --drop-ratio 0.6
    python experiments/run.py ablation --n 500 --epochs 50 --hidden 128 --drop-ratio 0.6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ConstraintType, ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import (
    DEVICE,
    GraphListDataset,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
)
from scripts.train_violation import (
    build_violation_graph,
    evaluate_violation,
    evaluate_proposal,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ablation groups
# ---------------------------------------------------------------------------

ALL_TYPES: set[ConstraintType] = set(ConstraintType)

ALIGNMENT_TYPES: set[ConstraintType] = {
    ConstraintType.ALIGN_LEFT,
    ConstraintType.ALIGN_RIGHT,
    ConstraintType.ALIGN_TOP,
    ConstraintType.ALIGN_BOTTOM,
    ConstraintType.CENTER_X,
    ConstraintType.CENTER_Y,
    ConstraintType.SAME_SIZE,
}

ABLATION_GROUPS: list[dict[str, Any]] = [
    {
        "name": "baseline",
        "label": "All 10 constraint types (control)",
        "allowed_types": ALL_TYPES,
    },
    {
        "name": "no_alignment",
        "label": "Remove all alignment types (LEFT/RIGHT/TOP/BOTTOM/CENTER_X/Y + SAME_SIZE)",
        "allowed_types": ALL_TYPES - ALIGNMENT_TYPES,
    },
    {
        "name": "no_containment",
        "label": "Remove CONTAINMENT only",
        "allowed_types": ALL_TYPES - {ConstraintType.CONTAINMENT},
    },
    {
        "name": "no_spacing",
        "label": "Remove SPACING only",
        "allowed_types": ALL_TYPES - {ConstraintType.SPACING},
    },
    {
        "name": "no_grid",
        "label": "Remove GRID only",
        "allowed_types": ALL_TYPES - {ConstraintType.GRID},
    },
    {
        "name": "only_alignment",
        "label": "Keep only alignment types (remove SPACING, CONTAINMENT, GRID)",
        "allowed_types": ALIGNMENT_TYPES,
    },
]


def count_constraints_by_type(
    all_jsons: list[Path],
    allowed_types: set[ConstraintType] | None = None,
    max_samples: int = 200,
) -> dict[str, int]:
    """Quickly count constraints per type for a sample of RICO JSONs.

    Returns a dict mapping type name → count.
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    n_processed = 0

    for path in all_jsons:
        if n_processed >= max_samples:
            break
        parsed = parse_rico_vh(path)
        if parsed is None:
            continue
        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]
        constraints = extract_all_constraints(gt_elements)
        for c in constraints:
            if allowed_types is None or c.constraint_type in allowed_types:
                counter[c.constraint_type.value] += 1
        n_processed += 1

    return dict(counter)


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------


def run_ablation(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run all ablation groups and return results."""
    rico_dir = Path(args.rico_dir)
    all_jsons = sorted(rico_dir.glob("*.json"))[: args.n]
    logger.info("RICO JSONs: %d", len(all_jsons))

    # --- Pre-compute constraint statistics ---
    logger.info("Computing constraint type statistics on first %d samples...", min(200, args.n))
    type_stats = count_constraints_by_type(all_jsons, max_samples=min(200, args.n))
    logger.info("Constraint distribution: %s", json.dumps(type_stats, indent=2))

    # --- Build graphs once (baseline) and cache them ---
    # For ablation we need to rebuild graphs with different constraint filters.
    # We'll rebuild each time since the filtering is cheap.

    results: list[dict[str, Any]] = []

    for group in ABLATION_GROUPS:
        group_name = group["name"]
        group_label = group["label"]
        allowed_types = group["allowed_types"]
        logger.info("")
        logger.info("=" * 60)
        logger.info("ABLATION GROUP: %s", group_name)
        logger.info("  %s", group_label)
        logger.info("  Allowed types (%d): %s",
                     len(allowed_types),
                     sorted(t.value for t in allowed_types))
        logger.info("=" * 60)

        t0 = time.time()

        # --- Build graphs with filtered constraints ---
        builder = BipartiteGraphBuilder()
        all_graphs: List[Tuple[Any, Dict[str, torch.Tensor]]] = []
        n_skipped = 0
        n_constraints_list: list[int] = []
        n_elements_list: list[int] = []
        n_edges_list: list[int] = []

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

            result = build_violation_graph(
                gt_elements,
                builder,
                drop_ratio=args.drop_ratio,
                seed=args.seed,
                allowed_constraint_types=allowed_types,
            )
            if result is None:
                n_skipped += 1
                continue

            all_graphs.append(result)
            n_constraints_list.append(result[0]["constraint"].x.shape[0])
            n_elements_list.append(result[0]["element"].x.shape[0])
            if "element" in result[0].node_types:
                n_edges_list.append(
                    result[0]["element", "to", "constraint"].edge_index.shape[1]
                )

        build_time = time.time() - t0
        n_graphs = len(all_graphs)

        if n_graphs < 2:
            logger.warning("  Skipping %s: only %d graphs built", group_name, n_graphs)
            results.append({
                "group": group_name,
                "label": group_label,
                "n_constraint_types": len(allowed_types),
                "n_graphs": n_graphs,
                "n_skipped": n_skipped,
                "error": "insufficient graphs",
            })
            continue

        avg_constraints = (
            sum(n_constraints_list) / len(n_constraints_list) if n_constraints_list else 0
        )
        avg_elements = (
            sum(n_elements_list) / len(n_elements_list) if n_elements_list else 0
        )
        avg_edges = (
            sum(n_edges_list) / len(n_edges_list) if n_edges_list else 0
        )

        logger.info(
            "  Built %d graphs (%d skipped) in %.1fs — "
            "%.1f±? elem, %.1f±? con, %.1f±? edges",
            n_graphs, n_skipped, build_time,
            avg_elements, avg_constraints, avg_edges,
        )

        # --- Train/val split ---
        split_idx = int(n_graphs * (1.0 - args.val_split))
        train_dataset = GraphListDataset(all_graphs[:split_idx])
        val_dataset = GraphListDataset(all_graphs[split_idx:])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=None, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=None, shuffle=False
        )
        logger.info("  Split: %d train / %d val", len(train_dataset), len(val_dataset))

        # --- Model ---
        model = BipartiteGNNCorrector(
            hidden_dim=args.hidden,
            dropout=0.1,
            coord_weight=0.0,
            existence_weight=0.0,
        ).to(DEVICE)
        model.loss_fn.violation_weight = 1.0
        model.loss_fn.coord_weight = 0.0
        model.loss_fn.existence_weight = 0.0
        model.loss_fn.alignment_weight = 0.0
        model.proposal_weight = 1.0
        model.proposal_type_weight = 0.5

        n_params = sum(p.numel() for p in model.parameters())
        logger.info("  Model: %d params", n_params)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4
        )

        # --- Training loop ---
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        final_epoch = 0
        final_train_loss = 0.0

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
            final_train_loss = avg_train_loss

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
            final_epoch = epoch

            logger.info(
                "  Epoch %2d/%d — train: %.4f | val: %.4f | acc: %.3f | "
                "prop_mse: %.4f | n_con: %.0f",
                epoch, args.epochs, avg_train_loss, avg_val_loss,
                metrics["acc"], prop_metrics["proposal_mse"], metrics["n"],
            )

            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        "  Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, patience,
                    )
                    break

        # --- Final evaluation ---
        final_metrics = evaluate_violation(model, val_dataset, DEVICE)
        final_prop = evaluate_proposal(model, val_dataset, DEVICE)

        result_entry = {
            "group": group_name,
            "label": group_label,
            "n_constraint_types": len(allowed_types),
            "n_graphs": n_graphs,
            "n_skipped": n_skipped,
            "avg_constraints_per_graph": round(avg_constraints, 1),
            "avg_elements_per_graph": round(avg_elements, 1),
            "avg_edges_per_graph": round(avg_edges, 1),
            "epochs_trained": final_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "violation_acc": round(final_metrics["acc"], 4),
            "violation_pos_frac": round(final_metrics["pos_frac"], 4),
            "violation_n": int(final_metrics["n"]),
            "proposal_mse": round(final_prop["proposal_mse"], 6),
            "type_acc": round(final_prop["type_acc"], 4),
            "train_time_sec": round(time.time() - t0, 1),
        }
        results.append(result_entry)

        logger.info("")
        logger.info("  ── Results for %s ──", group_name)
        logger.info("    Violation accuracy:  %.4f", final_metrics["acc"])
        logger.info("    Proposal MSE:        %.6f", final_prop["proposal_mse"])
        logger.info("    Type accuracy:       %.4f", final_prop["type_acc"])
        logger.info("    Best val loss:       %.6f", best_val_loss)
        logger.info("    Avg constraints:     %.1f", avg_constraints)
        logger.info("    Time:                %.1fs", time.time() - t0)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constraint type ablation experiment"
    )
    parser.add_argument("--n", type=int, default=500, help="RICO samples")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--drop-ratio", type=float, default=0.6,
        help="Fraction of elements to remove",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=str, default="experiments/ablation_results.json",
    )
    parser.add_argument(
        "--rico-dir", type=str, default="data/rico_local/combined",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )

    logger.info("=" * 55)
    logger.info("CONSTRAINT TYPE ABLATION — Phase 7.2")
    logger.info("=" * 55)
    logger.info(
        "Device: %s | n=%d | epochs=%d | hidden=%d | drop=%.2f | lr=%.4f",
        DEVICE, args.n, args.epochs, args.hidden, args.drop_ratio, args.lr,
    )

    results = run_ablation(args)

    # --- Print results table ---
    logger.info("")
    logger.info("=" * 80)
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info("=" * 80)

    header = (
        f"{'Group':<20s} {'#Types':>6s} {'#Cons':>6s} {'#Graphs':>7s} "
        f"{'Acc':>7s} {'PropMSE':>10s} {'TypeAcc':>8s} {'BestLoss':>10s} "
        f"{'Epochs':>7s} {'Time(s)':>8s}"
    )
    sep = "-" * len(header)

    logger.info(header)
    logger.info(sep)

    for r in results:
        if "error" in r:
            logger.info(
                "%-20s %6s %6s %7s %7s %10s %8s %10s %7s %8s",
                r["group"], "-", "-", str(r.get("n_graphs", 0)),
                "ERROR", "-", "-", "-", "-", "-",
            )
        else:
            logger.info(
                "%-20s %6d %6.0f %7d %7.4f %10.6f %8.4f %10.6f %7d %8.1f",
                r["group"],
                r["n_constraint_types"],
                r["avg_constraints_per_graph"],
                r["n_graphs"],
                r["violation_acc"],
                r["proposal_mse"],
                r["type_acc"],
                r["best_val_loss"],
                r["epochs_trained"],
                r["train_time_sec"],
            )

    logger.info(sep)
    logger.info("")

    # --- Save results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n": args.n,
                    "epochs": args.epochs,
                    "hidden": args.hidden,
                    "drop_ratio": args.drop_ratio,
                    "lr": args.lr,
                    "seed": args.seed,
                    "val_split": args.val_split,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    logger.info("Results saved to: %s", output_path)


if __name__ == "__main__":
    main()
