#!/usr/bin/env python3
"""Phase 4.9.5 — Systematic evaluation of structural element completion.

Compares the GNN-based element proposal (violation detection + missing
element bbox prediction) against baselines across multiple drop ratios.

Usage:
  python scripts/evaluate_completion.py --n 1000 --epochs 50
  python scripts/evaluate_completion.py --load-json experiments/completion_results.json
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
    _bbox_xyxy_to_xywh,
)

logger = logging.getLogger(__name__)

# ── baselines ──────────────────────────────────────────────────────────────


def baseline_nearest_neighbor(
    targets: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Copy the closest surviving element's bbox as the proposal.

    This baseline tests: how much structural info is already in the existing
    layout? If missing elements are geometrically close to survivors, this
    baseline is hard to beat.
    """
    if "gt_boxes" not in targets or "proposal_target" not in targets:
        return {"mse": 0.0, "iou": 0.0}
    gt_boxes = targets["gt_boxes"]  # (N_surv, 4) xywh
    mask = targets.get("proposal_violation_mask", torch.zeros(0, dtype=torch.bool))
    prop_tgt = targets["proposal_target"]  # (N_con, 4) xyxy
    if mask.sum() == 0 or gt_boxes.shape[0] < 2:
        return {"mse": 0.0, "iou": 0.0}

    # Convert prop_tgt from xyxy to xywh for comparison.
    p_masked = prop_tgt[mask][:, :4]  # (N_violated, 4) xyxy, drop type col
    gt_xyxy = torch.stack([
        gt_boxes[:, 0] - gt_boxes[:, 2] / 2,
        gt_boxes[:, 1] - gt_boxes[:, 3] / 2,
        gt_boxes[:, 0] + gt_boxes[:, 2] / 2,
        gt_boxes[:, 1] + gt_boxes[:, 3] / 2,
    ], dim=1)  # (N_surv, 4) xyxy

    # For each violation, find nearest survivor's bbox.
    p_compat = p_masked[:, None, :]  # (N_v, 1, 4)
    gt_compat = gt_xyxy[None, :, :]  # (1, N_s, 4)
    # L1 distance between bbox corners.
    dists = (p_compat - gt_compat).abs().sum(dim=2)  # (N_v, N_s)
    nearest_idx = dists.argmin(dim=1)
    nearest_boxes = gt_xyxy[nearest_idx]

    mse = F.mse_loss(nearest_boxes, p_masked).item()
    iou = _batch_iou(nearest_boxes, p_masked).mean().item()
    return {"mse": mse, "iou": iou}


def baseline_center(
    targets: Dict[str, torch.Tensor],
    img_size: Tuple[float, float] = (1440.0, 2560.0),
) -> Dict[str, float]:
    """Always predict layout center as the missing element.

    A trivial spatial prior — if layouts are centered, this could be hard
    to beat. Provides the lower bound for MSE and IoU.
    """
    if "proposal_target" not in targets:
        return {"mse": 0.0, "iou": 0.0}
    prop_tgt = targets.get("proposal_target", torch.zeros(0, 4))
    mask = targets.get("proposal_violation_mask", torch.zeros(0, dtype=torch.bool))
    if mask.sum() == 0:
        return {"mse": 0.0, "iou": 0.0}

    p_masked = prop_tgt[mask][:, :4]  # drop type col, (N_violated, 4)
    cx, cy = 0.5, 0.5
    w, h = 0.05, 0.05
    center_box = torch.tensor(
        [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]],
        dtype=torch.float32,
    ).expand(mask.sum(), -1)

    mse = F.mse_loss(center_box, p_masked).item()
    iou = _batch_iou(center_box, p_masked).mean().item()
    return {"mse": mse, "iou": iou}


@torch.no_grad()
def _batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for two sets of boxes in xyxy format."""
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-8)


# ── GNN evaluation ─────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_gnn(
    model: BipartiteGNNCorrector,
    dataset: GraphListDataset,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate violation accuracy and proposal quality."""
    model.eval()
    v_acc_all = []
    mse_all = []
    iou_all = []
    n_total = 0

    for data, targets in dataset:
        data = data.to(device)
        t = {k: v.to(device) for k, v in targets.items()}
        preds = model(data)

        # Violation accuracy.
        if "violation" in preds and "violation" in t:
            v_acc_all.append(
                ((preds["violation"].view(-1) > 0.5)
                 == (t["violation"].view(-1) > 0.5)).float()
            )
            n_total += t["violation"].numel()

        # Proposal.
        mask = t.get("proposal_violation_mask", torch.zeros(0, dtype=torch.bool))
        if "proposal" in preds and mask.sum() > 0:
            p_masked = preds["proposal"][mask]
            tgt_masked = t["proposal_target"][mask, :4]  # only bbox cols, not type idx
            mse_all.append(F.mse_loss(p_masked, tgt_masked))
            iou_all.append(_batch_iou(p_masked, tgt_masked).mean())

    return {
        "violation_acc": float(torch.cat(v_acc_all).mean().cpu()) if v_acc_all else 0.0,
        "proposal_mse": float(torch.stack(mse_all).mean().cpu()) if mse_all else 0.0,
        "proposal_iou": float(torch.stack(iou_all).mean().cpu()) if iou_all else 0.0,
        "n_violated": n_total,
    }


# ── Data loading ───────────────────────────────────────────────────────────


def load_rico_graphs(
    rico_dir: str,
    n: int,
    builder: Any,
    drop_ratio: float,
    seed: int,
) -> Tuple[GraphListDataset, GraphListDataset, List[Dict]]:
    """Load RICO, apply random dropping, return train/val split."""
    all_jsons = sorted(Path(rico_dir).glob("*.json"))[:n]
    all_graphs: List[Tuple[Any, Dict[str, torch.Tensor]]] = []
    baseline_info = []
    n_skipped = 0

    for path in all_jsons:
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
            gt_elements, builder, drop_ratio=drop_ratio, seed=seed
        )
        if result is None:
            n_skipped += 1
            continue
        all_graphs.append(result)
        baseline_info.append({
            "img_size": (float(img_w), float(img_h)),
            "targets": {k: v.clone() for k, v in result[1].items()},
        })

    split_idx = int(len(all_graphs) * 0.8)
    train_ds = GraphListDataset(all_graphs[:split_idx])
    val_ds = GraphListDataset(all_graphs[split_idx:])
    return train_ds, val_ds, baseline_info[split_idx:]


# ── main ───────────────────────────────────────────────────────────────────


def evaluate_drop_ratio(
    drop_ratio: float,
    n: int,
    epochs: int,
    hidden: int,
    lr: float,
    rico_dir: str,
    seed: int,
) -> Dict[str, Any]:
    """Train + evaluate at a single drop ratio. Returns metrics dict."""
    from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder

    builder = BipartiteGraphBuilder()
    logger.info("\n── drop=%.1f n=%d seed=%d ──", drop_ratio, n, seed)

    train_ds, val_ds, bl_info = load_rico_graphs(
        rico_dir, n, builder, drop_ratio, seed
    )
    logger.info("Train=%d Val=%d", len(train_ds), len(val_ds))

    if len(val_ds) < 1:
        return {"drop_ratio": drop_ratio, "error": "no val data"}

    model = BipartiteGNNCorrector(
        hidden_dim=hidden, dropout=0.1,
        coord_weight=0.0, existence_weight=0.0,
    ).to(DEVICE)
    model.loss_fn.violation_weight = 1.0
    model.loss_fn.coord_weight = 0.0
    model.loss_fn.existence_weight = 0.0
    model.loss_fn.alignment_weight = 0.0
    model.proposal_weight = 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float("inf")
    patience_cnt = 0

    train_loader = DataLoader(train_ds, batch_size=None, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=None, shuffle=False)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        nb = 0
        for data, targets in train_loader:
            data = data.to(DEVICE)
            targets = {k: v.to(DEVICE) for k, v in targets.items()}
            opt.zero_grad()
            preds = model(data)
            loss = model.compute_loss(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            nb += 1

        avg_metrics = evaluate_gnn(model, val_ds, DEVICE)
        # Log every 10 epochs.
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info("  ep %2d train=%.4f acc=%.3f mse=%.4f iou=%.4f",
                        epoch, train_loss / max(nb, 1),
                        avg_metrics["violation_acc"],
                        avg_metrics["proposal_mse"],
                        avg_metrics["proposal_iou"])

        if avg_metrics["proposal_mse"] < best_val_loss:
            best_val_loss = avg_metrics["proposal_mse"]
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                logger.info("  early stop @ ep %d", epoch)
                break

    final = evaluate_gnn(model, val_ds, DEVICE)

    # Baselines on val set.
    nn_mses, nn_ious = [], []
    ctr_mses, ctr_ious = [], []
    for info in bl_info:
        nn = baseline_nearest_neighbor(info["targets"])
        nn_mses.append(nn["mse"])
        nn_ious.append(nn["iou"])
        ctr = baseline_center(info["targets"], info["img_size"])
        ctr_mses.append(ctr["mse"])
        ctr_ious.append(ctr["iou"])

    return {
        "drop_ratio": drop_ratio,
        "n_samples": n,
        "seed": seed,
        "gnn_violation_acc": final["violation_acc"],
        "gnn_proposal_mse": final["proposal_mse"],
        "gnn_proposal_iou": final["proposal_iou"],
        "baseline_nn_mse": float(torch.tensor(nn_mses).mean()) if nn_mses else 0.0,
        "baseline_nn_iou": float(torch.tensor(nn_ious).mean()) if nn_ious else 0.0,
        "baseline_center_mse": float(torch.tensor(ctr_mses).mean()) if ctr_mses else 0.0,
        "baseline_center_iou": float(torch.tensor(ctr_ious).mean()) if ctr_ious else 0.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 4.9.5 — structural completion evaluation"
    )
    parser.add_argument("--n", type=int, default=1000, help="RICO samples per drop ratio")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--rico-dir", type=str, default="data/rico_local/combined")
    parser.add_argument("--drop-ratios", type=str, default="0.2,0.4,0.6,0.8")
    parser.add_argument("--seeds", type=str, default="42,73,99")
    parser.add_argument("--output", type=str, default="experiments/completion_results.json")
    parser.add_argument("--load-json", type=str, default=None,
                        help="Skip training, just report from existing JSON")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s", stream=sys.stdout,
    )

    if args.load_json:
        with open(args.load_json) as f:
            results = json.load(f)
        report(results)
        return

    drop_ratios = [float(x) for x in args.drop_ratios.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    all_results = []

    for dr in drop_ratios:
        for seed in seeds:
            result = evaluate_drop_ratio(
                drop_ratio=dr,
                n=args.n,
                epochs=args.epochs,
                hidden=args.hidden,
                lr=args.lr,
                rico_dir=args.rico_dir,
                seed=seed,
            )
            all_results.append(result)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("\nResults saved to %s", args.output)

    report(all_results)


def report(results: List[Dict[str, Any]]) -> None:
    """Print formatted report."""
    # Aggregate by drop_ratio.
    from itertools import groupby

    logger.info("")
    logger.info("=" * 100)
    logger.info("COMPLETION EVALUATION REPORT")
    logger.info("=" * 100)

    results.sort(key=lambda r: r["drop_ratio"])
    for dr, group in groupby(results, key=lambda r: r["drop_ratio"]):
        items = list(group)
        logger.info("\n── Drop ratio: %.1f (%d seeds) ──", dr, len(items))
        logger.info(
            "  %-25s %-12s %-12s %-12s %-12s %-12s",
            "Metric", "GNN", "NN-Baseline", "Center-Baseline", "Best-GNN", "Best-BL"
        )
        keys = [
            ("gnn_violation_acc", "Acc", "{:.3f}"),
            ("gnn_proposal_mse", "MSE", "{:.4f}"),
            ("gnn_proposal_iou", "IoU", "{:.4f}"),
        ]
        for key, label, fmt in keys:
            vals = [i.get(key, 0.0) for i in items]
            nn_key = "baseline_nn_iou" if "iou" in key else "baseline_nn_mse"
            nn_vals = [i.get(nn_key, 0.0) for i in items]
            ct_key = "baseline_center_iou" if "iou" in key else "baseline_center_mse"
            ct_vals = [i.get(ct_key, 0.0) for i in items]
            best_gnn = max(vals) if "iou" in key or "acc" in key else min(vals)
            best_nn = max(nn_vals) if nn_vals and nn_vals[0] > 0.001 else min(nn_vals)
            mean_str = fmt.format(sum(vals) / len(vals)) if vals else "N/A"
            nn_str = fmt.format(sum(nn_vals) / len(nn_vals)) if nn_vals else "N/A"
            ct_str = fmt.format(sum(ct_vals) / len(ct_vals)) if ct_vals else "N/A"
            logger.info(f"  %-25s {mean_str:<12s} {nn_str:<12s} {ct_str:<12s}", label)

    logger.info("")
    logger.info("=" * 100)


if __name__ == "__main__":
    main()
