#!/usr/bin/env python3
"""Prepare structural-completion demo assets for the web demo (Tab 3).

Part 1 — curve data: aggregates experiments/completion_results.json (8 runs:
drop_ratio x 2 seeds) into demo_data/completion/curve.json. All runs are
included, including the negative ones (GNN < NN at low drop ratios), so the
demo shows the honest full picture.

Part 2 — single-image demo: loads the joint checkpoint (violation+proposal),
drops 60% of GT elements on one RICO screenshot, runs the GNN proposal head,
and renders a comparison PNG:
  - surviving elements (grey)
  - dropped GT elements (green outline = ground truth)
  - GNN proposals (blue)
  - NN baseline (nearest surviving element's bbox, orange)
Writes demo_data/completion/demo_<img>.png.

Usage:
  python scripts/prepare_completion_demo.py [--image 10027] [--drop 0.6] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import DEVICE, extract_elements, normalize_bbox, parse_rico_vh

logger = logging.getLogger(__name__)

RICO_DIR = ROOT / "data" / "rico_local" / "combined"
RESULTS_JSON = ROOT / "experiments" / "completion_results.json"
CKPT = ROOT / "checkpoints" / "violation_detection_joint" / "best_model.pt"
OUT_DIR = ROOT / "demo_data" / "completion"


def build_curve() -> None:
    """Aggregate completion_results.json into curve.json (all runs, honest)."""
    if not RESULTS_JSON.exists():
        logger.warning("missing %s — skip curve", RESULTS_JSON)
        return
    results = json.loads(RESULTS_JSON.read_text())
    by_drop: dict[float, list[dict]] = {}
    for r in results:
        by_drop.setdefault(float(r["drop_ratio"]), []).append(r)

    curve = []
    for dr in sorted(by_drop):
        runs = by_drop[dr]
        gnn_iou = [r.get("gnn_proposal_iou", 0.0) for r in runs]
        nn_iou = [r.get("baseline_nn_iou", 0.0) for r in runs]
        gnn_mse = [r.get("gnn_proposal_mse", 0.0) for r in runs]
        nn_mse = [r.get("baseline_nn_mse", 0.0) for r in runs]
        curve.append({
            "drop_ratio": dr,
            "gnn_iou_mean": round(sum(gnn_iou) / len(gnn_iou), 4),
            "nn_iou_mean": round(sum(nn_iou) / len(nn_iou), 4),
            "gnn_iou_runs": [round(v, 4) for v in gnn_iou],
            "nn_iou_runs": [round(v, 4) for v in nn_iou],
            "gnn_mse_mean": round(sum(gnn_mse) / len(gnn_mse), 4),
            "nn_mse_mean": round(sum(nn_mse) / len(nn_mse), 4),
            "n_runs": len(runs),
        })

    (OUT_DIR / "curve.json").write_text(json.dumps({
        "metric": "IoU (higher = better)",
        "note": "All 8 runs shown; GNN is trained per drop_ratio (proposal_weight=1.0). "
                "GNN only beats NN at high drop ratios.",
        "points": curve,
    }, indent=1))
    logger.info("curve.json: %d drop-ratio points", len(curve))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None, help="single-image demo (unavailable: no completion ckpt saved)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_curve()
    if args.image:
        logger.warning("single-image demo not supported: completion checkpoint was not persisted "
                       "(completion_results.json comes from per-run training); joint ckpt is a "
                       "different model and would misrepresent completion quality.")
        logger.warning("Tab 3 shows the honest evaluation curve only.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
