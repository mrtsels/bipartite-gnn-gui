#!/usr/bin/env python3
"""Hyperparameter sweep for bipartite GNN training pipeline.

Runs 6 configurations across (hidden_dim, lr, noise_scale) combinations,
logs results to experiments/results.json, and prints a comparison table.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.run_experiment import ExperimentConfig, run_experiment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configs to sweep: (name, hidden_dim, lr, noise_scale, epochs)
# ---------------------------------------------------------------------------
CONFIGS = [
    ("hd64_small-noise",   64,   1e-3, 0.08, 50),
    ("hd128_small-noise",  128,  1e-3, 0.08, 50),
    ("hd64_big-noise",     64,   1e-3, 0.20, 50),
    ("hd128_big-noise",    128,  1e-3, 0.20, 50),
    ("hd128_low-lr",       128,  5e-4, 0.12, 50),
    ("hd256",              256,  1e-3, 0.12, 50),
]

RESULTS_FILE = Path(__file__).resolve().parent / "results.json"
CHECKPOINT_BASE = _PROJECT_ROOT / "checkpoints" / "sweep"


def _setup_logging() -> None:
    """Configure simple logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def _make_config(
    name: str,
    hidden_dim: int,
    lr: float,
    noise_scale: float,
    epochs: int,
) -> ExperimentConfig:
    """Create an ExperimentConfig for one sweep run."""
    cfg = ExperimentConfig()
    cfg.n_samples = 200
    cfg.val_split = 0.2
    cfg.hidden_dim = hidden_dim
    cfg.lr = lr
    cfg.noise_scale = noise_scale
    cfg.epochs = epochs
    cfg.checkpoint_dir = str(CHECKPOINT_BASE / name)
    cfg.rico_dir = "/Users/minimx/bipartite-gnn-gui/data/rico_local/combined"
    cfg.log_level = "WARNING"  # reduce noise during sweep
    return cfg


def _save_result(result_entry: dict) -> None:
    """Append one result entry to the JSON results file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            existing = json.load(f)
    else:
        existing = []
    existing.append(result_entry)
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)
    logger.info("Result appended to %s", RESULTS_FILE)


def _print_table(results: list[dict]) -> None:
    """Print a formatted comparison table of all sweep results."""
    sep = "─" * 120
    header = (
        f"{'name':<22} | {'best_val':>9} | {'recall':>6} | {'precision':>9} "
        f"| {'f1':>6} | {'pos_err':>7} | {'noop_rec':>8} | {'improv':>10}"
    )
    print()
    print(sep)
    print("  HYPERPARAMETER SWEEP RESULTS")
    print(sep)
    print(header)
    print(sep)
    for entry in results:
        r = entry["results"]
        name = entry["name"]
        best_val = r.get("best_val_loss", float("nan"))
        recall = r.get("recall", 0.0)
        precision = r.get("precision", 0.0)
        f1 = r.get("f1", 0.0)
        pos_err = r.get("position_error", 0.0)
        noop_rec = r.get("noop_recall", 0.0)
        # Improvement over NoOp position error
        noop_pos = r.get("noop_position_error", 0.0)
        if noop_pos > 0:
            improv = (noop_pos - pos_err) / noop_pos * 100.0
        else:
            improv = 0.0
        print(
            f"  {name:<20} | {best_val:>9.4f} | {recall:>6.3f} "
            f"| {precision:>9.3f} | {f1:>6.3f} | {pos_err:>7.3f} "
            f"| {noop_rec:>8.3f} | {improv:>+9.1f}%"
        )
    print(sep)
    print()

    # Best config summary
    best = min(results, key=lambda e: e["results"].get("best_val_loss", float("inf")))
    print(f"  🏆 Best config: {best['name']} (val_loss={best['results']['best_val_loss']:.4f})")
    print()


def main() -> None:
    """Run the sweep: iterate CONFIGS, train & evaluate each, collect results."""
    _setup_logging()

    results: list[dict] = []
    n_configs = len(CONFIGS)

    for i, (name, hidden_dim, lr, noise_scale, epochs) in enumerate(CONFIGS, 1):
        print()
        print("━" * 70)
        print(f"  SWEEP [{i}/{n_configs}] — {name}")
        print(f"  hidden_dim={hidden_dim}, lr={lr}, noise_scale={noise_scale}, "
              f"epochs={epochs}")
        print("━" * 70)

        cfg = _make_config(name, hidden_dim, lr, noise_scale, epochs)
        t_start = time.time()

        try:
            result = run_experiment(cfg)
            elapsed = time.time() - t_start

            if "error" in result:
                logger.error("Config %s failed: %s", name, result["error"])
                result_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": name,
                    "config": {
                        "hidden_dim": hidden_dim,
                        "lr": lr,
                        "noise_scale": noise_scale,
                        "epochs": epochs,
                        "n_samples": cfg.n_samples,
                        "val_split": cfg.val_split,
                    },
                    "results": {"error": result["error"]},
                    "elapsed_seconds": elapsed,
                }
            else:
                result_entry = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "name": name,
                    "config": {
                        "hidden_dim": hidden_dim,
                        "lr": lr,
                        "noise_scale": noise_scale,
                        "epochs": epochs,
                        "n_samples": cfg.n_samples,
                        "val_split": cfg.val_split,
                    },
                    "results": result,
                    "elapsed_seconds": elapsed,
                }

            _save_result(result_entry)
            results.append(result_entry)

            print(f"  ✓ {name} completed in {elapsed:.1f}s")
        except Exception as e:
            logger.exception("Config %s raised an exception", name)
            result_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "name": name,
                "config": {
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "noise_scale": noise_scale,
                    "epochs": epochs,
                },
                "results": {"error": str(e)},
                "elapsed_seconds": time.time() - t_start,
            }
            _save_result(result_entry)
            results.append(result_entry)

    # Print summary table
    print()
    print("=" * 70)
    print("  SWEEP COMPLETE")
    print("=" * 70)
    _print_table(results)
    print(f"  Results saved to: {RESULTS_FILE}")
    print()


if __name__ == "__main__":
    main()
