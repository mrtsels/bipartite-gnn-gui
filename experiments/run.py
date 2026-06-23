#!/usr/bin/env python3
"""Phase 7.1 — Unified experiment entry point.

Usage:
  python experiments/run.py train-violation --n 500 --epochs 50
  python experiments/run.py train-confidence --n 500 --epochs 10
  python experiments/run.py evaluate-completion --n 500
  python experiments/run.py ablation --n 500 --epochs 50
"""

import argparse
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--rico-dir", default="data/rico_local/combined")
    parser.add_argument("--log-level", default="INFO")


def cmd_train_violation(args: argparse.Namespace) -> None:
    import scripts.train_violation
    sys.argv = [
        "train_violation.py",
        "--n", str(args.n),
        "--epochs", str(args.epochs),
        "--hidden", str(args.hidden),
        "--drop-ratio", str(getattr(args, "drop_ratio", 0.6)),
        "--log-level", args.log_level,
        "--rico-dir", args.rico_dir,
    ]
    scripts.train_violation.main()


def cmd_train_confidence(args: argparse.Namespace) -> None:
    import scripts.train_confidence
    sys.argv = [
        "train_confidence.py",
        "--n", str(args.n),
        "--epochs", str(args.epochs),
        "--hidden", str(args.hidden),
        "--imposter-ratio", str(getattr(args, "imposter_ratio", 0.5)),
        "--log-level", args.log_level,
        "--rico-dir", args.rico_dir,
    ]
    scripts.train_confidence.main()


def cmd_evaluate_completion(args: argparse.Namespace) -> None:
    import scripts.evaluate_completion
    sys.argv = [
        "evaluate_completion.py",
        "--n", str(args.n),
        "--epochs", str(args.epochs),
        "--hidden", str(args.hidden),
        "--drop-ratios", str(getattr(args, "drop_ratios", "0.2,0.4,0.6,0.8")),
        "--seeds", str(getattr(args, "seeds", "42,73")),
        "--output", str(getattr(args, "output", "experiments/completion_results.json")),
        "--log-level", args.log_level,
        "--rico-dir", args.rico_dir,
    ]
    scripts.evaluate_completion.main()


def cmd_ablation(args: argparse.Namespace) -> None:
    import experiments.ablation_constraints
    sys.argv = [
        "ablation_constraints.py",
        "--n", str(args.n),
        "--epochs", str(args.epochs),
        "--hidden", str(args.hidden),
        "--drop-ratio", str(getattr(args, "drop_ratio", 0.6)),
        "--output", str(getattr(args, "output", "experiments/ablation_results.json")),
        "--log-level", args.log_level,
        "--rico-dir", args.rico_dir,
    ]
    experiments.ablation_constraints.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Bipartite GNN — Experiment Runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train-violation")
    add_common_args(p)
    p.add_argument("--drop-ratio", type=float, default=0.6)
    p.add_argument("--output-dir", default="checkpoints/violation_detection")
    p.set_defaults(func=cmd_train_violation)

    p = sub.add_parser("train-confidence")
    add_common_args(p)
    p.add_argument("--imposter-ratio", type=float, default=0.5)
    p.set_defaults(func=cmd_train_confidence)

    p = sub.add_parser("evaluate-completion")
    add_common_args(p)
    p.add_argument("--drop-ratios", default="0.2,0.4,0.6,0.8")
    p.add_argument("--seeds", default="42,73")
    p.add_argument("--output", default="experiments/completion_results.json")
    p.set_defaults(func=cmd_evaluate_completion)

    p = sub.add_parser("ablation")
    add_common_args(p)
    p.add_argument("--drop-ratio", type=float, default=0.6)
    p.add_argument("--output", default="experiments/ablation_results.json")
    p.set_defaults(func=cmd_ablation)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
