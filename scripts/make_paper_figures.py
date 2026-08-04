#!/usr/bin/env python3
"""Generate publication-quality figures for the Phase 12 paper.

Reads experiment JSONs directly (no hardcoded numbers) and writes PNGs to
paper/figures/. IEEE style: Times New Roman, 8pt, no chartjunk.

Figures:
  fig_completion.png  — completion sweep: GNN vs NN IoU by drop ratio (mean ± std, 2 seeds)
  fig_ablation.png    — constraint-type ablation: violation accuracy per constraint set
  fig_phase9.png      — training-mode comparison: joint vs violation-only vs proposal-only
                        (5 seeds, mean ± std; violation accuracy + proposal MSE)
  fig_real_vlm.png    — real-VLM end-to-end: precision/recall/F1 before vs after GNN
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"
OUT = REPO / "paper" / "figures"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    }
)

GNN_BLUE = "#1f77b4"
NN_GRAY = "#a0a0a0"


def load(name: str):
    with open(EXP / name) as f:
        return json.load(f)


def fig_completion() -> None:
    """GNN vs NN IoU across drop ratios (completion_results.json)."""
    rows = load("completion_results.json")
    by_drop: dict[float, dict[str, list[float]]] = {}
    for r in rows:
        d = by_drop.setdefault(
            r["drop_ratio"], {"gnn_iou": [], "nn_iou": [], "gnn_mse": [], "nn_mse": []}
        )
        d["gnn_iou"].append(r["gnn_proposal_iou"])
        d["nn_iou"].append(r["baseline_nn_iou"])
        d["gnn_mse"].append(r["gnn_proposal_mse"])
        d["nn_mse"].append(r["baseline_nn_mse"])

    drops = sorted(by_drop)
    gnn_mean = [np.mean(by_drop[d]["gnn_iou"]) for d in drops]
    gnn_std = [np.std(by_drop[d]["gnn_iou"]) for d in drops]
    nn_mean = [np.mean(by_drop[d]["nn_iou"]) for d in drops]
    nn_std = [np.std(by_drop[d]["nn_iou"]) for d in drops]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(drops))
    w = 0.32
    ax.bar(x - w / 2, gnn_mean, w, yerr=gnn_std, capsize=2, color=GNN_BLUE,
           label="GNN (ours)")
    ax.bar(x + w / 2, nn_mean, w, yerr=nn_std, capsize=2, color=NN_GRAY,
           label="NN baseline")
    ax.set_xticks(x, [f"{d:g}" for d in drops])
    ax.set_xlabel("Drop ratio")
    ax.set_ylabel("Proposal IoU")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_completion.png")
    plt.close(fig)
    print(f"completion: gnn={dict(zip(drops, [round(v,3) for v in gnn_mean]))}")


def fig_ablation() -> None:
    """Violation accuracy per constraint set (ablation_results.json)."""
    data = load("ablation_results.json")["results"]
    # Keep a readable subset in paper order.
    order = [
        ("All 10 constraint types (control)", "baseline"),
        ("Remove CONTAINMENT only", "no_containment"),
        ("Remove SPACING only", "no_spacing"),
        ("Remove GRID only", "no_grid"),
        ("Remove all alignment types", "no_alignment"),
        ("Keep only alignment types", "only_alignment"),
    ]
    by_group = {r["group"]: r for r in data}
    labels, accs = [], []
    for label, g in order:
        if g in by_group:
            labels.append(label.replace("all constraint types (control)", "10 types"))
            accs.append(by_group[g]["violation_acc"])

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    colors = [GNN_BLUE] + [NN_GRAY] * (len(accs) - 1)
    bars = ax.barh(np.arange(len(accs))[::-1], accs, color=colors, height=0.62)
    for bar, v in zip(bars, accs):
        ax.text(v + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=7)
    ax.set_xlim(0.85, 0.97)
    ax.set_yticks(np.arange(len(accs))[::-1], labels)
    ax.set_xlabel("Violation accuracy")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.2f}")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_ablation.png")
    plt.close(fig)
    print(f"ablation: {dict(zip(labels, [round(a,4) for a in accs]))}")


def fig_phase9() -> None:
    """Training-mode comparison, 5 seeds (phase9_913_results.json)."""
    data = load("phase9_913_results.json")
    modes = {
        "joint": "joint",
        "violation-only": "violation-only",
        "proposal-only": "proposal-only",
    }
    accs, mses = {}, {}
    for key, val in data.items():
        for mode, token in modes.items():
            if f"× {token}" in key:
                accs.setdefault(mode, []).append(val["val_acc"])
                mses.setdefault(mode, []).append(val["prop_mse"])

    mode_names = ["joint", "violation-only", "proposal-only"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2))
    for ax, values, ylabel, title in [
        (axes[0], accs, "Violation accuracy", "Violation head"),
        (axes[1], mses, "Proposal MSE", "Proposal head"),
    ]:
        means = [np.mean(values[m]) for m in mode_names]
        stds = [np.std(values[m]) for m in mode_names]
        x = np.arange(len(mode_names))
        ax.bar(x, means, 0.55, yerr=stds, capsize=2, color=GNN_BLUE)
        ax.set_xticks(x, [m.replace("-", "\n") for m in mode_names])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for xi, m, s in zip(x, means, stds):
            ax.text(xi, m + (0.02 if ylabel == "Violation accuracy" else 0.006),
                    f"{m:.3f}", ha="center", fontsize=7)
        if ylabel == "Violation accuracy":
            ax.set_ylim(0.3, 1.0)
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT / "fig_phase9.png")
    plt.close(fig)
    print(f"phase9 acc: {dict(zip(mode_names, [round(np.mean(accs[m]),4) for m in mode_names]))}")


def fig_real_vlm() -> None:
    """Real-VLM end-to-end before/after (recheck_visual_fusion_model.json)."""
    data = load("vlm_completion/recheck_visual_fusion_model.json")
    before, after = data["before"], data["after"]
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    b = [before[m] for m in metrics]
    a = [after[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(labels))
    w = 0.32
    ax.bar(x - w / 2, b, w, color=NN_GRAY, label="VLM only")
    ax.bar(x + w / 2, a, w, color=GNN_BLUE, label="VLM + GNN")
    for xi, v in zip(x - w / 2, b):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, a):
        ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 0.5)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_real_vlm.png")
    plt.close(fig)
    print(f"real-vlm: before F1={before['f1']:.3f} after F1={after['f1']:.3f} "
          f"delta={after['f1']-before['f1']:+.3f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_completion()
    fig_ablation()
    fig_phase9()
    fig_real_vlm()
    print("done ->", OUT)


if __name__ == "__main__":
    main()
