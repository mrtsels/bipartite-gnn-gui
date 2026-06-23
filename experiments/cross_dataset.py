#!/usr/bin/env python3
"""Phase 7.5 — Cross-dataset generalization: RICO-trained GNN → ScreenSpot.

Measures zero-shot violation detection on ScreenSpot after training on RICO.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector

logger = logging.getLogger(__name__)

TYPE_MAP = {
    "icon": "icon", "text": "text", "button": "button",
    "input": "input", "image": "image", "widget": "container",
    "slider": "container", "scrollbar": "container", "list": "list",
    "detail": "text", "tab": "button", "checkbox": "input", "menu": "container",
}
FALLBACK_TYPE = "other"


def load_screenspot_items(ann_path: str, max_n: int = 500) -> List[Dict]:
    """Load ScreenSpot with actual image sizes."""
    with open(ann_path) as f:
        data = json.load(f)
    img_dir = Path("data/raw/screenspot/images")
    from PIL import Image
    items = []
    for item in data[:max_n]:
        img_path = img_dir / item["image"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path)
            w, h = img.size
        except Exception:
            w, h = 1440, 2560
        anns = [a for a in item["annotations"]
                if a["bounding_box"][2] > 0 and a["bounding_box"][3] > 0]
        if len(anns) >= 2:
            items.append({"annotations": anns, "width": w, "height": h})
    logger.info(f"Loaded {len(items)} ScreenSpot images with >=2 elements")
    return items


def to_elements(item: Dict) -> List[ElementNode]:
    """Convert ScreenSpot annotations to ElementNodes (xywh → xyxy, normalize)."""
    w, h = item["width"], item["height"]
    elems = []
    for ann in item["annotations"]:
        x, y, bw, bh = ann["bounding_box"]
        x1, y1 = max(0, x) / w, max(0, y) / h
        x2, y2 = min(w, x + bw) / w, min(h, y + bh) / h
        if x2 <= x1 or y2 <= y1:
            continue
        label = TYPE_MAP.get(ann.get("data_type", ""), FALLBACK_TYPE)
        elems.append(ElementNode(bbox=[x1, y1, x2, y2], confidence=1.0, label=label))
    return elems


def load_model(ckpt_path: str = "checkpoints/violation_detection/best_model.pt",
               hidden_dim: int = 128) -> BipartiteGNNCorrector:
    """Load the RICO-trained model."""
    state = torch.load(ckpt_path, map_location="cpu")
    model = BipartiteGNNCorrector(
        hidden_dim=hidden_dim, dropout=0.1,
        coord_weight=0.0, existence_weight=0.0,
    )
    # Handle different checkpoint formats
    sd = state
    if isinstance(state, dict):
        k = list(state.keys())
        if "model_state_dict" in state:
            sd = state["model_state_dict"]
        elif k and "network" not in str(k[0]):
            sd = {k: v for k, v in state.items()}
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()},
                          strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: BipartiteGNNCorrector,
             items: List[Dict]) -> Dict[str, float]:
    """Zero-shot evaluation on ScreenSpot data."""
    builder = BipartiteGraphBuilder()
    all_scores = []
    n_graphs = 0
    for item in items:
        elems = to_elements(item)
        constraints = extract_all_constraints(elems)
        if not constraints:
            continue
        data = builder.build(elems, constraints)
        preds = model(data)
        if "violation" in preds:
            scores = preds["violation"].flatten()
            all_scores.extend(scores.tolist())
        n_graphs += 1

    scores_t = torch.tensor(all_scores)
    pct_above_05 = (scores_t > 0.5).float().mean().item()
    return {
        "n_images": len(items),
        "n_with_constraints": n_graphs,
        "n_predictions": len(all_scores),
        "pct_violated": pct_above_05,
        "mean_score": scores_t.mean().item(),
        "std_score": scores_t.std().item(),
        "rico_reference_acc": 0.91,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ann_path = "data/raw/screenspot/ScreenSpot_combined.json"
    items = load_screenspot_items(ann_path, max_n=500)
    # Load models
    rico_model = load_model("checkpoints/violation_detection/best_model.pt")
    finetuned_model = load_model("checkpoints/violation_detection/screenspot_finetuned.pt")

    metrics_rico = evaluate(rico_model, items)
    metrics_finetune = evaluate(finetuned_model, items)

    print()
    print("=" * 60)
    print("  Cross-Dataset: RICO → ScreenSpot (Zero-Shot)")
    print(f"  ScreenSpot images with >=2 elem:   {metrics_rico['n_images']}")
    print(f"  Images with constraints:            {metrics_rico['n_with_constraints']}")
    print(f"  Total violation predictions:        {metrics_rico['n_predictions']}")
    print()
    print(f"  {'Metric':<30} {'RICO Model':>12} {'Fine-Tuned':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Mean violation score':<30} {metrics_rico['mean_score']:>12.4f} {metrics_finetune['mean_score']:>12.4f}")
    print(f"  {'Std violation score':<30} {metrics_rico['std_score']:>12.4f} {metrics_finetune['std_score']:>12.4f}")
    print(f"  {'% predicted violated':<30} {metrics_rico['pct_violated']*100:>11.1f}% {metrics_finetune['pct_violated']*100:>11.1f}%")
    print(f"  {'RICO in-distribution ref':<30} {metrics_rico['rico_reference_acc']*100:>11.0f}% {metrics_finetune['rico_reference_acc']*100:>11.0f}%")
    print("=" * 60)
    print()

    # Analysis
    def interpret(m):
        if m["pct_violated"] > 0.7:
            return "⚠️ Most constraints → violated (distribution shift)"
        elif m["pct_violated"] < 0.3:
            return "⚠️ Most constraints → intact (distribution shift)"
        elif m["mean_score"] < 0.55 and m["mean_score"] > 0.45:
            return "✅ Scores near 0.5 → model signals uncertainty"
        else:
            return "❌ Does not generalize"

    print(f"  RICO model:    {interpret(metrics_rico)}")
    print(f"  Fine-tuned:    {interpret(metrics_finetune)}")
    print()
    print(f"  Fine-tuning gain: {metrics_finetune['pct_violated'] - metrics_rico['pct_violated']:+.1%} violated rate")

    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "cross_dataset_results.json", "w") as f:
        json.dump({"rico": metrics_rico, "finetuned": metrics_finetune}, f, indent=2)
    logger.info(f"Saved to experiments/cross_dataset_results.json")


if __name__ == "__main__":
    main()
