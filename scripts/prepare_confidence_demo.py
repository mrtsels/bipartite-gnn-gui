#!/usr/bin/env python3
"""Prepare confidence-scoring demo assets for the web demo (Tab 2).

Loads the dedicated confidence_scoring checkpoint, picks 2-3 RICO screenshots,
injects random imposters into the GT elements (same protocol as
scripts/train_confidence.py), runs the model, and writes:

  demo_data/confidence/{img_id}.json   — per-element {bbox, label, is_imposter, score}
  demo_data/confidence/{img_id}.png    — visualisation (real = blue, imposter = red,
                                          score label; darker fill = lower score)
  demo_data/confidence/summary.json    — AUROC per image + overall, imposter ratio

Honest labelling: the checkpoint was trained under synthetic conditions
(random imposters). Real-VLM AUROC is reported separately in the demo UI.

Usage:
  python scripts/prepare_confidence_demo.py [--images 10027 10067 10033] [--imposter-ratio 0.5] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import DEVICE, extract_elements, normalize_bbox, parse_rico_vh

logger = logging.getLogger(__name__)

RICO_DIR = ROOT / "data" / "rico_local" / "combined"
CKPT = ROOT / "checkpoints" / "confidence_scoring" / "best_model.pt"
OUT_DIR = ROOT / "demo_data" / "confidence"

_CANONICAL_TYPES = ["button", "text", "icon", "image", "input", "container", "list", "other"]


def _random_imposter(rng, types=None) -> ElementNode:
    """Random imposter element with random bbox and type (matches training)."""
    w = rng.uniform(0.03, 0.3)
    h = rng.uniform(0.02, 0.2)
    x1 = rng.uniform(0, 1 - w)
    y1 = rng.uniform(0, 1 - h)
    return ElementNode(
        bbox=[x1, y1, x1 + w, y1 + h],
        confidence=rng.uniform(0.5, 1.0),
        label=rng.choice(types or ["button"]),
    )


def _load_model() -> BipartiteGNNCorrector:
    """Load the confidence checkpoint with shape-filtered state loading."""
    state = torch.load(str(CKPT), map_location="cpu")
    if "model_state" in state:  # trainer-style checkpoint
        state = state["model_state"]
    keys = [k for k in state.keys() if k.startswith("encoder.") or "head." in k]
    # infer hidden dim from element_proj or first conv weight
    hd = None
    for k in ("encoder.element_proj.weight", "encoder.constraint_proj.weight"):
        if k in state:
            hd = state[k].shape[0]
            break
    if hd is None:
        # fall back: any 128/256 dim from first head layer
        for k in state:
            if "network.0.weight" in k:
                hd = state[k].shape[1]
                break
    hd = hd or 128
    model = BipartiteGNNCorrector(hidden_dim=hd, dropout=0.0)
    missing, unexpected = [], []
    filtered = {}
    for k, v in state.items():
        if k in model.state_dict():
            filtered[k] = v
        else:
            unexpected.append(k)
    for k in model.state_dict():
        if k not in filtered:
            missing.append(k)
    model.load_state_dict(filtered)
    model.eval()
    logger.info("confidence model: %d/%d keys loaded (hd=%d)", len(filtered), len(state), hd)
    return model


def _run_image(model: BipartiteGNNCorrector, builder: BipartiteGraphBuilder,
               img_id: str, imposter_ratio: float, rng) -> dict | None:
    """Run confidence scoring on one image; returns per-element records or None."""
    rico = RICO_DIR / f"{img_id}.json"
    if not rico.exists():
        logger.warning("missing RICO json: %s", rico)
        return None
    parsed = parse_rico_vh(rico)
    if parsed is None:
        return None
    rw, rh = parsed["width"], parsed["height"]
    raw = extract_elements(parsed["root"])
    if len(raw) < 2:
        return None
    gt_elems = [normalize_bbox(e, rw, rh) for e in raw]

    n_imp = max(1, int(len(gt_elems) * imposter_ratio))
    imposters = [_random_imposter(rng, _CANONICAL_TYPES) for _ in range(n_imp)]
    all_elems = gt_elems + imposters
    labels = [1.0] * len(gt_elems) + [0.0] * len(imposters)

    # shuffle so order is not learnable
    order = list(range(len(all_elems)))
    rng.shuffle(order)
    all_elems = [all_elems[i] for i in order]
    labels = [labels[i] for i in order]

    constraints = extract_all_constraints(all_elems)
    if len(constraints) == 0:
        return None
    data = builder.build(all_elems, constraints)
    if data is None:
        return None
    data = data.to(DEVICE)

    with torch.no_grad():
        pred = model(data)
    exist = pred.get("existence")
    if exist is None:
        logger.warning("no existence head output for %s", img_id)
        return None
    scores = F.sigmoid(exist).squeeze(-1).tolist() if exist.dim() > 1 else exist.tolist()

    records = []
    for i, (el, lab, sc) in enumerate(zip(all_elems, labels, scores)):
        records.append({
            "bbox": [round(float(v), 4) for v in el.bbox],
            "label": el.label,
            "is_imposter": bool(lab < 0.5),
            "score": round(float(sc), 4),
        })
    # AUROC + optimal threshold (Youden's J) for honest binary labelling
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = None
    # optimal threshold: maximize TPR - FPR
    best_t, best_j = 0.5, -1.0
    for t in [v / 100 for v in range(10, 96, 5)]:
        tp = sum(1 for lab, sc in zip(labels, scores) if lab > 0.5 and sc > t)
        fp = sum(1 for lab, sc in zip(labels, scores) if lab < 0.5 and sc > t)
        fn = sum(1 for lab, sc in zip(labels, scores) if lab > 0.5 and sc <= t)
        tn = sum(1 for lab, sc in zip(labels, scores) if lab < 0.5 and sc <= t)
        j = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        if j > best_j:
            best_j, best_t = j, t
    return {
        "id": img_id,
        "n_elements": len(records),
        "n_real": sum(1 for r in records if not r["is_imposter"]),
        "n_imposter": sum(1 for r in records if r["is_imposter"]),
        "imposter_ratio": imposter_ratio,
        "auroc": auroc,
        "threshold": round(best_t, 2),
        "elements": records,
    }


def _draw_image(img_id: str, data: dict, shot_size: tuple[int, int]) -> None:
    """Render real(blue)/imposter(red) boxes with scores onto the screenshot."""
    img = Image.open(RICO_DIR / f"{img_id}.jpg").convert("RGB")
    sw, sh = img.size
    # letterbox to shot size if ratio differs (shouldn't for demo picks)
    draw = ImageDraw.Draw(img)
    for el in data["elements"]:
        x1, y1, x2, y2 = el["bbox"]
        px = [int(x1 * sw), int(y1 * sh), int(x2 * sw), int(y2 * sh)]
        color = (66, 165, 245, 255) if not el["is_imposter"] else (255, 82, 82, 255)
        # fill alpha by score: high score = opaque, low = transparent
        alpha = int(30 + 120 * el["score"])
        fill = color[:3] + (alpha,)
        # PIL doesn't do per-rect alpha easily without overlay; use RGBA overlay
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        od.rectangle(px, outline=color, width=3)
        od.rectangle(px, fill=fill)
        img = Image.alpha_composite(img.convert("RGBA"), overlay)
        od2 = ImageDraw.Draw(img)
        od2.text((px[0] + 4, px[1] + 4), f"{el['score']:.2f}", fill=(255, 255, 255, 255))
    img.convert("RGB").save(OUT_DIR / f"{img_id}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["10027", "10067", "10033"])
    ap.add_argument("--imposter-ratio", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_model()
    builder = BipartiteGraphBuilder()
    rng = __import__("random").Random(args.seed)

    results = []
    for img_id in args.images:
        data = _run_image(model, builder, img_id, args.imposter_ratio, rng)
        if data is None:
            logger.warning("skip %s", img_id)
            continue
        (OUT_DIR / f"{img_id}.json").write_text(json.dumps(data, indent=1))
        shot = Image.open(RICO_DIR / f"{img_id}.jpg").size
        _draw_image(img_id, data, shot)
        results.append(data)
        t = data["threshold"]
        tp = sum(1 for r in data["elements"] if not r["is_imposter"] and r["score"] > t)
        fp = sum(1 for r in data["elements"] if r["is_imposter"] and r["score"] > t)
        tn = sum(1 for r in data["elements"] if r["is_imposter"] and r["score"] <= t)
        fn = sum(1 for r in data["elements"] if not r["is_imposter"] and r["score"] <= t)
        acc = (tp + tn) / max(len(data["elements"]), 1)
        logger.info("%s: AUROC=%s thr=%.2f acc=%.3f (tp=%d fp=%d tn=%d fn=%d)",
                    img_id, data["auroc"], t, acc, tp, fp, tn, fn)

    if results:
        summary = {
            "images": [r["id"] for r in results],
            "imposter_ratio": args.imposter_ratio,
            "n_total_elements": sum(r["n_elements"] for r in results),
            "note": "Synthetic condition: random imposters mixed into GT elements. "
                    "Real-VLM AUROC is lower (~0.60) — see demo UI.",
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"Saved {len(results)} images to {OUT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    main()
