#!/usr/bin/env python3
"""Pre-compute visual features for RICO elements using DINOv2-base.

For each RICO screenshot + JSON pair:
  1. Load the JPG screenshot.
  2. For each element from the GT JSON, crop the bbox region
     (extend 5 px on each side for context, clamped to image bounds).
  3. Resize crop to (224, 224).
  4. Encode with facebook/dinov2-base (transformers).
  5. Use model(**inputs).pooler_output → 768-dim per element.
  6. Save feature tensor to data/rico_local/visual_features_dinov2/<uid>.pt,
     shape (N_elements, 768).

Usage:
    python scripts/precompute_dinov2_features.py
    python scripts/precompute_dinov2_features.py --limit 500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (configurable via CLI or env)
# ---------------------------------------------------------------------------
_RICO_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/combined")
_OUT_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/visual_features_dinov2")


def _detect_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Cropping utilities
# ---------------------------------------------------------------------------


def _expand_bbox(
    x1: float, y1: float, x2: float, y2: float,
    pad: int = 5,
    img_w: int = 0, img_h: int = 0,
) -> tuple[int, int, int, int]:
    """Expand a bbox by *pad* pixels, clamped to image bounds.

    Args:
        x1, y1, x2, y2: Bounding box in pixel coords (float).
        pad: Pixels to add on each side.
        img_w, img_h: Image dimensions.

    Returns:
        ``(x1, y1, x2, y2)`` as ints, clamped.
    """
    return (
        max(0, int(x1) - pad),
        max(0, int(y1) - pad),
        min(img_w, int(x2) + pad),
        min(img_h, int(y2) + pad),
    )


# ---------------------------------------------------------------------------
# RICO JSON element extraction
# ---------------------------------------------------------------------------


def _extract_elements_from_json(
    data: dict[str, Any], img_w: int, img_h: int,
) -> list[dict[str, Any]]:
    """Extract visible leaf elements with pixel bboxes from a RICO tree.

    Returns list of dicts with keys: ``bbox`` (4-tuple of ints in pixel coords),
    ``label``, ``element_id``.
    """
    elements: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], depth: int = 0):
        if depth > 50:
            return
        children = node.get("children")
        is_leaf = not isinstance(children, list) or len(children) == 0

        if is_leaf:
            # Visibility filter.
            if node.get("visibility", "visible") != "visible":
                return
            if node.get("visible-to-user", True) is False:
                return

            bounds = node.get("bounds")
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
                return
            x1, y1, x2, y2 = map(int, bounds)
            if x2 <= x1 or y2 <= y1:
                return

            # Map class to label.
            cls: str = node.get("class", "")
            label = _rico_class_to_label(cls)

            elements.append({
                "bbox": (x1, y1, x2, y2),
                "label": label,
                "element_id": f"{node.get('text', '')}_{len(elements)}",
            })
        else:
            for child in children:
                if isinstance(child, dict):
                    walk(child, depth + 1)

    walk(data)
    return elements


def _rico_class_to_label(cls: str) -> str:
    """Map Android class name to canonical type (same as run_experiment)."""
    short = cls.rsplit(".", 1)[-1]
    mapping = {
        "Button": "button",
        "ImageButton": "icon",
        "ImageView": "image",
        "TextView": "text",
        "EditText": "input",
        "CheckBox": "checkbox",
        "Switch": "switch",
        "Spinner": "icon",
        "ProgressBar": "icon",
        "WebView": "container",
        "ListView": "list",
        "ScrollView": "container",
        "TabWidget": "tab",
        "RadioButton": "radio",
        "SeekBar": "slider",
    }
    for suffix, label in mapping.items():
        if short.endswith(suffix):
            return label
    return "other"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def precompute(
    rico_dir: Path = _RICO_DIR,
    out_dir: Path = _OUT_DIR,
    limit: int = 0,
    device: torch.device | None = None,
) -> None:
    """Run pre-computation for all (or a limited subset of) RICO images.

    Args:
        rico_dir: Directory containing ``.json`` and paired ``.jpg`` files.
        out_dir: Output directory for ``.pt`` files.
        limit: If > 0, only process this many images.
        device: Target device. Auto-detected if ``None``.
    """
    device = device or _detect_device()
    logger.info("Device: %s", device)

    # Discover JSON files.
    json_paths = sorted(Path(rico_dir).glob("*.json"))
    if limit > 0:
        json_paths = json_paths[:limit]
    logger.info("Found %d JSON files to process", len(json_paths))

    # Load DINOv2 model and processor.
    logger.info("Loading facebook/dinov2-base...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    model = Dinov2Model.from_pretrained("facebook/dinov2-base")
    model.eval()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "DINOv2-base loaded: %dM params, %d-dim features",
        n_params // 1_000_000,
        model.config.hidden_size,
    )

    # Create output directory.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_processed = 0
    n_skipped = 0
    n_total_elements = 0

    for json_path in json_paths:
        uid = json_path.stem  # e.g. "0", "1", ...

        # Check if already computed.
        out_path = out_dir / f"{uid}.pt"
        if out_path.exists():
            n_skipped += 1
            continue

        # Load JSON.
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Skipping %s: failed to load JSON — %s", uid, exc)
            n_skipped += 1
            continue

        # Extract root and screen dims.
        activity = data.get("activity", {})
        root = activity.get("root") or data.get("root")
        if not root:
            n_skipped += 1
            continue

        bounds = root.get("bounds", [0, 0, 0, 0])
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
            n_skipped += 1
            continue
        img_w, img_h = int(bounds[2]), int(bounds[3])
        if img_w <= 0 or img_h <= 0:
            n_skipped += 1
            continue

        # Extract elements (recursive walk).
        elements = _extract_elements_from_json(root, img_w, img_h)
        if not elements:
            n_skipped += 1
            continue

        # Load screenshot.
        img_path = rico_dir / f"{uid}.jpg"
        if not img_path.exists():
            img_path = rico_dir / f"{uid}.png"
        if not img_path.exists():
            n_skipped += 1
            continue

        try:
            screenshot = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.warning("Skipping %s: cannot open image — %s", uid, exc)
            n_skipped += 1
            continue

        actual_w, actual_h = screenshot.size
        # Scale factor in case JSON dims differ from image pixel dims.
        x_scale = actual_w / max(img_w, 1)
        y_scale = actual_h / max(img_h, 1)

        # Extract crops for each element.
        crop_pil: list[Image.Image] = []
        valid_element_indices: list[int] = []

        for idx, elem in enumerate(elements):
            x1, y1, x2, y2 = elem["bbox"]
            # Scale pixel coords to actual image size.
            x1_s = int(x1 * x_scale)
            y1_s = int(y1 * y_scale)
            x2_s = int(x2 * x_scale)
            y2_s = int(y2 * y_scale)

            # Expand by 5px with clamping.
            x1_e, y1_e, x2_e, y2_e = _expand_bbox(
                x1_s, y1_s, x2_s, y2_s, pad=5, img_w=actual_w, img_h=actual_h
            )

            # Guard: skip if crop would be empty.
            if x2_e <= x1_e or y2_e <= y1_e:
                # Fall back to un-expanded crop.
                x1_e, y1_e, x2_e, y2_e = x1_s, y1_s, x2_s, y2_s
                if x2_e <= x1_e or y2_e <= y1_e:
                    # Degenerate element — will use zero vector.
                    continue

            try:
                crop = screenshot.crop((x1_e, y1_e, x2_e, y2_e))
                crop_pil.append(crop)
                valid_element_indices.append(idx)
            except Exception:
                continue

        if not crop_pil:
            n_skipped += 1
            continue

        # Batch process: DINOv2 processor handles resize + normalization.
        # We process elements in mini-batches for memory efficiency.
        feats_list: list[torch.Tensor] = []

        batch_size = 32  # Process 32 crops at a time
        for start_idx in range(0, len(crop_pil), batch_size):
            batch_crops = crop_pil[start_idx : start_idx + batch_size]
            inputs = processor(images=batch_crops, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                # pooler_output: (batch_size, 768)
                feats_list.append(outputs.pooler_output.cpu())

        all_feats = torch.cat(feats_list, dim=0)  # (N_valid, 768)

        # Build full feature tensor with zeros for invalid elements.
        full_feats = torch.zeros(len(elements), all_feats.shape[1], dtype=torch.float32)
        for i, feat in zip(valid_element_indices, all_feats):
            full_feats[i] = feat

        # Save.
        torch.save(full_feats, out_path)
        n_processed += 1
        n_total_elements += full_feats.shape[0]

        if n_processed % 100 == 0:
            elapsed = time.time() - t0
            rate = n_processed / max(elapsed, 1e-6)
            logger.info(
                "  Processed %d/%d (%.1f img/s, %d total elements)",
                n_processed, len(json_paths), rate, n_total_elements,
            )

    dt = time.time() - t0
    logger.info("=" * 55)
    logger.info("Done in %.1fs", dt)
    logger.info("Processed: %d images (%d skipped)", n_processed, n_skipped)
    logger.info("Total elements: %d", n_total_elements)
    logger.info("Output: %s/*.pt", out_dir)
    logger.info("=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute DINOv2 visual features for RICO elements"
    )
    parser.add_argument("--rico-dir", type=str, default=str(_RICO_DIR),
                        help="RICO combined directory with JSON + JPG")
    parser.add_argument("--out-dir", type=str, default=str(_OUT_DIR),
                        help="Output directory for .pt files")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most this many images (0 = all)")
    parser.add_argument("--device", type=str, default="",
                        help="Torch device (mps/cuda/cpu; default: auto-detect)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    precompute(
        rico_dir=Path(args.rico_dir),
        out_dir=Path(args.out_dir),
        limit=args.limit,
        device=device,
    )


if __name__ == "__main__":
    main()
