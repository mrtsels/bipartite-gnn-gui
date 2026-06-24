#!/usr/bin/env python3
"""ScreenSpot Ground-Truth and VLM prediction loader.

Provides:
  - load_screenspot_gt(path_or_json, images_dir)
    → list of (image_stem, list[ElementNode])
  - load_screenspot_vlm(image_path, vlm_dir)
    → list[ElementNode] for the matching VLM predictions

Usage:
    from experiments.screenspot_loader import load_screenspot_gt, load_screenspot_vlm
    gts = load_screenspot_gt('/path/to/ScreenSpot_combined.json',
                             '/path/to/images/')
    print(f'Loaded {len(gts)} images')

    vlms = load_screenspot_vlm('mobile_xxx.json',
                               '/path/to/vlm_predictions/')
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Allow running from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bipartite_gnn_gui.graph.schema import ElementNode

logger = logging.getLogger(__name__)

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover
    PILImage = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Canonical type mapping for ScreenSpot data_type field
# ---------------------------------------------------------------------------
# ScreenSpot uses only "icon" and "text" in its data_type field.
# We keep them as-is and add a fallback for any future types.
_SCREENSPOT_TYPE_MAP: Dict[str, str] = {
    "icon": "icon",
    "text": "text",
    "button": "button",
}


def _normalize_screenspot_type(raw_type: str) -> str:
    """Map a ScreenSpot data_type string to the canonical type."""
    return _SCREENSPOT_TYPE_MAP.get(raw_type.strip().lower(), "other")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_screenspot_gt(
    path_or_json: Union[str, Path, List[Dict[str, Any]]],
    images_dir: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, List[ElementNode]]]:
    """Load ScreenSpot combined JSON into a list of (image_stem, elements) pairs.

    Parses ``ScreenSpot_combined.json`` which is a JSON array of entries::

        {
            "image": "pc_xxx.png",
            "annotations": [{
                "bounding_box": [x, y, w, h],
                "data_type": "icon",
                "objective_reference": "close",
                "data_source": "windows"
            }]
        }

    Bounding boxes are converted from ``[x, y, w, h]`` (pixels) to normalized
    ``[x1, y1, x2, y2]`` in ``[0, 1]``.  Image pixel dimensions are read from
    the PNG files in *images_dir*.

    Args:
        path_or_json: Path to ``ScreenSpot_combined.json``, or a pre-loaded
            list of entries.
        images_dir: Directory of PNG images.  When ``None``, defaults to
            ``<parent of JSON>/images/``.  Ignored when *path_or_json* is
            already a list.

    Returns:
        List of ``(image_stem, element_nodes)`` tuples, where *image_stem*
        is the filename without extension (e.g. ``"pc_xxx"``), used for
        matching with VLM predictions.
    """
    if PILImage is None:
        raise ImportError("Pillow is required to load ScreenSpot data; pip install Pillow")

    # Resolve JSON data
    if isinstance(path_or_json, (str, Path)):
        json_path = Path(path_or_json)
        with json_path.open("r", encoding="utf-8") as f:
            data: List[Dict[str, Any]] = json.load(f)
        if images_dir is None:
            images_dir = json_path.parent / "images"
    else:
        data = path_or_json
        if images_dir is None:
            images_dir = Path(".") / "images"

    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array, got {type(data).__name__}")

    results: List[Tuple[str, List[ElementNode]]] = []

    for entry_idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry at index %d", entry_idx)
            continue

        image_file = str(entry.get("image", ""))
        if not image_file:
            logger.warning("Skipping entry %d with missing 'image'", entry_idx)
            continue

        image_stem = Path(image_file).stem  # e.g. "pc_xxx"

        # Get image dimensions
        image_path = images_dir / image_file
        try:
            with PILImage.open(image_path) as img:
                img_w, img_h = img.size
        except Exception as exc:
            logger.warning("Failed to open %s: %s, skipping", image_path, exc)
            continue

        if img_w <= 0 or img_h <= 0:
            logger.warning("Invalid dimensions (%d, %d) for %s", img_w, img_h, image_file)
            continue

        annotations_raw: List[Dict[str, Any]] = entry.get("annotations", [])
        if not isinstance(annotations_raw, list):
            annotations_raw = []

        elements: List[ElementNode] = []
        for ann_idx, item in enumerate(annotations_raw):
            if not isinstance(item, dict):
                continue

            # bounding_box: [x, y, w, h] (pixel coordinates)
            bbox_raw = item.get("bounding_box")
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                logger.debug("Skipping ann %d in %s: invalid bbox", ann_idx, image_file)
                continue

            x_px, y_px, w_px, h_px = map(float, bbox_raw)
            if w_px <= 0 or h_px <= 0:
                continue

            # Convert to normalized [x1, y1, x2, y2]
            x1 = max(0.0, min(1.0, x_px / img_w))
            y1 = max(0.0, min(1.0, y_px / img_h))
            x2 = max(0.0, min(1.0, (x_px + w_px) / img_w))
            y2 = max(0.0, min(1.0, (y_px + h_px) / img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            # Map data_type
            raw_type = str(item.get("data_type", "")).strip()
            label = _normalize_screenspot_type(raw_type)

            elements.append(
                ElementNode(bbox=[x1, y1, x2, y2], label=label, confidence=1.0)
            )

        results.append((image_stem, elements))

    return results


# ---------------------------------------------------------------------------
# VLM prediction loader
# ---------------------------------------------------------------------------


def load_screenspot_vlm(
    image_identifier: Union[str, Path],
    vlm_dir: Union[str, Path] = "/Users/minimx/bipartite-gnn-gui/data/vlm_predictions/screenspot_qwen_flash",
) -> Optional[List[ElementNode]]:
    """Load VLM predictions for a ScreenSpot image.

    Matches by filename stem: if *image_identifier* is ``"pc_xxx.png"`` or
    ``"pc_xxx"``, loads ``pc_xxx.json`` from *vlm_dir*.

    Args:
        image_identifier: Image filename (e.g. ``"pc_xxx.png"``) or stem
            (e.g. ``"pc_xxx"``).
        vlm_dir: Directory containing VLM prediction JSON files.

    Returns:
        List of ``ElementNode`` from VLM predictions, or ``None`` if no
        matching file found or parse error.
    """
    vlm_dir = Path(vlm_dir)
    stem = Path(image_identifier).stem  # strip .png/.json etc.
    vlm_path = vlm_dir / f"{stem}.json"

    if not vlm_path.exists():
        return None

    try:
        with vlm_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to parse %s: %s", vlm_path, exc)
        return None

    img_w = float(data.get("image_width", 1))
    img_h = float(data.get("image_height", 1))
    if img_w <= 0 or img_h <= 0:
        logger.warning("Invalid image dimensions in %s: %s x %s", vlm_path, img_w, img_h)
        return None

    elements: List[ElementNode] = []
    for item in data.get("elements", []):
        if not isinstance(item, dict):
            continue

        # VLM predictions use bbox_xyxy (normalized or pixel) or bbox
        bbox_raw = item.get("bbox_xyxy") or item.get("bbox")
        if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            continue

        x1, y1, x2, y2 = map(float, bbox_raw)

        # Normalize if pixel values (check if > 1.0 indicates pixel coords)
        if x1 > 1.0 or y1 > 1.0 or x2 > 1.0 or y2 > 1.0:
            x1 = max(0.0, min(1.0, x1 / img_w))
            y1 = max(0.0, min(1.0, y1 / img_h))
            x2 = max(0.0, min(1.0, x2 / img_w))
            y2 = max(0.0, min(1.0, y2 / img_h))

        if x2 <= x1 or y2 <= y1:
            continue

        label = str(item.get("label", "other")).strip()
        if not label:
            label = "other"

        elements.append(
            ElementNode(bbox=[x1, y1, x2, y2], label=label, confidence=1.0)
        )

    return elements


def load_all_screenspot_vlm(
    vlm_dir: Union[str, Path] = "/Users/minimx/bipartite-gnn-gui/data/vlm_predictions/screenspot_qwen_flash",
    max_n: int = 600,
) -> Dict[str, List[ElementNode]]:
    """Load all ScreenSpot VLM predictions as a dict mapping stem → elements.

    Args:
        vlm_dir: Directory containing VLM prediction JSON files.
        max_n: Maximum number of files to load (default 600).

    Returns:
        Dict mapping image stem (e.g. ``"pc_xxx"``) to list of ElementNode.
    """
    vlm_dir = Path(vlm_dir)
    result: Dict[str, List[ElementNode]] = {}
    files = sorted(vlm_dir.glob("*.json"))[:max_n]
    for fpath in files:
        stem = fpath.stem
        elems = load_screenspot_vlm(stem, vlm_dir=vlm_dir)
        if elems is not None:
            result[stem] = elems
    return result


# ---------------------------------------------------------------------------
# Quick verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    screenspot_json = "/Users/minimx/mnt/thinkpad/1007/Documents/dataset/screenspot/ScreenSpot_combined.json"
    images_dir = "/Users/minimx/mnt/thinkpad/1007/Documents/dataset/screenspot/images"

    gts = load_screenspot_gt(screenspot_json, images_dir)
    total_elems = sum(len(e) for _, e in gts)
    print(f"Loaded {len(gts)} images, avg {total_elems / max(len(gts), 1):.1f} elements")

    # Show a few samples
    for stem, elems in gts[:5]:
        print(f"  {stem}: {len(elems)} elements")
        for el in elems[:3]:
            print(f"    {el.label}: {[round(v, 3) for v in el.bbox]}")

    # Show source distribution from annotations
    import json as _json
    with open(screenspot_json) as f:
        raw = _json.load(f)
    sources = {}
    for d in raw:
        for a in d.get("annotations", []):
            src = a.get("data_source", "unknown")
            sources[src] = sources.get(src, 0) + 1
    print(f"\nSources: {sources}")
    print(f"Types: {set(a.get('data_type') for d in raw for a in d.get('annotations', []))}")
