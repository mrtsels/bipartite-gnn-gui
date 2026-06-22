"""RICO dataset loader — View Hierarchy and Semantic Annotation parsing.

Loads RICO dataset JSON files and converts them into the unified
:class:`GroundTruth` format used by the rest of the pipeline.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .ground_truth import (
    GTElement,
    GroundTruth,
    GroundTruthParseError,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Android class → canonical type mapping (§3.5.3 of gt_format.md)
# ---------------------------------------------------------------------------

#: Exact-match mapping from fully-qualified Android class to canonical type.
_RICO_CLASS_MAP: Dict[str, str] = {
    "android.widget.Button": "button",
    "android.widget.ImageButton": "icon",
    "android.widget.ImageView": "image",
    "android.widget.TextView": "text",
    "android.widget.EditText": "input",
    "android.widget.CheckBox": "icon",
    "android.widget.Switch": "icon",
    "android.widget.Spinner": "icon",
    "android.widget.ProgressBar": "icon",
    "android.webkit.WebView": "container",
    "android.widget.ListView": "list",
    "android.widget.ScrollView": "container",
}

#: Suffix-based fallback for partially-qualified or variant class names.
_RICO_CLASS_SUFFIX_MAP: Dict[str, str] = {
    "Button": "button",
    "ImageButton": "icon",
    "ImageView": "image",
    "TextView": "text",
    "EditText": "input",
    "CheckBox": "icon",
    "Switch": "icon",
    "Spinner": "icon",
    "ProgressBar": "icon",
    "WebView": "container",
    "ListView": "list",
    "ScrollView": "container",
}


def rico_class_to_type(android_class: str) -> str:
    """Map an Android View class name to its canonical element type.

    Uses the mapping table from §3.5.3 of ``gt_format.md``.
    Unrecognized classes are mapped to ``"other"``.

    Args:
        android_class: Fully-qualified Android class name (e.g.
            ``"android.widget.Button"``).

    Returns:
        Canonical element type string (e.g. ``"button"``, ``"icon"``).
    """
    if not android_class or not isinstance(android_class, str):
        return "other"

    # Exact match first
    if android_class in _RICO_CLASS_MAP:
        return _RICO_CLASS_MAP[android_class]

    # Suffix-based fallback: extract short class name and check if it
    # ends with one of the known Android View class names.  Longer
    # suffixes are checked first (e.g. ImageButton before Button).
    short = android_class.rsplit(".", 1)[-1] if "." in android_class else android_class
    for suffix in sorted(_RICO_CLASS_SUFFIX_MAP, key=lambda s: -len(s)):
        if short.endswith(suffix):
            return _RICO_CLASS_SUFFIX_MAP[suffix]

    logger.debug("Unknown Android class '%s', mapping to 'other'", android_class)
    return "other"


# ---------------------------------------------------------------------------
# Bounds string parsing
# ---------------------------------------------------------------------------

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def parse_rico_bounds(bounds_str: str) -> Tuple[float, float, float, float]:
    """Parse a RICO bounds string into a 4-tuple of floats.

    Args:
        bounds_str: String in ``"[x1,y1][x2,y2]"`` format.

    Returns:
        ``(x1, y1, x2, y2)`` tuple.

    Raises:
        GroundTruthParseError: If the string cannot be parsed.
    """
    m = _BOUNDS_RE.match(bounds_str.strip())
    if m is None:
        raise GroundTruthParseError(
            f"Cannot parse RICO bounds string: {bounds_str!r}"
        )
    return tuple(map(float, m.groups()))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Leaf node extraction
# ---------------------------------------------------------------------------


def _find_leaf_nodes(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Recursively collect all leaf nodes from a RICO View Hierarchy tree.

    A leaf node is any node whose ``children`` key is missing, ``None``,
    not a list, or refers to an empty list.

    Args:
        node: A RICO View Hierarchy node dict.  Must contain ``bounds``
            and ``class`` keys; may contain ``children``, ``visibility``,
            ``text``, ``content-desc``, and ``clickable``.

    Returns:
        List of leaf node dicts (references into the original tree).
    """
    children = node.get("children")
    if not isinstance(children, list) or len(children) == 0:
        return [node]

    result: List[Dict[str, Any]] = []
    for child in children:
        if isinstance(child, dict):
            result.extend(_find_leaf_nodes(child))
    return result


# ---------------------------------------------------------------------------
# View Hierarchy loader
# ---------------------------------------------------------------------------


def parse_rico_view_hierarchy(
    vh_path: Union[str, Path],
    images_dir: Union[str, Path],
) -> GroundTruth:
    """Parse a single RICO View Hierarchy JSON into a ``GroundTruth``.

    Recursively traverses ``root.children`` to extract all visible
    leaf nodes, normalizes their pixel bbox coordinates to ``[0, 1]``,
    and maps Android class names to canonical element types.

    Args:
        vh_path: Path to the RICO View Hierarchy JSON file.
        images_dir: Directory containing the corresponding screenshot
            PNG images (used to construct ``image_path``).

    Returns:
        ``GroundTruth`` instance with normalized bboxes and canonical types.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid, missing required keys,
            or ``screen_width`` / ``screen_height`` is not positive.
    """
    vh_path = Path(vh_path)
    with vh_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    screen_id = str(data.get("screen_id", ""))
    screen_width = int(data.get("screen_width", 0))
    screen_height = int(data.get("screen_height", 0))

    if screen_width <= 0:
        raise GroundTruthParseError(
            f"screen_width must be positive, got {screen_width}"
        )
    if screen_height <= 0:
        raise GroundTruthParseError(
            f"screen_height must be positive, got {screen_height}"
        )

    root = data.get("root")
    if not isinstance(root, dict):
        raise GroundTruthParseError(
            "Missing or invalid 'root' key in RICO View Hierarchy"
        )

    leaf_nodes = _find_leaf_nodes(root)
    elements: List[GTElement] = []

    for i, node in enumerate(leaf_nodes):
        # ---- visibility filter ------------------------------------------
        visibility = node.get("visibility", "visible")
        if visibility != "visible":
            logger.debug(
                "Skipping leaf node %d: visibility='%s' (not 'visible')",
                i, visibility,
            )
            continue

        # ---- bounds parsing and normalization ---------------------------
        bounds_str = node.get("bounds", "")
        if not bounds_str or not isinstance(bounds_str, str):
            logger.warning("Skipping leaf node %d: missing bounds", i)
            continue

        try:
            x1, y1, x2, y2 = parse_rico_bounds(bounds_str)
        except GroundTruthParseError:
            logger.warning(
                "Skipping leaf node %d: invalid bounds string %r", i, bounds_str
            )
            continue

        # Filter zero-area bboxes (before normalization)
        if x2 <= x1 or y2 <= y1:
            logger.debug(
                "Skipping leaf node %d: zero-area bbox "
                "(x2=%.1f <= x1=%.1f or y2=%.1f <= y1=%.1f)",
                i, x2, x1, y2, y1,
            )
            continue

        # Normalize pixel coordinates to [0, 1] and clamp
        x1_norm = max(0.0, min(1.0, x1 / screen_width))
        y1_norm = max(0.0, min(1.0, y1 / screen_height))
        x2_norm = max(0.0, min(1.0, x2 / screen_width))
        y2_norm = max(0.0, min(1.0, y2 / screen_height))
        bbox = (x1_norm, y1_norm, x2_norm, y2_norm)

        # Recheck after normalization (guard against floating-point edge cases)
        if x2_norm <= x1_norm or y2_norm <= y1_norm:
            logger.debug(
                "Skipping leaf node %d: degenerate after normalization", i
            )
            continue

        # ---- type mapping -----------------------------------------------
        android_class = node.get("class", "")
        element_type = rico_class_to_type(android_class)

        # ---- text extraction --------------------------------------------
        # Prefer text field, fall back to content-desc
        text: Optional[str] = node.get("text")
        if text is not None and isinstance(text, str):
            text = text.strip()
            if not text:
                text = None
        if text is None:
            fallback = node.get("content-desc")
            if fallback is not None and isinstance(fallback, str):
                fallback = fallback.strip()
                if fallback:
                    text = fallback

        # ---- metadata ---------------------------------------------------
        metadata: Dict[str, Any] = {
            "class": android_class,
            "clickable": bool(node.get("clickable", False)),
        }

        # Build element_id from screen_id + leaf index
        element_id = f"{screen_id}_{i}"

        elements.append(
            GTElement(
                element_id=element_id,
                bbox=bbox,
                element_type=element_type,
                text_content=text,
                source_dataset="rico",
                metadata=metadata,
            )
        )

    images_dir = Path(images_dir)
    image_path = str(images_dir / f"{screen_id}.png")

    return GroundTruth(
        elements=elements,
        image_path=image_path,
        image_width=screen_width,
        image_height=screen_height,
        source="rico",
    )


# ---------------------------------------------------------------------------
# Semantic Annotation loader
# ---------------------------------------------------------------------------


def parse_rico_semantic(
    ann_path: Union[str, Path],
) -> GroundTruth:
    """Parse a RICO Semantic Annotation JSON into a ``GroundTruth``.

    Semantic Annotations use a flatter format with a direct list of
    per-image human-corrected annotations.  No recursive tree traversal
    is needed.

    Args:
        ann_path: Path to the RICO Semantic Annotation JSON file.

    Returns:
        ``GroundTruth`` instance with normalized bboxes and canonical types.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid or missing required keys.
    """
    ann_path = Path(ann_path)
    with ann_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    screen_id = str(data.get("screen_id", ""))
    screen_width = int(data.get("screen_width", 0))
    screen_height = int(data.get("screen_height", 0))

    if screen_width <= 0:
        raise GroundTruthParseError(
            f"screen_width must be positive, got {screen_width}"
        )
    if screen_height <= 0:
        raise GroundTruthParseError(
            f"screen_height must be positive, got {screen_height}"
        )

    annotations: List[Dict[str, Any]] = data.get("annotations", [])
    if not isinstance(annotations, list):
        raise GroundTruthParseError(
            "Missing or invalid 'annotations' key in RICO Semantic Annotation"
        )

    elements: List[GTElement] = []

    for i, ann in enumerate(annotations):
        # ---- bbox parsing and normalization -----------------------------
        bbox_raw = ann.get("bbox")
        if bbox_raw is None or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            logger.warning(
                "Skipping semantic annotation %d: missing/invalid bbox", i
            )
            continue

        x1_px, y1_px, x2_px, y2_px = map(float, bbox_raw)

        # Filter zero-area bboxes (before normalization)
        if x2_px <= x1_px or y2_px <= y1_px:
            logger.debug(
                "Skipping semantic annotation %d: zero-area bbox", i
            )
            continue

        x1 = max(0.0, min(1.0, x1_px / screen_width))
        y1 = max(0.0, min(1.0, y1_px / screen_height))
        x2 = max(0.0, min(1.0, x2_px / screen_width))
        y2 = max(0.0, min(1.0, y2_px / screen_height))
        bbox = (x1, y1, x2, y2)

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Skipping semantic annotation %d: degenerate after normalization "
                "(x2=%.4f <= x1=%.4f or y2=%.4f <= y1=%.4f)",
                i, x2, x1, y2, y1,
            )
            continue

        # ---- type mapping -----------------------------------------------
        android_class = str(ann.get("class", ""))
        element_type = rico_class_to_type(android_class)

        # ---- text extraction --------------------------------------------
        text: Optional[str] = ann.get("text")
        if text is not None and isinstance(text, str) and text.strip() == "":
            text = None

        # ---- metadata ---------------------------------------------------
        metadata: Dict[str, Any] = {"class": android_class}
        for key in ("clickable", "component_id", "icon_class", "icon_shape"):
            if key in ann:
                metadata[key] = ann[key]

        element_id = str(ann.get("element_id", f"{screen_id}_{i}"))

        elements.append(
            GTElement(
                element_id=element_id,
                bbox=bbox,
                element_type=element_type,
                text_content=text,
                source_dataset="rico",
                metadata=metadata,
            )
        )

    # Construct image path from screen_id (relative to RICO root)
    image_path = f"data/raw/rico/unique_uis/{screen_id}.png"

    return GroundTruth(
        elements=elements,
        image_path=image_path,
        image_width=screen_width,
        image_height=screen_height,
        source="rico",
    )


# ---------------------------------------------------------------------------
# Image ID extraction
# ---------------------------------------------------------------------------


def get_rico_image_id(vh_json: Dict[str, Any]) -> str:
    """Extract the image filename from a RICO View Hierarchy JSON dict.

    Args:
        vh_json: Parsed RICO View Hierarchy JSON dict (must contain
            ``screen_id``).

    Returns:
        The screenshot filename with ``.png`` extension (e.g.
        ``"screenshot_001.png"``).
    """
    screen_id = vh_json.get("screen_id", "")
    return f"{screen_id}.png"


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------


def load_rico_directory(
    rico_dir: Union[str, Path],
) -> List[GroundTruth]:
    """Load all screenshots from a RICO app directory.

    Scans for ``*.json`` files in the directory and parses each as a
    RICO View Hierarchy, using the same directory for screenshot images.

    Args:
        rico_dir: Path to a RICO app directory (e.g.
            ``data/raw/rico/unique_uis/com.example.app/``) containing
            screenshot PNGs and View Hierarchy JSONs.

    Returns:
        List of ``GroundTruth`` instances, one per JSON file found
        (sorted by filename).  Files that fail to parse are logged
        and skipped.

    Raises:
        FileNotFoundError: If ``rico_dir`` does not exist.
    """
    rico_dir = Path(rico_dir)
    if not rico_dir.is_dir():
        raise FileNotFoundError(f"RICO directory not found: {rico_dir}")

    json_files = sorted(rico_dir.glob("*.json"))
    results: List[GroundTruth] = []

    for json_path in json_files:
        try:
            gt = parse_rico_view_hierarchy(json_path, rico_dir)
            results.append(gt)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", json_path.name, exc)

    return results
