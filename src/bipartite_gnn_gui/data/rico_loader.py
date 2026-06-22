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

#: Mapping from RICO ``componentLabel`` values to canonical element types.
#: Used when semantic annotations are available (preferred over class-based).
_COMPONENT_LABEL_MAP: Dict[str, str] = {
    "Icon": "icon",
    "Text": "text",
    "Input": "input",
    "Drawer": "container",
    "Image": "image",
    "Button": "button",
    "List": "list",
    "Checkbox": "icon",
    "Switch": "icon",
    "On/Off": "icon",
    "Radio Button": "icon",
    "Text Button": "button",
    "Toolbar": "container",
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


def component_label_to_type(label: str) -> str:
    """Map a RICO ``componentLabel`` to a canonical element type.

    Args:
        label: Component label string (e.g. ``"Text"``, ``"Icon"``).

    Returns:
        Canonical element type, or ``"other"`` if unrecognized.
    """
    if not label or not isinstance(label, str):
        return "other"
    return _COMPONENT_LABEL_MAP.get(label, "other")


# ---------------------------------------------------------------------------
# Bounds parsing — accepts both string and list formats
# ---------------------------------------------------------------------------

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def parse_rico_bounds(
    bounds: Union[str, List[int], Tuple[int, ...], List[float], Tuple[float, ...]],
) -> Tuple[float, float, float, float]:
    """Parse RICO bounds into a 4-tuple of floats.

    Accepts **both** formats found in RICO data:

    * **Integer list** ``[x1, y1, x2, y2]`` (actual on-disk format).
    * **String** ``"[x1,y1][x2,y2]"`` (legacy format, for backward
      compatibility with test fixtures and older dumps).

    Args:
        bounds: Either a list/tuple of 4 numbers or a string in
            ``"[x1,y1][x2,y2]"`` format.

    Returns:
        ``(x1, y1, x2, y2)`` tuple of floats.

    Raises:
        GroundTruthParseError: If the input cannot be parsed.
    """
    # --- list / tuple path (actual on-disk format) ---------------------------
    if isinstance(bounds, (list, tuple)):
        if len(bounds) != 4:
            raise GroundTruthParseError(
                f"Expected 4-element bounds list, got {len(bounds)}: {bounds!r}"
            )
        return tuple(map(float, bounds))

    # --- string path (legacy format) ----------------------------------------
    bounds_str = str(bounds)
    m = _BOUNDS_RE.match(bounds_str.strip())
    if m is None:
        raise GroundTruthParseError(
            f"Cannot parse RICO bounds: {bounds!r}"
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
# Internal helpers — root extraction, text, visibility
# ---------------------------------------------------------------------------


def _extract_root(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the tree root node from a RICO JSON dict.

    Handles three formats:

    1. **View Hierarchy** – root is at ``data["activity"]["root"]``.
    2. **Semantic Annotations** – the top-level dict *is* the root node
       (has ``children`` and ``class`` keys).
    3. **Legacy** – root is at ``data["root"]``.

    Returns:
        The root node dict.

    Raises:
        GroundTruthParseError: If no root node can be found.
    """
    # Format 1: View Hierarchy (activity.root)
    activity = data.get("activity")
    if isinstance(activity, dict):
        root = activity.get("root")
        if isinstance(root, dict):
            return root

    # Format 3: Legacy wrapper (root)
    root = data.get("root")
    if isinstance(root, dict):
        return root

    # Format 2: Semantic Annotations — top-level IS the tree
    if "children" in data and "class" in data:
        return data

    raise GroundTruthParseError(
        "Cannot locate root node: expected 'activity.root', 'root', "
        "or top-level tree structure (with 'children' and 'class')"
    )


def _extract_text(node: Dict[str, Any]) -> Optional[str]:
    """Extract text content from a RICO node.

    Precedence: ``text`` → ``content-desc`` → ``componentLabel``.

    The ``content-desc`` field in RICO View Hierarchies is a **list**
    of nullable strings (e.g. ``[None]`` or ``["back button"]``).
    The first non-``None`` string is used.

    Returns cleaned text, or ``None`` if no text is available.
    """
    # Prefer text field
    text: Any = node.get("text")
    if text is not None and isinstance(text, str):
        text = text.strip()
        if text:
            return text

    # Fall back to content-desc (list on disk)
    content_desc_raw = node.get("content-desc")
    if isinstance(content_desc_raw, list):
        for item in content_desc_raw:
            if item is not None and isinstance(item, str):
                item = item.strip()
                if item:
                    return item
    elif content_desc_raw is not None and isinstance(content_desc_raw, str):
        content_desc_raw = content_desc_raw.strip()
        if content_desc_raw:
            return content_desc_raw

    # Last resort: componentLabel (e.g. "Text", "Icon")
    label = node.get("componentLabel")
    if label is not None and isinstance(label, str) and label.strip():
        return None  # componentLabel is a *type* hint, not text content

    return None


def _node_is_visible(node: Dict[str, Any]) -> bool:
    """Check whether a RICO node should be treated as visible.

    A node is skipped if:

    * ``visibility`` is present and not ``"visible"``, OR
    * ``visible-to-user`` is explicitly ``False``.

    Returns ``True`` for nodes that pass all visibility checks.
    """
    # Check visibility string
    visibility = node.get("visibility")
    if visibility is not None and visibility != "visible":
        return False

    # Check visible-to-user boolean (present in View Hierarchy format)
    visible_to_user = node.get("visible-to-user")
    if visible_to_user is not None and visible_to_user is not True:
        return False

    return True


def _derive_screen_id(vh_path: Path) -> str:
    """Derive a screen ID from a JSON file path (stem without extension)."""
    return vh_path.stem


def _find_paired_image(json_path: Path, images_dir: Path) -> str:
    """Find the screenshot image paired with a RICO JSON file.

    Tries ``.jpg`` first (View Hierarchy) then ``.png`` (Semantic Annotations).
    Falls back to ``.jpg`` if neither exists, so downstream code can report
    a clear error when the image is actually needed.

    Returns:
        Absolute path string to the paired image file.
    """
    stem = json_path.stem
    for ext in (".jpg", ".png"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.is_file():
            return str(candidate)
    # If neither exists, default to .jpg so the path is still meaningful
    return str(images_dir / f"{stem}.jpg")


# ---------------------------------------------------------------------------
# Unified View Hierarchy / Semantic Annotation parser
# ---------------------------------------------------------------------------

#: Canonical leaf types extracted from ``componentLabel`` – these types
#: are considered "descriptive" and take priority over Android class names.
_SEMANTIC_TYPE_SET = set(_COMPONENT_LABEL_MAP.values())


def _parse_rico_tree(
    vh_path: Path,
    images_dir: Path,
) -> GroundTruth:
    """Core parser for both View Hierarchy and Semantic Annotation JSON files.

    Handles three root-node wrapping styles (``activity.root``, ``root``,
    and bare top-level tree) and two type-mapping strategies
    (``componentLabel`` → class-based).

    Args:
        vh_path: Path to the RICO JSON file.
        images_dir: Directory containing paired screenshots.

    Returns:
        ``GroundTruth`` instance.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid or the root node cannot
            be located.
    """
    with vh_path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    # --- Derive screen_id from filename ------------------------------------
    screen_id = _derive_screen_id(vh_path)

    # --- Extract root node -------------------------------------------------
    root = _extract_root(data)

    # --- Derive screen dimensions from root bounds -------------------------
    root_bounds = root.get("bounds")
    if not isinstance(root_bounds, (list, tuple)) or len(root_bounds) != 4:
        raise GroundTruthParseError(
            f"Cannot derive screen dimensions: root node missing "
            f"valid 'bounds' list, got {root_bounds!r}"
        )
    screen_width = int(root_bounds[2])
    screen_height = int(root_bounds[3])

    if screen_width <= 0 or screen_height <= 0:
        raise GroundTruthParseError(
            f"Screen dimensions must be positive, "
            f"got width={screen_width}, height={screen_height}"
        )

    # --- Collect leaf nodes ------------------------------------------------
    leaf_nodes = _find_leaf_nodes(root)
    elements: List[GTElement] = []

    for i, node in enumerate(leaf_nodes):
        # ---- visibility filter --------------------------------------------
        if not _node_is_visible(node):
            logger.debug(
                "Skipping leaf node %d: not visible "
                "(visibility=%r, visible-to-user=%r)",
                i,
                node.get("visibility"),
                node.get("visible-to-user"),
            )
            continue

        # ---- bounds parsing -----------------------------------------------
        bounds_raw = node.get("bounds")
        if bounds_raw is None:
            logger.warning("Skipping leaf node %d: missing bounds", i)
            continue

        try:
            x1, y1, x2, y2 = parse_rico_bounds(bounds_raw)
        except GroundTruthParseError:
            logger.warning(
                "Skipping leaf node %d: invalid bounds %r", i, bounds_raw
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

        if x2_norm <= x1_norm or y2_norm <= y1_norm:
            logger.debug(
                "Skipping leaf node %d: degenerate after normalization", i
            )
            continue

        # ---- type mapping -------------------------------------------------
        # Prefer componentLabel (Semantic Annotations) over class-based mapping
        comp_label = node.get("componentLabel")
        if comp_label and isinstance(comp_label, str):
            element_type = component_label_to_type(comp_label)
        else:
            android_class = node.get("class", "")
            element_type = rico_class_to_type(android_class)

        # ---- text extraction ----------------------------------------------
        text = _extract_text(node)

        # ---- metadata -----------------------------------------------------
        android_class = node.get("class", "")
        metadata: Dict[str, Any] = {
            "class": android_class,
            "clickable": bool(node.get("clickable", False)),
        }
        if comp_label:
            metadata["componentLabel"] = comp_label

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

    # --- Construct image path -----------------------------------------------
    image_path = _find_paired_image(vh_path, images_dir)

    return GroundTruth(
        elements=elements,
        image_path=image_path,
        image_width=screen_width,
        image_height=screen_height,
        source="rico",
    )


def parse_rico_view_hierarchy(
    vh_path: Union[str, Path],
    images_dir: Union[str, Path],
) -> GroundTruth:
    """Parse a RICO View Hierarchy JSON into a ``GroundTruth``.

    Recursively traverses the tree (rooted at ``activity.root``,
    ``root``, or a bare top-level node) to extract all visible
    leaf nodes, normalizes their pixel bbox coordinates to ``[0, 1]``,
    and maps Android class names (or ``componentLabel`` values) to
    canonical element types.

    Args:
        vh_path: Path to the RICO View Hierarchy JSON file.
        images_dir: Directory containing the corresponding screenshot
            images (jpg or png).  Used to construct ``image_path``.

    Returns:
        ``GroundTruth`` instance with normalized bboxes and canonical types.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid, missing required keys,
            or screen dimensions are not positive.
    """
    return _parse_rico_tree(Path(vh_path), Path(images_dir))


# ---------------------------------------------------------------------------
# Semantic Annotation loader (delegates to unified tree parser)
# ---------------------------------------------------------------------------


def parse_rico_semantic(
    ann_path: Union[str, Path],
    images_dir: Union[str, Path],
) -> GroundTruth:
    """Parse a RICO Semantic Annotation JSON into a ``GroundTruth``.

    Semantic Annotations use the **same recursive tree structure** as
    View Hierarchies, but with ``componentLabel`` providing more
    descriptive element types (e.g. ``"Icon"``, ``"Text"``, ``"Drawer"``).

    This function delegates to :func:`parse_rico_view_hierarchy`, which
    automatically detects the wrapping style and prefers
    ``componentLabel`` when present.

    Args:
        ann_path: Path to the RICO Semantic Annotation JSON file.
        images_dir: Directory containing the paired screenshot images
            (png or jpg).

    Returns:
        ``GroundTruth`` instance with normalized bboxes and canonical types.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid or missing required keys.
    """
    return parse_rico_view_hierarchy(ann_path, images_dir)


# ---------------------------------------------------------------------------
# Image ID extraction
# ---------------------------------------------------------------------------


def get_rico_image_id(vh_json: Dict[str, Any]) -> str:
    """Extract the image filename from a RICO JSON dict.

    When ``screen_id`` is present it is used; otherwise the image filename
    must be derived from the JSON file path itself (see
    :func:`_derive_screen_id`).

    Args:
        vh_json: Parsed RICO JSON dict (may contain ``screen_id``).

    Returns:
        The screenshot filename with ``.jpg`` extension (e.g.
        ``"10101.jpg"``).  Callers should verify existence.
    """
    screen_id = vh_json.get("screen_id", "")
    return f"{screen_id}.jpg"


# ---------------------------------------------------------------------------
# Directory loader
# ---------------------------------------------------------------------------


def load_rico_directory(
    rico_dir: Union[str, Path],
) -> List[GroundTruth]:
    """Load all RICO screens from a flat JSON/images directory.

    Scans for ``*.json`` files at the top level of *rico_dir* and
    parses each as a RICO View Hierarchy (or Semantic Annotation),
    using the same directory to locate paired ``.jpg`` / ``.png`` images.

    Args:
        rico_dir: Path to a directory (e.g. ``data/raw/rico/combined/``)
            containing JSON + image files at the top level.

    Returns:
        List of ``GroundTruth`` instances, one per JSON file found
        (sorted by filename).  Files that fail to parse are logged
        and skipped.

    Raises:
        FileNotFoundError: If *rico_dir* does not exist.
    """
    rico_dir = Path(rico_dir)
    if not rico_dir.is_dir():
        raise FileNotFoundError(f"RICO directory not found: {rico_dir}")

    json_files = sorted(rico_dir.glob("*.json"))
    results: List[GroundTruth] = []

    for json_path in json_files:
        try:
            gt = _parse_rico_tree(json_path, rico_dir)
            results.append(gt)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", json_path.name, exc)

    return results
