"""VLM output parsing for Qwen3.5-2B and MiniMax-VL-01.

Provides dataclasses for parsed GUI element predictions and parser functions
for model-specific JSON output formats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VlmParseError(ValueError):
    """Raised when VLM output cannot be parsed due to fatal errors."""


# ---------------------------------------------------------------------------
# Element type taxonomy
# ---------------------------------------------------------------------------

#: Canonical element types and their recognized aliases (case-insensitive).
ELEMENT_TYPES: Dict[str, List[str]] = {
    "button": ["btn"],
    "text": ["label", "paragraph", "span"],
    "image": ["img", "picture"],
    "input": ["textbox", "search", "textarea", "textfield"],
    "icon": ["glyph"],
    "container": ["div", "section", "frame", "panel"],
    "card": [],
    "checkbox": ["check"],
    "radio": ["radiobutton"],
    "slider": ["range"],
    "switch": ["toggle"],
    "label": [],
    "tab": [],
    "menu": ["dropdown", "nav"],
    "divider": ["separator", "hr"],
    "list": [],
    "modal": ["dialog", "overlay"],
    "toast": ["snackbar", "notification"],
    "banner": ["announcement", "alertbar"],
    "other": [],
}


def _build_type_lookup() -> Dict[str, str]:
    """Build a case-insensitive lookup table mapping aliases to canonical types."""
    lookup: Dict[str, str] = {}
    for canonical, aliases in ELEMENT_TYPES.items():
        lookup[canonical.lower()] = canonical
        for alias in aliases:
            lookup[alias.lower()] = canonical
    return lookup


_TYPE_LOOKUP = _build_type_lookup()


def normalize_element_type(type_str: str) -> str:
    """Map a raw type string to its canonical element type (case-insensitive).

    Args:
        type_str: Raw element type string from VLM output.

    Returns:
        Canonical type name, or ``"other"`` if unrecognized.
    """
    key = type_str.strip().lower()
    if key in _TYPE_LOOKUP:
        return _TYPE_LOOKUP[key]
    logger.warning("Unknown element type '%s', mapping to 'other'", type_str)
    return "other"


def normalize_bbox(
    bbox: List[float],
    format: str = "xyxy",
    img_width: int = 0,
    img_height: int = 0,
) -> Tuple[float, float, float, float]:
    """Convert a bounding box to normalized xyxy format.

    Supports three input formats:

    - ``"xyxy"``: ``[x1, y1, x2, y2]``
    - ``"xywh"``: ``[x, y, w, h]`` (top-left corner + size)
    - ``"cxcywh"``: ``[cx, cy, w, h]`` (center + size)

    If *img_width* and *img_height* are positive, the coordinates are
    treated as pixel values and divided by the corresponding dimension
    to normalize to ``[0, 1]``.  Otherwise they are assumed to already
    be normalized.

    The result is clamped to ``[0.0, 1.0]``.

    Args:
        bbox: Four-element list of coordinates.
        format: Input format. One of ``"xyxy"``, ``"xywh"``, ``"cxcywh"``.
        img_width: Image width in pixels (0 if unknown / already normalized).
        img_height: Image height in pixels (0 if unknown / already normalized).

    Returns:
        Normalized ``(x1, y1, x2, y2)`` tuple with all values in ``[0, 1]``.

    Raises:
        VlmParseError: If the bbox list does not contain exactly 4 elements
            or the format string is unrecognized.
    """
    if len(bbox) != 4:
        raise VlmParseError(f"bbox must have 4 elements, got {len(bbox)}")

    x1, y1, x2, y2 = map(float, bbox)

    if format == "xywh":
        x1, y1, x2, y2 = x1, y1, x1 + x2, y1 + y2
    elif format == "cxcywh":
        cx, cy, w, h = x1, y1, x2, y2
        x1, y1 = cx - w / 2.0, cy - h / 2.0
        x2, y2 = cx + w / 2.0, cy + h / 2.0
    elif format != "xyxy":
        raise VlmParseError(
            f"Unknown bbox format '{format}'; expected 'xyxy', 'xywh', or 'cxcywh'"
        )

    # Normalize pixel values to [0, 1]
    if img_width > 0:
        x1 /= img_width
        x2 /= img_width
    if img_height > 0:
        y1 /= img_height
        y2 /= img_height

    # Clamp to [0, 1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VLMOutputElement:
    """A single GUI element predicted by a VLM.

    Attributes:
        element_id: Zero-based unique index within the parent ``VLMOutput``.
        bbox: Normalized bounding box ``(x1, y1, x2, y2)`` in ``[0, 1]``.
        element_type: Canonical element type from the shared taxonomy.
        text_content: Visible text on the element, or ``None``.
        confidence: Detection confidence in ``[0, 1]``.
        attributes: Free-form metadata dictionary (role, disabled, etc.).
        source: Model identifier (e.g. ``"qwen3.5-2b"``).
    """

    element_id: int
    bbox: Tuple[float, float, float, float]
    element_type: str
    text_content: Optional[str] = None
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class VLMOutput:
    """Parsed VLM predictions for a single screenshot.

    Attributes:
        image_id: Image identifier or filename.
        elements: Ordered list of predicted elements.
        model_name: VLM model identifier (e.g. ``"qwen3.5-2b"``).
        image_width: Original image pixel width (0 if unknown).
        image_height: Original image pixel height (0 if unknown).
        timestamp: ISO-8601 timestamp string, or empty string.
        parse_errors: Non-fatal parse issue descriptions.
    """

    image_id: str = ""
    elements: List[VLMOutputElement] = field(default_factory=list)
    model_name: str = ""
    image_width: int = 0
    image_height: int = 0
    timestamp: str = ""
    parse_errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal element parsers
# ---------------------------------------------------------------------------


def _parse_qwen_element(
    item: Dict[str, Any],
    raw_index: int,
    img_width: int,
    img_height: int,
    source: str,
) -> Tuple[Optional[VLMOutputElement], List[str]]:
    """Parse a single element from Qwen3.5-2B output.

    Args:
        item: Raw element dict from the VLM output.
        raw_index: Index in the original elements list (for error messages).
        img_width: Image width (0 if unknown).
        img_height: Image height (0 if unknown).
        source: Model identifier for the source field.

    Returns:
        Tuple of (parsed element or None if skipped, list of error messages).
    """
    errors: List[str] = []
    prefix = f"element[{raw_index}]"

    # Extract bbox (supports both bbox_xyxy and bbox field names)
    bbox_raw = item.get("bbox_xyxy", item.get("bbox"))
    if bbox_raw is None or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        errors.append(f"{prefix}: missing or invalid bbox, skipped")
        return None, errors

    try:
        bbox = normalize_bbox(
            list(bbox_raw), format="xyxy", img_width=img_width, img_height=img_height
        )
    except (VlmParseError, ValueError, TypeError) as exc:
        errors.append(f"{prefix}: invalid bbox ({exc}), skipped")
        return None, errors

    # Check for degeneracy
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        errors.append(
            f"{prefix}: degenerate bbox (x2={bbox[2]:.4f} <= x1={bbox[0]:.4f} "
            f"or y2={bbox[3]:.4f} <= y1={bbox[1]:.4f}), skipped"
        )
        return None, errors

    # Extract and normalize element type
    raw_type = item.get("label")
    if raw_type is None or not isinstance(raw_type, str) or not raw_type.strip():
        errors.append(f"{prefix}: missing or empty label, skipped")
        return None, errors
    element_type = normalize_element_type(raw_type)
    if element_type == "other" and raw_type.strip().lower() not in _TYPE_LOOKUP:
        errors.append(f"{prefix}: unknown type '{raw_type}', mapped to 'other'")

    # Extract text content (treat empty string as None)
    text_content: Optional[str] = item.get("text")
    if text_content is not None and isinstance(text_content, str) and text_content.strip() == "":
        text_content = None
        errors.append(f"{prefix}: empty text_content, treated as None")

    # Extract confidence
    confidence = item.get("confidence", 1.0)
    if confidence is None:
        confidence = 1.0
    confidence = max(0.0, min(1.0, float(confidence)))

    element = VLMOutputElement(
        element_id=0,  # placeholder; reassigned by caller
        bbox=bbox,
        element_type=element_type,
        text_content=text_content,
        confidence=confidence,
        attributes={},
        source=source,
    )
    return element, errors


def _parse_minimax_element(
    item: Dict[str, Any],
    raw_index: int,
    img_width: int,
    img_height: int,
    source: str,
) -> Tuple[Optional[VLMOutputElement], List[str]]:
    """Parse a single element from MiniMax-VL-01 output.

    Args:
        item: Raw element dict from the VLM output.
        raw_index: Index in the original elements list (for error messages).
        img_width: Image width in pixels.
        img_height: Image height in pixels.
        source: Model identifier for the source field.

    Returns:
        Tuple of (parsed element or None if skipped, list of error messages).
    """
    errors: List[str] = []
    prefix = f"element[{raw_index}]"

    # Extract bbox (pixel values)
    bbox_raw = item.get("bbox")
    if bbox_raw is None or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        errors.append(f"{prefix}: missing or invalid bbox, skipped")
        return None, errors

    try:
        bbox = normalize_bbox(
            list(bbox_raw), format="xyxy", img_width=img_width, img_height=img_height
        )
    except (VlmParseError, ValueError, TypeError) as exc:
        errors.append(f"{prefix}: invalid bbox ({exc}), skipped")
        return None, errors

    # Check for degeneracy
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        errors.append(
            f"{prefix}: degenerate bbox (x2={bbox[2]:.4f} <= x1={bbox[0]:.4f} "
            f"or y2={bbox[3]:.4f} <= y1={bbox[1]:.4f}), skipped"
        )
        return None, errors

    # Extract and normalize element type
    raw_type = item.get("category")
    if raw_type is None or not isinstance(raw_type, str) or not raw_type.strip():
        errors.append(f"{prefix}: missing or empty category, skipped")
        return None, errors

    element_type = normalize_element_type(raw_type)
    if element_type == "other":
        # Try prefix match for compound types like "button-primary"
        prefix_part = raw_type.split("-")[0].strip().lower()
        if prefix_part in _TYPE_LOOKUP:
            element_type = _TYPE_LOOKUP[prefix_part]
        else:
            errors.append(f"{prefix}: unknown type '{raw_type}', mapped to 'other'")

    # Extract text content
    text_content: Optional[str] = item.get("text_content")
    if text_content is not None and isinstance(text_content, str) and text_content.strip() == "":
        text_content = None

    # Extract confidence
    confidence = item.get("confidence", 1.0)
    if confidence is None:
        confidence = 1.0
    confidence = max(0.0, min(1.0, float(confidence)))

    # Extract attributes
    attributes: Dict[str, Any] = item.get("attributes")
    if attributes is None or not isinstance(attributes, dict):
        attributes = {}

    element = VLMOutputElement(
        element_id=0,  # placeholder; reassigned by caller
        bbox=bbox,
        element_type=element_type,
        text_content=text_content,
        confidence=confidence,
        attributes=attributes,
        source=source,
    )
    return element, errors


# ---------------------------------------------------------------------------
# Top-level parsers
# ---------------------------------------------------------------------------


def parse_qwen_output(data: Dict[str, Any]) -> VLMOutput:
    """Parse Qwen3.5-2B JSON output into a ``VLMOutput``.

    Args:
        data: Parsed JSON dict from Qwen3.5-2B. Expected keys:
            ``image_id`` (str), ``elements`` (list[dict]).

    Returns:
        Parsed ``VLMOutput`` with normalized elements.

    Raises:
        VlmParseError: If the top-level structure is invalid (missing or
            non-dict input, missing ``elements`` key).
    """
    if not isinstance(data, dict):
        raise VlmParseError(f"Expected dict, got {type(data).__name__}")

    elements_raw = data.get("elements")
    if elements_raw is None or not isinstance(elements_raw, (list, tuple)):
        raise VlmParseError("Missing or invalid 'elements' key in Qwen output")

    image_id = str(data.get("image_id", ""))
    img_width = int(data.get("image_width", 0))
    img_height = int(data.get("image_height", 0))
    timestamp = str(data.get("timestamp", ""))
    source = "qwen3.5-2b"

    elements: List[VLMOutputElement] = []
    parse_errors: List[str] = []

    for i, item in enumerate(elements_raw):
        if not isinstance(item, dict):
            parse_errors.append(f"element[{i}]: expected dict, skipped")
            continue
        element, item_errors = _parse_qwen_element(
            item, i, img_width, img_height, source
        )
        parse_errors.extend(item_errors)
        if element is not None:
            element.element_id = len(elements)
            elements.append(element)

    return VLMOutput(
        image_id=image_id,
        elements=elements,
        model_name=source,
        image_width=img_width,
        image_height=img_height,
        timestamp=timestamp,
        parse_errors=parse_errors,
    )


def parse_minimax_output(data: Dict[str, Any]) -> VLMOutput:
    """Parse MiniMax-VL-01 JSON output into a ``VLMOutput``.

    Args:
        data: Parsed JSON dict from MiniMax-VL-01. Expected keys:
            ``image_id`` (str), ``image_width`` (int), ``image_height`` (int),
            ``elements`` (list[dict]).

    Returns:
        Parsed ``VLMOutput`` with normalized elements.

    Raises:
        VlmParseError: If the top-level structure is invalid (missing or
            non-dict input, missing ``elements`` key).
    """
    if not isinstance(data, dict):
        raise VlmParseError(f"Expected dict, got {type(data).__name__}")

    elements_raw = data.get("elements")
    if elements_raw is None or not isinstance(elements_raw, (list, tuple)):
        raise VlmParseError("Missing or invalid 'elements' key in MiniMax output")

    image_id = str(data.get("image_id", ""))
    img_width = int(data.get("image_width", 0))
    img_height = int(data.get("image_height", 0))
    timestamp = str(data.get("timestamp", ""))
    source = "minimax-vl-01"

    elements: List[VLMOutputElement] = []
    parse_errors: List[str] = []

    for i, item in enumerate(elements_raw):
        if not isinstance(item, dict):
            parse_errors.append(f"element[{i}]: expected dict, skipped")
            continue
        element, item_errors = _parse_minimax_element(
            item, i, img_width, img_height, source
        )
        parse_errors.extend(item_errors)
        if element is not None:
            element.element_id = len(elements)
            elements.append(element)

    return VLMOutput(
        image_id=image_id,
        elements=elements,
        model_name=source,
        image_width=img_width,
        image_height=img_height,
        timestamp=timestamp,
        parse_errors=parse_errors,
    )
