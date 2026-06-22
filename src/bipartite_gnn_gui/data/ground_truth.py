"""Ground-truth annotation loading and prediction matching.

Provides dataclasses for ground-truth GUI elements and loaders for
GUI-360° and ScreenSpot annotation formats, plus Hungarian-algorithm
matching against VLM predictions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from ..data.vlm_output import VLMOutputElement, normalize_element_type
from ..utils.bbox import bbox_to_tensor, compute_iou

try:
    from PIL import Image as PILImage
except ImportError:  # pragma: no cover - optional dependency
    PILImage = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GroundTruthParseError(ValueError):
    """Raised when a ground-truth annotation file cannot be parsed due to
    fatal errors (invalid JSON, missing required keys, non-positive
    image dimensions).
    """


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GTElement:
    """A single ground-truth GUI element annotation.

    Attributes:
        element_id: Unique element identifier from the source dataset.
        bbox: Normalized bounding box ``(x1, y1, x2, y2)`` in ``[0, 1]``.
        element_type: Canonical element type from the shared taxonomy.
        text_content: OCR text or element description, or ``None``.
        source_dataset: Origin dataset identifier (``"gui360"`` or ``"screenspot"``).
        metadata: Original metadata merged from the source.
    """

    element_id: str
    bbox: Tuple[float, float, float, float]
    element_type: str
    text_content: Optional[str] = None
    source_dataset: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    """Ground-truth annotations for a single screenshot.

    Attributes:
        elements: Ordered list of ground-truth element annotations.
        image_path: Local filesystem path to the corresponding screenshot.
        image_width: Original image pixel width.
        image_height: Original image pixel height.
        source: Source dataset identifier (``"gui360"`` or ``"screenspot"``).
    """

    elements: List[GTElement] = field(default_factory=list)
    image_path: str = ""
    image_width: int = 0
    image_height: int = 0
    source: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _element_to_bbox(elem: Union[VLMOutputElement, GTElement]) -> Tensor:
    """Extract a 4-element bbox tensor from either element type.

    Args:
        elem: A ``VLMOutputElement`` or ``GTElement`` instance.

    Returns:
        Float32 tensor of shape ``(4,)`` containing ``(x1, y1, x2, y2)``.
    """
    return bbox_to_tensor(list(elem.bbox))


def _parse_common_annotation(
    item: Dict[str, Any],
) -> Optional[Tuple[str, Tuple[float, float, float, float], str, Optional[str], Dict[str, Any]]]:
    """Extract common fields from a raw annotation dict.

    Returns ``None`` if the annotation should be skipped (invalid bbox).

    Returns a tuple of
    ``(element_id, bbox, element_type, text_content, attributes)``
    with the bbox already validated and type already normalised.
    """
    element_id = item.get("element_id", "")

    bbox_raw = item.get("bbox")
    if bbox_raw is None or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
        logger.warning("Skipping annotation with missing/invalid bbox: %s", element_id)
        return None

    bbox = tuple(map(float, bbox_raw))
    x1, y1, x2, y2 = bbox

    # Validate — degenerate bboxes are skipped
    if x2 <= x1 or y2 <= y1:
        logger.warning(
            "Skipping degenerate bbox (x2=%.4f <= x1=%.4f or y2=%.4f <= y1=%.4f): %s",
            x2, x1, y2, y1, element_id,
        )
        return None

    # Normalise type
    raw_type = str(item.get("type", "")) if item.get("type") is not None else ""
    element_type = normalize_element_type(raw_type)
    if element_type == "other" and raw_type.strip():
        logger.warning("Unknown type '%s' for %s, mapped to 'other'", raw_type, element_id)

    # Text: empty string -> None
    text_content: Optional[str] = item.get("text")
    if text_content is not None and isinstance(text_content, str) and text_content == "":
        text_content = None

    # Attributes
    attributes = item.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}

    return (element_id, bbox, element_type, text_content, attributes)


# ---------------------------------------------------------------------------
# Format-specific loaders
# ---------------------------------------------------------------------------


def load_gui360_annotation(path: Union[str, Path]) -> GroundTruth:
    """Load a GUI-360 degree JSON annotation file into a unified GroundTruth.

    GUI-360 degree bboxes are already normalised to [0, 1] and are passed
    through without coordinate conversion.

    Args:
        path: Path to the GUI-360 degree JSON annotation file.

    Returns:
        GroundTruth instance with normalised bboxes and canonical types.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid, a required key is
            missing, or image_width / image_height is not positive.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    image_id = data.get("image_id", "")
    image_width = int(data.get("image_width", 0))
    image_height = int(data.get("image_height", 0))
    platform = str(data.get("platform", ""))

    if image_width <= 0:
        raise GroundTruthParseError(
            f"image_width must be positive, got {image_width}"
        )
    if image_height <= 0:
        raise GroundTruthParseError(
            f"image_height must be positive, got {image_height}"
        )

    annotations_raw: List[Dict[str, Any]] = data.get("annotations", [])
    elements: List[GTElement] = []

    for item in annotations_raw:
        parsed = _parse_common_annotation(item)
        if parsed is None:
            continue
        element_id, bbox, element_type, text_content, attributes = parsed

        # Merge platform + attributes
        metadata: Dict[str, Any] = {"platform": platform}
        metadata.update(attributes)

        elements.append(
            GTElement(
                element_id=element_id,
                bbox=bbox,
                element_type=element_type,
                text_content=text_content,
                source_dataset="gui360",
                metadata=metadata,
            )
        )

    image_path = f"data/raw/gui360/images/{image_id}"

    return GroundTruth(
        elements=elements,
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        source="gui360",
    )


def load_screenspot_annotation(path: Union[str, Path]) -> GroundTruth:
    """Load a ScreenSpot JSON annotation file into a unified GroundTruth.

    ScreenSpot uses absolute pixel bbox coordinates; these are normalised
    to [0, 1] at load time.

    Args:
        path: Path to the ScreenSpot JSON annotation file.

    Returns:
        GroundTruth instance with normalised bboxes and canonical types.

    Raises:
        FileNotFoundError: The file does not exist.
        GroundTruthParseError: The JSON is invalid, a required key is
            missing, or image_width / image_height is not positive.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    image_id = data.get("image_id", "")
    image_width = int(data.get("image_width", 0))
    image_height = int(data.get("image_height", 0))
    group = str(data.get("group", ""))

    if image_width <= 0:
        raise GroundTruthParseError(
            f"image_width must be positive, got {image_width}"
        )
    if image_height <= 0:
        raise GroundTruthParseError(
            f"image_height must be positive, got {image_height}"
        )

    annotations_raw: List[Dict[str, Any]] = data.get("annotations", [])
    elements: List[GTElement] = []

    for item in annotations_raw:
        element_id = item.get("element_id", "")

        bbox_raw = item.get("bbox")
        if bbox_raw is None or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
            logger.warning("Skipping annotation with missing/invalid bbox: %s", element_id)
            continue

        # Normalise pixel coordinates to [0, 1] and clamp
        x1_px, y1_px, x2_px, y2_px = map(float, bbox_raw)
        x1 = max(0.0, min(1.0, x1_px / image_width))
        y1 = max(0.0, min(1.0, y1_px / image_height))
        x2 = max(0.0, min(1.0, x2_px / image_width))
        y2 = max(0.0, min(1.0, y2_px / image_height))
        bbox = (x1, y1, x2, y2)

        # Validate after normalisation
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Skipping degenerate bbox after normalization "
                "(x2=%.4f <= x1=%.4f or y2=%.4f <= y1=%.4f): %s",
                x2, x1, y2, y1, element_id,
            )
            continue

        # Normalise type
        raw_type = str(item.get("type", "")) if item.get("type") is not None else ""
        element_type = normalize_element_type(raw_type)
        if element_type == "other" and raw_type.strip():
            logger.warning("Unknown type '%s' for %s, mapped to 'other'", raw_type, element_id)

        # Text: empty string -> None
        text_content: Optional[str] = item.get("text")
        if text_content is not None and isinstance(text_content, str) and text_content == "":
            text_content = None

        # Attributes
        attributes = item.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}

        # Merge group + attributes
        metadata: Dict[str, Any] = {"group": group}
        metadata.update(attributes)

        elements.append(
            GTElement(
                element_id=element_id,
                bbox=bbox,
                element_type=element_type,
                text_content=text_content,
                source_dataset="screenspot",
                metadata=metadata,
            )
        )

    image_path = f"data/raw/screenspot/images/{image_id}"

    return GroundTruth(
        elements=elements,
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        source="screenspot",
    )


def load_screenspot_combined(
    path: Union[str, Path],
    images_dir: Union[str, Path],
) -> List[GroundTruth]:
    """Load the ScreenSpot combined JSON format into a list of GroundTruth objects.

    The combined format is a single JSON array where each entry describes one
    screenshot and its annotations.  Image dimensions are read from the
    corresponding PNG files via PIL since the combined JSON does not include
    ``image_width`` / ``image_height`` fields.

    Bounding boxes are stored as ``[x, y, w, h]`` (top-left corner + size),
    converted to normalised ``[x1, y1, x2, y2]`` in ``[0, 1]``.

    Args:
        path: Path to the combined ScreenSpot JSON file (e.g.
            ``ScreenSpot_combined.json``).
        images_dir: Directory containing the PNG image files referenced
            by the ``"image"`` field of each entry.

    Returns:
        List of GroundTruth objects, one per image entry in the combined JSON.

    Raises:
        FileNotFoundError: The combined JSON file does not exist.
        GroundTruthParseError: The JSON is not a list (invalid combined format).
    """
    if PILImage is None:  # pragma: no cover
        raise ImportError("Pillow is required to load ScreenSpot combined format")

    path = Path(path)
    images_dir = Path(images_dir)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise GroundTruthParseError(
            f"Expected a JSON array for combined ScreenSpot format, "
            f"got {type(data).__name__}"
        )

    results: List[GroundTruth] = []

    for entry in data:
        if not isinstance(entry, dict):
            logger.warning("Skipping non-dict entry in combined ScreenSpot JSON")
            continue

        image_file = str(entry.get("image", ""))
        if not image_file:
            logger.warning("Skipping entry with missing 'image' field")
            continue

        # Open the image to get pixel dimensions
        image_path = images_dir / image_file
        try:
            with PILImage.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as exc:
            logger.warning(
                "Failed to open image %s: %s, skipping entry", image_path, exc
            )
            continue

        if image_width <= 0 or image_height <= 0:
            logger.warning(
                "Skipping entry with invalid image dimensions (%d, %d): %s",
                image_width, image_height, image_file,
            )
            continue

        annotations_raw: List[Dict[str, Any]] = entry.get("annotations", [])
        if not isinstance(annotations_raw, list):
            annotations_raw = []

        elements: List[GTElement] = []

        for idx, item in enumerate(annotations_raw):
            if not isinstance(item, dict):
                logger.warning(
                    "Skipping non-dict annotation %d in %s", idx, image_file
                )
                continue

            # bounding_box is [x, y, w, h] — convert to xyxy
            bbox_raw = item.get("bounding_box")
            if (
                bbox_raw is None
                or not isinstance(bbox_raw, (list, tuple))
                or len(bbox_raw) != 4
            ):
                logger.warning(
                    "Skipping annotation %d with missing/invalid "
                    "bounding_box in %s", idx, image_file,
                )
                continue

            x, y, w, h = map(float, bbox_raw)
            x1_px = x
            y1_px = y
            x2_px = x + w
            y2_px = y + h

            # Normalise to [0, 1] and clamp
            x1 = max(0.0, min(1.0, x1_px / image_width))
            y1 = max(0.0, min(1.0, y1_px / image_height))
            x2 = max(0.0, min(1.0, x2_px / image_width))
            y2 = max(0.0, min(1.0, y2_px / image_height))
            bbox = (x1, y1, x2, y2)

            if x2 <= x1 or y2 <= y1:
                logger.warning(
                    "Skipping degenerate bbox in %s annotation %d "
                    "(x2=%.4f <= x1=%.4f or y2=%.4f <= y1=%.4f)",
                    image_file, idx, x2, x1, y2, y1,
                )
                continue

            # Map field names: data_type → type
            raw_type = (
                str(item.get("data_type", ""))
                if item.get("data_type") is not None
                else ""
            )
            element_type = normalize_element_type(raw_type)
            if element_type == "other" and raw_type.strip():
                logger.warning(
                    "Unknown type '%s' in %s annotation %d, mapped to 'other'",
                    raw_type, image_file, idx,
                )

            # Map field names: objective_reference → text
            text_content: Optional[str] = item.get("objective_reference")
            if (
                text_content is not None
                and isinstance(text_content, str)
                and text_content == ""
            ):
                text_content = None

            # Map field names: data_source → group
            data_source = str(item.get("data_source", ""))

            element_id = f"{Path(image_file).stem}_{idx}"

            metadata: Dict[str, Any] = {
                "group": data_source,
                "data_source": data_source,
            }

            elements.append(
                GTElement(
                    element_id=element_id,
                    bbox=bbox,
                    element_type=element_type,
                    text_content=text_content,
                    source_dataset="screenspot",
                    metadata=metadata,
                )
            )

        results.append(
            GroundTruth(
                elements=elements,
                image_path=str(image_path),
                image_width=image_width,
                image_height=image_height,
                source="screenspot",
            )
        )

    return results


# ---------------------------------------------------------------------------
# Factory dispatcher
# ---------------------------------------------------------------------------


def load_ground_truth(path: Union[str, Path], source: Optional[str] = None) -> GroundTruth:
    """Load a ground-truth annotation file, auto-detecting the format.

    Args:
        path: Path to the annotation file.
        source: Dataset identifier ("gui360", "screenspot", or "rico").
            When None, the format is auto-detected from the file
            contents by checking for the presence of a "platform" key
            (GUI-360 degree), "group" key (ScreenSpot), or "root" key
            (RICO View Hierarchy).

    Returns:
        GroundTruth instance.

    Raises:
        GroundTruthParseError: If the format cannot be determined or
            parsing fails.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)

    if source is None:
        with path.open("r", encoding="utf-8") as f:
            data: Any = json.load(f)

        # Detect combined ScreenSpot format (JSON array)
        if isinstance(data, list):
            raise GroundTruthParseError(
                f"{path} is a ScreenSpot combined JSON array. "
                "Use load_screenspot_combined() with an images_dir instead."
            )

        # data is now known to be a dict
        if "root" in data:
            source = "rico"
        elif "platform" in data:
            source = "gui360"
        elif "group" in data:
            source = "screenspot"
        else:
            raise GroundTruthParseError(
                f"Cannot determine ground-truth format from {path}: "
                "missing 'root' (RICO), 'platform' (GUI-360), or 'group' (ScreenSpot)"
            )

    if source == "gui360":
        return load_gui360_annotation(path)
    elif source == "screenspot":
        return load_screenspot_annotation(path)
    elif source == "rico":
        from .rico_loader import parse_rico_view_hierarchy
        # Use the parent directory of the JSON as images_dir
        images_dir = path.parent
        return parse_rico_view_hierarchy(path, images_dir)
    else:
        raise GroundTruthParseError(
            f"Unknown ground-truth source '{source}'; "
            "expected 'gui360', 'screenspot', or 'rico'"
        )


# ---------------------------------------------------------------------------
# Prediction to ground-truth matching (Hungarian algorithm)
# ---------------------------------------------------------------------------


def match_predictions_to_ground_truth(
    predictions: Sequence[Union[VLMOutputElement, GTElement]],
    ground_truth: Sequence[GTElement],
    iou_threshold: float = 0.5,
    type_conditioned: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Match predicted elements to ground-truth elements using the Hungarian algorithm.

    Builds an IoU-based cost matrix C of shape (M, N) where
    C[i, j] = 1 - IoU(pred_i, gt_j).  Pairs with IoU below
    iou_threshold are given infinite cost and excluded from matching.

    When type_conditioned is True, pairs where the predicted and
    ground-truth element types differ (and neither is "other") are
    also given infinite cost.

    Args:
        predictions: Predicted elements (VLMOutputElement or GTElement).
        ground_truth: Ground-truth elements (GTElement).
        iou_threshold: Minimum IoU for a valid match (default 0.5).
        type_conditioned: If True, require matching types.

    Returns:
        Tuple of (matched_pairs, fp_indices, fn_indices):
        - matched_pairs: List of (pred_idx, gt_idx) for successful matches.
        - fp_indices: Indices of unmatched predictions (false positives).
        - fn_indices: Indices of unmatched ground-truth elements (false negatives).
    """
    M = len(predictions)
    N = len(ground_truth)

    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))

    # Build bbox tensors: (M, 4) and (N, 4)
    pred_boxes = torch.stack([_element_to_bbox(p) for p in predictions])
    gt_boxes = torch.stack([_element_to_bbox(g) for g in ground_truth])

    # Compute IoU matrix (M, N)
    iou_matrix = compute_iou(pred_boxes, gt_boxes)

    # Build cost matrix: C[i,j] = 1 - IoU(i,j)
    cost = 1.0 - iou_matrix
    INF = float("inf")

    # Apply IoU threshold
    cost[iou_matrix < iou_threshold] = INF

    # Apply type conditioning
    if type_conditioned:
        for i in range(M):
            pred_type = predictions[i].element_type
            for j in range(N):
                gt_type = ground_truth[j].element_type
                if pred_type != "other" and gt_type != "other" and pred_type != gt_type:
                    cost[i, j] = INF

    # Hungarian algorithm (minimises total cost)
    # Check if any feasible pair exists; scipy raises ValueError when
    # the cost matrix is entirely infeasible (all entries are INF).
    has_feasible = torch.isfinite(cost).any().item()

    if has_feasible:
        row_indices, col_indices = linear_sum_assignment(cost.numpy())
    else:
        row_indices, col_indices = [], []

    matched_pairs: List[Tuple[int, int]] = []
    matched_rows: set = set()
    matched_cols: set = set()

    for i, j in zip(row_indices, col_indices):
        if cost[i, j] < INF:
            matched_pairs.append((int(i), int(j)))
            matched_rows.add(int(i))
            matched_cols.add(int(j))

    fp_indices = [i for i in range(M) if i not in matched_rows]
    fn_indices = [j for j in range(N) if j not in matched_cols]

    return matched_pairs, fp_indices, fn_indices
