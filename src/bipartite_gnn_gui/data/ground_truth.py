"""Ground-truth annotation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..utils.bbox import bbox_to_tensor, compute_iou


@dataclass
class GTElement:
    """Single annotated GUI element."""

    bbox: list[float]
    label: str = "unknown"
    element_id: str | None = None


@dataclass
class GroundTruth:
    """Container for annotations."""

    elements: list[GTElement] = field(default_factory=list)
    source: str | None = None
    image_size: tuple[int, int] | None = None


def _parse_element(payload: Mapping[str, Any]) -> GTElement:
    bbox = list(payload.get("bbox", payload.get("box", [0.0, 0.0, 0.0, 0.0])))
    return GTElement(
        bbox=[float(value) for value in bbox],
        label=str(payload.get("label", payload.get("type", "unknown"))),
        element_id=payload.get("id"),
    )


def load_ground_truth(source: str | Path | Mapping[str, Any]) -> GroundTruth:
    """Load annotations from a JSON file or mapping."""

    if isinstance(source, Mapping):
        payload = source
        source_name = None
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        source_name = str(path)

    elements = [_parse_element(element) for element in payload.get("elements", payload.get("annotations", []))]
    image_size = tuple(payload["image_size"]) if payload.get("image_size") else None
    return GroundTruth(elements=elements, source=source_name, image_size=image_size)


def match_elements(predicted: Sequence[Mapping[str, Any] | GTElement], ground_truth: Sequence[Mapping[str, Any] | GTElement], iou_threshold: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedily match predicted elements to ground truth by IoU."""

    gt_remaining = set(range(len(ground_truth)))
    matches: list[tuple[int, int, float]] = []

    for pred_index, predicted_element in enumerate(predicted):
        pred_bbox = predicted_element.bbox if isinstance(predicted_element, GTElement) else list(predicted_element.get("bbox", [0.0, 0.0, 0.0, 0.0]))
        pred_tensor = bbox_to_tensor(pred_bbox).unsqueeze(0)

        best_match = None
        best_score = 0.0
        for gt_index in list(gt_remaining):
            gt_element = ground_truth[gt_index]
            gt_bbox = gt_element.bbox if isinstance(gt_element, GTElement) else list(gt_element.get("bbox", [0.0, 0.0, 0.0, 0.0]))
            score = float(compute_iou(pred_tensor, bbox_to_tensor(gt_bbox).unsqueeze(0)).item())
            if score > best_score:
                best_score = score
                best_match = gt_index

        if best_match is not None and best_score >= iou_threshold:
            matches.append((pred_index, best_match, best_score))
            gt_remaining.remove(best_match)

    return matches
