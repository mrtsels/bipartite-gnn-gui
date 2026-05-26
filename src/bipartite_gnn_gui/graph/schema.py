"""Graph schema objects for the bipartite GUI graph."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List

import torch
from torch import Tensor

from bipartite_gnn_gui.data.vlm_output import VLMOutputElement


class ConstraintType(str, Enum):
    """Supported spatial constraints."""

    ALIGN_LEFT = "align_left"
    ALIGN_RIGHT = "align_right"
    ALIGN_TOP = "align_top"
    ALIGN_BOTTOM = "align_bottom"
    CENTER_X = "center_x"
    CENTER_Y = "center_y"
    SAME_SIZE = "same_size"
    SPACING = "spacing"
    CONTAINMENT = "containment"
    GRID = "grid"


class EdgeType(str, Enum):
    """Edge categories for the bipartite graph."""

    ELEMENT_TO_CONSTRAINT = "element_to_constraint"
    CONSTRAINT_TO_ELEMENT = "constraint_to_element"


@dataclass
class ElementNode:
    """Node describing a GUI element.

    Attributes:
        bbox: Bounding box as ``[x1, y1, x2, y2]`` (normalized to [0, 1]).
        label: Element type label string (e.g. ``"button"``, ``"text"``).
        confidence: Detection confidence in [0, 1].
        element_id: Optional unique identifier.
        features: Extra key-value feature dictionary.
    """

    bbox: List[float]
    label: str = "unknown"
    confidence: float = 1.0
    element_id: str | None = None
    features: dict[str, float] = field(default_factory=dict)

    def to_tensor(self) -> Tensor:
        """Convert this node to a Float32 feature tensor.

        Returns:
            Tensor of shape ``(5,)`` containing
            ``[x1, y1, x2, y2, confidence]``.
        """
        return torch.tensor(
            [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.confidence],
            dtype=torch.float32,
        )

    @classmethod
    def from_vlm_element(cls, vlm_elem: VLMOutputElement) -> ElementNode:
        """Construct an ``ElementNode`` from a ``VLMOutputElement``.

        Args:
            vlm_elem: Parsed VLM output element.

        Returns:
            New ``ElementNode`` with fields mapped from the VLM element.
        """
        return cls(
            bbox=list(vlm_elem.bbox),
            label=vlm_elem.element_type,
            confidence=vlm_elem.confidence,
            element_id=str(vlm_elem.element_id) if vlm_elem.element_id is not None else None,
        )


@dataclass
class ConstraintNode:
    """Node describing a spatial constraint.

    Attributes:
        constraint_type: Type of constraint (one of 10 types).
        source_indices: Indices of source elements (0-based).
        target_indices: Indices of target elements (0-based).
        params: Constraint-specific parameters (e.g. ``{"tolerance": 0.02}``).
    """

    constraint_type: ConstraintType
    source_indices: List[int] = field(default_factory=list)
    target_indices: List[int] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)

    def to_tensor(self) -> Tensor:
        """Convert constraint parameters to a Float32 feature tensor.

        Returns:
            Tensor of shape ``(D,)`` where ``D = len(params)``, or
            ``(1,)`` with value ``[0.0]`` when ``params`` is empty.
        """
        if not self.params:
            return torch.zeros(1, dtype=torch.float32)
        return torch.tensor(list(self.params.values()), dtype=torch.float32)

    def to_onehot(self) -> Tensor:
        """Return a 10-dimensional one-hot encoding of the constraint type.

        The order matches the definition order of ``ConstraintType``.

        Returns:
            Float32 Tensor of shape ``(10,)`` with a single ``1.0`` at
            the index corresponding to ``self.constraint_type``.
        """
        types = list(ConstraintType)
        idx = types.index(self.constraint_type)
        onehot = torch.zeros(len(types), dtype=torch.float32)
        onehot[idx] = 1.0
        return onehot


@dataclass
class EdgeFeatures:
    """Features computed between two element nodes connected by an edge.

    Attributes:
        spatial_distance: Euclidean distance between element centers.
        relative_position: Tuple ``(dx, dy)`` representing the center
            offset of element B relative to element A, normalized by
            image dimensions.
        iou: Intersection over Union between the two bounding boxes.
    """

    spatial_distance: float
    relative_position: tuple[float, float]
    iou: float

    def to_tensor(self) -> Tensor:
        """Convert edge features to a Float32 feature tensor.

        Returns:
            Tensor of shape ``(4,)`` containing
            ``[spatial_distance, dx, dy, iou]``.
        """
        return torch.tensor(
            [
                self.spatial_distance,
                self.relative_position[0],
                self.relative_position[1],
                self.iou,
            ],
            dtype=torch.float32,
        )

    @classmethod
    def compute(cls, elem_a: ElementNode, elem_b: ElementNode) -> EdgeFeatures:
        """Compute edge features between two element nodes.

        Both bounding boxes are assumed to be in normalized xyxy format
        ``[x1, y1, x2, y2]`` with values in ``[0, 1]``.

        Args:
            elem_a: First element node.
            elem_b: Second element node.

        Returns:
            ``EdgeFeatures`` with computed spatial distance, relative
            position, and IoU.
        """
        x1_a, y1_a, x2_a, y2_a = elem_a.bbox
        x1_b, y1_b, x2_b, y2_b = elem_b.bbox

        # Centers
        cx_a = (x1_a + x2_a) / 2.0
        cy_a = (y1_a + y2_a) / 2.0
        cx_b = (x1_b + x2_b) / 2.0
        cy_b = (y1_b + y2_b) / 2.0

        # Spatial distance (Euclidean between centers)
        dx = cx_b - cx_a
        dy = cy_b - cy_a
        spatial_distance = math.sqrt(dx * dx + dy * dy)

        # Relative position (already normalized since bboxes are in [0, 1])
        relative_position = (dx, dy)

        # Intersection over Union
        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)
        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        intersection = inter_w * inter_h

        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union = area_a + area_b - intersection
        iou = intersection / union if union > 0.0 else 0.0

        return cls(
            spatial_distance=spatial_distance,
            relative_position=relative_position,
            iou=iou,
        )
