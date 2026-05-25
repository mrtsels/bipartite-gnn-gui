"""Graph schema objects for the bipartite GUI graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    """Node describing a GUI element."""

    bbox: list[float]
    label: str = "unknown"
    confidence: float = 1.0
    element_id: str | None = None
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class ConstraintNode:
    """Node describing a spatial constraint."""

    constraint_type: ConstraintType
    source_indices: list[int] = field(default_factory=list)
    target_indices: list[int] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)
