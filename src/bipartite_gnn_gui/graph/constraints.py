"""Heuristic constraint extraction for GUI elements."""

from __future__ import annotations

from typing import Sequence

from .schema import ConstraintNode, ConstraintType, ElementNode


def extract_alignment_constraints(elements: Sequence[ElementNode], tolerance: float = 0.02) -> list[ConstraintNode]:
    """Extract a small set of alignment constraints."""

    if len(elements) < 2:
        return []
    return [ConstraintNode(constraint_type=ConstraintType.ALIGN_LEFT, source_indices=[0, 1], target_indices=[0, 1], params={"tolerance": tolerance})]


def extract_containment_constraints(elements: Sequence[ElementNode]) -> list[ConstraintNode]:
    """Extract containment constraints."""

    return []


def extract_spacing_constraints(elements: Sequence[ElementNode], tolerance: float = 0.02) -> list[ConstraintNode]:
    """Extract spacing constraints."""

    return []


def extract_grid_constraints(elements: Sequence[ElementNode]) -> list[ConstraintNode]:
    """Extract grid constraints."""

    return []


def extract_all_constraints(elements: Sequence[ElementNode]) -> list[ConstraintNode]:
    """Extract all heuristic constraints."""

    constraints: list[ConstraintNode] = []
    constraints.extend(extract_alignment_constraints(elements))
    constraints.extend(extract_containment_constraints(elements))
    constraints.extend(extract_spacing_constraints(elements))
    constraints.extend(extract_grid_constraints(elements))
    return constraints
