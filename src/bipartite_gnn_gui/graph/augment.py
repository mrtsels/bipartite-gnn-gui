"""Graph augmentation transforms for robustness training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .schema import ConstraintNode, ElementNode

_MIN_BBOX_SIZE = 0.005


@dataclass
class NodeDropout:
    """Randomly drop element nodes with given probability.

    After dropping, constraint indices are remapped to account for the
    removed elements.  Constraints referencing only dropped elements are
    removed entirely.

    Args:
        p: Probability of dropping each element (mutually independent).
    """

    p: float = 0.1

    def __call__(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
        seed: int | None = None,
    ) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Apply node dropout.

        Args:
            elements: Input element nodes.
            constraints: Input constraint nodes.
            seed: Optional seed for deterministic behaviour.

        Returns:
            Tuple of (new_elements, new_constraints) with dropped
            elements removed and indices remapped.
        """
        rng = random.Random(seed) if seed is not None else random

        # Build keep mask
        keep_mask = [rng.random() >= self.p for _ in elements]

        # Build old-to-new index mapping for kept elements
        new_elements: list[ElementNode] = []
        old_to_new: dict[int, int] = {}
        dropped_set: set[int] = set()
        for old_idx, keep in enumerate(keep_mask):
            if keep:
                old_to_new[old_idx] = len(new_elements)
                new_elements.append(elements[old_idx])
            else:
                dropped_set.add(old_idx)

        # Remap constraint indices; drop constraints with no surviving refs
        new_constraints: list[ConstraintNode] = []
        for c in constraints:
            new_source = [
                old_to_new[i] for i in c.source_indices if i not in dropped_set
            ]
            new_target = [
                old_to_new[i] for i in c.target_indices if i not in dropped_set
            ]
            if not new_source and not new_target:
                continue
            new_constraints.append(
                ConstraintNode(
                    constraint_type=c.constraint_type,
                    source_indices=new_source,
                    target_indices=new_target,
                    params=c.params.copy(),
                )
            )

        return new_elements, new_constraints


@dataclass
class CoordinateJitter:
    """Add Gaussian noise to bounding box coordinates.

    Each coordinate of each element's bbox receives independent
    zero-mean Gaussian noise with std *std*.  Results are clamped
    to [0, 1] and degenerate boxes (w <= 0 or h <= 0) are fixed to a
    minimum size of 0.005.

    Args:
        std: Standard deviation of Gaussian noise (normalised coords).
    """

    std: float = 0.01

    def __call__(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
        seed: int | None = None,
    ) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Apply coordinate jitter.

        Args:
            elements: Input element nodes.
            constraints: Input constraint nodes (unchanged).
            seed: Optional seed for deterministic behaviour.

        Returns:
            Tuple of (jittered_elements, constraints).
        """
        rng = random.Random(seed) if seed is not None else random

        new_elements: list[ElementNode] = []
        for elem in elements:
            x1, y1, x2, y2 = elem.bbox

            # Independent Gaussian noise per coordinate
            x1 += rng.gauss(0.0, self.std)
            y1 += rng.gauss(0.0, self.std)
            x2 += rng.gauss(0.0, self.std)
            y2 += rng.gauss(0.0, self.std)

            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))

            # Fix degenerate width
            if x2 - x1 < _MIN_BBOX_SIZE:
                x2 = min(x1 + _MIN_BBOX_SIZE, 1.0)
                if x2 - x1 < _MIN_BBOX_SIZE:
                    x1 = max(x2 - _MIN_BBOX_SIZE, 0.0)

            # Fix degenerate height
            if y2 - y1 < _MIN_BBOX_SIZE:
                y2 = min(y1 + _MIN_BBOX_SIZE, 1.0)
                if y2 - y1 < _MIN_BBOX_SIZE:
                    y1 = max(y2 - _MIN_BBOX_SIZE, 0.0)

            new_elements.append(
                ElementNode(
                    bbox=[x1, y1, x2, y2],
                    label=elem.label,
                    confidence=elem.confidence,
                    element_id=elem.element_id,
                    features=elem.features.copy(),
                )
            )

        return new_elements, list(constraints)


@dataclass
class ConstraintPerturbation:
    """Randomly remove constraints with given probability.

    Args:
        remove_p: Probability of removing each constraint.
    """

    remove_p: float = 0.05

    def __call__(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
        seed: int | None = None,
    ) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Apply constraint perturbation.

        Args:
            elements: Input element nodes (unchanged).
            constraints: Input constraint nodes.
            seed: Optional seed for deterministic behaviour.

        Returns:
            Tuple of (elements, remaining_constraints).
        """
        rng = random.Random(seed) if seed is not None else random

        new_constraints = [
            c for c in constraints if rng.random() >= self.remove_p
        ]

        return list(elements), new_constraints


@dataclass
class GraphAugmentationPipeline:
    """Compose multiple augmentation transforms sequentially.

    Args:
        transforms: List of augmentation transforms to apply in order.
    """

    transforms: list

    def __call__(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
        seed: int | None = None,
    ) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Apply all transforms in sequence.

        When *seed* is provided, each transform receives ``seed + i``
        where ``i`` is the transform index.

        Args:
            elements: Input element nodes.
            constraints: Input constraint nodes.
            seed: Optional base seed for deterministic behaviour.

        Returns:
            Tuple of (augmented_elements, augmented_constraints).
        """
        result: tuple[Sequence[ElementNode], Sequence[ConstraintNode]] = (
            elements,
            constraints,
        )
        for i, t in enumerate(self.transforms):
            s = seed + i if seed is not None else None
            result = t(*result, seed=s)
        return result
