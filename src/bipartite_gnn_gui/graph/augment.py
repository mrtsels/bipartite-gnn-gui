"""Graph augmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .schema import ConstraintNode, ElementNode


@dataclass
class GraphAugmenter:
    """Apply light-weight stochastic augmentations."""

    node_dropout_rate: float = 0.0
    jitter_std: float = 0.0

    def augment(self, elements: Sequence[ElementNode], constraints: Sequence[ConstraintNode]) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Return a copy of the input graph components."""

        return list(elements), list(constraints)
