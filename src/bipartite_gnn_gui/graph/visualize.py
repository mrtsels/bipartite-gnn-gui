"""Graph visualization helpers."""

from __future__ import annotations

from typing import Any, Sequence

from .schema import ConstraintNode, ElementNode


def plot_bipartite_graph(elements: Sequence[ElementNode], constraints: Sequence[ConstraintNode], ax: Any | None = None) -> Any:
    """Plot a simple placeholder visualization."""

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(f"Elements: {len(elements)} | Constraints: {len(constraints)}")
    ax.axis("off")
    return ax
