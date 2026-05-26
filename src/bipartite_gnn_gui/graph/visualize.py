"""Graph visualization helpers.

Functions for plotting bipartite graphs overlaid on screenshots,
color-coding elements and constraints by type, and exporting the
graph structure as JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from .schema import ConstraintNode, ConstraintType, ElementNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

_ELEMENT_COLORS: dict[str, str] = {
    "button": "#e74c3c",
    "text": "#3498db",
    "image": "#2ecc71",
    "input": "#f39c12",
    "icon": "#9b59b6",
    "container": "#1abc9c",
}

_CONSTRAINT_COLORS: dict[str, str] = {
    "align_left": "#e74c3c",
    "align_right": "#e74c3c",
    "align_top": "#e74c3c",
    "align_bottom": "#e74c3c",
    "center_x": "#e74c3c",
    "center_y": "#e74c3c",
    "containment": "#3498db",
    "spacing": "#2ecc71",
    "grid": "#f39c12",
    "same_size": "#9b59b6",
}

_OTHER_COLOR = "#95a5a6"


def _hex_to_rgba(hex_color: str) -> tuple[float, float, float, float]:
    """Convert a hex color string ``#rrggbb`` to an RGBA tuple."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def _get_element_color(label: str) -> str:
    """Return the color for an element type label."""
    return _ELEMENT_COLORS.get(label.lower(), _OTHER_COLOR)


def _get_constraint_color(ctype: ConstraintType) -> str:
    """Return the color for a constraint type."""
    return _CONSTRAINT_COLORS.get(ctype.value, _OTHER_COLOR)


def _element_centroid(elem: ElementNode) -> tuple[float, float]:
    """Return ``(cx, cy)`` centroid of an element bbox."""
    x1, y1, x2, y2 = elem.bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _constraint_centroid(
    constraint: ConstraintNode, elements: list[ElementNode],
) -> tuple[float, float] | None:
    """Return centroid of all uniquely referenced elements for a constraint.

    Returns ``None`` when no valid elements are referenced.
    """
    indices = set(constraint.source_indices) | set(constraint.target_indices)
    valid: list[ElementNode] = []
    for i in indices:
        if 0 <= i < len(elements):
            valid.append(elements[i])
    if not valid:
        return None
    cx = sum(_element_centroid(e)[0] for e in valid) / len(valid)
    cy = sum(_element_centroid(e)[1] for e in valid) / len(valid)
    return (cx, cy)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_graph_on_screenshot(
    elements: Sequence[ElementNode],
    constraints: Sequence[ConstraintNode],
    image_path: str | None = None,
    ax: Any = None,
    show_bboxes: bool = True,
    show_edges: bool = True,
    color_by: str = "type",
) -> Any:
    """Plot GUI elements as bboxes overlaid on a screenshot.

    Args:
        elements: Sequence of element nodes to plot.
        constraints: Sequence of constraint nodes to plot.
        image_path: Optional path to a screenshot image for the background.
        ax: Optional matplotlib Axes to draw on.  Created if ``None``.
        show_bboxes: Whether to draw element bounding boxes.
        show_edges: Whether to draw constraint-to-element edges.
        color_by: ``"type"`` (color by element label) or ``"confidence"``
            (color by confidence score with a colormap).

    Returns:
        The ``Axes`` instance, or ``None`` if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch, FancyBboxPatch as Rectangle
        from matplotlib.patches import Patch
    except Exception:
        return None

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    # ---- background ----
    if image_path is not None:
        p = Path(image_path)
        if p.is_file():
            try:
                from PIL import Image as pil_image

                img = pil_image.open(p)
                ax.imshow(img)
            except Exception as exc:
                logger.warning("Failed to load image %s: %s", image_path, exc)
                ax.set_facecolor("white")
        else:
            logger.warning("Image path does not exist: %s", image_path)
            ax.set_facecolor("white")
    else:
        ax.set_facecolor("white")

    # ---- elements ----
    if not elements:
        ax.text(0.5, 0.5, "No elements", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
        ax.set_title(f"Elements: 0 | Constraints: {len(constraints)}")
        ax.axis("off")
        return ax

    elem_list = list(elements)

    # Determine bounding-box colors
    if color_by == "confidence":
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.cm.viridis
        bbox_colors = [cmap(norm(e.confidence)) for e in elem_list]
    else:
        # Default to type-based coloring
        bbox_colors = [_get_element_color(e.label) for e in elem_list]

    if show_bboxes:
        for elem, color in zip(elem_list, bbox_colors):
            x1, y1, x2, y2 = elem.bbox
            w, h = x2 - x1, y2 - y1
            rect = FancyBboxPatch(
                (x1, y1), w, h,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
                clip_on=True,
            )
            ax.add_patch(rect)

    # ---- constraints (compute positions) ----
    cons_list = list(constraints)
    constraint_positions: list[tuple[float, float] | None] = [
        _constraint_centroid(c, elem_list) for c in cons_list
    ]

    # ---- edges ----
    if show_edges and cons_list:
        for c_idx, c in enumerate(cons_list):
            pos = constraint_positions[c_idx]
            if pos is None:
                continue
            indices = set(c.source_indices) | set(c.target_indices)
            for e_idx in indices:
                if 0 <= e_idx < len(elem_list):
                    cx, cy = _element_centroid(elem_list[e_idx])
                    ax.plot(
                        [pos[0], cx], [pos[1], cy],
                        color="gray", linewidth=0.5, alpha=0.3,
                    )

    # ---- constraint markers ----
    for c_idx, c in enumerate(cons_list):
        pos = constraint_positions[c_idx]
        if pos is None:
            continue
        ccolor = _get_constraint_color(c.constraint_type)
        ax.plot(pos[0], pos[1], marker="s", markersize=6,
                color=ccolor, zorder=5)

    # ---- legend / colorbar ----
    if color_by == "type":
        seen: dict[str, str] = {}
        for e in elem_list:
            label = e.label.lower() if e.label else "unknown"
            if label not in seen:
                seen[label] = _get_element_color(e.label)
        if seen:
            from matplotlib.patches import Patch

            patches = [
                Patch(color=color, label=label)
                for label, color in sorted(seen.items())
            ]
            ax.legend(handles=patches, loc="upper right", fontsize=8)
    elif color_by == "confidence":
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        norm = Normalize(vmin=0.0, vmax=1.0)
        sm = ScalarMappable(norm=norm, cmap=plt.cm.viridis)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Confidence")

    ax.set_title(f"Elements: {len(elem_list)} | Constraints: {len(cons_list)}")
    ax.axis("off")
    return ax


def color_by_element_type(
    ax: Any,
    elements: Sequence[ElementNode],
) -> dict[str, tuple[float, float, float, float]]:
    """Annotate axes with colored markers at element centroids by type.

    Args:
        ax: Matplotlib Axes to annotate.
        elements: Sequence of element nodes.

    Returns:
        Mapping ``{type_name: rgba_tuple}`` of the colors used.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {}

    color_map: dict[str, tuple[float, float, float, float]] = {}
    elem_list = list(elements)

    for elem in elem_list:
        label = elem.label.lower() if elem.label else "unknown"
        if label not in color_map:
            color_map[label] = _hex_to_rgba(_get_element_color(elem.label))
        cx, cy = _element_centroid(elem)
        ax.plot(cx, cy, marker="o", markersize=4,
                color=_get_element_color(elem.label), zorder=3)

    return color_map


def color_by_constraint_type(
    ax: Any,
    constraints: Sequence[ConstraintNode],
) -> dict[str, tuple[float, float, float, float]]:
    """Annotate axes with colored markers per constraint type.

    .. note::
        Constraints do not have a natural spatial position.  This function
        returns the color mapping dict but does **not** place markers on
        the axes.

    Args:
        ax: Matplotlib Axes (unused for markers — kept for API consistency).
        constraints: Sequence of constraint nodes.

    Returns:
        Mapping ``{constraint_type.value: rgba_tuple}`` of the colors used.
    """
    color_map: dict[str, tuple[float, float, float, float]] = {}
    for c in constraints:
        ctype = c.constraint_type.value
        if ctype not in color_map:
            color_map[ctype] = _hex_to_rgba(_get_constraint_color(c.constraint_type))
    return color_map


def export_graph(
    elements: Sequence[ElementNode],
    constraints: Sequence[ConstraintNode],
    output_path: str | None = None,
) -> dict[str, Any]:
    """Export the bipartite graph structure as a JSON-serializable dict.

    Args:
        elements: Sequence of element nodes.
        constraints: Sequence of constraint nodes.
        output_path: Optional file path to write the JSON output.

    Returns:
        Dictionary with keys ``num_elements``, ``num_constraints``,
        ``num_edges``, ``elements``, ``constraints``, ``edges``.
    """
    elem_list = list(elements)
    cons_list = list(constraints)

    elem_data: list[dict[str, Any]] = []
    for i, e in enumerate(elem_list):
        elem_data.append({
            "id": i,
            "bbox": list(e.bbox),
            "type": e.label,
            "confidence": e.confidence,
        })

    cons_data: list[dict[str, Any]] = []
    for c in cons_list:
        cons_data.append({
            "type": c.constraint_type.value,
            "params": dict(c.params),
            "source_indices": list(c.source_indices),
            "target_indices": list(c.target_indices),
        })

    edges: list[dict[str, int]] = []
    for c_idx, c in enumerate(cons_list):
        seen_pairs: set[tuple[int, int]] = set()
        for e_idx in set(c.source_indices) | set(c.target_indices):
            if 0 <= e_idx < len(elem_list):
                pair = (e_idx, c_idx)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append({"element_idx": e_idx, "constraint_idx": c_idx})

    result: dict[str, Any] = {
        "num_elements": len(elem_list),
        "num_constraints": len(cons_list),
        "num_edges": len(edges),
        "elements": elem_data,
        "constraints": cons_data,
        "edges": edges,
    }

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(result, f, indent=2)

    return result
