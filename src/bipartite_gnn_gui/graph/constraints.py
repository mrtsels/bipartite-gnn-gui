"""Heuristic constraint extraction for GUI elements.

All bboxes are assumed to be in normalized xyxy format [x1, y1, x2, y2]
with values in [0, 1].
"""

from __future__ import annotations

from typing import Sequence

from .schema import ConstraintNode, ConstraintType, ElementNode

_DEGENERATE_THRESHOLD = 0.001


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cx(elem: ElementNode) -> float:
    return (elem.bbox[0] + elem.bbox[2]) / 2.0


def _cy(elem: ElementNode) -> float:
    return (elem.bbox[1] + elem.bbox[3]) / 2.0


def _w(elem: ElementNode) -> float:
    return elem.bbox[2] - elem.bbox[0]


def _h(elem: ElementNode) -> float:
    return elem.bbox[3] - elem.bbox[1]


def _cluster_by_threshold(
    values: list[float], tolerance: float,
) -> list[list[int]]:
    """Group indices by approximate equality of their associated values.

    Sorts by value and groups consecutive indices whose values differ by
    less than *tolerance*.  Only groups with 2+ members are returned.

    Args:
        values: One scalar per element.
        tolerance: Maximum difference to be considered equal.

    Returns:
        List of groups, where each group is a list of element indices.
    """
    if len(values) < 2:
        return []
    indexed = list(enumerate(values))
    indexed.sort(key=lambda x: x[1])

    groups: list[list[int]] = []
    current = [indexed[0][0]]
    for idx in range(1, len(indexed)):
        gap = indexed[idx][1] - indexed[idx - 1][1]
        if gap < tolerance:
            current.append(indexed[idx][0])
        else:
            if len(current) >= 2:
                groups.append(current)
            current = [indexed[idx][0]]
    if len(current) >= 2:
        groups.append(current)
    return groups


def _make_constraint(
    ctype: ConstraintType,
    indices: list[int],
    **extra_params: float,
) -> ConstraintNode:
    """Build a single ConstraintNode with symmetric source/target indices."""
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=sorted(indices),
        target_indices=sorted(indices),
        params=extra_params,
    )


# ---------------------------------------------------------------------------
# Alignment  (ALIGN_LEFT / RIGHT / TOP / BOTTOM / CENTER_X / CENTER_Y)
# ---------------------------------------------------------------------------


def extract_alignment_constraints(
    elements: Sequence[ElementNode],
    tolerance: float = 0.02,
) -> list[ConstraintNode]:
    """Extract alignment constraints (ALIGN_LEFT/RIGHT/TOP/BOTTOM + CENTER_X/Y).

    Groups elements whose relevant coordinate differs by less than
    *tolerance* and emits one ConstraintNode per aligned group.

    Args:
        elements: Sequence of element nodes.
        tolerance: Maximum coordinate difference to consider aligned.

    Returns:
        List of alignment ConstraintNodes.
    """
    if len(elements) < 2:
        return []

    type_coord_pairs: list[tuple[ConstraintType, list[float]]] = [
        (ConstraintType.ALIGN_LEFT, [e.bbox[0] for e in elements]),
        (ConstraintType.ALIGN_RIGHT, [e.bbox[2] for e in elements]),
        (ConstraintType.ALIGN_TOP, [e.bbox[1] for e in elements]),
        (ConstraintType.ALIGN_BOTTOM, [e.bbox[3] for e in elements]),
        (ConstraintType.CENTER_X, [_cx(e) for e in elements]),
        (ConstraintType.CENTER_Y, [_cy(e) for e in elements]),
    ]

    results: list[ConstraintNode] = []
    for ctype, values in type_coord_pairs:
        for group in _cluster_by_threshold(values, tolerance):
            results.append(
                _make_constraint(ctype, group, tolerance=tolerance)
            )
    return results


# ---------------------------------------------------------------------------
# Same-size
# ---------------------------------------------------------------------------


def _extract_same_size_constraints(
    elements: Sequence[ElementNode],
    tolerance: float = 0.02,
) -> list[ConstraintNode]:
    """Extract SAME_SIZE constraints.

    Two elements have the same size when
    ``max(|w_i-w_j|/w_j, |h_i-h_j|/h_j) < tolerance``.  Elements with
    degenerate width or height (< 0.001) are skipped.

    Returns:
        List of SAME_SIZE ConstraintNodes (one per group of 2+).
    """
    n = len(elements)
    if n < 2:
        return []

    widths = [_w(e) for e in elements]
    heights = [_h(e) for e in elements]

    # Union-Find to build groups of same-size elements
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        if widths[i] < _DEGENERATE_THRESHOLD or heights[i] < _DEGENERATE_THRESHOLD:
            continue
        for j in range(i + 1, n):
            if widths[j] < _DEGENERATE_THRESHOLD or heights[j] < _DEGENERATE_THRESHOLD:
                continue
            rel_w = abs(widths[i] - widths[j]) / max(widths[j], _DEGENERATE_THRESHOLD)
            rel_h = abs(heights[i] - heights[j]) / max(heights[j], _DEGENERATE_THRESHOLD)
            if max(rel_w, rel_h) < tolerance:
                union(i, j)

    # Collect groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        if widths[i] < _DEGENERATE_THRESHOLD or heights[i] < _DEGENERATE_THRESHOLD:
            continue
        root = find(i)
        groups.setdefault(root, []).append(i)

    results: list[ConstraintNode] = []
    for group in groups.values():
        if len(group) >= 2:
            results.append(
                _make_constraint(ConstraintType.SAME_SIZE, group, tolerance=tolerance)
            )
    return results


# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------


def _find_equidistant_runs(
    elements: Sequence[ElementNode],
    axis: str,
    tolerance: float,
) -> list[list[int]]:
    """Find runs of 3+ elements with approximately equal gaps.

    Args:
        elements: Element nodes.
        axis: ``"horizontal"`` (sort by x1) or ``"vertical"`` (sort by y1).
        tolerance: Maximum gap difference for equality.

    Returns:
        List of index groups, each with 3+ elements.
    """
    n = len(elements)
    if n < 3:
        return []

    coord_fn = (lambda e: e.bbox[0]) if axis == "horizontal" else (lambda e: e.bbox[1])
    gap_fn = (lambda e: _cx(e)) if axis == "horizontal" else (lambda e: _cy(e))

    indexed = sorted(enumerate(elements), key=lambda t: coord_fn(t[1]))
    # Compute gaps between consecutive centers
    coords = [gap_fn(t[1]) for t in indexed]
    gaps = [coords[i + 1] - coords[i] for i in range(n - 1)]

    runs: list[list[int]] = []
    current_run = [indexed[0][0], indexed[1][0]]
    for i in range(2, n):
        # Check whether gap i-1 is approximately equal to gap i-2
        gap_prev = gaps[i - 2]
        gap_curr = gaps[i - 1]
        if abs(gap_curr - gap_prev) < tolerance:
            current_run.append(indexed[i][0])
        else:
            if len(current_run) >= 3:
                runs.append(current_run)
            current_run = [indexed[i - 1][0], indexed[i][0]]

    if len(current_run) >= 3:
        runs.append(current_run)

    return runs


def extract_spacing_constraints(
    elements: Sequence[ElementNode],
    tolerance: float = 0.02,
) -> list[ConstraintNode]:
    """Extract spacing constraints (equidistant elements).

    Elements sorted along the horizontal or vertical axis whose
    consecutive center-gaps are approximately equal form a spacing
    constraint.

    Args:
        elements: Sequence of element nodes.
        tolerance: Maximum gap difference for equality.

    Returns:
        List of SPACING ConstraintNodes (one per equidistant run per axis).
    """
    results: list[ConstraintNode] = []
    for axis in ("horizontal", "vertical"):
        for group in _find_equidistant_runs(elements, axis, tolerance):
            results.append(
                _make_constraint(
                    ConstraintType.SPACING,
                    group,
                    tolerance=tolerance,
                    axis=1.0 if axis == "horizontal" else 0.0,
                )
            )
    return results


# ---------------------------------------------------------------------------
# Containment
# ---------------------------------------------------------------------------


def extract_containment_constraints(
    elements: Sequence[ElementNode],
) -> list[ConstraintNode]:
    """Extract containment (parent-child) constraints.

    Element *i* contains element *j* when every coordinate of *i* is on
    the outer side of *j* and *i* is strictly larger than *j*.

    Args:
        elements: Sequence of element nodes.

    Returns:
        List of CONTAINMENT ConstraintNodes (one per parent-child pair).
    """
    n = len(elements)
    if n < 2:
        return []

    results: list[ConstraintNode] = []
    for i in range(n):
        x1_i, y1_i, x2_i, y2_i = elements[i].bbox
        w_i = x2_i - x1_i
        h_i = y2_i - y1_i
        if w_i < _DEGENERATE_THRESHOLD or h_i < _DEGENERATE_THRESHOLD:
            continue
        area_i = w_i * h_i
        for j in range(n):
            if i == j:
                continue
            x1_j, y1_j, x2_j, y2_j = elements[j].bbox
            w_j = x2_j - x1_j
            h_j = y2_j - y1_j
            if w_j < _DEGENERATE_THRESHOLD or h_j < _DEGENERATE_THRESHOLD:
                continue

            # i contains j ?
            if x1_i <= x1_j and y1_i <= y1_j and x2_i >= x2_j and y2_i >= y2_j:
                if w_i > w_j and h_i > h_j:
                    area_j = w_j * h_j
                    area_diff = area_i - area_j
                    margin = min(area_diff / area_i, 1.0)
                    results.append(
                        _make_constraint(
                            ConstraintType.CONTAINMENT, [i, j], margin=margin
                        )
                    )
    return results


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def _cluster_by_cy(
    elements: Sequence[ElementNode],
    tolerance: float,
) -> list[list[int]]:
    """Group element indices into rows by approximate center-y equality."""
    cy_vals = [_cy(e) for e in elements]
    return _cluster_by_threshold(cy_vals, tolerance)


def extract_grid_constraints(
    elements: Sequence[ElementNode],
    tolerance: float = 0.05,
) -> list[ConstraintNode]:
    """Extract grid layout constraints.

    Detects a 2D row + column arrangement: elements are first grouped
    into rows by approximate center-y equality, then column alignment is
    verified across rows.

    Args:
        elements: Sequence of element nodes.
        tolerance: Maximum coordinate difference for row/column clustering.

    Returns:
        List of GRID ConstraintNodes (at most one per detected grid).
    """
    n = len(elements)
    if n < 4:
        return []

    # 1. Cluster into rows by cy
    rows = _cluster_by_cy(elements, tolerance)

    # Need at least 2 rows, each with at least 2 elements
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) < 2:
        return []

    # 2. For each row, sort elements by cx to get column positions
    row_cx_sorted: list[list[tuple[int, float]]] = []
    for row in rows:
        sorted_row = sorted(
            [(idx, _cx(elements[idx])) for idx in row], key=lambda x: x[1]
        )
        row_cx_sorted.append(sorted_row)

    # 3. Check that each row has the same number of columns
    n_cols = len(row_cx_sorted[0])
    for row in row_cx_sorted[1:]:
        if len(row) != n_cols:
            return []

    # 4. Check that column centers align across rows
    for col in range(n_cols):
        cx_col = [row[col][1] for row in row_cx_sorted]
        for i in range(1, len(cx_col)):
            if abs(cx_col[i] - cx_col[0]) > tolerance:
                return []

    # Build flat list of all indices in the grid
    all_indices: list[int] = []
    for row in row_cx_sorted:
        all_indices.extend(idx for idx, _ in row)

    return [
        _make_constraint(
            ConstraintType.GRID,
            all_indices,
            rows=float(len(rows)),
            columns=float(n_cols),
            tolerance=tolerance,
        )
    ]


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def _build_covered_pairs(
    constraints: list[ConstraintNode],
) -> set[tuple[int, int]]:
    """Build a set of all unordered element pairs covered by constraints."""
    pairs: set[tuple[int, int]] = set()
    for c in constraints:
        indices = sorted(set(c.source_indices + c.target_indices))
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                pairs.add((indices[i], indices[j]))
    return pairs


def extract_all_constraints(
    elements: Sequence[ElementNode],
    tolerance: float = 0.02,
) -> list[ConstraintNode]:
    """Extract all heuristic constraints with deduplication.

    Aggregates constraints from all extractors.  Alignment constraints
    that duplicate pairs already covered by higher-priority types
    (containment) are dropped.

    Args:
        elements: Sequence of element nodes.
        tolerance: Default tolerance for all sub-extractors.

    Returns:
        Combined list of ConstraintNodes.
    """
    results: list[ConstraintNode] = []

    # 1. Containment (highest priority)
    containment = extract_containment_constraints(elements)
    results.extend(containment)
    covered = _build_covered_pairs(containment)

    # 2. Same-size
    results.extend(_extract_same_size_constraints(elements, tolerance))

    # 3. Alignment — skip groups whose pairs are already covered
    alignment = extract_alignment_constraints(elements, tolerance)
    for c in alignment:
        indices = sorted(c.source_indices)
        has_covered = False
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                if (indices[i], indices[j]) in covered:
                    has_covered = True
                    break
            if has_covered:
                break
        if not has_covered:
            results.append(c)

    # 4. Spacing
    results.extend(extract_spacing_constraints(elements, tolerance))

    # 5. Grid
    results.extend(extract_grid_constraints(elements, tolerance=tolerance))

    return results
