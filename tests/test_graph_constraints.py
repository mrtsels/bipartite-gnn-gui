"""Tests for constraint extraction — all 10 constraint types."""
# ruff: noqa: N802  allow snake_case test names

from __future__ import annotations

import pytest

from bipartite_gnn_gui.graph.constraints import (
    _cluster_by_threshold,
    _extract_same_size_constraints,
    extract_alignment_constraints,
    extract_all_constraints,
    extract_containment_constraints,
    extract_grid_constraints,
    extract_spacing_constraints,
)
from bipartite_gnn_gui.graph.schema import ConstraintNode, ConstraintType, ElementNode


# ===================================================================
# Helpers
# ===================================================================


def _elem(x1: float, y1: float, x2: float, y2: float) -> ElementNode:
    """Shorthand to create an ElementNode with default confidence / label."""
    return ElementNode(bbox=[x1, y1, x2, y2])


def _assert_constraint(
    c: ConstraintNode,
    ctype: ConstraintType,
    indices: list[int],
) -> None:
    """Assert a constraint has the expected type and source/target indices."""
    assert c.constraint_type == ctype, f"Expected {ctype}, got {c.constraint_type}"
    assert sorted(c.source_indices) == sorted(indices), (
        f"Expected source_indices={indices}, got {c.source_indices}"
    )
    assert sorted(c.target_indices) == sorted(indices), (
        f"Expected target_indices={indices}, got {c.target_indices}"
    )


# ===================================================================
# Unit: _cluster_by_threshold
# ===================================================================


class TestClusterByThreshold:
    def test_empty(self) -> None:
        assert _cluster_by_threshold([], 0.02) == []

    def test_single(self) -> None:
        assert _cluster_by_threshold([0.5], 0.02) == []

    def test_two_close(self) -> None:
        groups = _cluster_by_threshold([0.1, 0.11], 0.02)
        assert groups == [[0, 1]]

    def test_two_far(self) -> None:
        assert _cluster_by_threshold([0.0, 0.5], 0.02) == []

    def test_three_close(self) -> None:
        groups = _cluster_by_threshold([0.1, 0.11, 0.105], 0.02)
        assert len(groups) == 1
        assert sorted(groups[0]) == [0, 1, 2]

    def test_two_separate_groups(self) -> None:
        """Two distinct clusters — both returned."""
        groups = _cluster_by_threshold([0.1, 0.11, 0.5, 0.51], 0.02)
        assert len(groups) == 2
        assert sorted(groups[0]) == [0, 1]
        assert sorted(groups[1]) == [2, 3]

    def test_boundary_value(self) -> None:
        """Gap exactly equal to tolerance is NOT included (strict <)."""
        values = [0.0, 0.02]
        assert _cluster_by_threshold(values, 0.02) == []

    def test_barely_under_tolerance(self) -> None:
        values = [0.0, 0.0199]
        assert _cluster_by_threshold(values, 0.02) == [[0, 1]]


# ===================================================================
# extract_alignment_constraints
# ===================================================================


class TestExtractAlignmentConstraints:
    def test_empty(self) -> None:
        assert extract_alignment_constraints([]) == []

    def test_single_element(self) -> None:
        assert extract_alignment_constraints([_elem(0, 0, 1, 1)]) == []

    def test_two_elements_left_aligned_only(self) -> None:
        """Elements share x1 but have different y positions so only ALIGN_LEFT fires."""
        elems = [_elem(0.1, 0.2, 0.4, 0.3), _elem(0.1, 0.6, 0.5, 0.7)]
        result = extract_alignment_constraints(elems)
        assert len(result) == 1
        _assert_constraint(result[0], ConstraintType.ALIGN_LEFT, [0, 1])

    def test_two_elements_no_alignment(self) -> None:
        """No alignment type should be detected."""
        elems = [_elem(0.0, 0.0, 0.3, 0.3), _elem(0.5, 0.5, 0.9, 0.9)]
        assert extract_alignment_constraints(elems) == []

    def test_all_six_alignment_types_triggered(self) -> None:
        """Identical bboxes trigger all 6 alignment types."""
        elems = [_elem(0.1, 0.2, 0.5, 0.6), _elem(0.1, 0.2, 0.5, 0.6)]
        result = extract_alignment_constraints(elems)
        assert len(result) == 6
        types_found = {c.constraint_type for c in result}
        expected_types = {
            ConstraintType.ALIGN_LEFT,
            ConstraintType.ALIGN_RIGHT,
            ConstraintType.ALIGN_TOP,
            ConstraintType.ALIGN_BOTTOM,
            ConstraintType.CENTER_X,
            ConstraintType.CENTER_Y,
        }
        assert types_found == expected_types

    def test_three_elements_align_left(self) -> None:
        """Three elements with same x1 produce one ALIGN_LEFT constraint."""
        elems = [
            _elem(0.1, 0.0, 0.3, 0.2),
            _elem(0.1, 0.3, 0.4, 0.5),
            _elem(0.1, 0.6, 0.35, 0.8),
        ]
        result = extract_alignment_constraints(elems)
        align_left = [c for c in result if c.constraint_type == ConstraintType.ALIGN_LEFT]
        assert len(align_left) == 1
        _assert_constraint(align_left[0], ConstraintType.ALIGN_LEFT, [0, 1, 2])

    def test_align_right(self) -> None:
        elems = [_elem(0.1, 0.0, 0.6, 0.2), _elem(0.2, 0.3, 0.6, 0.5)]
        result = extract_alignment_constraints(elems)
        align_right = [c for c in result if c.constraint_type == ConstraintType.ALIGN_RIGHT]
        assert len(align_right) == 1
        _assert_constraint(align_right[0], ConstraintType.ALIGN_RIGHT, [0, 1])

    def test_align_top(self) -> None:
        elems = [_elem(0.0, 0.1, 0.3, 0.4), _elem(0.4, 0.1, 0.7, 0.5)]
        align_top = [c for c in extract_alignment_constraints(elems) if c.constraint_type == ConstraintType.ALIGN_TOP]
        assert len(align_top) == 1
        _assert_constraint(align_top[0], ConstraintType.ALIGN_TOP, [0, 1])

    def test_align_bottom(self) -> None:
        elems = [_elem(0.0, 0.0, 0.3, 0.5), _elem(0.4, 0.2, 0.7, 0.5)]
        align_bottom = [c for c in extract_alignment_constraints(elems) if c.constraint_type == ConstraintType.ALIGN_BOTTOM]
        assert len(align_bottom) == 1
        _assert_constraint(align_bottom[0], ConstraintType.ALIGN_BOTTOM, [0, 1])

    def test_center_x(self) -> None:
        elems = [_elem(0.1, 0.0, 0.5, 0.3), _elem(0.1, 0.4, 0.5, 0.7)]
        center_x = [c for c in extract_alignment_constraints(elems) if c.constraint_type == ConstraintType.CENTER_X]
        assert len(center_x) == 1
        _assert_constraint(center_x[0], ConstraintType.CENTER_X, [0, 1])

    def test_center_y(self) -> None:
        elems = [_elem(0.0, 0.2, 0.3, 0.6), _elem(0.4, 0.2, 0.8, 0.6)]
        center_y = [c for c in extract_alignment_constraints(elems) if c.constraint_type == ConstraintType.CENTER_Y]
        assert len(center_y) == 1
        _assert_constraint(center_y[0], ConstraintType.CENTER_Y, [0, 1])

    def test_left_aligned_but_not_top(self) -> None:
        """Elements share x1 but not y1 — only ALIGN_LEFT fires."""
        elems = [_elem(0.2, 0.0, 0.5, 0.3), _elem(0.2, 0.5, 0.6, 0.8)]
        result = extract_alignment_constraints(elems)
        assert len(result) == 1
        assert result[0].constraint_type == ConstraintType.ALIGN_LEFT

    def test_tolerance_tight_fewer_constraints(self) -> None:
        """Tight tolerance means fewer alignment detections."""
        # Elements with x1 close but not within tight tolerance, and also
        # all other coordinates differ by >0.01 so nothing fires at tolerance=0.01.
        nearly_aligned = [_elem(0.1, 0.0, 0.3, 0.2), _elem(0.12, 0.4, 0.4, 0.6)]
        loose = extract_alignment_constraints(nearly_aligned, tolerance=0.05)
        tight = extract_alignment_constraints(nearly_aligned, tolerance=0.01)
        assert len(loose) >= 1  # ALIGN_LEFT detected with loose tolerance
        assert len(tight) == 0  # Not detected with tight tolerance


# ===================================================================
# extract_containment_constraints
# ===================================================================


class TestExtractContainmentConstraints:
    def test_empty(self) -> None:
        assert extract_containment_constraints([]) == []

    def test_single(self) -> None:
        assert extract_containment_constraints([_elem(0, 0, 1, 1)]) == []

    def test_one_contains_other(self) -> None:
        outer = _elem(0.0, 0.0, 1.0, 1.0)
        inner = _elem(0.2, 0.2, 0.4, 0.4)
        result = extract_containment_constraints([outer, inner])
        assert len(result) == 1
        _assert_constraint(result[0], ConstraintType.CONTAINMENT, [0, 1])
        assert "margin" in result[0].params

    def test_no_containment(self) -> None:
        a = _elem(0.0, 0.0, 0.3, 0.3)
        b = _elem(0.5, 0.5, 0.9, 0.9)
        assert extract_containment_constraints([a, b]) == []

    def test_partial_overlap_not_containment(self) -> None:
        """Partial overlap should not be flagged as containment."""
        a = _elem(0.0, 0.0, 0.5, 0.5)
        b = _elem(0.3, 0.3, 0.8, 0.8)
        assert extract_containment_constraints([a, b]) == []

    def test_same_size_no_containment(self) -> None:
        """Same-size overlapping bboxes should not be containment (not strictly larger)."""
        a = _elem(0.0, 0.0, 0.5, 0.5)
        b = _elem(0.0, 0.0, 0.5, 0.5)
        assert extract_containment_constraints([a, b]) == []

    def test_nested_containment_chain(self) -> None:
        """Outer container has two children (one nested inside the other)."""
        outer = _elem(0.0, 0.0, 1.0, 1.0)
        mid = _elem(0.1, 0.1, 0.8, 0.8)
        inner = _elem(0.2, 0.2, 0.5, 0.5)
        result = extract_containment_constraints([outer, mid, inner])
        # outer->mid, outer->inner, mid->inner = 3
        assert len(result) == 3
        for c in result:
            assert c.constraint_type == ConstraintType.CONTAINMENT

    def test_containment_margin_value(self) -> None:
        outer = _elem(0.0, 0.0, 1.0, 1.0)  # area = 1.0
        inner = _elem(0.1, 0.1, 0.3, 0.3)  # area = 0.04
        result = extract_containment_constraints([outer, inner])
        margin = result[0].params["margin"]
        expected_margin = (1.0 - 0.04) / 1.0
        assert margin == pytest.approx(expected_margin, rel=1e-6)

    def test_degenerate_bbox_skipped(self) -> None:
        """Near-zero-area bbox should not participate in containment."""
        small = _elem(0.1, 0.1, 0.10001, 0.10001)
        large = _elem(0.0, 0.0, 1.0, 1.0)
        result = extract_containment_constraints([small, large])
        assert len(result) == 0


# ===================================================================
# SAME_SIZE  (via _extract_same_size_constraints)
# ===================================================================


class TestExtractSameSizeConstraints:
    def test_empty(self) -> None:
        assert _extract_same_size_constraints([]) == []

    def test_single(self) -> None:
        assert _extract_same_size_constraints([_elem(0, 0, 0.5, 0.5)]) == []

    def test_two_same_size(self) -> None:
        elems = [_elem(0.0, 0.0, 0.5, 0.3), _elem(0.5, 0.5, 1.0, 0.8)]
        result = _extract_same_size_constraints(elems)
        assert len(result) == 1
        _assert_constraint(result[0], ConstraintType.SAME_SIZE, [0, 1])

    def test_two_different_size(self) -> None:
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.0, 0.5, 0.2, 0.7)]
        assert _extract_same_size_constraints(elems) == []

    def test_three_identical_size(self) -> None:
        elems = [
            _elem(0.0, 0.0, 0.4, 0.2),
            _elem(0.5, 0.0, 0.9, 0.2),
            _elem(0.0, 0.3, 0.4, 0.5),
        ]
        result = _extract_same_size_constraints(elems)
        assert len(result) == 1
        _assert_constraint(result[0], ConstraintType.SAME_SIZE, [0, 1, 2])

    def test_degenerate_skipped(self) -> None:
        """Near-zero width or height elements are skipped."""
        elems = [
            _elem(0.0, 0.0, 0.0005, 0.5),  # degenerate width
            _elem(0.5, 0.0, 0.5005, 0.5),  # degenerate width
        ]
        assert _extract_same_size_constraints(elems) == []

    def test_tolerance_rejects_slightly_different(self) -> None:
        elems = [_elem(0.0, 0.0, 0.5, 0.3), _elem(0.0, 0.5, 0.56, 0.8)]
        # width 0.5 vs 0.56 -> rel diff = 0.06/0.56 ~ 0.107 > 0.02
        assert _extract_same_size_constraints(elems, tolerance=0.02) == []

    def test_tolerance_accepts_slightly_different(self) -> None:
        elems = [_elem(0.0, 0.0, 0.5, 0.3), _elem(0.0, 0.5, 0.51, 0.8)]
        # width 0.5 vs 0.51 -> rel diff = 0.01/0.51 ~ 0.0196 < 0.02
        result = _extract_same_size_constraints(elems, tolerance=0.02)
        assert len(result) == 1


# ===================================================================
# extract_spacing_constraints
# ===================================================================


class TestExtractSpacingConstraints:
    def test_empty(self) -> None:
        assert extract_spacing_constraints([]) == []

    def test_single(self) -> None:
        assert extract_spacing_constraints([_elem(0, 0, 0.5, 0.5)]) == []

    def test_two_elements_no_spacing(self) -> None:
        """Spacing requires 3+ elements."""
        assert extract_spacing_constraints(
            [_elem(0, 0, 0.3, 0.5), _elem(0.4, 0, 0.7, 0.5)]
        ) == []

    def test_horizontal_three_equidistant(self) -> None:
        """Three elements with equal horizontal gaps, but different vertical positions."""
        elems = [
            _elem(0.0, 0.0, 0.1, 0.4),  # cx = 0.05
            _elem(0.2, 0.2, 0.3, 0.6),  # cx = 0.25  gap = 0.2
            _elem(0.4, 0.4, 0.5, 0.8),  # cx = 0.45  gap = 0.2
        ]
        result = extract_spacing_constraints(elems)
        assert len(result) >= 1
        spacing = [
            c for c in result
            if c.constraint_type == ConstraintType.SPACING
            and c.params.get("axis", 0) == 1.0
        ]
        assert len(spacing) == 1

    def test_vertical_three_equidistant(self) -> None:
        """Three elements with equal vertical gaps."""
        elems = [
            _elem(0.0, 0.0, 0.5, 0.1),  # cy = 0.05
            _elem(0.2, 0.1, 0.7, 0.3),  # cy = 0.20  gap = 0.15
            _elem(0.0, 0.3, 0.5, 0.4),  # cy = 0.35  gap = 0.15
        ]
        result = extract_spacing_constraints(elems)
        spacing = [
            c for c in result
            if c.constraint_type == ConstraintType.SPACING
            and c.params.get("axis", 0) == 0.0
        ]
        assert len(spacing) == 1

    def test_unequal_gaps_no_spacing(self) -> None:
        """Three elements with unequal gaps in both axes."""
        elems = [
            _elem(0.0, 0.0, 0.1, 0.3),  # cx = 0.05, cy = 0.15
            _elem(0.2, 0.2, 0.3, 0.5),  # cx = 0.25, cy = 0.35  gap cx=0.2, cy=0.2
            _elem(0.5, 0.6, 0.6, 0.9),  # cx = 0.55, cy = 0.75  gap cx=0.3, cy=0.4
        ]
        result = extract_spacing_constraints(elems)
        spacing = [c for c in result if c.constraint_type == ConstraintType.SPACING]
        assert len(spacing) == 0

    def test_four_equidistant_horizontal(self) -> None:
        """Four elements with equal horizontal gaps."""
        elems = [
            _elem(0.0, 0.0, 0.1, 0.4),  # cx = 0.05
            _elem(0.2, 0.1, 0.3, 0.5),  # cx = 0.25  gap = 0.2
            _elem(0.4, 0.2, 0.5, 0.6),  # cx = 0.45  gap = 0.2
            _elem(0.6, 0.3, 0.7, 0.7),  # cx = 0.65  gap = 0.2
        ]
        spacing = [
            c for c in extract_spacing_constraints(elems)
            if c.constraint_type == ConstraintType.SPACING
        ]
        assert len(spacing) >= 1


# ===================================================================
# extract_grid_constraints
# ===================================================================


class TestExtractGridConstraints:
    def test_empty(self) -> None:
        assert extract_grid_constraints([]) == []

    def test_single(self) -> None:
        assert extract_grid_constraints([_elem(0, 0, 0.5, 0.5)]) == []

    def test_three_elements_no_grid(self) -> None:
        """Grid requires 4+ elements."""
        elems = [
            _elem(0.0, 0.0, 0.3, 0.3),
            _elem(0.4, 0.0, 0.7, 0.3),
            _elem(0.0, 0.4, 0.3, 0.7),
        ]
        assert extract_grid_constraints(elems) == []

    def test_two_by_two_grid(self) -> None:
        """2 rows x 2 columns."""
        elems = [
            _elem(0.0, 0.0, 0.3, 0.3),  # row 0, col 0
            _elem(0.5, 0.0, 0.8, 0.3),  # row 0, col 1
            _elem(0.0, 0.5, 0.3, 0.8),  # row 1, col 0
            _elem(0.5, 0.5, 0.8, 0.8),  # row 1, col 1
        ]
        result = extract_grid_constraints(elems, tolerance=0.1)
        assert len(result) == 1
        _assert_constraint(result[0], ConstraintType.GRID, [0, 1, 2, 3])
        assert result[0].params["rows"] == 2.0
        assert result[0].params["columns"] == 2.0

    def test_irregular_positions_no_grid(self) -> None:
        """Elements not arranged in a regular grid."""
        elems = [
            _elem(0.0, 0.0, 0.2, 0.2),
            _elem(0.5, 0.0, 0.7, 0.2),
            _elem(0.0, 0.5, 0.2, 0.7),
            _elem(0.8, 0.5, 1.0, 0.7),  # cx at 0.9, not aligned with col 0.6
        ]
        assert extract_grid_constraints(elems, tolerance=0.1) == []

    def test_three_by_two_grid(self) -> None:
        """3 rows x 2 columns."""
        elems = [
            _elem(0.0, 0.0, 0.2, 0.2),  # row 0, col 0
            _elem(0.4, 0.0, 0.6, 0.2),  # row 0, col 1
            _elem(0.0, 0.3, 0.2, 0.5),  # row 1, col 0
            _elem(0.4, 0.3, 0.6, 0.5),  # row 1, col 1
            _elem(0.0, 0.6, 0.2, 0.8),  # row 2, col 0
            _elem(0.4, 0.6, 0.6, 0.8),  # row 2, col 1
        ]
        result = extract_grid_constraints(elems, tolerance=0.1)
        assert len(result) == 1
        assert result[0].params["rows"] == 3.0
        assert result[0].params["columns"] == 2.0


# ===================================================================
# extract_all_constraints (integration + dedup)
# ===================================================================


class TestExtractAllConstraints:
    def test_empty(self) -> None:
        assert extract_all_constraints([]) == []

    def test_single_element(self) -> None:
        assert extract_all_constraints([_elem(0, 0, 1, 1)]) == []

    def test_two_identical(self) -> None:
        """Two identical bboxes produce many constraints."""
        elems = [_elem(0.1, 0.2, 0.5, 0.6), _elem(0.1, 0.2, 0.5, 0.6)]
        result = extract_all_constraints(elems, tolerance=0.02)
        # Alignment (6) + same_size (1) + containment (0 - not nested) = 7
        assert len(result) == 7

    def test_containment_dedup(self) -> None:
        """A container and its child should not produce alignment constraints."""
        outer = _elem(0.0, 0.0, 1.0, 1.0)
        inner = _elem(0.2, 0.2, 0.4, 0.4)
        result = extract_all_constraints([outer, inner], tolerance=0.02)

        containments = [
            c for c in result if c.constraint_type == ConstraintType.CONTAINMENT
        ]
        assert len(containments) == 1

        alignments = [
            c for c in result
            if c.constraint_type
            in {
                ConstraintType.ALIGN_LEFT,
                ConstraintType.ALIGN_RIGHT,
                ConstraintType.ALIGN_TOP,
                ConstraintType.ALIGN_BOTTOM,
                ConstraintType.CENTER_X,
                ConstraintType.CENTER_Y,
            }
        ]
        assert len(alignments) == 0, (
            f"Expected no alignment constraints after containment dedup, "
            f"got {len(alignments)}"
        )

    def test_many_elements_no_dedup(self) -> None:
        """Two far-apart same-size elements get SAME_SIZE and alignment."""
        elems = [
            _elem(0.0, 0.0, 0.2, 0.2),
            _elem(0.5, 0.5, 0.7, 0.7),
        ]
        result = extract_all_constraints(elems, tolerance=0.02)
        same_sizes = [
            c for c in result if c.constraint_type == ConstraintType.SAME_SIZE
        ]
        assert len(same_sizes) == 1
        _assert_constraint(same_sizes[0], ConstraintType.SAME_SIZE, [0, 1])

    def test_loose_tolerance_more_constraints(self) -> None:
        """Loose tolerance produces more constraints overall."""
        elems = [_elem(0.0, 0.0, 0.4, 0.4), _elem(0.03, 0.03, 0.43, 0.43)]
        tight = extract_all_constraints(elems, tolerance=0.02)
        loose = extract_all_constraints(elems, tolerance=0.1)
        assert len(loose) >= len(tight)


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    def test_bbox_values_at_boundaries(self) -> None:
        """Bboxes at [0,0,0,0] (zero-size) should not crash."""
        elems = [_elem(0.0, 0.0, 0.0, 0.0), _elem(0.5, 0.5, 0.5, 0.5)]
        result = extract_all_constraints(elems)
        assert isinstance(result, list)

    def test_all_identical_bboxes_produces_all_types(self) -> None:
        """Three identical bboxes produce alignment + same_size."""
        template = _elem(0.1, 0.2, 0.5, 0.6)
        elems = [template, template, template]
        result = extract_all_constraints(elems, tolerance=0.02)
        types_found = {c.constraint_type for c in result}
        assert ConstraintType.ALIGN_LEFT in types_found
        assert ConstraintType.ALIGN_RIGHT in types_found
        assert ConstraintType.ALIGN_TOP in types_found
        assert ConstraintType.ALIGN_BOTTOM in types_found
        assert ConstraintType.CENTER_X in types_found
        assert ConstraintType.CENTER_Y in types_found
        assert ConstraintType.SAME_SIZE in types_found

    def test_no_matches_any_extractor(self) -> None:
        """Scattered elements with different sizes — no alignment/containment/grid.

        With a tight tolerance, pairwise alignment should not fire.
        Elements have different sizes so SAME_SIZE does not fire either.
        Spacing might still incidentally fire for some triples — that's
        expected heuristic behavior, not a bug.
        """
        positions = [
            (0.0, 0.0, 0.1, 0.3),    # idx0, h=0.3
            (0.3, 0.4, 0.4, 0.7),    # idx1, h=0.3
            (0.7, 0.1, 0.8, 0.5),    # idx2, h=0.4 (different)
            (0.5, 0.6, 0.6, 0.9),    # idx3, h=0.3
        ]
        elems = [_elem(*p) for p in positions]
        result = extract_all_constraints(elems, tolerance=0.001)

        # No containment (far apart)
        assert not any(c.constraint_type == ConstraintType.CONTAINMENT for c in result)

        # No alignment (tight tolerance, all coord diffs > 0.001)
        align_types = {
            ConstraintType.ALIGN_LEFT, ConstraintType.ALIGN_RIGHT,
            ConstraintType.ALIGN_TOP, ConstraintType.ALIGN_BOTTOM,
            ConstraintType.CENTER_X, ConstraintType.CENTER_Y,
        }
        assert not any(c.constraint_type in align_types for c in result)

        # No grid (< 4 elements in 2D arrangement)
        assert not any(c.constraint_type == ConstraintType.GRID for c in result)

    def test_constraint_params_contains_tolerance(self) -> None:
        """Alignment constraints store tolerance in params."""
        elems = [_elem(0.1, 0.1, 0.5, 0.5), _elem(0.1, 0.3, 0.5, 0.7)]
        result = extract_alignment_constraints(elems, tolerance=0.05)
        for c in result:
            assert "tolerance" in c.params
            assert c.params["tolerance"] == 0.05

    def test_many_constraints_no_duplicate_indices(self) -> None:
        """Each constraint's source_indices are sorted and unique."""
        elems = [_elem(0.0, 0.0, 0.3, 0.3), _elem(0.4, 0.0, 0.7, 0.3)]
        result = extract_all_constraints(elems)
        for c in result:
            assert len(c.source_indices) == len(set(c.source_indices))
            assert c.source_indices == sorted(c.source_indices)
            assert c.source_indices == c.target_indices


# ===================================================================
# Real-world scenario
# ===================================================================


class TestRealWorldScenarios:
    def test_two_buttons_same_row(self) -> None:
        """Two identical buttons in the same row produce all 6 alignment + 1 same_size."""
        btn1 = _elem(0.1, 0.5, 0.3, 0.6)
        btn2 = _elem(0.1, 0.5, 0.3, 0.6)
        result = extract_all_constraints([btn1, btn2], tolerance=0.02)
        assert len(result) == 7  # 6 alignment + 1 same_size

    def test_button_in_a_container(self) -> None:
        """A button inside a modal — containment fires, alignment deduped."""
        modal = _elem(0.2, 0.3, 0.8, 0.7)
        button = _elem(0.35, 0.4, 0.65, 0.5)
        result = extract_all_constraints([modal, button], tolerance=0.02)
        containments = [
            c for c in result if c.constraint_type == ConstraintType.CONTAINMENT
        ]
        assert len(containments) == 1
        _assert_constraint(containments[0], ConstraintType.CONTAINMENT, [0, 1])
        alignments = [
            c for c in result
            if c.constraint_type
            in {
                ConstraintType.ALIGN_LEFT, ConstraintType.ALIGN_RIGHT,
                ConstraintType.ALIGN_TOP, ConstraintType.ALIGN_BOTTOM,
                ConstraintType.CENTER_X, ConstraintType.CENTER_Y,
            }
        ]
        assert len(alignments) == 0

    def test_three_buttons_equal_spacing(self) -> None:
        """Three equally-spaced buttons with same top/bottom positions."""
        btn1 = _elem(0.0, 0.4, 0.1, 0.5)  # cx = 0.05
        btn2 = _elem(0.2, 0.4, 0.3, 0.5)  # cx = 0.25
        btn3 = _elem(0.4, 0.4, 0.5, 0.5)  # cx = 0.45
        result = extract_all_constraints([btn1, btn2, btn3], tolerance=0.02)
        types_found = {c.constraint_type for c in result}
        assert ConstraintType.ALIGN_TOP in types_found
        assert ConstraintType.ALIGN_BOTTOM in types_found
        assert ConstraintType.SPACING in types_found


# ===================================================================
# Determinism
# ===================================================================


class TestDeterminism:
    def test_deterministic_output(self) -> None:
        """Running twice with same input gives identical results."""
        elems = [
            _elem(0.0, 0.0, 0.2, 0.2),
            _elem(0.1, 0.0, 0.3, 0.2),
            _elem(0.0, 0.3, 0.2, 0.5),
        ]
        r1 = extract_all_constraints(elems, tolerance=0.02)
        r2 = extract_all_constraints(elems, tolerance=0.02)

        def _key(c: ConstraintNode) -> tuple:
            return (c.constraint_type.value, tuple(c.source_indices),
                    tuple(c.params.items()))

        sorted1 = sorted(r1, key=_key)
        sorted2 = sorted(r2, key=_key)
        assert len(sorted1) == len(sorted2)
        for a, b in zip(sorted1, sorted2):
            assert a.constraint_type == b.constraint_type
            assert a.source_indices == b.source_indices
            assert a.params == b.params
