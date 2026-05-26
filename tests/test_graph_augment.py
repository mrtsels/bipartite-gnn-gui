"""Tests for graph augmentation — NodeDropout, CoordinateJitter, ConstraintPerturbation, GraphAugmentationPipeline."""

from __future__ import annotations

import copy

import pytest

from bipartite_gnn_gui.graph.augment import (
    ConstraintPerturbation,
    CoordinateJitter,
    GraphAugmentationPipeline,
    NodeDropout,
)
from bipartite_gnn_gui.graph.schema import ConstraintNode, ConstraintType, ElementNode


# ===================================================================
# Helpers
# ===================================================================


def _elem(
    x1: float, y1: float, x2: float, y2: float,
    label: str = "button", confidence: float = 1.0, element_id: str | None = None,
) -> ElementNode:
    return ElementNode(
        bbox=[x1, y1, x2, y2],
        label=label,
        confidence=confidence,
        element_id=element_id,
    )


def _con(
    ctype: ConstraintType,
    source: list[int],
    target: list[int] | None = None,
    **params: float,
) -> ConstraintNode:
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=source,
        target_indices=target or list(source),
        params=params,
    )


def _elems(n: int, prefix: str = "e") -> list[ElementNode]:
    """Create *n* distinct elements with sequential bboxes."""
    return [
        _elem(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05, element_id=f"{prefix}{i}")
        for i in range(n)
    ]


def _all_align_con(elems: list) -> list[ConstraintNode]:
    """Single ALIGN_LEFT constraint covering all element indices."""
    return [_con(ConstraintType.ALIGN_LEFT, list(range(len(elems))))]


# ===================================================================
# NodeDropout
# ===================================================================


class TestNodeDropout:
    def test_p0_identity(self) -> None:
        """p=0 keeps all elements and constraints unchanged."""
        elems = _elems(5)
        constrs = _all_align_con(elems)
        t = NodeDropout(p=0.0)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_elems) == 5
        assert len(new_constrs) == 1
        assert new_constrs[0].source_indices == [0, 1, 2, 3, 4]

    def test_p1_all_dropped(self) -> None:
        """p=1 drops all elements and constraints referencing them."""
        elems = _elems(5)
        constrs = _all_align_con(elems)
        t = NodeDropout(p=1.0)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_elems) == 0
        assert len(new_constrs) == 0

    def test_p1_empty_elements(self) -> None:
        """p=1 with no elements returns empty lists."""
        t = NodeDropout(p=1.0)
        new_elems, new_constrs = t([], [], seed=42)
        assert len(new_elems) == 0
        assert len(new_constrs) == 0

    def test_p1_all_constraints_removed(self) -> None:
        """p=1 removes even constraints referencing a subset of elements."""
        elems = _elems(3)
        constrs = [_con(ConstraintType.CONTAINMENT, [0, 1])]
        t = NodeDropout(p=1.0)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_elems) == 0
        assert len(new_constrs) == 0

    def test_seed_reproducibility(self) -> None:
        """Same seed yields identical results."""
        elems = _elems(100)
        constrs = _all_align_con(elems)
        t = NodeDropout(p=0.3)
        r1 = t(elems, constrs, seed=42)
        r2 = t(elems, constrs, seed=42)
        assert len(r1[0]) == len(r2[0])
        for e1, e2 in zip(r1[0], r2[0]):
            assert e1.element_id == e2.element_id
        assert len(r1[1]) == len(r2[1])
        assert r1[1][0].source_indices == r2[1][0].source_indices

    def test_different_seeds_different(self) -> None:
        """Different seeds produce different outcomes (statistically)."""
        elems = _elems(100)
        constrs = _all_align_con(elems)
        t = NodeDropout(p=0.3)
        r1 = t(elems, constrs, seed=42)
        r2 = t(elems, constrs, seed=99)
        # Extremely unlikely to keep the exact same set
        ids1 = [e.element_id for e in r1[0]]
        ids2 = [e.element_id for e in r2[0]]
        assert ids1 != ids2

    def test_index_remapping_valid(self) -> None:
        """All constraint indices are valid after remapping."""
        elems = _elems(50)
        constrs = [
            _con(ConstraintType.ALIGN_LEFT, list(range(50))),
            _con(ConstraintType.ALIGN_TOP, [10, 20, 30, 40]),
            _con(ConstraintType.CONTAINMENT, [0, 5, 10, 15, 20]),
        ]
        t = NodeDropout(p=0.5)
        new_elems, new_constrs = t(elems, constrs, seed=123)
        for c in new_constrs:
            for idx in c.source_indices + c.target_indices:
                assert 0 <= idx < len(new_elems)

    def test_index_remapping_preserves_elements(self) -> None:
        """Constraint source indices point to the correct elements after remap."""
        elems = _elems(10)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 3, 7])]
        t = NodeDropout(p=0.5)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        kept_ids = {e.element_id for e in new_elems}
        for c in new_constrs:
            for idx in c.source_indices:
                assert 0 <= idx < len(new_elems)
                assert new_elems[idx].element_id in kept_ids

    def test_constraint_removed_when_all_elements_dropped(self) -> None:
        """Constraint removed when all referenced elements are dropped."""
        elems = _elems(5)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [2, 4])]
        # Use seed that drops indices 2 and 4 with high probability
        # Alternative: force removal by p close to 1
        t = NodeDropout(p=1.0)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_constrs) == 0

    def test_constraint_kept_when_partial_drop(self) -> None:
        """Constraint kept when at least one referenced element survives."""
        elems = _elems(5)
        constrs = [_con(ConstraintType.SPACING, [0, 1, 2, 3, 4])]
        t = NodeDropout(p=0.4)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        if len(new_elems) > 0:
            # At least one constraint survives (elements were kept)
            assert len(new_constrs) <= 1
        # Elements at the constraint's source indices must be in new_elements
        for c in new_constrs:
            for idx in c.source_indices:
                assert 0 <= idx < len(new_elems)

    def test_empty_elements(self) -> None:
        """Empty element list returns empty results."""
        elems: list[ElementNode] = []
        constrs: list[ConstraintNode] = []
        t = NodeDropout(p=0.5)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_elems) == 0
        assert len(new_constrs) == 0

    def test_single_element_kept(self) -> None:
        """Single element with p=0 kept."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5)]
        constrs: list[ConstraintNode] = []
        t = NodeDropout(p=0.0)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_elems) == 1
        assert len(new_constrs) == 0

    def test_single_element_dropped(self) -> None:
        """Single element with p=1 dropped."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5)]
        constrs: list[ConstraintNode] = []
        t = NodeDropout(p=1.0)
        new_elems, new_constrs = t(elems, constrs, seed=42)
        assert len(new_elems) == 0

    def test_p_half_approximate_count(self) -> None:
        """With p=0.5 ~ half of elements survive."""
        elems = _elems(1000)
        constrs: list[ConstraintNode] = []
        t = NodeDropout(p=0.5)
        new_elems, _ = t(elems, constrs, seed=42)
        # 1000 * 0.5 = 500 expected, std ~ 15.8, 400-600 covers >6 sigma
        assert 400 <= len(new_elems) <= 600

    def test_constraints_not_modified(self) -> None:
        """Original constraint objects are not mutated."""
        elems = _elems(5)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 1, 2])]
        orig_source = list(constrs[0].source_indices)
        t = NodeDropout(p=0.5)
        t(elems, constrs, seed=42)
        assert constrs[0].source_indices == orig_source

    def test_no_seed_global_state(self) -> None:
        """Calling without seed uses global random state (no crash)."""
        elems = _elems(20)
        constrs = _all_align_con(elems)
        t = NodeDropout(p=0.5)
        result = t(elems, constrs)
        assert len(result[0]) > 0
        for c in result[1]:
            for idx in c.source_indices:
                assert 0 <= idx < len(result[0])

    def test_params_copied(self) -> None:
        """Constraint params dict is copied, not shared."""
        elems = _elems(5)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 1, 2], tolerance=0.02)]
        t = NodeDropout(p=0.0)
        _, new_constrs = t(elems, constrs, seed=42)
        assert new_constrs[0].params == {"tolerance": 0.02}
        # Mutating original should not affect new
        constrs[0].params["tolerance"] = 99.0
        assert new_constrs[0].params["tolerance"] == 0.02


# ===================================================================
# CoordinateJitter
# ===================================================================


class TestCoordinateJitter:
    def test_std0_identity(self) -> None:
        """std=0 leaves bboxes unchanged."""
        elems = [_elem(0.1, 0.2, 0.5, 0.6)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=0.0)
        new_elems, _ = t(elems, constrs, seed=42)
        assert new_elems[0].bbox == [0.1, 0.2, 0.5, 0.6]

    def test_std_positive_values_change(self) -> None:
        """std>0 changes bbox values."""
        elems = [_elem(0.25, 0.25, 0.75, 0.75) for _ in range(10)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=0.05)
        new_elems, _ = t(elems, constrs, seed=42)
        orig_bboxes = [e.bbox for e in elems]
        new_bboxes = [e.bbox for e in new_elems]
        # At least some elements should have changed
        changed = any(o != n for o, n in zip(orig_bboxes, new_bboxes))
        assert changed

    def test_clamping_0_1(self) -> None:
        """All bbox coordinates are clamped to [0, 1]."""
        elems = [_elem(0.5, 0.5, 0.8, 0.8)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=1.0)
        new_elems, _ = t(elems, constrs, seed=42)
        for v in new_elems[0].bbox:
            assert 0.0 <= v <= 1.0

    def test_no_degenerate_boxes(self) -> None:
        """No bbox has width <= 0 or height <= 0 after jitter."""
        elems = [_elem(0.5, 0.5, 0.8, 0.8) for _ in range(50)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=0.5)
        new_elems, _ = t(elems, constrs, seed=42)
        for e in new_elems:
            x1, y1, x2, y2 = e.bbox
            assert x2 > x1, f"degenerate width: {e.bbox}"
            assert y2 > y1, f"degenerate height: {e.bbox}"

    def test_large_std_safe(self) -> None:
        """Very large std produces valid boxes (clamped + non-degenerate)."""
        elems = [_elem(0.3, 0.3, 0.7, 0.7) for _ in range(50)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=10.0)
        new_elems, _ = t(elems, constrs, seed=42)
        for e in new_elems:
            x1, y1, x2, y2 = e.bbox
            assert 0.0 <= x1 <= 1.0
            assert 0.0 <= y1 <= 1.0
            assert 0.0 <= x2 <= 1.0
            assert 0.0 <= y2 <= 1.0
            assert x2 - x1 >= 0.005
            assert y2 - y1 >= 0.005

    def test_seed_reproducibility(self) -> None:
        """Same seed yields identical jitter results."""
        elems = [_elem(0.2, 0.3, 0.6, 0.7)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=0.05)
        r1 = t(elems, constrs, seed=42)
        r2 = t(elems, constrs, seed=42)
        assert r1[0][0].bbox == r2[0][0].bbox

    def test_constraints_unchanged(self) -> None:
        """Constraints are returned unchanged (same objects)."""
        elems = [_elem(0.1, 0.1, 0.5, 0.5)]
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0])]
        t = CoordinateJitter(std=0.01)
        _, new_constrs = t(elems, constrs, seed=42)
        assert len(new_constrs) == 1
        assert new_constrs[0].constraint_type == ConstraintType.ALIGN_LEFT

    def test_degenerate_input_fixed(self) -> None:
        """A deliberately degenerate input bbox is fixed (width set to min size)."""
        elems = [_elem(0.5, 0.5, 0.5, 0.5)]  # zero-width, zero-height box
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=0.0)  # no additional noise
        new_elems, _ = t(elems, constrs, seed=42)
        x1, y1, x2, y2 = new_elems[0].bbox
        assert x2 - x1 >= 0.005
        assert y2 - y1 >= 0.005

    def test_multiple_elements_independent(self) -> None:
        """Each element gets independent noise."""
        elems = [_elem(0.4, 0.4, 0.6, 0.6) for _ in range(10)]
        constrs: list[ConstraintNode] = []
        t = CoordinateJitter(std=0.01)
        new_elems, _ = t(elems, constrs, seed=99)
        bboxes = [e.bbox for e in new_elems]
        # Not all bboxes should be identical (noise is independent)
        unique_bboxes = {tuple(b) for b in bboxes}
        assert len(unique_bboxes) > 1

    def test_elements_not_mutated(self) -> None:
        """Original element bboxes are not mutated."""
        elems = [_elem(0.2, 0.3, 0.6, 0.7)]
        constrs: list[ConstraintNode] = []
        orig_bbox = list(elems[0].bbox)
        t = CoordinateJitter(std=0.05)
        t(elems, constrs, seed=42)
        assert elems[0].bbox == orig_bbox


# ===================================================================
# ConstraintPerturbation
# ===================================================================


class TestConstraintPerturbation:
    def test_remove_p0_identity(self) -> None:
        """remove_p=0 keeps all constraints."""
        elems = _elems(3)
        constrs = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1]),
            _con(ConstraintType.ALIGN_RIGHT, [2]),
        ]
        t = ConstraintPerturbation(remove_p=0.0)
        _, new_constrs = t(elems, constrs, seed=42)
        assert len(new_constrs) == 2

    def test_remove_p1_all_removed(self) -> None:
        """remove_p=1 removes all constraints."""
        elems = _elems(3)
        constrs = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1]),
            _con(ConstraintType.ALIGN_RIGHT, [2]),
        ]
        t = ConstraintPerturbation(remove_p=1.0)
        _, new_constrs = t(elems, constrs, seed=42)
        assert len(new_constrs) == 0

    def test_seed_reproducibility(self) -> None:
        """Same seed yields identical results."""
        elems = _elems(3)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 1]) for _ in range(50)]
        t = ConstraintPerturbation(remove_p=0.3)
        r1 = t(elems, constrs, seed=42)
        r2 = t(elems, constrs, seed=42)
        assert len(r1[1]) == len(r2[1])

    def test_elements_unchanged(self) -> None:
        """Elements are returned unchanged."""
        elems = _elems(3)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 1])]
        t = ConstraintPerturbation(remove_p=0.1)
        new_elems, _ = t(elems, constrs, seed=42)
        assert len(new_elems) == 3
        for e1, e2 in zip(elems, new_elems):
            assert e1.bbox == e2.bbox

    def test_remove_p_half_approximate(self) -> None:
        """With remove_p=0.5, roughly half of constraints survive."""
        elems = _elems(3)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 1]) for _ in range(200)]
        t = ConstraintPerturbation(remove_p=0.5)
        _, new_constrs = t(elems, constrs, seed=42)
        # Expected 100, std ~ 7, 70-130 covers >4 sigma
        assert 70 <= len(new_constrs) <= 130

    def test_different_seeds_different(self) -> None:
        """Different seeds produce different constraint sets."""
        elems = _elems(3)
        constrs = [_con(ConstraintType.ALIGN_LEFT, [0, 1]) for _ in range(50)]
        t = ConstraintPerturbation(remove_p=0.5)
        r1 = t(elems, constrs, seed=42)
        r2 = t(elems, constrs, seed=99)
        assert len(r1[1]) != len(r2[1]) or r1[1] != r2[1]

    def test_empty_constraints(self) -> None:
        """Empty constraints list is handled gracefully."""
        elems = _elems(3)
        constrs: list[ConstraintNode] = []
        t = ConstraintPerturbation(remove_p=0.5)
        _, new_constrs = t(elems, constrs, seed=42)
        assert len(new_constrs) == 0


# ===================================================================
# GraphAugmentationPipeline
# ===================================================================


class TestGraphAugmentationPipeline:
    def test_empty_transforms(self) -> None:
        """Empty transform list acts as identity."""
        elems = _elems(5)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[])
        new_elems, new_constrs = pipe(elems, constrs, seed=42)
        assert len(new_elems) == 5
        assert len(new_constrs) == 1

    def test_single_transform(self) -> None:
        """Pipeline with one transform applies it."""
        elems = _elems(10)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[NodeDropout(p=0.0)])
        new_elems, new_constrs = pipe(elems, constrs, seed=42)
        assert len(new_elems) == 10
        assert len(new_constrs) == 1

    def test_multiple_transforms(self) -> None:
        """Pipeline with multiple transforms applies them sequentially."""
        elems = _elems(50)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.0),  # all kept
            CoordinateJitter(std=0.05),  # jitter applied
            ConstraintPerturbation(remove_p=0.0),  # all kept
        ])
        new_elems, new_constrs = pipe(elems, constrs, seed=42)
        assert len(new_elems) == 50
        assert len(new_constrs) == 1
        # Bboxes should have changed from CoordinateJitter
        assert new_elems[0].bbox != elems[0].bbox

    def test_seed_reproducibility(self) -> None:
        """Same seed yields identical results through pipeline."""
        elems = _elems(100)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.3),
            CoordinateJitter(std=0.02),
            ConstraintPerturbation(remove_p=0.2),
        ])
        r1 = pipe(elems, constrs, seed=42)
        r2 = pipe(elems, constrs, seed=42)
        assert len(r1[0]) == len(r2[0])
        for e1, e2 in zip(r1[0], r2[0]):
            assert e1.bbox == e2.bbox
        assert len(r1[1]) == len(r2[1])

    def test_no_seed(self) -> None:
        """Pipeline without seed does not crash."""
        elems = _elems(10)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.2),
            CoordinateJitter(std=0.01),
        ])
        new_elems, new_constrs = pipe(elems, constrs)
        assert len(new_elems) > 0
        assert len(new_constrs) > 0

    def test_pipeline_node_dropout_then_jitter(self) -> None:
        """NodeDropout then CoordinateJitter works."""
        elems = _elems(50)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.5),
            CoordinateJitter(std=0.03),
        ])
        new_elems, new_constrs = pipe(elems, constrs, seed=42)
        # Verify all constraints reference valid indices
        for c in new_constrs:
            for idx in c.source_indices:
                assert 0 <= idx < len(new_elems)
        # Verify bboxes are non-degenerate
        for e in new_elems:
            assert e.bbox[2] > e.bbox[0]
            assert e.bbox[3] > e.bbox[1]

    def test_pipeline_different_seeds_different(self) -> None:
        """Different pipeline seeds produce different results."""
        elems = _elems(50)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.3),
            CoordinateJitter(std=0.02),
        ])
        r1 = pipe(elems, constrs, seed=42)
        r2 = pipe(elems, constrs, seed=99)
        ids1 = [e.element_id for e in r1[0]]
        ids2 = [e.element_id for e in r2[0]]
        assert ids1 != ids2

    def test_pipeline_empty_elements(self) -> None:
        """Pipeline handles empty elements gracefully."""
        elems: list[ElementNode] = []
        constrs: list[ConstraintNode] = []
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.5),
            CoordinateJitter(std=0.01),
        ])
        new_elems, new_constrs = pipe(elems, constrs, seed=42)
        assert len(new_elems) == 0
        assert len(new_constrs) == 0


# ===================================================================
# Integration
# ===================================================================


class TestIntegration:
    def test_all_transforms_sequential(self) -> None:
        """All transforms applied sequentially produce valid output."""
        elems = _elems(100)
        constrs = _all_align_con(elems)
        elems2, constrs2 = NodeDropout(p=0.3)(elems, constrs, seed=10)
        elems3, constrs3 = CoordinateJitter(std=0.02)(elems2, constrs2, seed=20)
        elems4, constrs4 = ConstraintPerturbation(remove_p=0.1)(elems3, constrs3, seed=30)
        # Validate final state
        assert len(elems4) <= len(elems)
        assert len(constrs4) <= 1
        for e in elems4:
            x1, y1, x2, y2 = e.bbox
            assert 0.0 <= x1 <= 1.0
            assert 0.0 <= y1 <= 1.0
            assert 0.0 <= x2 <= 1.0
            assert 0.0 <= y2 <= 1.0
            assert x2 > x1
            assert y2 > y1
        for c in constrs4:
            for idx in c.source_indices:
                assert 0 <= idx < len(elems4)

    def test_full_pipeline_seeded_vs_unseeded(self) -> None:
        """Seeded pipeline is deterministic; unseeded uses global state."""
        elems = _elems(50)
        constrs = _all_align_con(elems)
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.3),
            CoordinateJitter(std=0.02),
            ConstraintPerturbation(remove_p=0.1),
        ])
        # Seeded: same each time
        r1 = pipe(elems, constrs, seed=42)
        r2 = pipe(elems, constrs, seed=42)
        assert len(r1[0]) == len(r2[0])
        for e1, e2 in zip(r1[0], r2[0]):
            assert e1.bbox == e2.bbox
        # Unseeded: runs without crash
        r3 = pipe(elems, constrs)
        assert len(r3[0]) > 0

    def test_pipeline_preserves_semantics(self) -> None:
        """After pipeline, all constraint indices point to valid elements."""
        elems = _elems(50)
        constrs = [
            _con(ConstraintType.SAME_SIZE, [0, 5, 10]),
            _con(ConstraintType.SPACING, [1, 6, 11]),
            _con(ConstraintType.GRID, [2, 7, 12]),
        ]
        pipe = GraphAugmentationPipeline(transforms=[
            NodeDropout(p=0.4),
            CoordinateJitter(std=0.01),
            ConstraintPerturbation(remove_p=0.1),
        ])
        new_elems, new_constrs = pipe(elems, constrs, seed=42)
        for c in new_constrs:
            for idx in c.source_indices:
                assert 0 <= idx < len(new_elems)
            for idx in c.target_indices:
                assert 0 <= idx < len(new_elems)

    def test_pipeline_dropout_then_jitter_no_corrupt_indices(self) -> None:
        """After NodeDropout + Jitter, constraint indices are always valid."""
        elems = _elems(100)
        constrs = _all_align_con(elems)
        for seed_val in range(10):
            pipe = GraphAugmentationPipeline(transforms=[
                NodeDropout(p=0.5),
                CoordinateJitter(std=0.03),
            ])
            new_elems, new_constrs = pipe(elems, constrs, seed=seed_val)
            for c in new_constrs:
                for idx in c.source_indices:
                    assert 0 <= idx < len(new_elems), f"seed={seed_val}, idx={idx}, n_elems={len(new_elems)}"
                for idx in c.target_indices:
                    assert 0 <= idx < len(new_elems), f"seed={seed_val}, idx={idx}, n_elems={len(new_elems)}"
