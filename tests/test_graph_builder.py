"""Tests for BipartiteGraphBuilder — full HeteroData construction."""

from __future__ import annotations

import math

import pytest
import torch

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import (
    ConstraintNode,
    ConstraintType,
    EdgeFeatures,
    ElementNode,
)


# ===================================================================
# Helpers
# ===================================================================


def _elem(
    x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0,
) -> ElementNode:
    """Shorthand to create an ElementNode."""
    return ElementNode(bbox=[x1, y1, x2, y2], confidence=confidence)


def _con(
    ctype: ConstraintType,
    source: list[int],
    target: list[int] | None = None,
    **params: float,
) -> ConstraintNode:
    """Shorthand to create a ConstraintNode."""
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=source,
        target_indices=target or source,
        params=params,
    )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def builder() -> BipartiteGraphBuilder:
    return BipartiteGraphBuilder()


@pytest.fixture
def two_elems() -> list[ElementNode]:
    return [
        _elem(0.0, 0.0, 0.5, 0.5, confidence=0.9),
        _elem(0.3, 0.3, 0.8, 0.8, confidence=0.8),
    ]


@pytest.fixture
def one_constraint() -> list[ConstraintNode]:
    return [
        _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
    ]


# ===================================================================
# Basic build
# ===================================================================


class TestBasicBuild:
    """Basic build with 2 elements + 1 constraint."""

    def test_shapes(self, builder: BipartiteGraphBuilder, two_elems: list[ElementNode],
                    one_constraint: list[ConstraintNode]) -> None:
        """Verify all tensor shapes."""
        data = builder.build(two_elems, one_constraint)
        assert data["element"].x.shape == (2, 5)
        assert data["constraint"].x.shape == (1, 11)
        assert data["element", "to", "constraint"].edge_index.shape == (2, 4)
        assert data["element", "to", "constraint"].edge_attr.shape == (4, 4)

    def test_dtypes(self, builder: BipartiteGraphBuilder, two_elems: list[ElementNode],
                    one_constraint: list[ConstraintNode]) -> None:
        """Verify correct tensor dtypes."""
        data = builder.build(two_elems, one_constraint)
        assert data["element"].x.dtype == torch.float32
        assert data["constraint"].x.dtype == torch.float32
        assert data["element", "to", "constraint"].edge_index.dtype == torch.long
        assert data["element", "to", "constraint"].edge_attr.dtype == torch.float32

    def test_element_feature_values(self, builder: BipartiteGraphBuilder,
                                    two_elems: list[ElementNode],
                                    one_constraint: list[ConstraintNode]) -> None:
        """Element features match to_tensor() of each element."""
        data = builder.build(two_elems, one_constraint)
        expected = torch.stack([e.to_tensor() for e in two_elems])
        assert torch.equal(data["element"].x, expected)

    def test_constraint_feature_values(self, builder: BipartiteGraphBuilder,
                                       two_elems: list[ElementNode],
                                       one_constraint: list[ConstraintNode]) -> None:
        """Constraint features = one-hot + first param."""
        data = builder.build(two_elems, one_constraint)
        c = one_constraint[0]
        expected = torch.cat(
            [c.to_onehot(), torch.tensor([c.params["tolerance"]], dtype=torch.float32)]
        )
        assert torch.equal(data["constraint"].x[0], expected)

    def test_forward_edge_index(self, builder: BipartiteGraphBuilder,
                                two_elems: list[ElementNode],
                                one_constraint: list[ConstraintNode]) -> None:
        """Forward edges: source=element idx, target=constraint idx."""
        data = builder.build(two_elems, one_constraint)
        ei = data["element", "to", "constraint"].edge_index
        # constraint 0 with source=[0,1], target=[0,1] => edges: (0,0), (1,0), (0,0), (1,0)
        expected = torch.tensor([[0, 1, 0, 1], [0, 0, 0, 0]], dtype=torch.long)
        assert torch.equal(ei, expected)

    def test_reverse_edge_is_flip(self, builder: BipartiteGraphBuilder,
                                  two_elems: list[ElementNode],
                                  one_constraint: list[ConstraintNode]) -> None:
        """Reverse edge index is the flip of forward edge index."""
        data = builder.build(two_elems, one_constraint)
        forward = data["element", "to", "constraint"].edge_index
        reverse = data["constraint", "to", "element"].edge_index
        assert torch.equal(reverse, torch.flip(forward, dims=[0]))

    def test_metadata(self, builder: BipartiteGraphBuilder,
                      two_elems: list[ElementNode],
                      one_constraint: list[ConstraintNode]) -> None:
        """Metadata attributes are set correctly."""
        data = builder.build(two_elems, one_constraint)
        assert data.num_elements == 2
        assert data.num_constraints == 1
        # num_edges is a PyG property (total across all edge types)
        assert data.num_edges == 8  # 4 forward + 4 reverse

    def test_edge_attr_shape(self, builder: BipartiteGraphBuilder,
                             two_elems: list[ElementNode],
                             one_constraint: list[ConstraintNode]) -> None:
        """Edge attributes have shape (E, 4)."""
        data = builder.build(two_elems, one_constraint)
        ea = data["element", "to", "constraint"].edge_attr
        assert ea.shape == (4, 4)

    def test_edge_attr_values(self, builder: BipartiteGraphBuilder,
                              two_elems: list[ElementNode],
                              one_constraint: list[ConstraintNode]) -> None:
        """Edge attributes match EdgeFeatures.compute() between paired elements."""
        data = builder.build(two_elems, one_constraint)
        ea = data["element", "to", "constraint"].edge_attr
        # Constraint [0,1] => edges: (0->c) paired with 1, (1->c) paired with 0,
        # (0->c) paired with 1, (1->c) paired with 0
        ef_0_1 = EdgeFeatures.compute(two_elems[0], two_elems[1])
        ef_1_0 = EdgeFeatures.compute(two_elems[1], two_elems[0])
        expected = torch.stack([ef_0_1.to_tensor(), ef_1_0.to_tensor(),
                                ef_0_1.to_tensor(), ef_1_0.to_tensor()])
        assert torch.allclose(ea, expected, atol=1e-6)

    def test_edge_attr_iou_value(self, builder: BipartiteGraphBuilder,
                                 two_elems: list[ElementNode],
                                 one_constraint: list[ConstraintNode]) -> None:
        """Verify IoU in edge attributes matches manual calculation."""
        data = builder.build(two_elems, one_constraint)
        ea = data["element", "to", "constraint"].edge_attr
        # elem0=[0,0,0.5,0.5], elem1=[0.3,0.3,0.8,0.8]
        # Intersection: [0.3,0.3,0.5,0.5] => 0.2*0.2=0.04
        # Union: 0.25 + 0.25 - 0.04 = 0.46
        # IoU: 0.04/0.46 ≈ 0.0869565
        expected_iou = 0.04 / 0.46
        assert ea[0, 3].item() == pytest.approx(expected_iou, rel=1e-5)


# ===================================================================
# Empty / edge cases
# ===================================================================


class TestEmptyElements:
    """Build with empty elements, non-empty constraints."""

    def test_zero_element_nodes(self, builder: BipartiteGraphBuilder,
                                one_constraint: list[ConstraintNode]) -> None:
        data = builder.build([], one_constraint)
        assert data["element"].x.shape == (0, 5)
        assert data["element"].x.dtype == torch.float32

    def test_no_edges(self, builder: BipartiteGraphBuilder,
                      one_constraint: list[ConstraintNode]) -> None:
        data = builder.build([], one_constraint)
        assert data["element", "to", "constraint"].edge_index.shape == (2, 0)
        assert data["constraint", "to", "element"].edge_index.shape == (2, 0)

    def test_no_edge_attr(self, builder: BipartiteGraphBuilder,
                          one_constraint: list[ConstraintNode]) -> None:
        data = builder.build([], one_constraint)
        assert data["element", "to", "constraint"].edge_attr.shape == (0, 4)

    def test_constraints_still_present(self, builder: BipartiteGraphBuilder,
                                       one_constraint: list[ConstraintNode]) -> None:
        data = builder.build([], one_constraint)
        assert data["constraint"].x.shape == (1, 11)
        assert data.num_constraints == 1
        assert data.num_elements == 0


class TestEmptyConstraints:
    """Build with non-empty elements, empty constraints."""

    def test_zero_constraint_nodes(self, builder: BipartiteGraphBuilder,
                                   two_elems: list[ElementNode]) -> None:
        data = builder.build(two_elems, [])
        assert data["constraint"].x.shape == (0, 11)
        assert data["constraint"].x.dtype == torch.float32

    def test_no_edges(self, builder: BipartiteGraphBuilder,
                      two_elems: list[ElementNode]) -> None:
        data = builder.build(two_elems, [])
        assert data["element", "to", "constraint"].edge_index.shape == (2, 0)
        assert data["constraint", "to", "element"].edge_index.shape == (2, 0)

    def test_no_edge_attr(self, builder: BipartiteGraphBuilder,
                          two_elems: list[ElementNode]) -> None:
        data = builder.build(two_elems, [])
        assert data["element", "to", "constraint"].edge_attr.shape == (0, 4)

    def test_elements_still_present(self, builder: BipartiteGraphBuilder,
                                    two_elems: list[ElementNode]) -> None:
        data = builder.build(two_elems, [])
        assert data["element"].x.shape == (2, 5)
        assert data.num_elements == 2
        assert data.num_constraints == 0


class TestEmptyBoth:
    """Build with both lists empty."""

    def test_zero_elements_and_constraints(self, builder: BipartiteGraphBuilder) -> None:
        data = builder.build([], [])
        assert data["element"].x.shape == (0, 5)
        assert data["constraint"].x.shape == (0, 11)

    def test_no_edges(self, builder: BipartiteGraphBuilder) -> None:
        data = builder.build([], [])
        assert data["element", "to", "constraint"].edge_index.shape == (2, 0)
        assert data["constraint", "to", "element"].edge_index.shape == (2, 0)
        assert data["element", "to", "constraint"].edge_attr.shape == (0, 4)

    def test_metadata(self, builder: BipartiteGraphBuilder) -> None:
        data = builder.build([], [])
        assert data.num_elements == 0
        assert data.num_constraints == 0
        assert data.num_edges == 0


class TestNoEdgesWithConstraints:
    """Constraints exist but reference no valid elements."""

    def test_all_out_of_range_indices(self, builder: BipartiteGraphBuilder,
                                      two_elems: list[ElementNode]) -> None:
        """Constraint references element index 99 which doesn't exist."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [99], tolerance=0.02)]
        data = builder.build(two_elems, cons)
        assert data["constraint"].x.shape == (1, 11)
        assert data["element"].x.shape == (2, 5)
        # The out-of-range edge should be skipped
        assert data["element", "to", "constraint"].edge_index.shape == (2, 0)
        assert data["element", "to", "constraint"].edge_attr.shape == (0, 4)

    def test_partial_out_of_range(self, builder: BipartiteGraphBuilder,
                                  two_elems: list[ElementNode]) -> None:
        """One valid and one out-of-range index: only valid index creates edges."""
        cons = [_con(ConstraintType.CENTER_X, [0, 99], tolerance=0.02)]
        data = builder.build(two_elems, cons)
        # Source=[0,99], target=[0,99] => valid indices = [0, 0] (2 edges)
        assert data["element", "to", "constraint"].edge_index.shape == (2, 2)
        assert data["element", "to", "constraint"].edge_attr.shape == (2, 4)


# ===================================================================
# Constraint node features
# ===================================================================


class TestConstraintNodeFeatures:
    """Constraint feature dimension and encoding."""

    def test_one_hot_encoding(self, builder: BipartiteGraphBuilder,
                              two_elems: list[ElementNode]) -> None:
        """Constraint one-hot encoding matches ConstraintNode.to_onehot()."""
        cons = [_con(ConstraintType.GRID, [0, 1], rows=2.0, columns=2.0, tolerance=0.05)]
        data = builder.build(two_elems, cons)
        first_10 = data["constraint"].x[0, :10]
        expected = cons[0].to_onehot()
        assert torch.equal(first_10, expected)

    def test_first_param_value(self, builder: BipartiteGraphBuilder,
                               two_elems: list[ElementNode]) -> None:
        """The 11th feature is the first param value."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02)]
        data = builder.build(two_elems, cons)
        assert data["constraint"].x[0, 10].item() == pytest.approx(0.02, rel=1e-6)

    def test_first_param_with_multiple_params(self, builder: BipartiteGraphBuilder,
                                              two_elems: list[ElementNode]) -> None:
        """Only the first param value is used when multiple params exist."""
        cons = [
            _con(ConstraintType.SPACING, [0, 1, 2],
                 tolerance=0.02, axis=1.0)
        ]
        elems = two_elems + [_elem(0.6, 0.6, 0.9, 0.9)]
        # "tolerance" is the first param (dict ordering)
        data = builder.build(elems, cons)
        assert data["constraint"].x[0, 10].item() == pytest.approx(0.02, rel=1e-6)

    def test_no_params_default_zero(self, builder: BipartiteGraphBuilder,
                                    two_elems: list[ElementNode]) -> None:
        """Constraint without params gets 0.0 as the 11th feature."""
        c = ConstraintNode(
            constraint_type=ConstraintType.CONTAINMENT,
            source_indices=[0],
            target_indices=[1],
        )
        data = builder.build(two_elems, [c])
        assert data["constraint"].x[0, 10].item() == pytest.approx(0.0, abs=1e-6)

    def test_dimension(self, builder: BipartiteGraphBuilder,
                       two_elems: list[ElementNode]) -> None:
        """Multiple constraints produce correct shape."""
        cons = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
            _con(ConstraintType.CENTER_X, [0], tolerance=0.02),
        ]
        data = builder.build(two_elems, cons)
        assert data["constraint"].x.shape == (2, 11)


# ===================================================================
# Element node features
# ===================================================================


class TestElementNodeFeatures:
    """Element feature correctness."""

    def test_single_element(self, builder: BipartiteGraphBuilder) -> None:
        elems = [_elem(0.1, 0.2, 0.3, 0.4, confidence=0.95)]
        data = builder.build(elems, [])
        expected = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.95], dtype=torch.float32)
        assert torch.equal(data["element"].x[0], expected)

    def test_multiple_elements(self, builder: BipartiteGraphBuilder) -> None:
        elems = [
            _elem(0.0, 0.0, 0.5, 0.5, confidence=0.9),
            _elem(0.2, 0.3, 0.7, 0.8, confidence=0.85),
            _elem(0.1, 0.1, 0.3, 0.4, confidence=0.95),
        ]
        data = builder.build(elems, [])
        expected = torch.stack([e.to_tensor() for e in elems])
        assert torch.equal(data["element"].x, expected)


# ===================================================================
# Multiple constraints
# ===================================================================


class TestMultipleConstraints:
    """Multiple constraints with overlapping element indices."""

    def test_two_constraints_separate(self, builder: BipartiteGraphBuilder) -> None:
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.5, 0.0, 1.0, 0.5),
                 _elem(0.0, 0.5, 0.5, 1.0)]
        cons = [
            _con(ConstraintType.ALIGN_TOP, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_LEFT, [0, 2], tolerance=0.02),
        ]
        data = builder.build(elems, cons)
        # Constraint 0: [0,1,0,1] = 4 edges. Constraint 1: [0,2,0,2] = 4 edges. Total = 8
        assert data["element", "to", "constraint"].edge_index.shape == (2, 8)
        assert data["element", "to", "constraint"].edge_attr.shape == (8, 4)
        assert data.num_edges == 16  # 8 forward + 8 reverse

    def test_overlapping_indices(self, builder: BipartiteGraphBuilder) -> None:
        """Same element appears in multiple constraints."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.5, 0.0, 1.0, 0.5)]
        cons = [
            _con(ConstraintType.ALIGN_TOP, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_BOTTOM, [0, 1], tolerance=0.02),
        ]
        data = builder.build(elems, cons)
        # Both constraints have 4 edges each => 8 total forward edges
        assert data["element", "to", "constraint"].edge_index.shape[1] == 8
        # Forward edges should alternate: constraint 0 edges then constraint 1 edges
        ei = data["element", "to", "constraint"].edge_index
        assert torch.all(ei[1, :4] == 0)  # first 4 belong to constraint 0
        assert torch.all(ei[1, 4:] == 1)  # next 4 belong to constraint 1

    def test_edge_attr_for_multiple(self, builder: BipartiteGraphBuilder) -> None:
        """Edge attributes computed for each edge independently."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.5, 0.0, 1.0, 0.5),
                 _elem(0.0, 0.5, 0.5, 1.0)]
        cons = [
            _con(ConstraintType.ALIGN_TOP, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_LEFT, [0, 2], tolerance=0.02),
        ]
        data = builder.build(elems, cons)
        ea = data["element", "to", "constraint"].edge_attr
        assert ea.shape == (8, 4)
        # All rows should be different (different element pairings)
        # At minimum, rows for different constraints should have different values
        # Constraint 0 pairs (0↔1), constraint 1 pairs (0↔2)
        # dx values should differ: 0→1 has dx=0.5, 0→2 has dx=0.0
        # dy values should differ: 0→1 has dy=0.0, 0→2 has dy=0.5
        assert ea[0, 1].item() == pytest.approx(0.5, abs=1e-5)  # dx for elem0→elem1
        assert ea[4, 1].item() == pytest.approx(0.0, abs=1e-5)  # dx for elem0→elem2 (cy same)
        assert ea[4, 2].item() == pytest.approx(0.5, abs=1e-5)  # dy for elem0→elem2


# ===================================================================
# Edge features details
# ===================================================================


class TestEdgeFeatures:
    """Detailed edge feature verification."""

    def test_three_elements_chain(self, builder: BipartiteGraphBuilder) -> None:
        """Constraint with 3 elements: pairings are (0↔1), (1↔2), (2↔0)."""
        elems = [
            _elem(0.0, 0.0, 0.2, 0.2),
            _elem(0.3, 0.3, 0.5, 0.5),
            _elem(0.6, 0.6, 0.8, 0.8),
        ]
        # Symmetric (source=target) means 6 edges (each of 3 elements appears twice)
        cons = [_con(ConstraintType.ALIGN_LEFT, [0, 1, 2], tolerance=0.02)]
        data = builder.build(elems, cons)
        ea = data["element", "to", "constraint"].edge_attr
        assert ea.shape == (6, 4)
        # Edge 0: elem 0 → c, paired with elem 1
        ef_0_1 = EdgeFeatures.compute(elems[0], elems[1])
        assert torch.allclose(ea[0], ef_0_1.to_tensor(), atol=1e-6)
        # Edge 1: elem 1 → c, paired with elem 2
        ef_1_2 = EdgeFeatures.compute(elems[1], elems[2])
        assert torch.allclose(ea[1], ef_1_2.to_tensor(), atol=1e-6)
        # Edge 2: elem 2 → c, paired with elem 0
        ef_2_0 = EdgeFeatures.compute(elems[2], elems[0])
        assert torch.allclose(ea[2], ef_2_0.to_tensor(), atol=1e-6)

    def test_single_element_constraint_self_pair(self, builder: BipartiteGraphBuilder) -> None:
        """Constraint with one element: pairs with itself."""
        elems = [_elem(0.1, 0.2, 0.5, 0.8)]
        cons = [_con(ConstraintType.CENTER_X, [0], tolerance=0.02)]
        data = builder.build(elems, cons)
        ea = data["element", "to", "constraint"].edge_attr
        assert ea.shape == (2, 4)
        # Self-pair: distance=0, dx=0, dy=0, iou=1.0
        assert ea[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert ea[0, 1].item() == pytest.approx(0.0, abs=1e-6)
        assert ea[0, 2].item() == pytest.approx(0.0, abs=1e-6)
        assert ea[0, 3].item() == pytest.approx(1.0, abs=1e-6)

    def test_containment_asymmetric(self, builder: BipartiteGraphBuilder) -> None:
        """Containment has different source and target."""
        elems = [
            _elem(0.0, 0.0, 1.0, 1.0),  # parent (idx 0)
            _elem(0.2, 0.2, 0.4, 0.4),  # child (idx 1)
        ]
        # containment: source=[parent], target=[child]
        c = ConstraintNode(
            constraint_type=ConstraintType.CONTAINMENT,
            source_indices=[0],
            target_indices=[1],
            params={"margin": 0.96},
        )
        data = builder.build(elems, [c])
        # valid_indices = [0, 1] => 2 edges
        ei = data["element", "to", "constraint"].edge_index
        assert ei.shape == (2, 2)
        # Edge 0: parent (0) paired with child (1)
        # Edge 1: child (1) paired with parent (0)
        assert ei[0, 0].item() == 0
        assert ei[0, 1].item() == 1
        assert ei[1, 0].item() == 0
        assert ei[1, 1].item() == 0

        ea = data["element", "to", "constraint"].edge_attr
        ef_parent_child = EdgeFeatures.compute(elems[0], elems[1])
        ef_child_parent = EdgeFeatures.compute(elems[1], elems[0])
        assert torch.allclose(ea[0], ef_parent_child.to_tensor(), atol=1e-6)
        assert torch.allclose(ea[1], ef_child_parent.to_tensor(), atol=1e-6)


# ===================================================================
# Independence
# ===================================================================


class TestBuildIndependence:
    """Multiple builds produce independent HeteroData objects."""

    def test_two_builds_independent(self, builder: BipartiteGraphBuilder,
                                    two_elems: list[ElementNode]) -> None:
        """Calling build twice with same inputs produces separate objects."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02)]
        data1 = builder.build(two_elems, cons)
        data2 = builder.build(two_elems, cons)
        # Different objects
        assert data1 is not data2
        # Same values
        assert torch.equal(data1["element"].x, data2["element"].x)
        assert torch.equal(
            data1["element", "to", "constraint"].edge_index,
            data2["element", "to", "constraint"].edge_index,
        )

    def test_mutate_after_build_no_effect(self, builder: BipartiteGraphBuilder,
                                          two_elems: list[ElementNode]) -> None:
        """Modifying input lists after build does not change built data."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02)]
        data = builder.build(two_elems, cons)
        # Mutate originals
        two_elems.append(_elem(0.9, 0.9, 1.0, 1.0))
        cons.append(_con(ConstraintType.CENTER_X, [0, 1], tolerance=0.02))
        # Original data should be unchanged
        assert data.num_elements == 2
        assert data.num_constraints == 1
        assert data["element"].x.shape == (2, 5)

    def test_different_constraints_different_graphs(self, builder: BipartiteGraphBuilder,
                                                    two_elems: list[ElementNode]) -> None:
        """Different constraint inputs produce different edge structures."""
        cons1 = [_con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02)]
        cons2 = [_con(ConstraintType.CENTER_X, [0, 1], tolerance=0.03)]
        data1 = builder.build(two_elems, cons1)
        data2 = builder.build(two_elems, cons2)
        # Different constraint features (different one-hot + different param)
        assert not torch.equal(data1["constraint"].x, data2["constraint"].x)
        # Same edge structure (same indices)
        assert torch.equal(
            data1["element", "to", "constraint"].edge_index,
            data2["element", "to", "constraint"].edge_index,
        )


# ===================================================================
# All constraint types
# ===================================================================


class TestAllConstraintTypes:
    """Ensure build works with each of the 10 constraint types."""

    @pytest.mark.parametrize("ctype", list(ConstraintType))
    def test_each_type(self, builder: BipartiteGraphBuilder, ctype: ConstraintType) -> None:
        """Each constraint type produces valid HeteroData."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.3, 0.3, 0.8, 0.8)]
        params: dict[str, float] = {"tolerance": 0.02}
        if ctype == ConstraintType.GRID:
            params = {"rows": 2.0, "columns": 2.0, "tolerance": 0.05}
        elif ctype == ConstraintType.SPACING:
            params = {"tolerance": 0.02, "axis": 1.0}
        elif ctype == ConstraintType.CONTAINMENT:
            params = {"margin": 0.5}
        cons = [ConstraintNode(
            constraint_type=ctype,
            source_indices=[0, 1],
            target_indices=[0, 1],
            params=params,
        )]
        data = builder.build(elems, cons)
        assert data["element"].x.shape == (2, 5)
        assert data["constraint"].x.shape == (1, 11)
        assert data["element", "to", "constraint"].edge_index.shape[1] == 4
        assert data["element", "to", "constraint"].edge_attr.shape[1] == 4


# ===================================================================
# Integration with constraint extraction
# ===================================================================


class TestIntegration:
    """End-to-end integration with extract_all_constraints."""

    def test_extract_and_build(self, builder: BipartiteGraphBuilder) -> None:
        """Full pipeline: elements → extract constraints → build graph."""
        elems = [
            _elem(0.0, 0.0, 0.5, 0.5),
            _elem(0.5, 0.0, 1.0, 0.5),
            _elem(0.0, 0.5, 0.5, 1.0),
            _elem(0.5, 0.5, 1.0, 1.0),
        ]
        constraints = extract_all_constraints(elems, tolerance=0.02)
        data = builder.build(elems, constraints)
        # Should have at least alignment constraints for a 2x2 grid
        assert data.num_elements == 4
        assert data.num_constraints >= 1
        assert data["element"].x.shape == (4, 5)
        assert data["constraint"].x.shape[1] == 11
        # Each constraint has at least 2 element indices
        assert data["element", "to", "constraint"].edge_index.shape[0] == 2
        # Edge features are present and valid
        ea = data["element", "to", "constraint"].edge_attr
        assert ea.shape[1] == 4
        assert torch.all(ea >= -1.0)  # dx, dy are in [-1, 1]
        assert torch.all(ea[:, 3] >= 0.0)  # IoU is non-negative
        assert torch.all(ea[:, 3] <= 1.0)  # IoU is at most 1.0

    def test_empty_elements_extract_and_build(self, builder: BipartiteGraphBuilder) -> None:
        """Empty elements → empty constraints → empty graph."""
        constraints = extract_all_constraints([])
        data = builder.build([], constraints)
        assert data["element"].x.shape == (0, 5)
        assert data["constraint"].x.shape == (0, 11)

    def test_single_element_extract_and_build(self, builder: BipartiteGraphBuilder) -> None:
        """Single element → no constraints (need 2+ for most)."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5)]
        constraints = extract_all_constraints(elems)
        data = builder.build(elems, constraints)
        assert data.num_elements == 1
        assert data.num_constraints == 0
        assert data["element", "to", "constraint"].edge_index.shape[1] == 0


# ===================================================================
# Edge cases
# ===================================================================


class TestEdgeCases:
    """Unusual or boundary conditions."""

    def test_constraint_referencing_negative_index(self, builder: BipartiteGraphBuilder,
                                                   two_elems: list[ElementNode]) -> None:
        """Negative element index is out of range and should be skipped."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [-1, 0], tolerance=0.02)]
        data = builder.build(two_elems, cons)
        # Only index 0 is valid, source=[-1,0], target=[-1,0] => valid=[0,0] => 2 edges
        assert data["element", "to", "constraint"].edge_index.shape == (2, 2)

    def test_many_elements(self, builder: BipartiteGraphBuilder) -> None:
        """Build with 10 elements and 3 constraints."""
        elems = [_elem(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05)
                 for i in range(10)]
        cons = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_TOP, [2, 3, 4], tolerance=0.02),
            _con(ConstraintType.CENTER_X, [5, 6, 7, 8], tolerance=0.02),
        ]
        data = builder.build(elems, cons)
        assert data.num_elements == 10
        assert data.num_constraints == 3
        # Constraint 0: 4 edges, Constraint 1: 6 edges, Constraint 2: 8 edges = 18 total
        assert data["element", "to", "constraint"].edge_index.shape[1] == 18
        assert data["element", "to", "constraint"].edge_attr.shape[0] == 18

    def test_node_types_present(self, builder: BipartiteGraphBuilder,
                                two_elems: list[ElementNode],
                                one_constraint: list[ConstraintNode]) -> None:
        """Both node types and both edge types are present in the data."""
        data = builder.build(two_elems, one_constraint)
        assert "element" in data.node_types
        assert "constraint" in data.node_types
        assert ("element", "to", "constraint") in data.edge_types
        assert ("constraint", "to", "element") in data.edge_types

    def test_no_edge_attr_on_reverse(self, builder: BipartiteGraphBuilder,
                                     two_elems: list[ElementNode],
                                     one_constraint: list[ConstraintNode]) -> None:
        """Reverse edge store has no edge_attr (only edge_index)."""
        data = builder.build(two_elems, one_constraint)
        reverse = data["constraint", "to", "element"]
        assert hasattr(reverse, "edge_index")
        assert not hasattr(reverse, "edge_attr")

    def test_constraint_with_empty_indices(self, builder: BipartiteGraphBuilder,
                                           two_elems: list[ElementNode]) -> None:
        """Constraint with empty source/target produces no edges."""
        c = ConstraintNode(
            constraint_type=ConstraintType.CONTAINMENT,
            source_indices=[],
            target_indices=[],
        )
        data = builder.build(two_elems, [c])
        assert data["constraint"].x.shape == (1, 11)
        assert data["element", "to", "constraint"].edge_index.shape == (2, 0)


# ===================================================================
# Constraint feature edge cases
# ===================================================================


class TestConstraintFeaturesEdgeCases:
    """Boundary behavior for constraint features."""

    def test_params_dict_single_entry(self, builder: BipartiteGraphBuilder,
                                      two_elems: list[ElementNode]) -> None:
        """Constraint with a single param correctly sets 11th feature."""
        cons = [_con(ConstraintType.CONTAINMENT, [0, 1], margin=0.5)]
        data = builder.build(two_elems, cons)
        assert data["constraint"].x[0, 10].item() == pytest.approx(0.5, rel=1e-6)

    def test_params_dict_value_zero(self, builder: BipartiteGraphBuilder,
                                    two_elems: list[ElementNode]) -> None:
        """Constraint with tolerance=0.0 still correctly stored."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.0)]
        data = builder.build(two_elems, cons)
        assert data["constraint"].x[0, 10].item() == pytest.approx(0.0, abs=1e-6)

    def test_one_hot_for_each_type(self, builder: BipartiteGraphBuilder,
                                   two_elems: list[ElementNode]) -> None:
        """Each constraint type produces a different one-hot encoding."""
        cons = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_RIGHT, [0, 1], tolerance=0.02),
            _con(ConstraintType.CONTAINMENT, [0, 1], margin=0.5),
        ]
        data = builder.build(two_elems, cons)
        # All one-hot vectors should be different
        oh0 = data["constraint"].x[0, :10]
        oh1 = data["constraint"].x[1, :10]
        oh2 = data["constraint"].x[2, :10]
        assert not torch.equal(oh0, oh1)
        assert not torch.equal(oh0, oh2)
        assert not torch.equal(oh1, oh2)


# ===================================================================
# Reverse edge index
# ===================================================================


class TestReverseEdgeIndex:
    """Reverse edge index correctness."""

    def test_two_constraints_reverse(self, builder: BipartiteGraphBuilder) -> None:
        """Reverse edges for multiple constraints are correct flips."""
        elems = [_elem(0.0, 0.0, 0.5, 0.5), _elem(0.5, 0.0, 1.0, 0.5)]
        cons = [
            _con(ConstraintType.ALIGN_TOP, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_BOTTOM, [0, 1], tolerance=0.02),
        ]
        data = builder.build(elems, cons)
        forward = data["element", "to", "constraint"].edge_index
        reverse = data["constraint", "to", "element"].edge_index
        expected_reverse = torch.flip(forward, dims=[0])
        assert torch.equal(reverse, expected_reverse)

    def test_empty_reverse(self, builder: BipartiteGraphBuilder) -> None:
        """Empty edge index produces empty reverse."""
        data = builder.build([], [])
        assert data["constraint", "to", "element"].edge_index.shape == (2, 0)

    def test_reverse_dtype(self, builder: BipartiteGraphBuilder,
                           two_elems: list[ElementNode],
                           one_constraint: list[ConstraintNode]) -> None:
        """Reverse edge index has long dtype."""
        data = builder.build(two_elems, one_constraint)
        assert data["constraint", "to", "element"].edge_index.dtype == torch.long
