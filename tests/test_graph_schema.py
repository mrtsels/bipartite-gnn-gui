"""Tests for graph schema — ElementNode, ConstraintNode, EdgeFeatures, enums."""

from __future__ import annotations

import math

import pytest
import torch

from bipartite_gnn_gui.data.vlm_output import VLMOutputElement
from bipartite_gnn_gui.graph.schema import (
    ConstraintNode,
    ConstraintType,
    EdgeFeatures,
    EdgeType,
    ElementNode,
)

# ---------------------------------------------------------------------------
# ConstraintType
# ---------------------------------------------------------------------------


class TestConstraintType:
    def test_ten_types(self) -> None:
        """There are exactly 10 constraint types."""
        assert len(list(ConstraintType)) == 10

    def test_str_values(self) -> None:
        """Each member has the expected string value."""
        pairs = [
            (ConstraintType.ALIGN_LEFT, "align_left"),
            (ConstraintType.ALIGN_RIGHT, "align_right"),
            (ConstraintType.ALIGN_TOP, "align_top"),
            (ConstraintType.ALIGN_BOTTOM, "align_bottom"),
            (ConstraintType.CENTER_X, "center_x"),
            (ConstraintType.CENTER_Y, "center_y"),
            (ConstraintType.SAME_SIZE, "same_size"),
            (ConstraintType.SPACING, "spacing"),
            (ConstraintType.CONTAINMENT, "containment"),
            (ConstraintType.GRID, "grid"),
        ]
        for constraint_type, expected in pairs:
            assert constraint_type.value == expected

    def test_is_str_enum(self) -> None:
        """ConstraintType values are usable as plain strings."""
        assert ConstraintType.ALIGN_LEFT == "align_left"
        assert isinstance(ConstraintType.ALIGN_RIGHT, str)

    def test_unique_values(self) -> None:
        """All string values are unique."""
        values = [c.value for c in ConstraintType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# EdgeType
# ---------------------------------------------------------------------------


class TestEdgeType:
    def test_two_types(self) -> None:
        assert len(list(EdgeType)) == 2

    def test_str_values(self) -> None:
        assert EdgeType.ELEMENT_TO_CONSTRAINT.value == "element_to_constraint"
        assert EdgeType.CONSTRAINT_TO_ELEMENT.value == "constraint_to_element"


# ---------------------------------------------------------------------------
# ElementNode
# ---------------------------------------------------------------------------


class TestElementNode:
    def test_creation_defaults(self) -> None:
        node = ElementNode(bbox=[0.1, 0.2, 0.5, 0.8])
        assert node.bbox == [0.1, 0.2, 0.5, 0.8]
        assert node.label == "unknown"
        assert node.confidence == 1.0
        assert node.element_id is None
        assert node.features == {}

    def test_creation_explicit(self) -> None:
        node = ElementNode(
            bbox=[0.0, 0.0, 1.0, 1.0],
            label="button",
            confidence=0.95,
            element_id="elem_0",
            features={"area": 1.0},
        )
        assert node.label == "button"
        assert node.confidence == 0.95
        assert node.element_id == "elem_0"
        assert node.features == {"area": 1.0}

    def test_to_tensor_shape(self) -> None:
        node = ElementNode(bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.85)
        t = node.to_tensor()
        assert t.shape == (5,)
        assert t.dtype == torch.float32

    def test_to_tensor_values(self) -> None:
        node = ElementNode(bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.85)
        t = node.to_tensor()
        expected = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.85], dtype=torch.float32)
        assert torch.equal(t, expected)

    def test_from_vlm_element(self) -> None:
        vlm = VLMOutputElement(
            element_id=3,
            bbox=(0.15, 0.25, 0.55, 0.75),
            element_type="button",
            text_content="Submit",
            confidence=0.92,
        )
        node = ElementNode.from_vlm_element(vlm)
        assert node.bbox == [0.15, 0.25, 0.55, 0.75]
        assert node.label == "button"
        assert node.confidence == 0.92
        assert node.element_id == "3"

    def test_from_vlm_element_zero_id(self) -> None:
        vlm = VLMOutputElement(
            element_id=0,
            bbox=(0.0, 0.0, 1.0, 1.0),
            element_type="text",
            confidence=0.5,
        )
        node = ElementNode.from_vlm_element(vlm)
        assert node.element_id == "0"


# ---------------------------------------------------------------------------
# ConstraintNode
# ---------------------------------------------------------------------------


class TestConstraintNode:
    def test_creation_defaults(self) -> None:
        node = ConstraintNode(constraint_type=ConstraintType.ALIGN_LEFT)
        assert node.constraint_type == ConstraintType.ALIGN_LEFT
        assert node.source_indices == []
        assert node.target_indices == []
        assert node.params == {}

    def test_creation_explicit(self) -> None:
        node = ConstraintNode(
            constraint_type=ConstraintType.CENTER_X,
            source_indices=[0, 1],
            target_indices=[2],
            params={"tolerance": 0.02},
        )
        assert node.source_indices == [0, 1]
        assert node.target_indices == [2]
        assert node.params == {"tolerance": 0.02}

    def test_to_tensor_with_params(self) -> None:
        node = ConstraintNode(
            constraint_type=ConstraintType.ALIGN_TOP,
            params={"tolerance": 0.02, "weight": 1.0},
        )
        t = node.to_tensor()
        assert t.shape == (2,)
        assert t.dtype == torch.float32
        assert torch.equal(t, torch.tensor([0.02, 1.0], dtype=torch.float32))

    def test_to_tensor_empty_params(self) -> None:
        node = ConstraintNode(constraint_type=ConstraintType.CONTAINMENT)
        t = node.to_tensor()
        assert t.shape == (1,)
        assert t.dtype == torch.float32
        assert t[0].item() == 0.0

    def test_to_onehot_shape(self) -> None:
        node = ConstraintNode(constraint_type=ConstraintType.ALIGN_LEFT)
        oh = node.to_onehot()
        assert oh.shape == (10,)
        assert oh.dtype == torch.float32

    def test_to_onehot_align_left(self) -> None:
        node = ConstraintNode(constraint_type=ConstraintType.ALIGN_LEFT)
        oh = node.to_onehot()
        expected = torch.zeros(10, dtype=torch.float32)
        expected[0] = 1.0  # ALIGN_LEFT is first in definition order
        assert torch.equal(oh, expected)

    def test_to_onehot_grid(self) -> None:
        node = ConstraintNode(constraint_type=ConstraintType.GRID)
        oh = node.to_onehot()
        expected = torch.zeros(10, dtype=torch.float32)
        expected[9] = 1.0  # GRID is last in definition order
        assert torch.equal(oh, expected)

    def test_to_onehot_single_one(self) -> None:
        """One-hot has exactly one 1.0 and nine 0.0 values."""
        for ct in ConstraintType:
            node = ConstraintNode(constraint_type=ct)
            oh = node.to_onehot()
            assert oh.sum().item() == pytest.approx(1.0)
            assert (oh == 0.0).sum().item() == 9


# ---------------------------------------------------------------------------
# EdgeFeatures
# ---------------------------------------------------------------------------


class TestEdgeFeatures:
    def test_creation(self) -> None:
        ef = EdgeFeatures(
            spatial_distance=0.5,
            relative_position=(0.3, -0.1),
            iou=0.25,
        )
        assert ef.spatial_distance == 0.5
        assert ef.relative_position == (0.3, -0.1)
        assert ef.iou == 0.25

    def test_to_tensor_shape(self) -> None:
        ef = EdgeFeatures(
            spatial_distance=0.5,
            relative_position=(0.3, -0.1),
            iou=0.25,
        )
        t = ef.to_tensor()
        assert t.shape == (4,)
        assert t.dtype == torch.float32

    def test_to_tensor_values(self) -> None:
        ef = EdgeFeatures(
            spatial_distance=0.5,
            relative_position=(0.3, -0.1),
            iou=0.25,
        )
        t = ef.to_tensor()
        expected = torch.tensor([0.5, 0.3, -0.1, 0.25], dtype=torch.float32)
        assert torch.equal(t, expected)

    def test_compute_non_overlapping(self) -> None:
        """Two boxes far apart — IoU should be 0.0."""
        a = ElementNode(bbox=[0.0, 0.0, 0.1, 0.1])  # top-left
        b = ElementNode(bbox=[0.8, 0.8, 0.9, 0.9])  # bottom-right
        ef = EdgeFeatures.compute(a, b)
        assert ef.iou == 0.0
        # Centers at (0.05, 0.05) and (0.85, 0.85)
        dx = 0.85 - 0.05  # 0.8
        dy = 0.85 - 0.05  # 0.8
        expected_distance = math.sqrt(dx * dx + dy * dy)
        assert ef.spatial_distance == pytest.approx(expected_distance, rel=1e-6)
        assert ef.relative_position[0] == pytest.approx(dx, rel=1e-6)
        assert ef.relative_position[1] == pytest.approx(dy, rel=1e-6)

    def test_compute_identical_boxes(self) -> None:
        """Two identical boxes — IoU should be 1.0, distance 0.0."""
        a = ElementNode(bbox=[0.1, 0.2, 0.5, 0.6])
        b = ElementNode(bbox=[0.1, 0.2, 0.5, 0.6])
        ef = EdgeFeatures.compute(a, b)
        assert ef.iou == 1.0
        assert ef.spatial_distance == 0.0
        assert ef.relative_position == (0.0, 0.0)

    def test_compute_partial_overlap(self) -> None:
        """Two boxes with known partial overlap."""
        # Box A: (0.0, 0.0, 0.4, 0.4)  area = 0.16
        # Box B: (0.2, 0.2, 0.6, 0.6)  area = 0.16
        # Intersection: (0.2, 0.2, 0.4, 0.4)  area = 0.04
        # Union: 0.16 + 0.16 - 0.04 = 0.28
        # IoU: 0.04 / 0.28 ≈ 0.142857
        a = ElementNode(bbox=[0.0, 0.0, 0.4, 0.4])
        b = ElementNode(bbox=[0.2, 0.2, 0.6, 0.6])
        ef = EdgeFeatures.compute(a, b)
        assert ef.iou == pytest.approx(0.04 / 0.28, rel=1e-6)
        # Centers: A=(0.2, 0.2), B=(0.4, 0.4)
        assert ef.relative_position == (0.2, 0.2)
        assert ef.spatial_distance == pytest.approx(math.sqrt(0.08), rel=1e-6)

    def test_compute_contained_box(self) -> None:
        """One box fully inside another — IoU = area(inner) / area(outer)."""
        outer = ElementNode(bbox=[0.0, 0.0, 1.0, 1.0])  # area = 1.0
        inner = ElementNode(bbox=[0.2, 0.2, 0.4, 0.4])  # area = 0.04
        ef = EdgeFeatures.compute(outer, inner)
        assert ef.iou == pytest.approx(0.04 / 1.0, rel=1e-6)
        # Centers: outer=(0.5, 0.5), inner=(0.3, 0.3)
        assert ef.relative_position[0] == pytest.approx(-0.2, rel=1e-6)
        assert ef.relative_position[1] == pytest.approx(-0.2, rel=1e-6)


# ---------------------------------------------------------------------------
# Integration: full round-trip consistency
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_element_to_tensor_round_trip(self) -> None:
        """ElementNode to_tensor() produces correct values we can verify."""
        node = ElementNode(bbox=[0.25, 0.25, 0.75, 0.75], confidence=0.9)
        t = node.to_tensor()
        assert t[0].item() == 0.25  # x1
        assert t[1].item() == 0.25  # y1
        assert t[2].item() == 0.75  # x2
        assert t[3].item() == 0.75  # y2
        assert t[4].item() == pytest.approx(0.9, rel=1e-5)  # confidence
