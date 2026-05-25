"""Tests for bounding box utility functions."""

from __future__ import annotations

import math

import pytest
import torch

from bipartite_gnn_gui.utils.bbox import (
    apply_delta,
    bbox_to_tensor,
    clamp_coords,
    compute_center_distance,
    compute_iou,
    tensor_to_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


# ---------------------------------------------------------------------------
# bbox_to_tensor / tensor_to_bbox
# ---------------------------------------------------------------------------


class TestConversions:
    def test_bbox_to_tensor_basic(self) -> None:
        t = bbox_to_tensor([1.0, 2.0, 3.0, 4.0])
        assert t.dtype == torch.float32
        assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    def test_bbox_to_tensor_device(self) -> None:
        t = bbox_to_tensor([0.0, 0.0, 1.0, 1.0], device=torch.device("cpu"))
        assert t.device.type == "cpu"

    def test_tensor_to_bbox(self) -> None:
        t = torch.tensor([5.0, 6.0, 7.0, 8.0])
        result = tensor_to_bbox(t)
        assert result == (5.0, 6.0, 7.0, 8.0)

    def test_tensor_to_bbox_detached(self) -> None:
        t = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        result = tensor_to_bbox(t)
        assert result == (1.0, 2.0, 3.0, 4.0)

    def test_roundtrip(self) -> None:
        original = (0.1, 0.2, 0.3, 0.4)
        result = tensor_to_bbox(bbox_to_tensor(original))
        assert result == pytest.approx(original, rel=1e-5)


# ---------------------------------------------------------------------------
# xywh_to_xyxy / xyxy_to_xywh  (centre-based)
# ---------------------------------------------------------------------------


class TestFormatConversions:
    """All conversions use centre-based xywh: (cx, cy, w, h)."""

    def test_xywh_to_xyxy_centred_square(self) -> None:
        """(cx=0.5, cy=0.5, w=0.2, h=0.2) → (0.4, 0.4, 0.6, 0.6)."""
        xywh = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        expected = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
        result = xywh_to_xyxy(xywh)
        assert torch.allclose(result, expected)

    def test_xyxy_to_xywh_centred_square(self) -> None:
        xyxy = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
        expected = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        result = xyxy_to_xywh(xyxy)
        assert torch.allclose(result, expected)

    def test_xywh_to_xyxy_batch(self) -> None:
        xywh = torch.tensor([
            [0.5, 0.5, 0.4, 0.4],  # centre, w=h=0.4
            [0.2, 0.3, 0.1, 0.2],  # offset
        ])
        expected = torch.tensor([
            [0.3, 0.3, 0.7, 0.7],
            [0.15, 0.2, 0.25, 0.4],
        ])
        result = xywh_to_xyxy(xywh)
        assert torch.allclose(result, expected)

    def test_xyxy_to_xywh_batch(self) -> None:
        xyxy = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.2, 0.3, 0.8, 0.9],
        ])
        expected = torch.tensor([
            [0.5, 0.5, 1.0, 1.0],
            [0.5, 0.6, 0.6, 0.6],
        ])
        result = xyxy_to_xywh(xyxy)
        assert torch.allclose(result, expected)

    def test_xywh_xyxy_roundtrip(self) -> None:
        """xywh → xyxy → xywh should be identity."""
        xywh = torch.tensor([
            [0.5, 0.5, 0.2, 0.3],
            [0.1, 0.9, 0.15, 0.12],
            [0.75, 0.25, 0.5, 0.5],
        ])
        result = xyxy_to_xywh(xywh_to_xyxy(xywh))
        assert torch.allclose(result, xywh, atol=1e-6)

    def test_xyxy_xywh_roundtrip(self) -> None:
        """xyxy → xywh → xyxy should be identity."""
        xyxy = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.5, 0.8, 0.9],
        ])
        result = xywh_to_xyxy(xyxy_to_xywh(xyxy))
        assert torch.allclose(result, xyxy, atol=1e-6)

    def test_zero_size_box(self) -> None:
        """Box with w=0, h=0 collapses to a point."""
        xywh = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        result = xywh_to_xyxy(xywh)
        assert torch.allclose(result, expected)

    def test_arbitrary_leading_dims(self) -> None:
        """Supports (B, N, 4) shapes."""
        xywh = torch.randn(2, 3, 4).abs() * 0.1 + 0.5
        result = xywh_to_xyxy(xywh)
        assert result.shape == (2, 3, 4)
        recover = xyxy_to_xywh(result)
        assert torch.allclose(recover, xywh, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_iou
# ---------------------------------------------------------------------------


class TestComputeIoU:
    """compute_iou expects xyxy format exclusively."""

    # -- overlapping boxes --------------------------------------------------

    def test_perfect_overlap(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        b2 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        iou = compute_iou(b1, b2)
        assert torch.allclose(iou, torch.tensor([[1.0]]))

    def test_partial_overlap(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        b2 = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
        iou = compute_iou(b1, b2)
        # area1=4, area2=4, intersection=1 → IoU=1/7≈0.1429
        assert torch.allclose(iou, torch.tensor([[1.0 / 7.0]]), atol=1e-4)

    def test_one_contained_in_other(self) -> None:
        big = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
        small = torch.tensor([[2.0, 2.0, 4.0, 4.0]])
        iou = compute_iou(small, big)
        # area_small=4, area_big=100, intersection=4 → IoU=4/100=0.04
        assert torch.allclose(iou, torch.tensor([[0.04]]), atol=1e-4)

    # -- non-overlapping boxes ----------------------------------------------

    def test_no_overlap(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        b2 = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        iou = compute_iou(b1, b2)
        assert torch.equal(iou, torch.tensor([[0.0]]))

    def test_touching_edges_zero_iou(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        b2 = torch.tensor([[1.0, 0.0, 2.0, 1.0]])  # x1==x2
        iou = compute_iou(b1, b2)
        assert torch.equal(iou, torch.tensor([[0.0]]))

    # -- degenerate boxes ---------------------------------------------------

    def test_degenerate_zero_area_box1(self) -> None:
        b1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])  # zero-width
        b2 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        iou = compute_iou(b1, b2)
        assert torch.equal(iou, torch.tensor([[0.0]]))

    def test_degenerate_zero_area_box2(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        b2 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        iou = compute_iou(b1, b2)
        assert torch.equal(iou, torch.tensor([[0.0]]))

    def test_degenerate_both_zero_area(self) -> None:
        b1 = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        b2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        iou = compute_iou(b1, b2)
        assert torch.equal(iou, torch.tensor([[0.0]]))

    def test_degenerate_box2_lt_box1(self) -> None:
        """x2 < x1 for a box — should still compute area=0 cleanly."""
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        b2 = torch.tensor([[1.0, 1.0, 0.5, 0.5]])  # invalid: x2 < x1
        iou = compute_iou(b1, b2)
        # b2 area should be clamped to 0
        assert torch.equal(iou, torch.tensor([[0.0]]))

    def test_degenerate_zero_width(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # w=0
        b2 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        iou = compute_iou(b1, b2)
        assert torch.equal(iou, torch.tensor([[0.0]]))

    # -- matrix shapes ------------------------------------------------------

    def test_n_m_matrix(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])  # (2,4)
        b2 = torch.tensor([[0.5, 0.5, 1.5, 1.5], [2.0, 2.0, 4.0, 4.0], [0.0, 0.0, 1.0, 1.0]])  # (3,4)
        iou = compute_iou(b1, b2)
        assert iou.shape == (2, 3)

    def test_single_box_each(self) -> None:
        """Flat (4,) tensors should be handled gracefully."""
        b1 = torch.tensor([0.0, 0.0, 2.0, 2.0])
        b2 = torch.tensor([1.0, 1.0, 3.0, 3.0])
        iou = compute_iou(b1, b2)
        assert iou.shape == (1, 1)
        assert torch.allclose(iou, torch.tensor([[1.0 / 7.0]]), atol=1e-4)

    def test_single_vs_batch(self) -> None:
        b1 = torch.tensor([0.0, 0.0, 2.0, 2.0])  # (4,)
        b2 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])  # (2,4)
        iou = compute_iou(b1, b2)
        assert iou.shape == (1, 2)

    # -- correctness --------------------------------------------------------

    def test_symmetric(self) -> None:
        """IoU matrix should be approximately symmetric for square sets."""
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
        b2 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
        iou = compute_iou(b1, b2)
        assert torch.allclose(iou, iou.T, atol=1e-6)

    def test_range(self) -> None:
        """All IoU values must be in [0, 1]."""
        b1 = torch.rand(50, 4) * 2
        b2 = torch.rand(30, 4) * 2
        iou = compute_iou(b1, b2)
        assert (iou >= 0.0).all()
        assert (iou <= 1.0).all()


# ---------------------------------------------------------------------------
# apply_delta
# ---------------------------------------------------------------------------


class TestApplyDelta:
    def test_add_delta(self) -> None:
        boxes = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        deltas = torch.tensor([[0.0, 0.1, -0.05, 0.05]])
        result = apply_delta(boxes, deltas)
        expected = torch.tensor([[0.5, 0.6, 0.15, 0.25]])
        assert torch.allclose(result, expected)

    def test_no_op(self) -> None:
        boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        deltas = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        result = apply_delta(boxes, deltas)
        assert torch.equal(result, boxes)

    def test_batch(self) -> None:
        boxes = torch.rand(10, 4)
        deltas = torch.randn(10, 4) * 0.01
        result = apply_delta(boxes, deltas)
        assert result.shape == (10, 4)
        assert torch.allclose(result, boxes + deltas)

    def test_broadcast_deltas(self) -> None:
        boxes = torch.rand(5, 4)
        delta = torch.tensor([0.01, 0.02, 0.03, 0.04])
        result = apply_delta(boxes, delta)
        assert result.shape == (5, 4)
        # each row gets the same delta
        for i in range(5):
            assert torch.allclose(result[i], boxes[i] + delta)


# ---------------------------------------------------------------------------
# compute_center_distance
# ---------------------------------------------------------------------------


class TestComputeCenterDistance:
    def test_same_box_zero_distance(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        b2 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        dist = compute_center_distance(b1, b2)
        assert torch.allclose(dist, torch.tensor([[0.0]]))

    def test_different_centers(self) -> None:
        b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])  # centre (1,1)
        b2 = torch.tensor([[3.0, 3.0, 5.0, 5.0]])  # centre (4,4)
        dist = compute_center_distance(b1, b2)
        # sqrt((4-1)^2 + (4-1)^2) = sqrt(18) ≈ 4.2426
        assert torch.allclose(dist, torch.tensor([[18.0 ** 0.5]]), atol=1e-4)

    def test_matrix_shape(self) -> None:
        b1 = torch.rand(3, 4) * 10
        b2 = torch.rand(5, 4) * 10
        dist = compute_center_distance(b1, b2)
        assert dist.shape == (3, 5)

    def test_single_flat_tensor(self) -> None:
        b1 = torch.tensor([0.0, 0.0, 2.0, 2.0])
        b2 = torch.tensor([3.0, 3.0, 5.0, 5.0])
        dist = compute_center_distance(b1, b2)
        assert dist.shape == (1, 1)

    def test_non_negative(self) -> None:
        b1 = torch.rand(10, 4) * 100
        b2 = torch.rand(10, 4) * 100
        dist = compute_center_distance(b1, b2)
        assert (dist >= 0.0).all()


# ---------------------------------------------------------------------------
# clamp_coords
# ---------------------------------------------------------------------------


class TestClampCoords:
    def test_default_range(self) -> None:
        boxes = torch.tensor([[-0.5, 0.2, 1.5, 0.8]])
        result = clamp_coords(boxes)
        expected = torch.tensor([[0.0, 0.2, 1.0, 0.8]])
        assert torch.equal(result, expected)

    def test_custom_range(self) -> None:
        boxes = torch.tensor([[-5.0, 10.0, 0.0, 100.0]])
        result = clamp_coords(boxes, min_val=-1.0, max_val=50.0)
        expected = torch.tensor([[-1.0, 10.0, 0.0, 50.0]])
        assert torch.equal(result, expected)

    def test_all_inside_no_change(self) -> None:
        boxes = torch.tensor([[0.1, 0.2, 0.8, 0.9]])
        result = clamp_coords(boxes)
        assert torch.equal(result, boxes)

    def test_batch(self) -> None:
        boxes = torch.tensor([
            [-0.5, 0.2, 0.8, 1.5],
            [0.1, -2.0, 0.9, 1.1],
            [0.3, 0.4, 0.6, 5.0],
        ])
        result = clamp_coords(boxes)
        expected = torch.tensor([
            [0.0, 0.2, 0.8, 1.0],
            [0.1, 0.0, 0.9, 1.0],
            [0.3, 0.4, 0.6, 1.0],
        ])
        assert torch.equal(result, expected)

    def test_xywh_format(self) -> None:
        """clamp_coords works on any format since it's element-wise."""
        boxes = torch.tensor([[-0.5, 0.2, -0.1, 1.5]])  # cx, cy, w, h
        result = clamp_coords(boxes)
        expected = torch.tensor([[0.0, 0.2, 0.0, 1.0]])
        assert torch.equal(result, expected)

    def test_preserves_dtype(self) -> None:
        boxes = torch.tensor([[0.1, 0.2, 0.8, 0.9]], dtype=torch.float64)
        result = clamp_coords(boxes)
        assert result.dtype == torch.float64


# ---------------------------------------------------------------------------
# Integration / round-trip scenarios
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end scenarios combining multiple bbox utilities."""

    def test_xywh_delta_clamp_flow(self) -> None:
        """Simulate a VLM refinement: xywh box → apply delta → clamp."""
        # Original VLM output (centre-based xywh)
        vlm_box = torch.tensor([[0.55, 0.45, 0.2, 0.3]])  # slightly off
        # GNN-predicted deltas
        delta = torch.tensor([[0.0, 0.05, -0.02, 0.0]])
        refined = apply_delta(vlm_box, delta)
        # Clamp to valid range
        corrected = clamp_coords(refined)
        # w should not go negative
        assert (corrected[..., 2] >= 0.0).all()
        assert (corrected[..., 3] >= 0.0).all()

    def test_xywh_to_xyxy_and_iou(self) -> None:
        """Convert centre-based xywh to xyxy, then compute IoU."""
        pred_xywh = torch.tensor([[0.5, 0.5, 0.2, 0.2]])  # cx=0.5, cy=0.5, w=h=0.2
        gt_xywh = torch.tensor([[0.5, 0.5, 0.4, 0.4]])     # same centre, twice as big

        pred_xyxy = xywh_to_xyxy(pred_xywh)  # (0.4,0.4,0.6,0.6)
        gt_xyxy = xywh_to_xyxy(gt_xywh)      # (0.3,0.3,0.7,0.7)

        iou = compute_iou(pred_xyxy, gt_xyxy)
        # pred area = 0.04, gt area = 0.16, intersection = 0.04
        # IoU = 0.04 / 0.16 = 0.25
        assert torch.allclose(iou, torch.tensor([[0.25]]), atol=1e-4)

    def test_center_distance_from_xywh(self) -> None:
        """Compute centre distances from centre-based xywh boxes."""
        orig = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
        other = torch.tensor([[0.7, 0.5, 0.2, 0.2]])

        orig_xyxy = xywh_to_xyxy(orig)
        other_xyxy = xywh_to_xyxy(other)

        dist = compute_center_distance(orig_xyxy, other_xyxy)
        # Distance between centres (0.5,0.5) and (0.7,0.5) = 0.2
        assert torch.allclose(dist, torch.tensor([[0.2]]), atol=1e-4)


# ---------------------------------------------------------------------------
# Additional edge-case / smoke tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_iou_extreme_values(self) -> None:
        """Very large and very small coordinates should not cause NaNs."""
        b1 = torch.tensor([[0.0, 0.0, 1e-6, 1e-6]])
        b2 = torch.tensor([[0.0, 0.0, 1e6, 1e6]])
        iou = compute_iou(b1, b2)
        assert not torch.isnan(iou).any()
        assert (iou >= 0.0).all()

    def test_center_distance_same_flat(self) -> None:
        """Two identical flat boxes → centre distance zero."""
        box = torch.tensor([0.0, 0.0, 10.0, 10.0])
        dist = compute_center_distance(box, box)
        assert torch.allclose(dist, torch.tensor([[0.0]]))

    def test_clamp_coords_no_op_when_inside(self) -> None:
        """Box values already in [0,1] are not changed."""
        boxes = torch.rand(100, 4).clamp(0.01, 0.99)
        result = clamp_coords(boxes)
        assert torch.equal(result, boxes)

    def test_xywh_conversion_with_3d_tensor(self) -> None:
        """Test (batch, num_elements, 4) shape."""
        boxes = torch.rand(2, 5, 4)
        xyxy = xywh_to_xyxy(boxes)
        assert xyxy.shape == (2, 5, 4)
        back = xyxy_to_xywh(xyxy)
        assert torch.allclose(back, boxes, atol=1e-6)
