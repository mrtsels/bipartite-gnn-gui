"""Tests for loss functions — coord, violation, existence, alignment, CombinedLoss."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.model.losses import (
    CombinedLoss,
    BipartiteGNNLoss,
    compute_alignment_consistency_loss,
    compute_coord_loss,
    compute_existence_loss,
    compute_violation_loss,
)


# ===================================================================
# coord loss
# ===================================================================


class TestCoordLoss:
    def test_scalar_output(self) -> None:
        pred = torch.randn(8, 4)
        target = torch.randn(8, 4)
        loss = compute_coord_loss(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_perfect_match_is_zero(self) -> None:
        pred = torch.ones(4, 4)
        target = torch.ones(4, 4)
        loss = compute_coord_loss(pred, target)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_larger_error_gives_larger_loss(self) -> None:
        pred_close = torch.zeros(4, 4)
        target = torch.zeros(4, 4)
        target[0, 0] = 0.1
        loss1 = compute_coord_loss(pred_close, target)

        target[0, 0] = 1.0
        loss2 = compute_coord_loss(pred_close, target)
        assert loss2 > loss1

    def test_empty_input(self) -> None:
        pred = torch.zeros(0, 4)
        target = torch.zeros(0, 4)
        # Empty inputs should not raise (NaN is acceptable for zero-size tensors).
        try:
            compute_coord_loss(pred, target)
        except Exception:
            pytest.fail("compute_coord_loss raised on empty input")


# ===================================================================
# violation loss
# ===================================================================


class TestViolationLoss:
    def test_scalar_output(self) -> None:
        pred = torch.sigmoid(torch.randn(6, 1))
        target = torch.randint(0, 2, (6, 1)).float()
        loss = compute_violation_loss(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_perfect_prediction_low_loss(self) -> None:
        pred = torch.tensor([[0.01], [0.99]])
        target = torch.tensor([[0.0], [1.0]])
        loss = compute_violation_loss(pred, target)
        assert loss.item() < 0.1

    def test_empty_input(self) -> None:
        pred = torch.zeros(0, 1)
        target = torch.zeros(0, 1)
        # Empty inputs should not raise.
        try:
            compute_violation_loss(pred, target)
        except Exception:
            pytest.fail("compute_violation_loss raised on empty input")


# ===================================================================
# existence loss
# ===================================================================


class TestExistenceLoss:
    def test_scalar_output(self) -> None:
        pred = torch.sigmoid(torch.randn(10, 1))
        target = torch.randint(0, 2, (10, 1)).float()
        loss = compute_existence_loss(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_empty_input(self) -> None:
        pred = torch.zeros(0, 1)
        target = torch.zeros(0, 1)
        # Empty inputs should not raise.
        try:
            compute_existence_loss(pred, target)
        except Exception:
            pytest.fail("compute_existence_loss raised on empty input")


# ===================================================================
# alignment consistency loss
# ===================================================================


class TestAlignmentConsistencyLoss:
    def test_aligned_elements_zero_loss(self) -> None:
        """Perfectly aligned elements should yield near-zero loss."""
        deltas = torch.zeros(4, 4)
        bboxes = torch.tensor([
            [0.25, 0.5, 0.5, 0.4],  # cx=0.25, cy=0.5, w=0.5, h=0.4 → x1=0.0
            [0.55, 0.5, 0.5, 0.4],  # cx=0.55, cy=0.5, w=0.5, h=0.4 → x1=0.3
            [0.25, 0.2, 0.5, 0.4],  # cx=0.25, cy=0.2, w=0.5, h=0.4 → x1=0.0
            [0.55, 0.2, 0.5, 0.4],  # cx=0.55, cy=0.2, w=0.5, h=0.4 → x1=0.3
        ])
        # Constraint with elements 0 and 2 (both x1=0.0 = ALIGN_LEFT).
        edge_index = torch.tensor([[0, 2], [0, 0]], dtype=torch.long)
        loss = compute_alignment_consistency_loss(
            deltas, bboxes, edge_index, constraint_type="align_left",
        )
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_misaligned_elements_positive_loss(self) -> None:
        """Misaligned elements should give positive loss after misaligned deltas."""
        deltas = torch.zeros(4, 4)
        bboxes = torch.tensor([
            [0.2, 0.5, 0.5, 0.4],   # x1 = 0.2 - 0.25 = -0.05 → 0
            [0.55, 0.5, 0.5, 0.4],  # x1 = 0.55 - 0.25 = 0.3
            [0.3, 0.5, 0.5, 0.4],   # x1 = 0.3 - 0.25 = 0.05
            [0.55, 0.2, 0.5, 0.4],  # x1 = 0.55 - 0.25 = 0.3
        ])
        edge_index = torch.tensor([[0, 2], [0, 0]], dtype=torch.long)
        loss = compute_alignment_consistency_loss(
            deltas, bboxes, edge_index, constraint_type="align_left",
        )
        assert loss.item() > 0.0

    def test_empty_deltas(self) -> None:
        loss = compute_alignment_consistency_loss(
            torch.zeros(0, 4), torch.zeros(0, 4),
            torch.zeros(2, 0, dtype=torch.long),
        )
        assert loss.item() == 0.0

    def test_empty_edge_index(self) -> None:
        deltas = torch.zeros(4, 4)
        bboxes = torch.zeros(4, 4)
        loss = compute_alignment_consistency_loss(
            deltas, bboxes, torch.zeros(2, 0, dtype=torch.long),
        )
        assert loss.item() == 0.0

    def test_unknown_constraint_type_returns_zero(self) -> None:
        deltas = torch.randn(4, 4)
        bboxes = torch.randn(4, 4)
        edge_index = torch.tensor([[0, 1], [0, 0]], dtype=torch.long)
        loss = compute_alignment_consistency_loss(
            deltas, bboxes, edge_index, constraint_type="unknown",
        )
        assert loss.item() == 0.0

    def test_single_element_constraint(self) -> None:
        """Single element in constraint should give zero loss (no pair to compare)."""
        deltas = torch.zeros(2, 4)
        bboxes = torch.zeros(2, 4)
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        loss = compute_alignment_consistency_loss(
            deltas, bboxes, edge_index,
        )
        assert loss.item() == 0.0


# ===================================================================
# CombinedLoss
# ===================================================================


class TestCombinedLoss:
    def test_all_components(self) -> None:
        loss_fn = CombinedLoss()
        pred = {
            "coord": torch.randn(8, 4),
            "violation": torch.sigmoid(torch.randn(5, 1)),
            "existence": torch.sigmoid(torch.randn(8, 1)),
        }
        target = {
            "coord": torch.randn(8, 4),
            "violation": torch.randint(0, 2, (5, 1)).float(),
            "existence": torch.randint(0, 2, (8, 1)).float(),
        }
        loss = loss_fn(pred, target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_missing_components_skipped(self) -> None:
        """Missing keys in prediction or target are skipped gracefully."""
        loss_fn = CombinedLoss()
        pred = {"coord": torch.randn(8, 4)}
        target = {"coord": torch.randn(8, 4)}
        loss = loss_fn(pred, target)
        assert loss.dim() == 0

    def test_empty_pred_target(self) -> None:
        loss_fn = CombinedLoss()
        loss = loss_fn({}, {})
        assert loss.item() == 0.0

    def test_custom_weights(self) -> None:
        loss1 = CombinedLoss(coord_weight=10.0, violation_weight=0.1)
        loss2 = CombinedLoss(coord_weight=1.0, violation_weight=1.0)
        pred = {
            "coord": torch.randn(4, 4),
            "violation": torch.sigmoid(torch.randn(2, 1)),
        }
        target = {
            "coord": torch.randn(4, 4),
            "violation": torch.randint(0, 2, (2, 1)).float(),
        }
        # Higher coord weight should give larger loss.
        assert loss1(pred, target) > loss2(pred, target)

    def test_with_alignment_consistency(self) -> None:
        loss_fn = CombinedLoss(alignment_weight=0.5)
        pred = {
            "coord": torch.randn(4, 4),
        }
        target = {
            "coord": torch.randn(4, 4),
        }
        bboxes = torch.randn(4, 4)
        edge_index = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
        loss = loss_fn(pred, target, original_bboxes=bboxes,
                        edge_index=edge_index)
        assert loss.dim() == 0

    def test_gradient_flow(self) -> None:
        loss_fn = CombinedLoss()
        pred = {"coord": torch.randn(4, 4, requires_grad=True)}
        target = {"coord": torch.randn(4, 4)}
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred["coord"].grad is not None

    def test_bipartite_gnn_loss_alias(self) -> None:
        """BipartiteGNNLoss is an alias for CombinedLoss."""
        assert BipartiteGNNLoss is CombinedLoss
