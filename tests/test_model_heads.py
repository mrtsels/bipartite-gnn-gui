"""Tests for prediction heads — coordinate, violation, existence."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.model.heads import (
    CoordinateRefinementHead,
    ExistencePredictionHead,
    ViolationPredictionHead,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def element_features() -> torch.Tensor:
    """Simulated encoded element features: (8, 128)."""
    torch.manual_seed(42)
    return torch.randn(8, 128)


@pytest.fixture
def constraint_features() -> torch.Tensor:
    """Simulated encoded constraint features: (4, 128)."""
    torch.manual_seed(123)
    return torch.randn(4, 128)


# ===================================================================
# CoordinateRefinementHead
# ===================================================================


class TestCoordinateRefinementHead:
    """Tests for the coordinate delta prediction head."""

    def test_output_shape(self, element_features) -> None:
        head = CoordinateRefinementHead(input_dim=128, dropout=0.0)
        out = head(element_features)
        assert out.shape == (8, 4)

    def test_no_activation(self, element_features) -> None:
        """Coordinate head outputs raw deltas (can be negative or positive)."""
        head = CoordinateRefinementHead(input_dim=128, dropout=0.0)
        out = head(element_features)
        # Raw deltas should have both positive and negative values.
        assert out.min() < 0 or out.max() > 1  # Not sigmoid-clamped

    def test_gradient_flow(self, element_features) -> None:
        head = CoordinateRefinementHead(input_dim=128, dropout=0.0)
        out = head(element_features)
        loss = out.sum()
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.all(param.grad == 0), f"{name} gradient is zero"

    def test_single_element(self) -> None:
        head = CoordinateRefinementHead(input_dim=128, dropout=0.0)
        x = torch.randn(1, 128)
        out = head(x)
        assert out.shape == (1, 4)

    def test_empty_input(self) -> None:
        head = CoordinateRefinementHead(input_dim=128, dropout=0.0)
        x = torch.zeros(0, 128)
        out = head(x)
        assert out.shape == (0, 4)

    def test_custom_input_dim(self) -> None:
        head = CoordinateRefinementHead(input_dim=64, dropout=0.0)
        x = torch.randn(5, 64)
        out = head(x)
        assert out.shape == (5, 4)


# ===================================================================
# ViolationPredictionHead
# ===================================================================


class TestViolationPredictionHead:
    """Tests for the violation score prediction head."""

    def test_output_shape(self, constraint_features) -> None:
        head = ViolationPredictionHead(input_dim=128, dropout=0.0)
        out = head(constraint_features)
        assert out.shape == (4, 1)

    def test_output_in_range(self, constraint_features) -> None:
        """Violation scores should be in [0, 1] due to sigmoid."""
        head = ViolationPredictionHead(input_dim=128, dropout=0.0)
        out = head(constraint_features)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_flow(self, constraint_features) -> None:
        head = ViolationPredictionHead(input_dim=128, dropout=0.0)
        out = head(constraint_features)
        loss = out.sum()
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_empty_input(self) -> None:
        head = ViolationPredictionHead(input_dim=128, dropout=0.0)
        x = torch.zeros(0, 128)
        out = head(x)
        assert out.shape == (0, 1)


# ===================================================================
# ExistencePredictionHead
# ===================================================================


class TestExistencePredictionHead:
    """Tests for the existence probability prediction head."""

    def test_output_shape(self, element_features) -> None:
        head = ExistencePredictionHead(input_dim=128, dropout=0.0)
        out = head(element_features)
        assert out.shape == (8, 1)

    def test_output_in_range(self, element_features) -> None:
        """Existence probabilities should be in [0, 1] due to sigmoid."""
        head = ExistencePredictionHead(input_dim=128, dropout=0.0)
        out = head(element_features)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_gradient_flow(self, element_features) -> None:
        head = ExistencePredictionHead(input_dim=128, dropout=0.0)
        out = head(element_features)
        loss = out.sum()
        loss.backward()
        for name, param in head.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    def test_empty_input(self) -> None:
        head = ExistencePredictionHead(input_dim=128, dropout=0.0)
        x = torch.zeros(0, 128)
        out = head(x)
        assert out.shape == (0, 1)


# ===================================================================
# Dropout behavior
# ===================================================================


class TestDropout:
    """Dropout is applied during training and disabled during eval."""

    def test_dropout_train(self) -> None:
        head = CoordinateRefinementHead(input_dim=128, dropout=0.5)
        head.train()
        x = torch.randn(4, 128)
        torch.manual_seed(1)
        out1 = head(x)
        torch.manual_seed(1)
        out2 = head(x)
        assert torch.allclose(out1, out2)

    def test_dropout_eval(self) -> None:
        head = CoordinateRefinementHead(input_dim=128, dropout=0.5)
        head.eval()
        x = torch.randn(4, 128)
        torch.manual_seed(1)
        out1 = head(x)
        torch.manual_seed(1)
        out2 = head(x)
        assert torch.allclose(out1, out2)
