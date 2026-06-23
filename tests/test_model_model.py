"""Tests for BipartiteGNNCorrector — end-to-end model."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.schema import (
    ConstraintNode,
    ConstraintType,
    ElementNode,
)
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector


def _elem(x1, y1, x2, y2, confidence=1.0) -> ElementNode:
    return ElementNode(bbox=[x1, y1, x2, y2], confidence=confidence)


def _con(ctype, source, target=None, **params) -> ConstraintNode:
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=source,
        target_indices=target or source,
        params=params,
    )


def _build_test_graph(n_elem=4, n_con=2):
    elements = [
        _elem(0.1 + i * 0.05, 0.1 + i * 0.05, 0.3 + i * 0.05, 0.3 + i * 0.05)
        for i in range(n_elem)
    ]
    constraints = [
        _con(ConstraintType.ALIGN_LEFT,
             [i % n_elem, (i + 1) % n_elem], tolerance=0.02)
        for i in range(n_con)
    ]
    builder = BipartiteGraphBuilder()
    return builder.build(elements, constraints)


@pytest.fixture
def model():
    return BipartiteGNNCorrector(
        element_dim=5, constraint_dim=11, hidden_dim=64,
        num_layers=2, dropout=0.0,
    )


@pytest.fixture
def graph():
    return _build_test_graph()


class TestForwardPass:
    """Shape and content checks for the forward pass."""

    def test_output_keys(self, model, graph) -> None:
        out = model(graph)
        assert "coord" in out
        assert "violation" in out
        assert "existence" in out

    def test_output_shapes(self, model, graph) -> None:
        out = model(graph)
        n_elem = graph["element"].x.shape[0]
        n_con = graph["constraint"].x.shape[0]
        assert out["coord"].shape == (n_elem, 4)
        assert out["violation"].shape == (n_con, 1)
        assert out["existence"].shape == (n_elem, 1)

    def test_violation_in_range(self, model, graph) -> None:
        out = model(graph)
        assert out["violation"].min() >= 0.0
        assert out["violation"].max() <= 1.0

    def test_existence_in_range(self, model, graph) -> None:
        out = model(graph)
        assert out["existence"].min() >= 0.0
        assert out["existence"].max() <= 1.0

    def test_params_require_grad(self, model) -> None:
        for name, param in model.named_parameters():
            assert param.requires_grad, f"{name} requires_grad is False"

    def test_empty_input(self) -> None:
        model = BipartiteGNNCorrector(
            element_dim=5, constraint_dim=11, hidden_dim=64,
            num_layers=1, dropout=0.0,
        )
        builder = BipartiteGraphBuilder()
        data = builder.build([], [])
        out = model(data)
        assert "coord" in out or out["coord"].numel() == 0


class TestComputeLoss:
    """Loss computation on the model."""

    def test_returns_scalar(self, model, graph) -> None:
        pred = model(graph)
        targets = {
            "coord": torch.randn(4, 4),
            "violation": torch.randint(0, 2, (2, 1)).float(),
            "existence": torch.randint(0, 2, (4, 1)).float(),
        }
        loss = model.compute_loss(pred, targets)
        assert loss.dim() == 0

    def test_loss_backward(self, model, graph) -> None:
        pred = model(graph)
        targets = {
            "coord": torch.randn(4, 4),
            "violation": torch.randint(0, 2, (2, 1)).float(),
            "existence": torch.randint(0, 2, (4, 1)).float(),
            # Add mask + proposal targets so all heads get gradients.
            "mask_completion_target": torch.randn(4, 5),
            "mask_completion_mask": torch.tensor([True, True, False, False]),
            "proposal_target": torch.randn(2, 4),
            "proposal_violation_mask": torch.tensor([True, False]),
        }
        model.mask_weight = 1.0
        model.proposal_weight = 1.0
        loss = model.compute_loss(pred, targets)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"


class TestTrainStep:
    """Single training step with optimizer."""

    def test_train_step_returns_loss(self, model, graph) -> None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        targets = {
            "coord": torch.randn(4, 4),
            "violation": torch.randint(0, 2, (2, 1)).float(),
            "existence": torch.randint(0, 2, (4, 1)).float(),
        }
        loss = model.train_step(graph, targets, optimizer, grad_clip=1.0)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_validation_step_returns_loss(self, model, graph) -> None:
        targets = {
            "coord": torch.randn(4, 4),
            "violation": torch.randint(0, 2, (2, 1)).float(),
            "existence": torch.randint(0, 2, (4, 1)).float(),
        }
        loss = model.validation_step(graph, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0.0
