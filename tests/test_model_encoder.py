"""Tests for BipartiteGraphSAGE — heterogeneous SAGEConv encoder."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.schema import (
    ConstraintNode,
    ConstraintType,
    ElementNode,
)
from bipartite_gnn_gui.model.encoder import BipartiteGraphSAGE


# ===================================================================
# Helpers
# ===================================================================


def _elem(
    x1: float, y1: float, x2: float, y2: float, confidence: float = 1.0,
) -> ElementNode:
    return ElementNode(bbox=[x1, y1, x2, y2], confidence=confidence)


def _con(
    ctype: ConstraintType,
    source: list[int],
    target: list[int] | None = None,
    **params: float,
) -> ConstraintNode:
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=source,
        target_indices=target or source,
        params=params,
    )


def _build_graph(
    num_elements: int = 4,
    num_constraints: int = 2,
):
    elements = [
        _elem(0.1 + i * 0.05, 0.1 + i * 0.05, 0.3 + i * 0.05, 0.3 + i * 0.05)
        for i in range(num_elements)
    ]
    constraints = [
        _con(ConstraintType.ALIGN_LEFT, [i % num_elements, (i + 1) % num_elements],
             tolerance=0.02)
        for i in range(num_constraints)
    ]
    builder = BipartiteGraphBuilder()
    data = builder.build(elements, constraints)
    return data, elements, constraints


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def small_graph():
    data, _, _ = _build_graph()
    return data


@pytest.fixture
def encoder():
    return BipartiteGraphSAGE(
        element_dim=5,
        constraint_dim=11,
        hidden_dim=64,
        num_layers=2,
        dropout=0.0,
    )


# ===================================================================
# Forward pass shapes
# ===================================================================


class TestForwardShapes:
    """Verify output tensor shapes from forward pass."""

    def test_element_shape(self, encoder, small_graph) -> None:
        output = encoder(small_graph)
        num_elements = small_graph["element"].x.shape[0]
        assert output["element"].shape == (num_elements, 64)

    def test_constraint_shape(self, encoder, small_graph) -> None:
        output = encoder(small_graph)
        num_constraints = small_graph["constraint"].x.shape[0]
        assert output["constraint"].shape == (num_constraints, 64)

    def test_output_keys(self, encoder, small_graph) -> None:
        output = encoder(small_graph)
        assert set(output.keys()) == {"element", "constraint"}

    def test_output_dtype(self, encoder, small_graph) -> None:
        output = encoder(small_graph)
        assert output["element"].dtype == torch.float32
        assert output["constraint"].dtype == torch.float32

    def test_single_element(self) -> None:
        elements = [_elem(0.1, 0.2, 0.5, 0.8)]
        constraints = [_con(ConstraintType.CENTER_X, [0], tolerance=0.02)]
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2, dropout=0.0)
        output = enc(data)
        assert output["element"].shape == (1, 64)
        assert output["constraint"].shape == (1, 64)


# ===================================================================
# Gradient flow
# ===================================================================


class TestGradientFlow:
    """Ensure gradients propagate through the encoder."""

    def test_parameters_require_grad(self, encoder) -> None:
        for name, param in encoder.named_parameters():
            assert param.requires_grad, f"Parameter {name} does not require grad"

    def test_loss_backward(self, encoder, small_graph) -> None:
        output = encoder(small_graph)
        loss = output["element"].sum() + output["constraint"].sum()
        loss.backward()
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
            assert not torch.all(param.grad == 0), (
                f"Parameter {name} gradient is all zeros"
            )

    def test_gradient_updates_weights(self, encoder, small_graph) -> None:
        original_params = {
            name: param.clone() for name, param in encoder.named_parameters()
        }
        output = encoder(small_graph)
        loss = output["element"].sum()
        loss.backward()
        with torch.no_grad():
            for param in encoder.parameters():
                g = param.grad if param.grad is not None else torch.zeros_like(param)
                param -= 0.01 * g
        for name, param in encoder.named_parameters():
            assert not torch.allclose(
                param, original_params[name]
            ), f"Parameter {name} was not updated"


# ===================================================================
# Empty / edge cases
# ===================================================================


class TestEmptyInputs:
    """Empty element or constraint nodes."""

    def test_no_constraints(self) -> None:
        elements = [_elem(0.0, 0.0, 0.5, 0.5)]
        constraints: list[ConstraintNode] = []
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2, dropout=0.0)
        output = enc(data)
        assert output["element"].shape == (1, 64)
        assert output["constraint"].shape == (0, 64)

    def test_no_elements(self) -> None:
        elements: list[ElementNode] = []
        constraints = [_con(ConstraintType.ALIGN_LEFT, [0], tolerance=0.02)]
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2, dropout=0.0)
        output = enc(data)
        assert output["element"].shape == (0, 64)
        assert output["constraint"].shape == (1, 64)

    def test_both_empty(self) -> None:
        builder = BipartiteGraphBuilder()
        data = builder.build([], [])
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2, dropout=0.0)
        output = enc(data)
        assert output["element"].shape == (0, 64)
        assert output["constraint"].shape == (0, 64)

    def test_empty_no_edges(self) -> None:
        builder = BipartiteGraphBuilder()
        data = builder.build([], [])
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2, dropout=0.0)
        output = enc(data)
        assert output["element"].numel() == 0
        assert output["constraint"].numel() == 0


# ===================================================================
# Reset parameters
# ===================================================================


class TestResetParameters:
    """Verify reset_parameters() produces non-degenerate values."""

    def test_reset_changes_params(self) -> None:
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2, dropout=0.0)
        orig_weight = enc.element_proj.weight.clone()
        orig_bias = enc.element_proj.bias.clone()

        enc.reset_parameters()

        assert not torch.equal(enc.element_proj.weight, orig_weight)
        assert not torch.equal(enc.element_proj.bias, orig_bias)

    def test_reset_deterministic(self) -> None:
        torch.manual_seed(42)
        enc1 = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                   hidden_dim=64, num_layers=2, dropout=0.0)
        torch.manual_seed(42)
        enc2 = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                   hidden_dim=64, num_layers=2, dropout=0.0)
        for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
            assert torch.equal(p1, p2)


# ===================================================================
# Configurable dimensions
# ===================================================================


class TestConfigurableDims:
    """Verify that input dims, hidden dim, and num_layers are respected."""

    def test_custom_hidden_dim(self, small_graph) -> None:
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=256, num_layers=2,
                                  dropout=0.0)
        output = enc(small_graph)
        assert output["element"].shape[-1] == 256
        assert output["constraint"].shape[-1] == 256

    def test_single_layer(self, small_graph) -> None:
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=1,
                                  dropout=0.0)
        output = enc(small_graph)
        assert output["element"].shape[-1] == 64
        assert output["constraint"].shape[-1] == 64


# ===================================================================
# Dropout behavior
# ===================================================================


class TestDropout:
    """Dropout is applied during training, disabled during eval."""

    def test_dropout_same_seed_consistent(self) -> None:
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2,
                                  dropout=0.5)
        data, _, _ = _build_graph()
        enc.train()
        torch.manual_seed(1)
        out1 = enc(data)
        torch.manual_seed(1)
        out2 = enc(data)
        assert torch.allclose(out1["element"], out2["element"])

    def test_dropout_eval_deterministic(self) -> None:
        enc = BipartiteGraphSAGE(element_dim=5, constraint_dim=11,
                                  hidden_dim=64, num_layers=2,
                                  dropout=0.5)
        data, _, _ = _build_graph()
        enc.eval()
        torch.manual_seed(1)
        out1 = enc(data)
        torch.manual_seed(1)
        out2 = enc(data)
        assert torch.allclose(out1["element"], out2["element"])
