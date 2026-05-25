"""Bipartite encoder used by the GUI corrector."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, torch


class BipartiteGraphSAGE(nn.Module):
    """Small feed-forward stand-in for the planned GraphSAGE encoder."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128, output_dim: int | None = None, num_layers: int = 2) -> None:
        super().__init__()
        output_dim = hidden_dim if output_dim is None else output_dim
        total_layers = max(num_layers, 1)

        def _build_encoder() -> nn.Sequential:
            layers: list[nn.Module] = []
            cur_dim = input_dim
            for layer_index in range(total_layers):
                next_dim = output_dim if layer_index == total_layers - 1 else hidden_dim
                layers.append(nn.Linear(cur_dim, next_dim))
                if layer_index < total_layers - 1:
                    layers.append(nn.ReLU())
                cur_dim = next_dim
            return nn.Sequential(*layers)

        self.element_encoder = _build_encoder()
        self.constraint_encoder = _build_encoder()

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        """Encode element and constraint node features."""

        if hasattr(data, "x_dict"):
            element_x = data.x_dict.get("element")
            constraint_x = data.x_dict.get("constraint")
        else:
            element_x = data["element"].x if "element" in data else None
            constraint_x = data["constraint"].x if "constraint" in data else None

        encoded: dict[str, torch.Tensor] = {}
        if element_x is not None:
            encoded["element"] = self.element_encoder(element_x.float())
        if constraint_x is not None:
            encoded["constraint"] = self.constraint_encoder(constraint_x.float())
        return encoded
