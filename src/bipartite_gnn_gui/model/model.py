"""End-to-end GUI correction model."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, torch

from .encoder import BipartiteGraphSAGE
from .heads import CoordinateRefinementHead, ExistencePredictionHead, ViolationPredictionHead


class BipartiteGNNCorrector(nn.Module):
    """Combine encoder and prediction heads."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = BipartiteGraphSAGE(input_dim=input_dim, hidden_dim=hidden_dim)
        self.coordinate_head = CoordinateRefinementHead(input_dim=hidden_dim)
        self.violation_head = ViolationPredictionHead(input_dim=hidden_dim)
        self.existence_head = ExistencePredictionHead(input_dim=hidden_dim)

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        """Run the full correction pipeline."""

        encoded = self.encoder(data)
        outputs: dict[str, torch.Tensor] = {}
        if "element" in encoded:
            outputs["coord"] = self.coordinate_head(encoded["element"])
            outputs["existence"] = self.existence_head(encoded["element"])
        if "constraint" in encoded:
            outputs["violation"] = self.violation_head(encoded["constraint"])
        return outputs
