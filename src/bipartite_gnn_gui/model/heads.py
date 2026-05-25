"""Prediction heads used by the corrector."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, torch


class _MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CoordinateRefinementHead(_MLPHead):
    """Predict coordinate deltas."""

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__(input_dim=input_dim, output_dim=4)


class ViolationPredictionHead(_MLPHead):
    """Predict violation scores."""

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__(input_dim=input_dim, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))


class ExistencePredictionHead(_MLPHead):
    """Predict element existence probabilities."""

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__(input_dim=input_dim, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))
