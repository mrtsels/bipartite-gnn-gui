"""Loss helpers for GUI correction training."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import F, Tensor, torch


def compute_coord_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on coordinate deltas."""

    return F.mse_loss(prediction, target)


def compute_violation_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary cross entropy on violation scores."""

    return F.binary_cross_entropy(prediction, target)


def compute_existence_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary cross entropy on existence probabilities."""

    return F.binary_cross_entropy(prediction, target)


@dataclass
class BipartiteGNNLoss:
    """Weighted combination of the model losses."""

    coord_weight: float = 1.0
    violation_weight: float = 1.0
    existence_weight: float = 1.0

    def __call__(self, prediction: dict[str, Tensor], target: dict[str, Tensor]) -> Tensor:
        total = torch.tensor(0.0)
        if "coord" in prediction and "coord" in target:
            total = total + self.coord_weight * compute_coord_loss(prediction["coord"], target["coord"])
        if "violation" in prediction and "violation" in target:
            total = total + self.violation_weight * compute_violation_loss(prediction["violation"], target["violation"])
        if "existence" in prediction and "existence" in target:
            total = total + self.existence_weight * compute_existence_loss(prediction["existence"], target["existence"])
        return total
