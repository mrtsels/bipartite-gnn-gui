"""Training loop scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, torch


@dataclass
class Trainer:
    """Minimal trainer placeholder."""

    model: Any
    loss_fn: Any | None = None

    def fit(self, *_: Any, **__: Any) -> None:
        """No-op fit method for now."""

        return None
