"""Coordinate and feature preprocessing helpers."""

from __future__ import annotations

from typing import Sequence

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch


def normalize_coordinates(box: Sequence[float], width: float, height: float) -> list[float]:
    """Normalize absolute coordinates to the [0, 1] range."""

    x, y, w, h = box
    return [x / width, y / height, w / width, h / height]


def extract_element_features(element: dict[str, object]) -> Tensor:
    """Convert a GUI element payload into a small feature tensor."""

    bbox = element.get("bbox", [0.0, 0.0, 0.0, 0.0])
    confidence = float(element.get("confidence", 1.0))
    values = list(bbox)[:4] + [confidence]
    return torch.tensor([float(value) for value in values], dtype=torch.float32)
