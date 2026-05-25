"""General helpers for reproducibility and bbox utilities."""

from __future__ import annotations

import os
import random
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch

from .bbox import (
    apply_delta,
    bbox_to_tensor,
    clamp_coords,
    compute_center_distance,
    compute_iou,
    tensor_to_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_deterministic(seed: int = 42) -> None:
    """Seed and request deterministic PyTorch behavior when possible."""

    set_seed(seed)
    torch.use_deterministic_algorithms(True)


def compute_iou_pair(box1: Sequence[float] | Tensor, box2: Sequence[float] | Tensor) -> float:
    """Compute IoU for two individual boxes."""

    box1_tensor = box1 if isinstance(box1, Tensor) else bbox_to_tensor(box1)
    box2_tensor = box2 if isinstance(box2, Tensor) else bbox_to_tensor(box2)
    return float(compute_iou(box1_tensor.unsqueeze(0), box2_tensor.unsqueeze(0)).item())


__all__ = [
    "set_seed",
    "set_deterministic",
    "bbox_to_tensor",
    "tensor_to_bbox",
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "compute_iou_pair",
    "compute_iou",
    "apply_delta",
    "compute_center_distance",
    "clamp_coords",
]
