"""Bounding box helpers used across the project."""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch


def bbox_to_tensor(bbox: Sequence[float], device: torch.device | None = None) -> Tensor:
    """Convert a 4-value bbox sequence to a tensor."""

    return torch.tensor(list(bbox), dtype=torch.float32, device=device)


def tensor_to_bbox(tensor: Tensor) -> Tuple[float, float, float, float]:
    """Convert a bbox tensor to a Python tuple."""

    values = tensor.detach().cpu().flatten().tolist()
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def xywh_to_xyxy(box: Tensor) -> Tensor:
    """Convert [x, y, w, h] boxes to [x1, y1, x2, y2]."""

    x, y, w, h = box.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_xywh(box: Tensor) -> Tensor:
    """Convert [x1, y1, x2, y2] boxes to [x, y, w, h]."""

    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """Compute pairwise IoU between two box tensors."""

    box1_xyxy = box1
    box2_xyxy = box2

    if box1_xyxy.shape[-1] == 4 and torch.any(box1_xyxy[..., 2] < box1_xyxy[..., 0]):
        box1_xyxy = xywh_to_xyxy(box1_xyxy)
    if box2_xyxy.shape[-1] == 4 and torch.any(box2_xyxy[..., 2] < box2_xyxy[..., 0]):
        box2_xyxy = xywh_to_xyxy(box2_xyxy)

    box1_xyxy = box1_xyxy.unsqueeze(-2)
    box2_xyxy = box2_xyxy.unsqueeze(-3)

    inter_x1 = torch.maximum(box1_xyxy[..., 0], box2_xyxy[..., 0])
    inter_y1 = torch.maximum(box1_xyxy[..., 1], box2_xyxy[..., 1])
    inter_x2 = torch.minimum(box1_xyxy[..., 2], box2_xyxy[..., 2])
    inter_y2 = torch.minimum(box1_xyxy[..., 3], box2_xyxy[..., 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    area1 = torch.clamp(box1_xyxy[..., 2] - box1_xyxy[..., 0], min=0.0) * torch.clamp(
        box1_xyxy[..., 3] - box1_xyxy[..., 1], min=0.0
    )
    area2 = torch.clamp(box2_xyxy[..., 2] - box2_xyxy[..., 0], min=0.0) * torch.clamp(
        box2_xyxy[..., 3] - box2_xyxy[..., 1], min=0.0
    )

    union = area1 + area2 - inter_area
    return torch.where(union > 0, inter_area / union, torch.zeros_like(inter_area))


def apply_delta(box: Tensor, delta: Tensor) -> Tensor:
    """Apply a refinement delta to an [x, y, w, h] box."""

    return box + delta
