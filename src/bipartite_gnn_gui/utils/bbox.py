"""Bounding box utility functions.

Coordinate format conventions
------------------------------
- **xyxy**: ``(x1, y1, x2, y2)`` — top-left and bottom-right corners.
- **xywh** (center-based): ``(cx, cy, w, h)`` — centre coordinates and
  width/height.  Converted via ``x1 = cx - w/2``, ``x2 = cx + w/2``, etc.

``compute_iou``, ``compute_center_distance`` all expect **xyxy** format.
``apply_delta`` works on **xywh** boxes and deltas (element-wise addition).
"""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch


# ---------------------------------------------------------------------------
# Conversion helpers (Python ↔ Tensor)
# ---------------------------------------------------------------------------


def bbox_to_tensor(
    bbox: Sequence[float],
    device: torch.device | None = None,
) -> Tensor:
    """Convert a 4-value bbox sequence to a float32 tensor."""
    return torch.tensor(list(bbox), dtype=torch.float32, device=device)


def tensor_to_bbox(tensor: Tensor) -> Tuple[float, float, float, float]:
    """Convert a bbox tensor to a Python tuple."""
    values = tensor.detach().cpu().flatten().tolist()
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


# ---------------------------------------------------------------------------
# Format conversions (centre-based xywh ↔ xyxy)
# ---------------------------------------------------------------------------


def xywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert centre-based ``(cx, cy, w, h)`` to ``(x1, y1, x2, y2)``.

    .. math::
       x_1 = cx - w/2 \\qquad x_2 = cx + w/2 \\\\
       y_1 = cy - h/2 \\qquad y_2 = cy + h/2

    Supports arbitrary leading batch dimensions.
    """
    cx, cy, w, h = boxes.unbind(-1)
    half_w = w * 0.5
    half_h = h * 0.5
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def xyxy_to_xywh(boxes: Tensor) -> Tensor:
    """Convert ``(x1, y1, x2, y2)`` to centre-based ``(cx, cy, w, h)``.

    .. math::
       cx = (x_1 + x_2)/2 \\qquad cy = (y_1 + y_2)/2 \\\\
       w = x_2 - x_1 \\qquad h = y_2 - y_1
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1
    )


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def compute_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of boxes in **xyxy** format.

    Parameters
    ----------
    boxes1:
        Tensor of shape ``(..., 4)`` (e.g. ``(N, 4)``).  Each row is
        ``(x1, y1, x2, y2)``.
    boxes2:
        Tensor of shape ``(..., 4)`` (e.g. ``(M, 4)``).

    Returns
    -------
    Tensor of shape ``(..., N, ..., M)`` where N and M are the last
    non-box dimensions of ``boxes1`` and ``boxes2`` respectively.
    IoU values are in ``[0, 1]``.  Degenerate boxes (area ≤ 0) always
    yield IoU = 0.

    Examples
    --------
    >>> b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    >>> b2 = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    >>> compute_iou(b1, b2)
    tensor([[0.1429]])  # (1,1)
    """
    # Ensure at least 2D — unsqueeze if flat (4,)
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    # Create broadcasting dimensions: (N, 1, 4) vs (1, M, 4) → (N, M, 4)
    b1 = boxes1.unsqueeze(-2)
    b2 = boxes2.unsqueeze(-3)

    # Intersection
    inter_x1 = torch.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = torch.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = torch.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = torch.minimum(b1[..., 3], b2[..., 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter_area = inter_w * inter_h

    # Areas (clamp for degenerate boxes where x2 < x1)
    area1 = (
        torch.clamp(b1[..., 2] - b1[..., 0], min=0.0)
        * torch.clamp(b1[..., 3] - b1[..., 1], min=0.0)
    )
    area2 = (
        torch.clamp(b2[..., 2] - b2[..., 0], min=0.0)
        * torch.clamp(b2[..., 3] - b2[..., 1], min=0.0)
    )

    union = area1 + area2 - inter_area
    return torch.where(union > 0, inter_area / union, torch.zeros_like(inter_area))


# ---------------------------------------------------------------------------
# Delta application
# ---------------------------------------------------------------------------


def apply_delta(boxes: Tensor, deltas: Tensor) -> Tensor:
    """Apply coordinate refinement deltas to centre-based xywh boxes.

    Parameters
    ----------
    boxes:
        Boxes in ``(cx, cy, w, h)`` format, shape ``(..., 4)``.
    deltas:
        Refinement deltas ``(Δcx, Δcy, Δw, Δh)``, same shape as
        ``boxes`` (broadcastable).

    Returns
    -------
    ``boxes + deltas`` — element-wise addition.
    """
    return boxes + deltas


# ---------------------------------------------------------------------------
# Distance utilities
# ---------------------------------------------------------------------------


def compute_center_distance(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise Euclidean (L2) distance between box **centres**.

    Both inputs are expected in **xyxy** format.  The centre of each
    box is computed as ``((x1 + x2)/2, (y1 + y2)/2)``.

    Parameters
    ----------
    boxes1:
        Tensor of shape ``(N, 4)``, xyxy.
    boxes2:
        Tensor of shape ``(M, 4)``, xyxy.

    Returns
    -------
    Tensor of shape ``(N, M)`` containing centre-to-centre distances.
    """
    if boxes1.dim() == 1:
        boxes1 = boxes1.unsqueeze(0)
    if boxes2.dim() == 1:
        boxes2 = boxes2.unsqueeze(0)

    # Centres: (x1 + x2) / 2, (y1 + y2) / 2
    c1 = torch.stack(
        [(boxes1[..., 0] + boxes1[..., 2]) * 0.5,
         (boxes1[..., 1] + boxes1[..., 3]) * 0.5],
        dim=-1,
    )  # (N, 2)
    c2 = torch.stack(
        [(boxes2[..., 0] + boxes2[..., 2]) * 0.5,
         (boxes2[..., 1] + boxes2[..., 3]) * 0.5],
        dim=-1,
    )  # (M, 2)

    # (N, 1, 2) - (1, M, 2) → (N, M, 2) → norm → (N, M)
    return torch.norm(c1.unsqueeze(1) - c2.unsqueeze(0), dim=-1)


# ---------------------------------------------------------------------------
# Clamping
# ---------------------------------------------------------------------------


def clamp_coords(boxes: Tensor, min_val: float = 0.0, max_val: float = 1.0) -> Tensor:
    """Clamp all coordinate values to ``[min_val, max_val]``.

    Works with any box format (xyxy, xywh, etc.) since it simply
    clamps every element independently.

    Parameters
    ----------
    boxes:
        Tensor of any shape ending with ``4``.
    min_val:
        Lower bound (default 0.0).
    max_val:
        Upper bound (default 1.0).

    Returns
    -------
    Clamped tensor with the same shape and dtype.
    """
    return boxes.clamp(min=min_val, max=max_val)
