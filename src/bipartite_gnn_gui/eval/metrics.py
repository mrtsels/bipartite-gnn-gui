"""Evaluation metrics for GUI structure correction."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch

from ..utils.bbox import compute_iou as _compute_iou


def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """Compute IoU between two bbox tensors."""

    return _compute_iou(box1, box2)


@dataclass
class PositionError:
    """Euclidean position error."""

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.norm(prediction[..., :2] - target[..., :2], dim=-1).mean()


@dataclass
class SizeError:
    """Euclidean size error."""

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.norm(prediction[..., 2:4] - target[..., 2:4], dim=-1).mean()


@dataclass
class AlignmentError:
    """Placeholder alignment error."""

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.abs(prediction - target).mean()


@dataclass
class ElementRecall:
    """Fraction of matched ground-truth elements."""

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
            device = prediction_boxes.device if prediction_boxes.numel() else target_boxes.device
            return torch.tensor(0.0, device=device)
        iou = compute_iou(prediction_boxes, target_boxes)
        return (iou.max(dim=-1).values >= self.iou_threshold).float().mean()


@dataclass
class ElementPrecision:
    """Fraction of predicted elements matched to ground truth."""

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
            device = prediction_boxes.device if prediction_boxes.numel() else target_boxes.device
            return torch.tensor(0.0, device=device)
        iou = compute_iou(prediction_boxes, target_boxes)
        return (iou.max(dim=-2).values >= self.iou_threshold).float().mean()


def compute_all_metrics(prediction_boxes: Tensor, target_boxes: Tensor) -> dict[str, float]:
    """Compute the standard metrics bundle."""

    return {
        "recall": float(ElementRecall()(prediction_boxes, target_boxes).item()),
        "precision": float(ElementPrecision()(prediction_boxes, target_boxes).item()),
        "position_error": float(PositionError()(prediction_boxes, target_boxes).item()),
        "size_error": float(SizeError()(prediction_boxes, target_boxes).item()),
        "alignment_error": float(AlignmentError()(prediction_boxes, target_boxes).item()),
    }
