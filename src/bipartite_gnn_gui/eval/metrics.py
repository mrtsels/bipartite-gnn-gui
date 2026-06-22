"""Evaluation metrics for GUI structure correction.

Defines metric callables for comparing predicted and ground-truth GUI element
bounding boxes.  All metrics operate on **xyxy** boxes ``(x1, y1, x2, y2)`` in
``[0, 1]`` normalised coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch

from ..utils.bbox import compute_iou as _compute_iou


# ---------------------------------------------------------------------------
# IoU – thin wrapper around the utility layer
# ---------------------------------------------------------------------------


def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """Compute pairwise IoU between two sets of xyxy bounding boxes.

    Args:
        box1: ``(N, 4)`` tensor of ``[x1, y1, x2, y2]``.
        box2: ``(M, 4)`` tensor of ``[x1, y1, x2, y2]``.

    Returns:
        ``(N, M)`` tensor of IoU values.
    """
    return _compute_iou(box1, box2)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


@dataclass
class PositionError:
    """Euclidean position error:  ‖(x̂₁, ŷ₁) − (x₁, y₁)‖₂  per element."""

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute mean top-left corner distance.

        Args:
            prediction: ``(N, 4)`` predicted xyxy boxes.
            target: ``(N, 4)`` ground-truth xyxy boxes.

        Returns:
            Scalar mean Euclidean distance.
        """
        if prediction.numel() == 0 or target.numel() == 0:
            device = prediction.device if prediction.numel() else target.device
            return torch.tensor(0.0, device=device)
        n = min(prediction.shape[0], target.shape[0])
        return torch.norm(prediction[:n, :2] - target[:n, :2], dim=-1).mean()


@dataclass
class SizeError:
    """Euclidean size error:  ‖(ŵ, ĥ) − (w, h)‖₂  per element.

    Size is derived as ``(x₂ − x₁, y₂ − y₁)`` from xyxy boxes.
    """

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute mean size (width, height) error.

        Args:
            prediction: ``(N, 4)`` predicted xyxy boxes.
            target: ``(N, 4)`` ground-truth xyxy boxes.

        Returns:
            Scalar mean Euclidean size error.
        """
        if prediction.numel() == 0 or target.numel() == 0:
            device = prediction.device if prediction.numel() else target.device
            return torch.tensor(0.0, device=device)
        n = min(prediction.shape[0], target.shape[0])
        pred_size = prediction[:n, 2:4] - prediction[:n, :2]
        tgt_size = target[:n, 2:4] - target[:n, :2]
        return torch.norm(pred_size - tgt_size, dim=-1).mean()


# ---------------------------------------------------------------------------
# Alignment detection helpers
# ---------------------------------------------------------------------------

# The six alignment types match graph/schema.py:ConstraintType:
#   ALIGN_LEFT, ALIGN_RIGHT, ALIGN_TOP, ALIGN_BOTTOM, CENTER_X, CENTER_Y

_ALIGNMENT_CHECKS: List[Tuple[str, int, bool]] = [
    # (name, coord_index, use_center)
    ("align_left",    0, False),   # x1 alignment
    ("align_right",   2, False),   # x2 alignment
    ("align_top",     1, False),   # y1 alignment
    ("align_bottom",  3, False),   # y2 alignment
    ("center_x",      0, True),    # centre-x alignment
    ("center_y",      1, True),    # centre-y alignment
]


def _detect_alignments(
    boxes: Tensor,
    tolerance: float = 0.02,
) -> List[Tuple[int, int, str]]:
    """Detect alignment relationships among a set of boxes.

    For each pair ``(i, j)`` with ``i < j``, checks whether any of the
    six alignment types hold within *tolerance*.

    Args:
        boxes: ``(N, 4)`` tensor of xyxy boxes in ``[0, 1]``.
        tolerance: Maximum deviation for two coordinates to be
            considered aligned (default 0.02).

    Returns:
        List of ``(i, j, alignment_name)`` triples.
    """
    n = boxes.shape[0]
    if n < 2:
        return []

    # Pre-compute centres
    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5

    alignments: List[Tuple[int, int, str]] = []

    for i in range(n):
        for j in range(i + 1, n):
            for name, idx, use_center in _ALIGNMENT_CHECKS:
                if use_center:
                    ci = cx if idx == 0 else cy
                    cj_val = cx[j] if idx == 0 else cy[j]
                    if abs(ci[i] - cj_val) < tolerance:
                        alignments.append((i, j, name))
                else:
                    if abs(boxes[i, idx] - boxes[j, idx]) < tolerance:
                        alignments.append((i, j, name))

    return alignments


@dataclass
class AlignmentError:
    """Constraint-aware alignment error.

    Detects alignment relationships (LEFT, RIGHT, TOP, BOTTOM,
    CENTER_X, CENTER_Y) among **target** boxes within *tolerance*,
    then measures the per-edge deviation among the corresponding
    **prediction** boxes and returns the mean.

    Args:
        tolerance: Maximum deviation for two coordinates to be
            considered aligned in the target (default 0.02).
    """

    tolerance: float = 0.02

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute mean alignment deviation.

        Args:
            prediction: ``(N, 4)`` predicted xyxy boxes.
            target: ``(N, 4)`` ground-truth xyxy boxes.

        Returns:
            Scalar mean deviation across all detected alignment
            relationships, or ``0.0`` if none are found / inputs
            are empty.
        """
        if prediction.numel() == 0 or target.numel() == 0:
            device = prediction.device if prediction.numel() else target.device
            return torch.tensor(0.0, device=device)

        n = min(prediction.shape[0], target.shape[0])
        if n < 2:
            return torch.tensor(0.0, device=prediction.device)

        pred = prediction[:n]
        tgt = target[:n]

        # Pre-compute centres for deviation checks
        pred_cx = (pred[:, 0] + pred[:, 2]) * 0.5
        pred_cy = (pred[:, 1] + pred[:, 3]) * 0.5

        deviations: List[Tensor] = []

        # For each pair, check which alignments exist in target,
        # then compute the deviation in prediction
        for i in range(n):
            for j in range(i + 1, n):
                for name, idx, use_center in _ALIGNMENT_CHECKS:
                    if use_center:
                        tgt_cx = (tgt[:, 0] + tgt[:, 2]) * 0.5
                        tgt_cy = (tgt[:, 1] + tgt[:, 3]) * 0.5
                        tc_i = tgt_cx[i] if idx == 0 else tgt_cy[i]
                        tc_j = tgt_cx[j] if idx == 0 else tgt_cy[j]
                        if abs(tc_i - tc_j) < self.tolerance:
                            pc_i = pred_cx[i] if idx == 0 else pred_cy[i]
                            pc_j = pred_cx[j] if idx == 0 else pred_cy[j]
                            deviations.append(torch.abs(pc_i - pc_j))
                    else:
                        if abs(tgt[i, idx] - tgt[j, idx]) < self.tolerance:
                            deviations.append(
                                torch.abs(pred[i, idx] - pred[j, idx])
                            )

        if not deviations:
            return torch.tensor(0.0, device=prediction.device)

        return torch.stack(deviations).mean()


# ---------------------------------------------------------------------------
# Detection metrics
# ---------------------------------------------------------------------------


@dataclass
class ElementRecall:
    """Fraction of matched ground-truth elements.

    For each ground-truth box, finds the predicted box with maximum
    IoU; counts as matched if IoU ≥ *iou_threshold*.

    Args:
        iou_threshold: Minimum IoU for a match (default 0.5).
    """

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
            device = prediction_boxes.device if prediction_boxes.numel() else target_boxes.device
            return torch.tensor(0.0, device=device)
        iou = compute_iou(prediction_boxes, target_boxes)
        # For each GT (column), max over predictions → recall per GT
        return (iou.max(dim=-2).values >= self.iou_threshold).float().mean()


@dataclass
class ElementPrecision:
    """Fraction of predicted elements matched to ground truth.

    For each predicted box, finds the ground-truth box with maximum
    IoU; counts as matched if IoU ≥ *iou_threshold*.

    Args:
        iou_threshold: Minimum IoU for a match (default 0.5).
    """

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
            device = prediction_boxes.device if prediction_boxes.numel() else target_boxes.device
            return torch.tensor(0.0, device=device)
        iou = compute_iou(prediction_boxes, target_boxes)
        # For each prediction (row), max over GTs → precision per pred
        return (iou.max(dim=-1).values >= self.iou_threshold).float().mean()


@dataclass
class F1Score:
    """Harmonic mean of ElementRecall and ElementPrecision.

    Args:
        iou_threshold: IoU threshold forwarded to inner metrics (default 0.5).
    """

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        recall = ElementRecall(self.iou_threshold)(prediction_boxes, target_boxes)
        precision = ElementPrecision(self.iou_threshold)(prediction_boxes, target_boxes)
        denom = recall + precision
        if denom == 0:
            return torch.tensor(0.0, device=denom.device)
        return 2.0 * recall * precision / denom


# ---------------------------------------------------------------------------
# Metrics bundle
# ---------------------------------------------------------------------------


@dataclass
class MetricsBundle:
    """Structured container for all evaluation metrics.

    Attributes:
        recall: Element recall (0–1).
        precision: Element precision (0–1).
        f1: Harmonic mean of recall and precision (0–1).
        position_error: Mean top-left Euclidean error.
        size_error: Mean size (w, h) Euclidean error.
        alignment_error: Mean constraint-aware alignment deviation.
    """

    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    position_error: float = 0.0
    size_error: float = 0.0
    alignment_error: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "position_error": self.position_error,
            "size_error": self.size_error,
            "alignment_error": self.alignment_error,
        }


# ---------------------------------------------------------------------------
# Composite metric computation
# ---------------------------------------------------------------------------


def compute_all_metrics(
    prediction_boxes: Tensor,
    target_boxes: Tensor,
    iou_threshold: float = 0.5,
    alignment_tolerance: float = 0.02,
) -> MetricsBundle:
    """Compute the standard metrics bundle.

    Args:
        prediction_boxes: ``(N, 4)`` predicted xyxy boxes, or empty tensor.
        target_boxes: ``(M, 4)`` ground-truth xyxy boxes, or empty tensor.
        iou_threshold: IoU threshold for recall/precision/f1 (default 0.5).
        alignment_tolerance: Tolerance for alignment detection (default 0.02).

    Returns:
        ``MetricsBundle`` with all six metric values.  Handles edge cases
        (empty predictions, empty targets, different box counts) gracefully.

    Raises:
        ValueError: If *prediction_boxes* or *target_boxes* is not a 2-d
            tensor with 4 columns.
    """
    _validate_box_tensor(prediction_boxes, "prediction_boxes")
    _validate_box_tensor(target_boxes, "target_boxes")

    recall_fn = ElementRecall(iou_threshold)
    precision_fn = ElementPrecision(iou_threshold)
    f1_fn = F1Score(iou_threshold)
    pos_fn = PositionError()
    size_fn = SizeError()
    align_fn = AlignmentError(alignment_tolerance)

    return MetricsBundle(
        recall=float(recall_fn(prediction_boxes, target_boxes).item()),
        precision=float(precision_fn(prediction_boxes, target_boxes).item()),
        f1=float(f1_fn(prediction_boxes, target_boxes).item()),
        position_error=float(pos_fn(prediction_boxes, target_boxes).item()),
        size_error=float(size_fn(prediction_boxes, target_boxes).item()),
        alignment_error=float(align_fn(prediction_boxes, target_boxes).item()),
    )


def _validate_box_tensor(t: Tensor, name: str) -> None:
    """Raise ValueError if *t* is not a 2-d tensor with 4 columns (or empty)."""
    if t.numel() == 0:
        # Empty tensors are valid — make sure dim is 2
        if t.dim() == 0:
            raise ValueError(f"{name} must be a 2-d tensor, got scalar")
        if t.dim() == 1:
            raise ValueError(f"{name} must be a 2-d tensor, got 1-d")
        return
    if t.dim() != 2:
        raise ValueError(
            f"{name} must be a 2-d (N, 4) tensor, got shape {tuple(t.shape)}"
        )
    if t.shape[-1] != 4:
        raise ValueError(
            f"{name} last dim must be 4, got {t.shape[-1]}"
        )
