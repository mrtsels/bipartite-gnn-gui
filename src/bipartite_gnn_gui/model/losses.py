"""Loss functions for GUI layout correction training.

Provides four component losses and a weighted combination:
    1. Coordinate MSE loss (on predicted deltas).
    2. Violation BCE loss (on constraint-level violation scores).
    3. Existence BCE loss (on element-level existence probabilities).
    4. Alignment consistency loss (penalises post-refinement misalignment).
"""

from __future__ import annotations

from typing import Dict

try:
    import torch
    from torch import Tensor
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import F, Tensor, torch


def compute_coord_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean squared error on coordinate refinement deltas.

    Args:
        prediction: ``(N_elem, 4)`` predicted deltas ``[Δcx, Δcy, Δw, Δh]``.
        target: ``(N_elem, 4)`` ground-truth deltas.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(prediction, target)


def compute_violation_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary cross-entropy on violation scores.

    Args:
        prediction: ``(N_con, 1)`` predicted violation scores in ``[0, 1]``.
        target: ``(N_con, 1)`` binary labels (0 = valid, 1 = violated).

    Returns:
        Scalar BCE loss. Returns 0.0 if no constraints exist.
    """
    if prediction.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=prediction.device)
    return F.binary_cross_entropy(prediction, target)


def compute_existence_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary cross-entropy on existence probabilities.

    Args:
        prediction: ``(N_elem, 1)`` predicted existence probabilities in ``[0, 1]``.
        target: ``(N_elem, 1)`` binary labels (0 = spurious, 1 = real).

    Returns:
        Scalar BCE loss. Returns 0.0 if no elements exist.
    """
    if prediction.numel() == 0 or target.numel() == 0:
        return torch.tensor(0.0, device=prediction.device)
    return F.binary_cross_entropy(prediction, target)


def compute_alignment_consistency_loss(
    predicted_deltas: Tensor,
    original_bboxes: Tensor,
    edge_index: Tensor,
    constraint_type: str = "align_left",
    tolerance: float = 0.02,
) -> Tensor:
    """Penalise constraint violations that persist after delta correction.

    For alignment constraints (e.g. ALIGN_LEFT), elements sharing the
    constraint should have their left edges remain aligned after the
    predicted deltas are applied. This loss measures the post-correction
    edge-position variance among constrained elements.

    Args:
        predicted_deltas: ``(N_elem, 4)`` coordinate deltas in xywh format.
        original_bboxes: ``(N_elem, 4)`` original bboxes in xywh format.
        edge_index: ``(2, E)`` element-to-constraint edge index where the
            first row contains element indices and the second row contains
            constraint indices.
        constraint_type: Which constraint type to evaluate alignment on.
        tolerance: Tolerance for acceptable misalignment.

    Returns:
        Scalar alignment consistency loss.
    """
    if predicted_deltas.numel() == 0 or edge_index.numel() == 0:
        return torch.tensor(0.0, device=predicted_deltas.device)

    # Apply deltas to get corrected bboxes.
    corrected = original_bboxes + predicted_deltas  # (N_elem, 4) xywh

    # Convert xywh to xyxy for edge-position comparison.
    cx, cy, w, h = corrected[:, 0], corrected[:, 1], corrected[:, 2], corrected[:, 3]
    x1 = cx - w / 2.0
    x2 = cx + w / 2.0
    y1 = cy - h / 2.0
    y2 = cy + h / 2.0

    # Select the relevant edge position based on constraint type.
    edge_pos_map: Dict[str, int] = {
        "align_left": 0,
        "align_right": 1,
        "align_top": 2,
        "align_bottom": 3,
    }
    if constraint_type not in edge_pos_map:
        return torch.tensor(0.0, device=predicted_deltas.device)

    pos_idx = edge_pos_map[constraint_type]
    pos_values = [x1, x2, y1, y2][pos_idx]  # (N_elem,)

    # Gather element indices from edges.
    elem_indices = edge_index[0]  # (E,)
    constr_indices = edge_index[1]  # (E,)

    total_loss = torch.tensor(0.0, device=predicted_deltas.device)
    unique_constraints = constr_indices.unique()

    if unique_constraints.numel() == 0:
        return total_loss

    for cid in unique_constraints:
        mask = constr_indices == cid
        elem_for_constraint = elem_indices[mask]  # element indices for this constraint
        if elem_for_constraint.numel() < 2:
            continue
        # Variance of edge positions among elements in this constraint.
        pos = pos_values[elem_for_constraint]
        mean_pos = pos.mean()
        per_elem_loss = F.mse_loss(pos, mean_pos.expand_as(pos), reduction="mean")
        # Only penalize if max deviation exceeds tolerance.
        if (pos - mean_pos).abs().max() > tolerance:
            total_loss = total_loss + per_elem_loss

    return total_loss


class CombinedLoss:
    """Weighted combination of four loss components for GUI layout correction.

    :math:`L_{total} = w_c L_{coord} + w_v L_{violation} + w_e L_{existence} + w_a L_{alignment}`

    Args:
        coord_weight: Weight for coordinate refinement loss.
        violation_weight: Weight for violation detection loss.
        existence_weight: Weight for existence probability loss.
        alignment_weight: Weight for alignment consistency loss.
        alignment_tolerance: Tolerance for acceptable misalignment in the
            consistency loss.
    """

    def __init__(
        self,
        coord_weight: float = 1.0,
        violation_weight: float = 1.0,
        existence_weight: float = 1.0,
        alignment_weight: float = 0.5,
        alignment_tolerance: float = 0.02,
    ) -> None:
        self.coord_weight = coord_weight
        self.violation_weight = violation_weight
        self.existence_weight = existence_weight
        self.alignment_weight = alignment_weight
        self.alignment_tolerance = alignment_tolerance

    def __call__(
        self,
        prediction: dict[str, Tensor],
        target: dict[str, Tensor],
        original_bboxes: Tensor | None = None,
        edge_index: Tensor | None = None,
        constraint_type: str = "align_left",
    ) -> Tensor:
        """Compute the combined multi-task loss.

        Args:
            prediction: Dict from ``BipartiteGNNCorrector.forward()`` with keys:
                ``"coord"``, ``"violation"``, ``"existence"``.
            target: Ground-truth dict with matching keys.
            original_bboxes: Optional ``(N_elem, 4)`` original bboxes in xywh
                format, used for alignment consistency loss.
            edge_index: Optional ``(2, E)`` element-to-constraint edge index,
                used for alignment consistency loss.
            constraint_type: Alignment constraint type for consistency loss.

        Returns:
            Scalar tensor with total weighted loss.
        """
        total = torch.tensor(0.0, device=self._device(prediction))

        if "coord" in prediction and "coord" in target:
            total = total + self.coord_weight * compute_coord_loss(
                prediction["coord"], target["coord"]
            )

        if "violation" in prediction and "violation" in target:
            total = total + self.violation_weight * compute_violation_loss(
                prediction["violation"], target["violation"]
            )

        if "existence" in prediction and "existence" in target:
            total = total + self.existence_weight * compute_existence_loss(
                prediction["existence"], target["existence"]
            )

        if (
            "coord" in prediction
            and original_bboxes is not None
            and edge_index is not None
            and self.alignment_weight > 0.0
        ):
            total = total + self.alignment_weight * compute_alignment_consistency_loss(
                prediction["coord"],
                original_bboxes,
                edge_index,
                constraint_type=constraint_type,
                tolerance=self.alignment_tolerance,
            )

        return total

    @staticmethod
    def _device(prediction: dict[str, Tensor]) -> torch.device:
        """Extract device from the first available prediction tensor."""
        for key in ("coord", "violation", "existence"):
            if key in prediction:
                return prediction[key].device
        return torch.device("cpu")


# Backward-compatible alias.
BipartiteGNNLoss = CombinedLoss
