"""End-to-end GUI layout correction model.

Assembles the encoder and three prediction heads into a single
``nn.Module`` that maps a ``HeteroData`` graph to correction outputs.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, Tensor, torch

from .attention import SplitAndFuse
from .encoder import BipartiteGraphSAGE
from .heads import (
    CoordinateRefinementHead,
    ElementProposalHead,
    ExistencePredictionHead,
    MaskCompletionHead,
    ViolationPredictionHead,
)
from .losses import CombinedLoss, compute_mask_loss, compute_proposal_loss
from .losses import compute_proposal_type_loss


class BipartiteGNNCorrector(nn.Module):
    """End-to-end heterogeneous bipartite GNN for GUI spatial error correction.

    Combines:
        - ``BipartiteGraphSAGE`` encoder for two-hop message passing.
        - ``CoordinateRefinementHead`` for per-element delta prediction.
        - ``ViolationPredictionHead`` for per-constraint violation scores.
        - ``ExistencePredictionHead`` for per-element existence probabilities.

    Args:
        element_dim: Feature dimension for element nodes (default 5).
        constraint_dim: Feature dimension for constraint nodes (default 11).
        hidden_dim: Hidden dimension shared by encoder and all heads
            (default 128).
        num_layers: Number of message-passing rounds (default 2).
        dropout: Dropout probability for encoder and heads (default 0.1).
    """

    def __init__(
        self,
        element_dim: int = 5,
        constraint_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        coord_weight: float = 1.0,
        existence_weight: float = 1.0,
        fusion_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.element_dim = element_dim
        self.constraint_dim = constraint_dim
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim

        # Optional cross-attention fusion of structural + visual features.
        # When fusion_dim is set, the encoder receives fusion_dim-d vectors
        # instead of raw element_dim-d vectors.  When None (default), the
        # encoder receives element_dim-d vectors as before (backward compat).
        self.fusion: SplitAndFuse | None = None
        encoder_element_dim = element_dim
        if fusion_dim is not None:
            self.fusion = SplitAndFuse(
                struct_dim=element_dim,
                visual_dim=192,
                fusion_dim=fusion_dim,
                dropout=dropout,
            )
            encoder_element_dim = fusion_dim

        self.encoder = BipartiteGraphSAGE(
            element_dim=encoder_element_dim,
            constraint_dim=constraint_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.coordinate_head = CoordinateRefinementHead(
            input_dim=hidden_dim, dropout=dropout
        )
        self.violation_head = ViolationPredictionHead(
            input_dim=hidden_dim, dropout=dropout
        )
        self.existence_head = ExistencePredictionHead(
            input_dim=hidden_dim, dropout=dropout
        )
        # Optional mask completion head for self-supervised pretraining.
        self.mask_head = MaskCompletionHead(
            input_dim=hidden_dim, dropout=dropout
        )
        # Optional element proposal head for structural completion.
        self.proposal_head = ElementProposalHead(
            input_dim=hidden_dim, dropout=dropout
        )
        self.loss_fn = CombinedLoss(
            coord_weight=coord_weight,
            existence_weight=existence_weight,
        )
        self.mask_weight: float = 0.0  # disabled by default
        self.proposal_type_weight: float = 0.0  # type prediction weight (disabled by default)

    def forward(self, data: Any) -> dict[str, Tensor]:
        """Run complete correction inference on a graph.

        Args:
            data: A ``HeteroData`` graph with element and constraint
                node features and edge indices.

        Returns:
            Dict with keys:
                - ``"coord"``: ``(N_elem, 4)`` coordinate deltas.
                - ``"violation"``: ``(N_con, 1)`` violation scores.
                - ``"existence"``: ``(N_elem, 1)`` existence probabilities.
                - ``"proposal"``: ``(N_con, 4)`` proposed bbox for missing elements.
                - ``"proposal_type"``: ``(N_con, N_TYPES)`` type logits for missing elements.
        """
        # Optional cross-attention fusion of structural + visual features.
        # When fusion is enabled, element features are projected through
        # SplitAndFuse before the GNN encoder.  This handles both the
        # concatenated (struct + visual) and pure-structural cases.
        if self.fusion is not None:
            data["element"].x = self.fusion(data["element"].x)

        encoded = self.encoder(data)
        outputs: dict[str, Tensor] = {}
        if "element" in encoded:
            outputs["coord"] = self.coordinate_head(encoded["element"])
            outputs["existence"] = self.existence_head(encoded["element"])
            outputs["mask_completion"] = self.mask_head(encoded["element"])
        if "constraint" in encoded:
            outputs["violation"] = self.violation_head(encoded["constraint"])
            proposed = self.proposal_head(encoded["constraint"])
            outputs["proposal"] = proposed[:, :4]       # (N_con, 4) bbox
            outputs["proposal_type"] = proposed[:, 4:]  # (N_con, N_TYPES) type logits
        return outputs

    def compute_loss(
        self,
        predictions: dict[str, Tensor],
        targets: dict[str, Tensor],
        original_bboxes: Tensor | None = None,
        edge_index: Tensor | None = None,
    ) -> Tensor:
        """Compute the combined multi-task loss.

        Args:
            predictions: Output from ``self.forward()``.
            targets: Ground-truth dict with matching keys.
            original_bboxes: Optional ``(N_elem, 4)`` original bboxes
                for alignment consistency loss.
            edge_index: Optional ``(2, E)`` edge index for alignment
                consistency loss.

        Returns:
            Scalar total loss tensor.
        """
        total = self.loss_fn(predictions, targets, original_bboxes, edge_index)

        # Mask completion loss (self-supervised pretraining).
        if (
            self.mask_weight > 0.0
            and "mask_completion" in predictions
            and "mask_completion_target" in targets
            and "mask_completion_mask" in targets
        ):
            mask_loss = compute_mask_loss(
                predictions["mask_completion"],
                targets["mask_completion_target"],
                targets["mask_completion_mask"],
            )
            total = total + self.mask_weight * mask_loss

        # Proposal loss (structural completion training).
        if (
            hasattr(self, "proposal_weight")
            and self.proposal_weight > 0.0
            and "proposal" in predictions
            and "proposal_target" in targets
            and "proposal_violation_mask" in targets
        ):
            prop_loss = compute_proposal_loss(
                predictions["proposal"],
                targets["proposal_target"],
                targets["proposal_violation_mask"],
            )
            total = total + self.proposal_weight * prop_loss

        # Proposal type loss (element type prediction).
        if (
            self.proposal_type_weight > 0.0
            and "proposal_type" in predictions
            and "proposal_type_target" in targets
            and "proposal_violation_mask" in targets
        ):
            prop_type_loss = compute_proposal_type_loss(
                predictions["proposal_type"],
                targets["proposal_type_target"],
                targets["proposal_violation_mask"],
            )
            total = total + self.proposal_type_weight * prop_type_loss

        return total

    def train_step(
        self,
        data: Any,
        targets: dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        grad_clip: float = 1.0,
        amp_enabled: bool = False,
    ) -> Tensor:
        """Execute a single training step (forward, backward, optimizer step).

        Args:
            data: Input ``HeteroData`` graph.
            targets: Ground-truth target dict.
            optimizer: Optimizer to use for the parameter update.
            grad_clip: Max L2 gradient norm for clipping.
            amp_enabled: Whether to use automatic mixed precision.

        Returns:
            Scalar loss value for this step.
        """
        self.train()
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=amp_enabled):  # type: ignore[attr-defined]
            predictions = self(data)
            loss = self.compute_loss(predictions, targets)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        optimizer.step()

        return loss.detach()

    def validation_step(self, data: Any, targets: dict[str, Tensor]) -> Tensor:
        """Compute loss for a single validation batch (no gradients).

        Args:
            data: Input ``HeteroData`` graph.
            targets: Ground-truth target dict.

        Returns:
            Scalar loss value.
        """
        self.eval()
        with torch.no_grad():
            predictions = self(data)
            loss = self.compute_loss(predictions, targets)
        return loss.detach()
