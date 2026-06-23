"""Prediction heads for the GUI layout corrector.

Three independent MLP heads operate on element or constraint embeddings
produced by the encoder to predict coordinate deltas, violation scores,
and existence probabilities.
"""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, torch


class _MLPHead(nn.Module):
    """Two-layer MLP base with ReLU, optional dropout, and optional activation.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension.
        dropout: Dropout probability (0.0 = no dropout).
        output_activation: Optional activation applied to output (e.g., sigmoid).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        output_activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape ``(N, input_dim)``.

        Returns:
            Output tensor of shape ``(N, output_dim)``.
        """
        out = self.network(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out


class CoordinateRefinementHead(_MLPHead):
    """Predict per-element coordinate refinement deltas.

    Takes encoded element features and produces a 4-d delta vector
    ``(Δcx, Δcy, Δw, Δh)`` for each element.  No output activation —
    deltas can be positive or negative.

    Args:
        input_dim: Dimensionality of encoded element features (default 128).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=4,
            dropout=dropout,
            output_activation=None,
        )


class ViolationPredictionHead(_MLPHead):
    """Predict per-constraint violation scores in ``[0, 1]``.

    Takes encoded constraint features and produces a scalar probability
    indicating how likely the constraint is violated.

    Args:
        input_dim: Dimensionality of encoded constraint features (default 128).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=1,
            dropout=dropout,
            output_activation=nn.Sigmoid(),
        )


class ExistencePredictionHead(_MLPHead):
    """Predict per-element existence probabilities in ``[0, 1]``.

    Takes encoded element features and produces a scalar probability
    indicating how likely the element is a genuine GUI component.

    Args:
        input_dim: Dimensionality of encoded element features (default 128).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=1,
            dropout=dropout,
            output_activation=nn.Sigmoid(),
        )


class MaskCompletionHead(_MLPHead):
    """Predict original features for masked element nodes.

    Takes encoded element embeddings (from the GNN encoder) and attempts
    to recover the original 5-d ``[x1, y1, x2, y2, confidence]`` features
    that were masked.  Loss is only computed on masked positions, so the
    model learns to infer missing elements from graph context alone.

    Args:
        input_dim: Dimensionality of encoded element features (default 128).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=input_dim,
            output_dim=5,  # [x1, y1, x2, y2, confidence]
            dropout=dropout,
            output_activation=nn.Sigmoid(),  # constrain to [0, 1]
        )


N_TYPES = 8  # RICO semantic types: button, text, icon, image, input, container, list, other


class ElementProposalHead(nn.Module):
    """Predict bounding box **and** element type for missing participants.

    Operates on **constraint embeddings** (not element embeddings).  For
    each constraint predicted as violated, this head proposes both the
    missing participant element's bounding box and its semantic type.

    Architecture:
        - A 2-layer MLP that maps ``(hidden_dim) → (4 + N_TYPES)`` for each
          constraint node.
        - First 4 outputs are ``(x1, y1, x2, y2)`` bounded to [0, 1] via
          Sigmoid.
        - Last ``N_TYPES`` outputs are un-normalised logits for type
          classification (CrossEntropyLoss handles the softmax internally).

    Args:
        input_dim: Dimensionality of constraint embeddings (default 128).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(self, input_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 4 + N_TYPES),  # (x1, y1, x2, y2) + type logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Constraint embeddings ``(N_con, input_dim)``.

        Returns:
            Tensor ``(N_con, 4 + N_TYPES)`` where:
                - ``[..., :4]`` is the proposed bbox ``[x1, y1, x2, y2]``
                  bounded to [0, 1] (post-sigmoid).
                - ``[..., 4:]`` are un-normalised type logits.
        """
        raw = self.network(x)
        bbox_out = raw[:, :4].sigmoid()
        type_out = raw[:, 4:]
        return torch.cat([bbox_out, type_out], dim=1)
