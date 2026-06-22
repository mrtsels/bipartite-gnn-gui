"""Heterogeneous bipartite GraphSAGE encoder using PyG SAGEConv layers.

Performs two rounds of bipartite message passing:
    1. element → constraint  (constraint aggregates element evidence)
    2. constraint → element  (elements receive constraint-aware updates)
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import F, nn, torch

from torch_geometric.nn import SAGEConv


class BipartiteGraphSAGE(nn.Module):
    """Heterogeneous GraphSAGE encoder for bipartite GUI layout graphs.

    Two-layer message passing over a ``HeteroData`` object with
    ``"element"`` and ``"constraint"`` node types.

    Architecture:
        1. Type-specific linear projection to a shared hidden dimension.
        2. Per round: sequential bipartite message passing —
           first ``element → constraint``, then ``constraint → element``.
        3. LayerNorm and ReLU after each convolution, with Dropout for
           regularisation.

    Args:
        element_dim: Input feature dimension for element nodes (default 5).
        constraint_dim: Input feature dimension for constraint nodes (default 11).
        hidden_dim: Hidden and output dimension (default 128).
        num_layers: Number of message-passing rounds (default 2).
        dropout: Dropout probability applied after each layer (default 0.1).

    Shape:
        - Input: ``HeteroData`` with ``data["element"].x`` shaped
          ``(N_elem, element_dim)`` and ``data["constraint"].x`` shaped
          ``(N_con, constraint_dim)``.
        - Output:
          ``{"element": (N_elem, hidden_dim), "constraint": (N_con, hidden_dim)}``.
    """

    def __init__(
        self,
        element_dim: int = 5,
        constraint_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.element_dim = element_dim
        self.constraint_dim = constraint_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Type-specific initial projections to common hidden dimension.
        self.element_proj = nn.Linear(element_dim, hidden_dim)
        self.constraint_proj = nn.Linear(constraint_dim, hidden_dim)

        # Bipartite message-passing layers — sequential per round.
        self.e_to_c_convs = nn.ModuleList()
        self.c_to_e_convs = nn.ModuleList()
        self.e_norms = nn.ModuleList()
        self.c_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.e_to_c_convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.c_to_e_convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.e_norms.append(nn.LayerNorm(hidden_dim))
            self.c_norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters."""
        self.element_proj.reset_parameters()
        self.constraint_proj.reset_parameters()
        for conv in self.e_to_c_convs:
            conv.reset_parameters()
        for conv in self.c_to_e_convs:
            conv.reset_parameters()

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        """Encode element and constraint node features via message passing.

        Args:
            data: A ``HeteroData`` object containing at minimum:
                - ``data["element"].x``: ``(N_elem, element_dim)`` float tensor.
                - ``data["constraint"].x``: ``(N_con, constraint_dim)`` float tensor.
                - ``data["element", "to", "constraint"].edge_index``: ``(2, E)``.
                - ``data["constraint", "to", "element"].edge_index``: ``(2, E)``.

        Returns:
            Dict with keys ``"element"`` and ``"constraint"`` mapping to
            encoded feature tensors of shape ``(N, hidden_dim)``.
        """
        # Initial type-specific projections.
        x_elem = self.element_proj(data["element"].x)
        x_con = self.constraint_proj(data["constraint"].x)

        edge_e2c = data["element", "to", "constraint"].edge_index
        edge_c2e = data["constraint", "to", "element"].edge_index

        for i in range(self.num_layers):
            # Hop 1: element → constraint (constraint aggregates from elements).
            con_msg = self.e_to_c_convs[i](
                (x_elem, x_con), edge_e2c
            )
            x_con = self.c_norms[i](con_msg) if x_con.numel() > 0 else con_msg
            x_con = F.relu(x_con)
            x_con = self.dropout(x_con)

            # Hop 2: constraint → element (elements receive constraint updates).
            elem_msg = self.c_to_e_convs[i](
                (x_con, x_elem), edge_c2e
            )
            x_elem = self.e_norms[i](elem_msg) if x_elem.numel() > 0 else elem_msg
            x_elem = F.relu(x_elem)
            x_elem = self.dropout(x_elem)

        return {"element": x_elem, "constraint": x_con}
