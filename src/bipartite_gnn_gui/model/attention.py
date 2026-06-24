"""Cross-attention fusion between structural and visual element features.

Replaces simple concatenation of ViT visual features with a learned
cross-attention mechanism that lets structural and visual modalities
interact before the GNN encoder.

Architecture (Pre-fusion — recommended over post-fusion, see docstring):

    struct_feats          visual_feats
        |                      |
    Linear(5→D)           Linear(192→D)
        |                      |
    LayerNorm              LayerNorm
        |                      |
      ReLU                   ReLU
        |                      |
    Dropout                Dropout
        |                      |
      QUERY ──→ MultiheadAttn ←─── KEY / VALUE
        |                      |
    ResidualAdd + LayerNorm
        |
    fused_feats (D,) ──→ GNN encoder

Why pre-fusion?
  The encoder's bipartite message passing propagates *semantic* information
  (element appearance, type, spatial layout) through the constraint graph.
  If visual features enter only after the encoder (post-fusion), the GNN
  has already completed message passing without knowing what each element
  *looks like* — a button with a "Submit" caption and an icon with a
  chevron arrow are treated identically as generic elements.  Fusing
  before the encoder lets visual semantics guide which constraint
  messages are relevant, improving both element and constraint embeddings.

  Pre-fusion also keeps the encoder's input dimension cleanly separated
  from the raw modalities: the encoder always sees ``fusion_dim`` vectors,
  independent of whether visual features are available or not.
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover — optional dependency fallback
    from bipartite_gnn_gui._compat import F, nn, torch


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between structural and visual element features.

    Two MLP towers project structural (``struct_dim``) and visual
    (``visual_dim``) features into a common ``fusion_dim`` space.
    Cross-attention then lets each element's structural representation
    query its visual representation (and vice-versa via the symmetric
    ``fusion_mode``), followed by a residual-add and layer norm.

    When ``visual_feats is None`` (fallback), only the structural tower
    is used — a simple projection from ``struct_dim → fusion_dim``.
    This guarantees the model degrades gracefully when visual features
    are unavailable at inference time.

    Args:
        struct_dim: Structural feature dimension (default 5:
            ``[x1, y1, x2, y2, confidence]``).  Set to 13 if/when
            bbox + one-hot type + confidence encoding is added.
        visual_dim: Visual embedding dimension (default 192 from
            ``vit_tiny_patch16_224`` patch-pooled features).
        fusion_dim: Common projection / attention dimension (default 64).
            This becomes the output dimension fed to the GNN encoder.
        num_heads: Number of attention heads (default 4).  ``fusion_dim``
            must be divisible by ``num_heads``.
        dropout: Dropout probability throughout (default 0.1).

    Shape:
        - ``struct_feats``: ``(N, struct_dim)``
        - ``visual_feats``: ``(N, visual_dim)`` or ``None``
        - Output: ``(N, fusion_dim)``
    """

    def __init__(
        self,
        struct_dim: int = 5,
        visual_dim: int = 192,
        fusion_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if fusion_dim % num_heads != 0:
            raise ValueError(
                f"fusion_dim ({fusion_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.struct_dim = struct_dim
        self.visual_dim = visual_dim
        self.fusion_dim = fusion_dim

        # ---- Structural tower ----
        self.struct_proj = nn.Sequential(
            nn.Linear(struct_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- Visual tower (only created if visual features exist) ----
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- Cross-attention: struct → Q, visual → KV ----
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(fusion_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        struct_feats: torch.Tensor,
        visual_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse structural and optional visual features via cross-attention.

        Args:
            struct_feats: ``(N, struct_dim)`` structural features.
            visual_feats: ``(N, visual_dim)`` visual features, or
                ``None`` for pure-structural fallback.

        Returns:
            ``(N, fusion_dim)`` fused element features.
        """
        s = self.struct_proj(struct_feats)  # (N, fusion_dim)

        if visual_feats is not None:
            v = self.visual_proj(visual_feats)  # (N, fusion_dim)

            # Cross-attention: each element's structural representation
            # queries its own visual representation.  We unsqueeze to
            # shape (N, 1, D) so that ``batch_first=True`` treats each
            # element as a batch entry with a sequence length of 1.
            attn_out, _ = self.cross_attn(
                query=s.unsqueeze(1),   # (N, 1, fusion_dim)
                key=v.unsqueeze(1),     # (N, 1, fusion_dim)
                value=v.unsqueeze(1),   # (N, 1, fusion_dim)
                need_weights=False,
            )
            attn_out = attn_out.squeeze(1)  # (N, fusion_dim)

            # Residual connection + layer norm.
            s = self.attn_norm(s + self.attn_dropout(attn_out))

        return s  # (N, fusion_dim)

    def reset_parameters(self) -> None:
        """Reset all learnable parameters for reproducibility."""
        for module in self.struct_proj:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        for module in self.visual_proj:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        self.cross_attn._reset_parameters()  # type: ignore[attr-defined]
        self.attn_norm.reset_parameters()


class SplitAndFuse(nn.Module):
    """Utility wrapper that splits combined element features and runs fusion.

    This handles the case where element features arrive as a single
    concatenated tensor (e.g., 197-d = 5-d structural + 192-d visual),
    which is the format produced by ``BipartiteGraphBuilder`` when
    ``visual_features`` is provided.

    Usage::

        fusion = SplitAndFuse(struct_dim=5, visual_dim=192, fusion_dim=64)
        fused = fusion(elem_feats)   # elem_feats is (N, 197) or (N, 5)

    When ``elem_feats.shape[1] == struct_dim`` (no visual features),
    the tensor is passed to the inner fusion module with
    ``visual_feats=None`` for pure-structural fallback.

    When ``elem_feats.shape[1] == struct_dim + visual_dim``, the tensor
    is split and both modalities are fused via cross-attention.

    Args:
        struct_dim: Number of structural dimensions at the beginning of
            the input tensor (default 5).
        visual_dim: Number of visual dimensions after the structural
            portion (default 192).
        fusion_dim: Output dimension of the cross-attention fusion
            (default 64).
        num_heads: Number of attention heads (default 4).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        struct_dim: int = 5,
        visual_dim: int = 192,
        fusion_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.struct_dim = struct_dim
        self.visual_dim = visual_dim
        self.fusion = CrossAttentionFusion(
            struct_dim=struct_dim,
            visual_dim=visual_dim,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Fuse, automatically detecting presence of visual features.

        Args:
            x: Element features ``(N, D)`` where ``D`` is either
                ``struct_dim`` (no visual) or
                ``struct_dim + visual_dim`` (with visual).

        Returns:
            ``(N, fusion_dim)`` fused features.
        """
        if x.shape[1] == self.struct_dim:
            # Pure-structural fallback: no visual features present.
            return self.fusion(x, visual_feats=None)
        elif x.shape[1] == self.struct_dim + self.visual_dim:
            # Combined structural + visual: split and fuse.
            struct = x[:, : self.struct_dim]
            visual = x[:, self.struct_dim :]
            return self.fusion(struct, visual_feats=visual)
        else:
            raise ValueError(
                f"Expected input dim {self.struct_dim} "
                f"(no visual) or {self.struct_dim + self.visual_dim} "
                f"(with visual), got {x.shape[1]}"
            )

    def reset_parameters(self) -> None:
        """Reset fusion sub-module parameters."""
        self.fusion.reset_parameters()
