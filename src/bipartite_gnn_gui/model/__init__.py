"""
Model module — GraphSAGE-based GNN for GUI structure error correction.

Architecture:
    1. (Optional) CrossAttentionFusion: cross-attention between structural
       (5-d bbox+conf) and visual (192-d ViT) features when available,
       producing fusion_dim-d vectors. Falls back to pure-structural
       projection when visual features are absent.
    2. Heterogeneous GraphSAGE encoder: two layers of bipartite message passing
       (element → constraint → element) producing refined node embeddings.
    3. CoordinateRefinementHead: MLP that predicts Δ𝐱ᵢ = (Δx, Δy, Δw, Δh)
       from element embeddings.
    4. ViolationPredictionHead: MLP that predicts per-constraint violation scores.
    5. ExistencePredictionHead: MLP predicting element existence probability.
    6. MaskCompletionHead: MLP that predicts masked element features for
       self-supervised structural completion pretraining.

Submodules:
    attention  — Cross-attention fusion between struct & visual features.
    encoder    — Heterogeneous GraphSAGE with bipartite message passing.
    heads      — Refinement and prediction heads (coordinate, violation, existence, mask).
    model      — End-to-end BipartiteGNNCorrector combining encoder + all heads.
    losses     — Combined loss: ℒ = ℒ_coord + λ₁ℒ_violation + λ₂ℒ_alignment + λ₃ℒ_existence + λ₄ℒ_mask.
    trainer    — Training loop with scheduling, checkpointing, and logging.
    inference  — End-to-end inference: VLM JSON → graph → GNN → corrected JSON.
"""

from .attention import CrossAttentionFusion, SplitAndFuse
from .encoder import BipartiteGraphSAGE
from .heads import (
    CoordinateRefinementHead,
    ElementProposalHead,
    ExistencePredictionHead,
    MaskCompletionHead,
    ViolationPredictionHead,
)
from .losses import (
    CombinedLoss,
    compute_alignment_consistency_loss,
    compute_coord_loss,
    compute_existence_loss,
    compute_mask_loss,
    compute_violation_loss,
)
from .model import BipartiteGNNCorrector
from .trainer import Trainer

__all__ = [
    "BipartiteGraphSAGE",
    "BipartiteGNNCorrector",
    "BipartiteGNNLoss",
    "CombinedLoss",
    "CoordinateRefinementHead",
    "CrossAttentionFusion",
    "ElementProposalHead",
    "ExistencePredictionHead",
    "InferencePipeline",
    "MaskCompletionHead",
    "SplitAndFuse",
    "Trainer",
    "ViolationPredictionHead",
    "compute_alignment_consistency_loss",
    "compute_coord_loss",
    "compute_existence_loss",
    "compute_mask_loss",
    "compute_violation_loss",
    "correct_layout",
]
