"""
Model module — GraphSAGE-based GNN for GUI structure error correction.

Architecture:
    1. Heterogeneous GraphSAGE encoder: two layers of bipartite message passing
       (element → constraint → element) producing refined node embeddings.
    2. CoordinateRefinementHead: MLP that predicts Δ𝐱ᵢ = (Δx, Δy, Δw, Δh)
       from element embeddings.
    3. ViolationPredictionHead: MLP that predicts per-constraint violation scores.
    4. ExistencePredictionHead: MLP predicting element existence probability.
    5. MaskCompletionHead: MLP that predicts masked element features for
       self-supervised structural completion pretraining.

Submodules:
    encoder   — Heterogeneous GraphSAGE with bipartite message passing.
    heads     — Refinement and prediction heads (coordinate, violation, existence, mask).
    model     — End-to-end BipartiteGNNCorrector combining encoder + all heads.
    losses    — Combined loss: ℒ = ℒ_coord + λ₁ℒ_violation + λ₂ℒ_alignment + λ₃ℒ_existence + λ₄ℒ_mask.
    trainer   — Training loop with scheduling, checkpointing, and logging.
    inference — End-to-end inference: VLM JSON → graph → GNN → corrected JSON.
"""

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
    "ExistencePredictionHead",
    "InferencePipeline",
    "MaskCompletionHead",
    "Trainer",
    "ViolationPredictionHead",
    "compute_alignment_consistency_loss",
    "compute_coord_loss",
    "compute_existence_loss",
    "compute_mask_loss",
    "compute_violation_loss",
    "correct_layout",
]
