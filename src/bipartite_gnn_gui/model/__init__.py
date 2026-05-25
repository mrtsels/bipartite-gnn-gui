"""
Model module — GraphSAGE-based GNN for GUI structure error correction.

Architecture:
    1. Heterogeneous GraphSAGE encoder: two layers of bipartite message passing
       (element → constraint → element) producing refined node embeddings.
    2. CoordinateRefinementHead: MLP that predicts Δ𝐱ᵢ = (Δx, Δy, Δw, Δh)
       from element embeddings.
    3. ViolationPredictionHead: MLP that predicts per-constraint violation scores.
    4. ExistencePredictionHead: MLP predicting element existence probability.

Submodules:
    encoder   — Heterogeneous GraphSAGE with bipartite message passing.
    heads     — Refinement and prediction heads (coordinate, violation, existence).
    model     — End-to-end BipartiteGNNCorrector combining encoder + all heads.
    losses    — Combined loss: ℒ = ℒ_coord + λ₁ℒ_violation + λ₂ℒ_alignment + λ₃ℒ_existence.
    trainer   — Training loop with scheduling, checkpointing, and logging.
    inference — End-to-end inference: VLM JSON → graph → GNN → corrected JSON.
"""

from .encoder import BipartiteGraphSAGE
from .heads import CoordinateRefinementHead, ViolationPredictionHead, ExistencePredictionHead
from .model import BipartiteGNNCorrector
from .losses import BipartiteGNNLoss, compute_coord_loss, compute_violation_loss
from .trainer import Trainer
from .inference import correct_layout

__all__ = [
    "BipartiteGraphSAGE",
    "CoordinateRefinementHead",
    "ViolationPredictionHead",
    "ExistencePredictionHead",
    "BipartiteGNNCorrector",
    "BipartiteGNNLoss",
    "compute_coord_loss",
    "compute_violation_loss",
    "Trainer",
    "correct_layout",
]
