"""
bipartite-gnn-gui — Heterogeneous Bipartite GNN for GUI Structure Error Correction.

A post-correction framework that refines noisy GUI element predictions from
lightweight Vision-Language Models (e.g., Qwen3.5-2B, MiniMax-VL-01) by:

1. Constructing a heterogeneous bipartite graph from VLM JSON output,
   with element nodes and spatial constraint nodes.
2. Applying GraphSAGE message passing across the bipartite structure.
3. Predicting coordinate refinement deltas Δ𝐱 = (Δx, Δy, Δw, Δh) to correct
   each element's bounding box.

Submodules:
    data    — Dataset loading, preprocessing, and unified data interfaces.
    graph   — Heterogeneous bipartite graph construction and constraint extraction.
    model   — GraphSAGE encoder, refinement heads, training, and inference.
    eval    — Evaluation metrics (PositionError, SizeError, AlignmentError, etc.).
    utils   — Configuration, logging, and miscellaneous utility functions.
"""

__version__ = "0.1.0"
