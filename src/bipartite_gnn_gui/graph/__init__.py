"""
Graph module — Heterogeneous bipartite graph construction and constraint extraction.

Builds a heterogeneous bipartite graph G = (Vₑ ∪ V_c, E) from VLM JSON output:
- Vₑ — Element nodes (buttons, text, images, inputs, etc.) with spatial + type features.
- V_c — Constraint nodes (alignment, containment, spacing, grid) encoding GUI priors.
- E   — Bipartite edges connecting elements to the constraints they participate in.

Submodules:
    schema      — Node type definitions, feature schemas, and edge type definitions.
    constraints — Constraint extraction from ground-truth and heuristic rules.
    builder     — Conversion of element + constraint lists to PyG HeteroData objects.
    visualize   — Graph visualization overlaid on screenshots.
    augment     — Graph augmentation for robustness (node dropout, jitter, etc.).
"""

from .schema import ElementNode, ConstraintNode, ConstraintType, EdgeType
from .constraints import (
    extract_alignment_constraints,
    extract_containment_constraints,
    extract_spacing_constraints,
    extract_grid_constraints,
    extract_all_constraints,
)
from .builder import BipartiteGraphBuilder
from .visualize import plot_bipartite_graph
from .augment import GraphAugmenter

__all__ = [
    "ElementNode",
    "ConstraintNode",
    "ConstraintType",
    "EdgeType",
    "extract_alignment_constraints",
    "extract_containment_constraints",
    "extract_spacing_constraints",
    "extract_grid_constraints",
    "extract_all_constraints",
    "BipartiteGraphBuilder",
    "plot_bipartite_graph",
    "GraphAugmenter",
]
