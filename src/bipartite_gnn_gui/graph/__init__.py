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

from .schema import EdgeFeatures, ElementNode, ConstraintNode, ConstraintType, EdgeType
from .constraints import (
    extract_alignment_constraints,
    extract_containment_constraints,
    extract_spacing_constraints,
    extract_grid_constraints,
    extract_all_constraints,
)
from .builder import BipartiteGraphBuilder
from .visualize import (
    color_by_constraint_type,
    color_by_element_type,
    export_graph,
    plot_graph_on_screenshot,
)
from .augment import (
    ConstraintPerturbation,
    CoordinateJitter,
    GraphAugmentationPipeline,
    NodeDropout,
)

__all__ = [
    "EdgeFeatures",
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
    "color_by_element_type",
    "color_by_constraint_type",
    "export_graph",
    "plot_graph_on_screenshot",
    "NodeDropout",
    "CoordinateJitter",
    "ConstraintPerturbation",
    "GraphAugmentationPipeline",
]
