"""Conversion from nodes to a heterogeneous bipartite graph object."""

from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import torch

from .schema import ConstraintNode, ElementNode

try:
    from torch_geometric.data import HeteroData
except Exception:  # pragma: no cover - fallback for lightweight environments
    class HeteroData(dict):
        """Minimal fallback when PyG is unavailable."""

        def __getattr__(self, item: str) -> Any:
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key: str, value: Any) -> None:
            self[key] = value


class BipartiteGraphBuilder:
    """Build a bipartite graph from elements and constraints."""

    def build(self, elements: Sequence[ElementNode], constraints: Sequence[ConstraintNode]) -> HeteroData:
        """Create a graph object with node and edge stores."""

        data = HeteroData()
        element_features = [element.bbox + [element.confidence] for element in elements]
        constraint_features = [list(constraint.params.values()) or [0.0] for constraint in constraints]

        data["element"].x = torch.tensor(element_features, dtype=torch.float32) if element_features else torch.zeros((0, 5), dtype=torch.float32)
        data["constraint"].x = torch.tensor(constraint_features, dtype=torch.float32) if constraint_features else torch.zeros((0, 1), dtype=torch.float32)

        if elements and constraints:
            source_index = []
            target_index = []
            for constraint_index, constraint in enumerate(constraints):
                for element_index in constraint.source_indices + constraint.target_indices:
                    source_index.append(element_index)
                    target_index.append(constraint_index)
            edge_index = torch.tensor([source_index, target_index], dtype=torch.long) if source_index else torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data[("element", "to", "constraint")].edge_index = edge_index
        data[("constraint", "to", "element")].edge_index = torch.flip(edge_index, dims=[0]) if edge_index.numel() else torch.zeros((2, 0), dtype=torch.long)
        return data
