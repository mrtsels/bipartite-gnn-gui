"""Conversion from nodes to a heterogeneous bipartite graph object."""

from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import torch

from .schema import ConstraintNode, EdgeFeatures, ElementNode

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
    """Build a heterogeneous bipartite graph from elements and constraints."""

    def build(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
    ) -> HeteroData:
        """Build a heterogeneous bipartite graph.

        Constructs element nodes (5-d features from bbox + confidence),
        constraint nodes (11-d features from one-hot type + first param),
        and bipartite edges from each element involved in a constraint to
        that constraint node.  Edges carry 4-d spatial features computed
        via ``EdgeFeatures.compute`` between paired elements.

        Args:
            elements: ElementNode list (N_elem).
            constraints: ConstraintNode list (N_con).

        Returns:
            HeteroData with keys for "element", "constraint",
            ("element", "to", "constraint"), and
            ("constraint", "to", "element").
        """
        data = HeteroData()
        num_elements = len(elements)
        num_constraints = len(constraints)

        # ---- Element node features ----
        if elements:
            elem_feats = torch.stack([e.to_tensor() for e in elements])
        else:
            elem_feats = torch.zeros((0, 5), dtype=torch.float32)
        data["element"].x = elem_feats

        # ---- Constraint node features ----
        if constraints:
            con_feats_list = []
            for c in constraints:
                onehot = c.to_onehot()  # (10,)
                first_param = next(iter(c.params.values()), 0.0)
                con_feats_list.append(
                    torch.cat(
                        [onehot, torch.tensor([first_param], dtype=torch.float32)]
                    )
                )
            con_feats = torch.stack(con_feats_list)  # (N_con, 11)
        else:
            con_feats = torch.zeros((0, 11), dtype=torch.float32)
        data["constraint"].x = con_feats

        # ---- Edge indices and features ----
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_attr_list: list[torch.Tensor] = []

        for c_idx, constraint in enumerate(constraints):
            # Collect all element indices for this constraint,
            # skipping any that reference out-of-range elements.
            all_indices = constraint.source_indices + constraint.target_indices
            valid_indices = [
                idx for idx in all_indices if 0 <= idx < num_elements
            ]
            if not valid_indices:
                continue

            # Unique sorted indices used for edge-feature pairing.
            # Each edge's element is paired with the *next* element
            # in this sorted list (wrapping around).
            unique_valid = sorted(set(valid_indices))
            pos_map = {idx: i for i, idx in enumerate(unique_valid)}
            n_unique = len(unique_valid)

            for elem_idx in valid_indices:
                edge_src.append(elem_idx)
                edge_dst.append(c_idx)

                # Pair with the next element in the unique set
                pos = pos_map[elem_idx]
                partner_idx = unique_valid[(pos + 1) % n_unique]
                ef = EdgeFeatures.compute(elements[elem_idx], elements[partner_idx])
                edge_attr_list.append(ef.to_tensor())

        num_edges = len(edge_src)

        if num_edges > 0:
            forward_edges = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            edge_attr = torch.stack(edge_attr_list)
        else:
            forward_edges = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float32)

        data["element", "to", "constraint"].edge_index = forward_edges
        data["element", "to", "constraint"].edge_attr = edge_attr

        # Reverse edges (flipped)
        if num_edges > 0:
            reverse_edges = torch.flip(forward_edges, dims=[0])
        else:
            reverse_edges = torch.zeros((2, 0), dtype=torch.long)
        data["constraint", "to", "element"].edge_index = reverse_edges

        # Metadata
        data.num_elements = num_elements
        data.num_constraints = num_constraints
        data.num_edges = num_edges

        return data
