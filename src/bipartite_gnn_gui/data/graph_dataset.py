"""Dataset wrapper converting GUIDataset flat dicts to HeteroData + targets.

Provides:
- GraphDataset: wraps a GUIDataset, converting each flat sample
  (element_features, vlm_boxes, gt_boxes, ...) into a (HeteroData, targets)
  tuple suitable for the BipartiteGNNCorrector model and Trainer.
- collate_graph_samples: custom collation that preserves the (data, targets)
  tuple structure across batches.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    import torch
    from torch import Tensor
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Dataset, Tensor, torch

from bipartite_gnn_gui.data.dataset import GUIDataset
from bipartite_gnn_gui.data.vlm_output import ELEMENT_TYPES
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _boxes_to_element_nodes(
    bboxes: Tensor,
    type_indices: Tensor,
    taxonomy: List[str],
    prefix: str = "elem",
) -> List[ElementNode]:
    """Convert tensors of boxes and type indices to a list of ElementNodes.

    Args:
        bboxes: ``(N, 4)`` xyxy bounding boxes in normalized [0, 1].
        type_indices: ``(N,)`` integer class indices.
        taxonomy: Ordered list of canonical type strings
            (e.g. from ``ELEMENT_TYPES.keys()``).
        prefix: String prefix for ``element_id``.

    Returns:
        List of ``ElementNode`` objects, one per box.
    """
    nodes: List[ElementNode] = []
    N = bboxes.size(0)
    for i in range(N):
        type_str = "other"
        idx = int(type_indices[i].item()) if i < type_indices.size(0) else 0
        if 0 <= idx < len(taxonomy):
            type_str = taxonomy[idx]
        nodes.append(
            ElementNode(
                bbox=bboxes[i].tolist(),
                label=type_str,
                confidence=1.0,
                element_id=f"{prefix}_{i}",
            )
        )
    return nodes


# ---------------------------------------------------------------------------
# GraphDataset
# ---------------------------------------------------------------------------


class GraphDataset(Dataset):
    """Wrap a GUIDataset, converting each flat sample to ``(HeteroData, targets)``.

    Each call to ``__getitem__``:
      1. Loads the cached ``.pt`` sample from ``GUIDataset``.
      2. Converts ``vlm_boxes`` + ``element_types`` into ``ElementNode`` objects.
      3. Converts ``gt_boxes`` + ``element_types`` into ``ElementNode`` objects
         for constraint extraction.
      4. Extracts heuristic constraints from the GT elements.
      5. Builds a ``HeteroData`` bipartite graph from (VLM elements, constraints).
      6. Constructs the target dict from ``gt_boxes``.

    Args:
        guidataset: An initialised ``GUIDataset`` with cached ``.pt`` files.
        builder: ``BipartiteGraphBuilder`` instance for graph construction.
        taxonomy: Ordered list of canonical element type names.
            Defaults to the keys of ``ELEMENT_TYPES``.
        noise_fn: Optional callable ``(gt_elements: List[ElementNode])
            -> List[ElementNode]`` that generates noisy VLM predictions from
            ground-truth elements.  When provided, ``vlm_boxes`` from the
            cached sample are **ignored** and replaced with the noisy output.
    """

    def __init__(
        self,
        guidataset: GUIDataset,
        builder: Optional[BipartiteGraphBuilder] = None,
        taxonomy: Optional[List[str]] = None,
        noise_fn: Optional[Callable[[List[ElementNode]], List[ElementNode]]] = None,
    ) -> None:
        self.guidataset = guidataset
        self.builder = builder or BipartiteGraphBuilder()
        self.taxonomy = taxonomy or list(ELEMENT_TYPES.keys())
        self.noise_fn = noise_fn
        self._num_types = len(self.taxonomy)

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Tensor]]:
        """Return ``(hetero_data, targets)`` for one sample.

        Returns:
            Tuple of:
            - ``hetero_data``: ``HeteroData`` graph built from VLM elements
              and GT-extracted constraints.
            - ``targets``: dict with keys ``"coord"`` ``(N, 4)``,
              ``"existence"`` ``(N, 1)``, and ``"violation"`` ``(N_con, 1)``.
        """
        sample = self.guidataset[index]

        vlm_boxes: Tensor = sample["vlm_boxes"]  # (N, 4)
        gt_boxes: Tensor = sample["gt_boxes"]  # (N, 4)
        element_types: Tensor = sample["element_types"]  # (N,)
        N = vlm_boxes.size(0)

        # ---- Build GT element nodes (for constraint extraction) ----
        gt_elements = _boxes_to_element_nodes(
            gt_boxes, element_types, self.taxonomy, prefix="gt"
        )

        # ---- Build VLM element nodes ----
        if self.noise_fn is not None:
            vlm_elements = self.noise_fn(gt_elements)
        else:
            vlm_elements = _boxes_to_element_nodes(
                vlm_boxes, element_types, self.taxonomy, prefix="vlm"
            )

        # ---- Extract constraints from GT structure ----
        constraints = extract_all_constraints(gt_elements)

        # ---- Build HeteroData graph ----
        hetero_data = self.builder.build(vlm_elements, constraints)

        # ---- Build targets ----
        # Convert VLM and GT boxes from xyxy to cxcywh for delta computation.
        def _to_cxcywh(boxes: Tensor) -> Tensor:
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            return torch.stack([cx, cy, w, h], dim=-1)

        vlm_xywh = _to_cxcywh(vlm_boxes)
        gt_xywh = _to_cxcywh(gt_boxes)
        delta = gt_xywh - vlm_xywh  # model target: GT_offset - VLM_offset

        N_con = len(constraints)
        targets: Dict[str, Tensor] = {
            "coord": delta,              # (N, 4) model predicts Δcx, Δcy, Δw, Δh
            "gt_boxes": gt_boxes,        # (N, 4) raw GT xyxy for evaluation
            "existence": torch.ones(N, 1, dtype=torch.float32),
            "violation": torch.zeros(N_con, 1, dtype=torch.float32),
        }

        return hetero_data, targets

    def __len__(self) -> int:
        return len(self.guidataset)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------


def collate_graph_samples(
    batch: List[Tuple[Any, Dict[str, Tensor]]],
) -> List[Tuple[Any, Dict[str, Tensor]]]:
    """Identity collation: each (HeteroData, targets) tuple is kept intact.

    Since heterogeneous graphs have variable numbers of nodes/edges, they
    cannot be naively stacked.  This collator returns the batch as a list
    of individual tuples, preserving each graph independently.

    The ``Trainer`` in ``bipartite_gnn_gui.model.trainer`` iterates over
    the yielded list and processes each ``(data, targets)`` pair.

    Args:
        batch: A list of ``(HeteroData, targets_dict)`` tuples.

    Returns:
        The same list of tuples (no stacking).
    """
    return batch
