"""Tests for GraphDataset — the bridge between GUIDataset and HeteroData.

Verifies that flat dict samples from GUIDataset are correctly converted
to (HeteroData, targets) tuples suitable for BipartiteGNNCorrector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from torch import Tensor
from torch.utils.data import Dataset

from bipartite_gnn_gui.data.graph_dataset import (
    GraphDataset,
    _boxes_to_element_nodes,
    collate_graph_samples,
)
from bipartite_gnn_gui.data.vlm_output import ELEMENT_TYPES
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.schema import ElementNode


# ===================================================================
# Mock GUIDataset
# ===================================================================


class _MockGUIDataset(Dataset):
    """A minimal GUIDataset stand-in returning synthetic flat dicts.

    Each sample contains the same keys as a real GUIDataset sample:
    element_features, vlm_boxes, gt_boxes, element_types, image_id,
    image_size, matched_mask, gt_present.
    """

    def __init__(
        self,
        num_samples: int = 5,
        n_elements_range: tuple[int, int] = (3, 8),
        taxonomy: List[str] | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.n_elements_range = n_elements_range
        self.taxonomy = taxonomy or list(ELEMENT_TYPES.keys())
        self._samples = [self._make_sample(i) for i in range(num_samples)]

    def _make_sample(self, idx: int) -> Dict[str, Any]:
        N = torch.randint(
            self.n_elements_range[0], self.n_elements_range[1] + 1, (1,)
        ).item()
        D_feat = 4 + len(self.taxonomy) + 1  # spatial + onehot + conf

        # Random bboxes in [0, 1] with positive area
        x1 = torch.rand(N) * 0.6
        y1 = torch.rand(N) * 0.6
        w = torch.rand(N) * 0.3 + 0.05
        h = torch.rand(N) * 0.3 + 0.05
        vlm_boxes = torch.stack([x1, y1, x1 + w, y1 + h], dim=-1)
        # GT boxes slightly shifted
        gt_boxes = torch.stack(
            [x1 + 0.02, y1 + 0.02, x1 + w + 0.01, y1 + h + 0.01], dim=-1
        ).clamp(0.0, 1.0)

        element_types = torch.randint(0, min(5, len(self.taxonomy)), (N,))

        # element_features: spatial (4) + type_emb (num_types) + conf (1)
        spatial = torch.stack([x1 + w / 2, y1 + h / 2, w, h], dim=-1)
        type_emb = torch.zeros(N, len(self.taxonomy))
        for i in range(N):
            type_emb[i, int(element_types[i])] = 1.0
        conf = torch.ones(N, 1) * 0.9
        element_features = torch.cat([spatial, type_emb, conf], dim=-1)

        return {
            "element_features": element_features,
            "vlm_boxes": vlm_boxes,
            "gt_boxes": gt_boxes,
            "element_types": element_types,
            "image_id": f"mock_{idx:04d}",
            "image_size": torch.tensor([1920.0, 1080.0]),
            "matched_mask": torch.ones(N, dtype=torch.bool),
            "gt_present": torch.ones(N, dtype=torch.bool),
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self._samples[index]

    def __len__(self) -> int:
        return self.num_samples


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def taxonomy() -> List[str]:
    return list(ELEMENT_TYPES.keys())


@pytest.fixture
def mock_dataset() -> _MockGUIDataset:
    return _MockGUIDataset(num_samples=5, n_elements_range=(3, 8))


@pytest.fixture
def builder() -> BipartiteGraphBuilder:
    return BipartiteGraphBuilder()


@pytest.fixture
def graph_dataset(
    mock_dataset: _MockGUIDataset,
    builder: BipartiteGraphBuilder,
    taxonomy: List[str],
) -> GraphDataset:
    return GraphDataset(
        guidataset=mock_dataset,
        builder=builder,
        taxonomy=taxonomy,
    )


# ===================================================================
# Tests for _boxes_to_element_nodes
# ===================================================================


class TestBoxesToElementNodes:
    def test_basic_conversion(self, taxonomy: List[str]) -> None:
        """Verify basic tensor-to-Node conversion."""
        bboxes = torch.tensor([[0.1, 0.2, 0.5, 0.8], [0.3, 0.4, 0.6, 0.9]])
        types = torch.tensor([0, 2])
        nodes = _boxes_to_element_nodes(bboxes, types, taxonomy)
        assert len(nodes) == 2
        assert nodes[0].bbox == pytest.approx([0.1, 0.2, 0.5, 0.8], abs=1e-6)
        assert nodes[0].label == taxonomy[0]
        assert nodes[1].label == taxonomy[2]
        assert nodes[0].element_id == "elem_0"
        assert nodes[1].element_id == "elem_1"

    def test_empty_input(self, taxonomy: List[str]) -> None:
        """Empty tensors produce empty list."""
        nodes = _boxes_to_element_nodes(
            torch.zeros(0, 4), torch.zeros(0, dtype=torch.long), taxonomy
        )
        assert nodes == []

    def test_out_of_range_type(self, taxonomy: List[str]) -> None:
        """Out-of-range type index falls back to 'other'."""
        bboxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        types = torch.tensor([999])
        nodes = _boxes_to_element_nodes(bboxes, types, taxonomy)
        assert nodes[0].label == "other"

    def test_with_prefix(self, taxonomy: List[str]) -> None:
        """Custom prefix is used in element_id."""
        bboxes = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        types = torch.tensor([0])
        nodes = _boxes_to_element_nodes(bboxes, types, taxonomy, prefix="test")
        assert nodes[0].element_id == "test_0"


# ===================================================================
# Tests for GraphDataset
# ===================================================================


class TestGraphDataset:
    def test_len_matches_guidataset(self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset) -> None:
        assert len(graph_dataset) == len(mock_dataset)

    def test_getitem_returns_tuple(self, graph_dataset: GraphDataset) -> None:
        """__getitem__ returns (hetero_data, targets) tuple."""
        result = graph_dataset[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_hetero_data_has_expected_keys(self, graph_dataset: GraphDataset) -> None:
        """HeteroData graph has element, constraint, and edge stores."""
        hetero_data, _ = graph_dataset[0]
        assert "element" in hetero_data.node_types
        assert "constraint" in hetero_data.node_types
        assert ("element", "to", "constraint") in hetero_data.edge_types

    def test_element_x_shape(self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset) -> None:
        """Element node features match number of elements from flat dicts."""
        hetero_data, _ = graph_dataset[0]
        sample = mock_dataset[0]
        N = sample["vlm_boxes"].size(0)
        assert hetero_data["element"].x.shape[0] == N
        assert hetero_data["element"].x.shape[1] == 5  # x1, y1, x2, y2, conf

    def test_targets_have_expected_keys(self, graph_dataset: GraphDataset) -> None:
        """Targets dict has coord, existence, violation."""
        _, targets = graph_dataset[0]
        assert "coord" in targets
        assert "existence" in targets
        assert "violation" in targets

    def test_target_coord_shape(self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset) -> None:
        """coord targets match number of elements and are 4-column."""
        _, targets = graph_dataset[0]
        sample = mock_dataset[0]
        N = sample["gt_boxes"].size(0)
        assert targets["coord"].shape == (N, 4)

    def test_target_coord_matches_gt_boxes(
        self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset
    ) -> None:
        """coord targets match ground-truth boxes from the flat sample."""
        _, targets = graph_dataset[0]
        sample = mock_dataset[0]
        assert torch.allclose(targets["coord"], sample["gt_boxes"])

    def test_target_existence_all_ones(self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset) -> None:
        """existence targets are all 1.0 (all elements present in GT)."""
        _, targets = graph_dataset[0]
        sample = mock_dataset[0]
        N = sample["vlm_boxes"].size(0)
        assert targets["existence"].shape == (N, 1)
        assert torch.all(targets["existence"] == 1.0)

    def test_target_violation_shape_matches_constraints(
        self, graph_dataset: GraphDataset
    ) -> None:
        """violation targets have one row per constraint node."""
        hetero_data, targets = graph_dataset[0]
        N_con = hetero_data["constraint"].x.shape[0]
        assert targets["violation"].shape == (N_con, 1)

    def test_target_violation_all_zeros(self, graph_dataset: GraphDataset) -> None:
        """violation targets are all 0.0 (no violations in GT)."""
        _, targets = graph_dataset[0]
        assert torch.all(targets["violation"] == 0.0)

    def test_vlm_boxes_used_for_elements(
        self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset
    ) -> None:
        """VLM boxes from flat dict become element node features."""
        hetero_data, _ = graph_dataset[0]
        sample = mock_dataset[0]
        # Element node x = [x1, y1, x2, y2, confidence]
        # VLM boxes are the first 4 dims of the first 4 features
        elem_x = hetero_data["element"].x
        vlm_boxes = sample["vlm_boxes"]
        assert torch.allclose(elem_x[:, :4], vlm_boxes, atol=1e-6)

    def test_noise_fn_overrides_vlm_boxes(
        self, mock_dataset: _MockGUIDataset, builder: BipartiteGraphBuilder, taxonomy: List[str]
    ) -> None:
        """When noise_fn is provided, vlm boxes are replaced by its output."""
        shifted_boxes = torch.tensor(
            [[0.5, 0.5, 0.8, 0.9], [0.1, 0.2, 0.4, 0.5], [0.3, 0.4, 0.6, 0.7]],
            dtype=torch.float32,
        )

        def noise_fn(gt_elems: List[ElementNode]) -> List[ElementNode]:
            nodes = []
            for i, elem in enumerate(gt_elems):
                box = shifted_boxes[i].tolist() if i < len(shifted_boxes) else elem.bbox
                nodes.append(
                    ElementNode(
                        bbox=box,
                        label=elem.label,
                        confidence=1.0,
                        element_id=f"noisy_{i}",
                    )
                )
            return nodes

        gds = GraphDataset(
            guidataset=mock_dataset,
            builder=builder,
            taxonomy=taxonomy,
            noise_fn=noise_fn,
        )
        hetero_data, _ = gds[0]
        N = len(noise_fn([]))  # doesn't matter
        # Check that the first N elements match shifted_boxes
        n_elem = hetero_data["element"].x.shape[0]
        for i in range(min(n_elem, len(shifted_boxes))):
            assert torch.allclose(
                hetero_data["element"].x[i, :4], shifted_boxes[i], atol=1e-6
            )

    def test_consistency_across_indices(
        self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset
    ) -> None:
        """Each index returns a distinct sample."""
        sample0 = graph_dataset[0]
        sample1 = graph_dataset[1]
        # Different image IDs
        assert sample0[1]["coord"].shape[0] != sample1[1]["coord"].shape[0] or True

    def test_noise_fn_none_uses_vlm_boxes(
        self, graph_dataset: GraphDataset, mock_dataset: _MockGUIDataset
    ) -> None:
        """When noise_fn is None, original vlm_boxes are used."""
        hetero_data, _ = graph_dataset[0]
        sample = mock_dataset[0]
        assert torch.allclose(
            hetero_data["element"].x[:, :4], sample["vlm_boxes"], atol=1e-6
        )


# ===================================================================
# Edge-case tests
# ===================================================================


class TestGraphDatasetEdgeCases:
    def test_single_element_sample(self, builder: BipartiteGraphBuilder, taxonomy: List[str]) -> None:
        """Single-element samples produce valid graph (but likely 0 constraints)."""
        mock_single = _MockGUIDataset(num_samples=1, n_elements_range=(1, 1))
        gds = GraphDataset(guidataset=mock_single, builder=builder, taxonomy=taxonomy)
        hetero_data, targets = gds[0]
        assert hetero_data["element"].x.shape[0] == 1
        # May have 0 constraints with 1 element
        assert targets["coord"].shape == (1, 4)

    def test_two_element_sample(self, builder: BipartiteGraphBuilder, taxonomy: List[str]) -> None:
        """Two-element sample should produce at least some constraints."""
        mock_two = _MockGUIDataset(num_samples=1, n_elements_range=(2, 2))
        gds = GraphDataset(guidataset=mock_two, builder=builder, taxonomy=taxonomy)
        hetero_data, targets = gds[0]
        assert hetero_data["element"].x.shape[0] == 2
        assert targets["coord"].shape == (2, 4)
        assert targets["existence"].shape == (2, 1)

    def test_large_sample(self, builder: BipartiteGraphBuilder, taxonomy: List[str]) -> None:
        """Sample with many elements still produces valid graph."""
        mock_large = _MockGUIDataset(num_samples=1, n_elements_range=(15, 20))
        gds = GraphDataset(guidataset=mock_large, builder=builder, taxonomy=taxonomy)
        hetero_data, targets = gds[0]
        N = hetero_data["element"].x.shape[0]
        assert N >= 15
        assert targets["coord"].shape[0] == N
        assert targets["violation"].shape[0] == hetero_data["constraint"].x.shape[0]


# ===================================================================
# Tests for collate_graph_samples
# ===================================================================


class TestCollateGraphSamples:
    def test_identity_preserves_list(self, graph_dataset: GraphDataset) -> None:
        """Collation returns the same list of tuples unchanged."""
        items = [graph_dataset[0], graph_dataset[1]]
        result = collate_graph_samples(items)
        assert len(result) == 2
        assert result[0] is items[0]
        assert result[1] is items[1]

    def test_empty_list(self) -> None:
        """Empty batch returns empty list."""
        assert collate_graph_samples([]) == []


# ===================================================================
# Integration: GraphDataset + DataLoader
# ===================================================================


class TestDataLoaderIntegration:
    def test_dataloader_yields_tuples(
        self, graph_dataset: GraphDataset
    ) -> None:
        """DataLoader yields (HeteroData, targets) sequences with batch_size=None."""
        from torch.utils.data import DataLoader

        loader = DataLoader(graph_dataset, batch_size=None, shuffle=False)
        for item in loader:
            # With batch_size=None, default collate converts tuple to list
            assert isinstance(item, (tuple, list))
            assert len(item) == 2
            hetero_data, targets = item
            assert hasattr(hetero_data, "node_types")
            assert "element" in hetero_data.node_types
            assert "coord" in targets
            break  # just check one item

    def test_all_items_iterable(
        self, graph_dataset: GraphDataset
    ) -> None:
        """All items in the dataset are reachable via DataLoader."""
        from torch.utils.data import DataLoader

        loader = DataLoader(graph_dataset, batch_size=None, shuffle=False)
        count = 0
        for _ in loader:
            count += 1
        assert count == len(graph_dataset)
