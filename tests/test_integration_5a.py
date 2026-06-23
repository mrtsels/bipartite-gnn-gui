"""Phase 5A — Integration tests for the original VLM correction pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector

# ── helpers ─────────────────────────────────────────────────────────────────


def _elem(x1: float, y1: float, x2: float, y2: float, label: str = "button") -> ElementNode:
    return ElementNode(
        bbox=[x1, y1, x2, y2],
        confidence=1.0,
        label=label,
    )


def _make_synthetic_vlm_json(path: Path, n_elements: int = 5) -> None:
    """Write a minimal VLM-style JSON for testing."""
    elements = []
    for i in range(n_elements):
        x1, y1 = 50 + i * 100, 100 + i * 80
        x2, y2 = x1 + 80, y1 + 40
        elements.append({
            "bbox_xyxy": [x1, y1, x2, y2],
            "label": ["button", "text", "icon", "input", "image"][i % 5],
            "confidence": 0.9 + 0.02 * i,
        })
    data = {"image_id": "test_001", "width": 1440, "height": 2560, "elements": elements}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ── 5A.1: Data pipeline integration ────────────────────────────────────────


class TestDataPipeline5A1:
    """Verify VLM JSON → parsed dict → normalized elements → ready for dataset."""

    @pytest.fixture(scope="class")
    def vlm_json(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        p = tmp_path_factory.mktemp("vlm") / "test_vlm.json"
        _make_synthetic_vlm_json(p)
        return p

    def test_json_loads_and_has_required_keys(self, vlm_json: Path) -> None:
        with open(vlm_json) as f:
            data = json.load(f)
        for key in ("image_id", "width", "height", "elements"):
            assert key in data, f"missing key: {key}"

    def test_parsed_elements_have_valid_bboxes(self, vlm_json: Path) -> None:
        with open(vlm_json) as f:
            data = json.load(f)
        for elem in data["elements"]:
            bbox = elem["bbox_xyxy"]
            assert len(bbox) == 4
            assert bbox[0] < bbox[2], "x1 >= x2"
            assert bbox[1] < bbox[3], "y1 >= y2"
            for key in ("label", "confidence"):
                assert key in elem, f"missing key: {key}"

    def test_can_construct_element_nodes(self, vlm_json: Path) -> None:
        with open(vlm_json) as f:
            data = json.load(f)
        elements = [
            ElementNode(bbox=e["bbox_xyxy"], confidence=e.get("confidence", 1.0), label=e["label"])
            for e in data["elements"]
        ]
        assert len(elements) > 0
        for el in elements:
            assert el.bbox[2] > el.bbox[0]
            assert el.bbox[3] > el.bbox[1]


# ── 5A.2: Graph building integration ───────────────────────────────────────


class TestGraphBuilding5A2:
    """Verify HeteroData construction from synthetic elements."""

    def test_equal_elements_produce_valid_graph(self) -> None:
        elements = [
            _elem(0.0, 0.0, 0.5, 0.5, "icon"),
            _elem(0.0, 0.6, 0.5, 0.8, "text"),
            _elem(0.6, 0.0, 0.8, 0.5, "button"),
        ]
        constraints = extract_all_constraints(elements)
        assert len(constraints) > 0, "no constraints extracted"

        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)

        assert "element" in data.node_types
        assert "constraint" in data.node_types
        assert ("element", "to", "constraint") in data.edge_types
        assert ("constraint", "to", "element") in data.edge_types

        N_elem = data["element"].x.shape[0]
        N_con = data["constraint"].x.shape[0]
        assert N_elem == 3
        assert N_con > 0

    def test_single_element_raises_no_constraints(self) -> None:
        elements = [_elem(0.0, 0.0, 0.5, 0.5)]
        constraints = extract_all_constraints(elements)
        assert len(constraints) == 0

    def test_graph_has_correct_feature_dims(self) -> None:
        elements = [
            _elem(0.0, 0.0, 0.5, 0.5),
            _elem(0.0, 0.6, 0.5, 0.8),
        ]
        constraints = extract_all_constraints(elements)
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)

        assert data["element"].x.shape[-1] == 5  # [x1,y1,x2,y2,confidence]
        # constraint features
        assert data["constraint"].x.shape[-1] >= 1


# ── 5A.4: End-to-end pipeline ──────────────────────────────────────────────


class TestEndToEnd5A4:
    """VLM JSON → model prediction → output dict with expected keys."""

    def test_model_forward_on_synthetic_graph(self) -> None:
        elements = [
            _elem(0.0, 0.0, 0.3, 0.3),
            _elem(0.0, 0.4, 0.3, 0.6),
            _elem(0.5, 0.0, 1.0, 0.3),
            _elem(0.0, 0.7, 1.0, 1.0),
        ]
        constraints = extract_all_constraints(elements)
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)

        model = BipartiteGNNCorrector(hidden_dim=16)
        model.eval()
        with torch.no_grad():
            outputs = model(data)

        for key in ("coord", "violation", "existence", "mask_completion", "proposal"):
            assert key in outputs, f"missing output: {key}"

        N_elem = len(elements)
        N_con = data["constraint"].x.shape[0]
        assert outputs["coord"].shape == (N_elem, 4)
        assert outputs["existence"].shape == (N_elem, 1)
        assert outputs["proposal"].shape == (N_con, 4)

    def test_loss_backward_on_synthetic_graph(self) -> None:
        elements = [
            _elem(0.0, 0.0, 0.3, 0.3),
            _elem(0.0, 0.4, 0.3, 0.6),
            _elem(0.5, 0.0, 1.0, 0.3),
        ]
        constraints = extract_all_constraints(elements)
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)

        model = BipartiteGNNCorrector(hidden_dim=16)
        model.mask_weight = 1.0
        model.proposal_weight = 1.0
        preds = model(data)

        N_elem = data["element"].x.shape[0]
        N_con = data["constraint"].x.shape[0]
        targets = {
            "coord": torch.randn(N_elem, 4),
            "violation": torch.rand(N_con, 1),
            "existence": torch.rand(N_elem, 1),
            "mask_completion_target": torch.randn(N_elem, 5),
            "mask_completion_mask": torch.tensor([True] + [False] * (N_elem - 1), dtype=torch.bool),
            "proposal_target": torch.randn(N_con, 4),
            "proposal_violation_mask": torch.tensor([True] + [False] * (N_con - 1), dtype=torch.bool),
        }

        loss = model.compute_loss(preds, targets)
        loss.backward()
        assert loss.dim() == 0
        assert loss.item() > 0

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"


# ── 5A.5: Baseline smoke test ──────────────────────────────────────────────


class TestBaselines5A5:
    """Verify every baseline can be instantiated and run on synthetic data."""

    def test_noop_baseline(self) -> None:
        from bipartite_gnn_gui.eval.baselines import NoOpBaseline
        bl = NoOpBaseline({"img_size": [1440, 2560]}, torch.device("cpu"))
        vlm = {
            "elements": [{"bbox": [0.0, 0.0, 100.0, 100.0], "label": "button"}],
            "image_id": "test",
            "image_width": 1440, "image_height": 2560,
        }
        result = bl.correct_single(vlm)
        assert len(result["elements"]) == 1
        elem = result["elements"][0]
        assert "bbox" in elem
        assert "confidence" in elem

    def test_identity_baseline(self) -> None:
        from bipartite_gnn_gui.eval.baselines import IdentityBaseline
        bl = IdentityBaseline({"dummy": True}, torch.device("cpu"))
        vlm = {"elements": [{"bbox": [0, 0, 100, 100], "label": "button"}]}
        result = bl.correct_single(vlm)
        assert len(result["elements"]) == 1

    def test_random_jitter_baseline(self) -> None:
        from bipartite_gnn_gui.eval.baselines import RandomJitterBaseline
        bl = RandomJitterBaseline({"dummy": True, "baseline_noise_scale": 0.1}, torch.device("cpu"))
        vlm = {"elements": [{"bbox": [0, 0, 100, 100], "label": "button"}]}
        result = bl.correct_single(vlm)
        elem = result["elements"][0]
        # random jitter should change bbox
        assert elem["bbox"] != [0, 0, 100, 100]
        assert len(elem["bbox"]) == 4
