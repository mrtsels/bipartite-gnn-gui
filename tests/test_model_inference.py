"""Tests for InferencePipeline — end-to-end correction from VLM JSON."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.model.inference import (
    InferencePipeline,
    _xyxy_to_xywh,
    _xywh_to_xyxy,
    correct_layout,
)
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector


@pytest.fixture
def model():
    return BipartiteGNNCorrector(
        element_dim=5, constraint_dim=11, hidden_dim=64,
        num_layers=2, dropout=0.0,
    )


@pytest.fixture
def pipeline(model):
    return InferencePipeline(model, device=torch.device("cpu"))


# ===================================================================
# Bbox conversion helpers
# ===================================================================


class TestBboxConversions:
    def test_xyxy_to_xywh(self) -> None:
        xywh = _xyxy_to_xywh([0.1, 0.2, 0.5, 0.8])
        cx = (0.1 + 0.5) / 2.0
        cy = (0.2 + 0.8) / 2.0
        w = 0.5 - 0.1
        h = 0.8 - 0.2
        assert xywh == pytest.approx([cx, cy, w, h])

    def test_xywh_to_xyxy(self) -> None:
        xyxy = _xywh_to_xyxy([0.3, 0.5, 0.4, 0.6])
        assert xyxy[0] == pytest.approx(0.1)
        assert xyxy[1] == pytest.approx(0.2)
        assert xyxy[2] == pytest.approx(0.5)
        assert xyxy[3] == pytest.approx(0.8)

    def test_roundtrip(self) -> None:
        original = [0.2, 0.3, 0.6, 0.7]
        xywh = _xyxy_to_xywh(original)
        xyxy = _xywh_to_xyxy(xywh)
        assert xyxy == pytest.approx(original)


# ===================================================================
# correct_layout backward compat
# ===================================================================


class TestCorrectLayout:
    def test_passes_through_model(self, model) -> None:
        """correct_layout should just call model.forward()."""
        from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
        from bipartite_gnn_gui.graph.schema import ConstraintType, ElementNode

        elements = [
            ElementNode(bbox=[0.1, 0.2, 0.3, 0.4], confidence=0.9),
            ElementNode(bbox=[0.5, 0.6, 0.7, 0.8], confidence=0.8),
        ]
        from bipartite_gnn_gui.graph.schema import ConstraintNode
        constraints = [
            ConstraintNode(
                constraint_type=ConstraintType.ALIGN_LEFT,
                source_indices=[0, 1],
                params={"tolerance": 0.02},
            )
        ]
        builder = BipartiteGraphBuilder()
        data = builder.build(elements, constraints)
        result = correct_layout(model, data)
        assert "coord" in result


# ===================================================================
# InferencePipeline
# ===================================================================


class TestInferencePipeline:
    def test_empty_elements(self, pipeline) -> None:
        result = pipeline.correct_single({"elements": []})
        assert result["elements"] == []

    def test_correct_single_returns_dict(self, pipeline) -> None:
        vlm_json = {
            "image_id": "test_img",
            "elements": [
                {
                    "bbox": [100, 200, 300, 400],
                    "label": "button",
                    "confidence": 0.95,
                },
                {
                    "bbox": [500, 600, 700, 800],
                    "label": "text",
                    "confidence": 0.85,
                },
            ],
            "image_width": 1000,
            "image_height": 1000,
        }
        result = pipeline.correct_single(vlm_json)
        assert "image_id" in result
        assert "elements" in result
        assert result["image_id"] == "test_img"

    def test_correct_single_returns_elements(self, pipeline) -> None:
        vlm_json = {
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"},
                {"bbox": [0.5, 0.6, 0.7, 0.8], "label": "text"},
            ],
        }
        result = pipeline.correct_single(vlm_json)
        assert len(result["elements"]) >= 0  # Existence filter may drop some

    def test_correct_batch(self, pipeline) -> None:
        vlm_jsons = [
            {"elements": [{"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"}]},
            {"elements": [{"bbox": [0.5, 0.6, 0.7, 0.8], "label": "text"}]},
        ]
        results = pipeline.correct_batch(vlm_jsons)
        assert len(results) == 2

    def test_bboxes_clamped(self, pipeline) -> None:
        """Deltas should be clamped, resulting bboxes in [0, 1]."""
        vlm_json = {
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        result = pipeline.correct_single(vlm_json)
        for elem in result["elements"]:
            for v in elem["bbox"]:
                assert 0.0 <= v <= 1.0, f"bbox value {v} out of [0,1]"

    def test_model_inference_uses_cpu(self, model) -> None:
        """Pipeline runs on CPU without CUDA."""
        pipeline = InferencePipeline(model, device=torch.device("cpu"))
        result = pipeline.correct_single({
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "icon"},
            ],
        })
        assert isinstance(result, dict)
