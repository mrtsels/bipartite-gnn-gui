"""Tests for the Evaluator and EvaluationResult."""

from __future__ import annotations

import pytest
import torch

from bipartite_gnn_gui.eval.evaluator import (
    EvaluationResult,
    Evaluator,
)
from bipartite_gnn_gui.eval.metrics import MetricsBundle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    num_samples: int = 2,
    num_elems: int = 3,
) -> dict:
    """Build a fake batch dict mimicking GUIDataset collate output."""
    B = num_samples
    N = num_elems
    # Create dummy boxes: each sample has N elements
    vlm_boxes = torch.rand(B, N, 4)
    gt_boxes = vlm_boxes + 0.01 * torch.randn(B, N, 4)
    gt_boxes = gt_boxes.clamp(0.0, 1.0)
    element_types = torch.randint(0, 5, (B, N))
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    image_ids = [f"img_{i}" for i in range(B)]
    image_sizes = torch.tensor([[1920, 1080] for _ in range(B)], dtype=torch.float32)
    gt_present = [torch.ones(N, dtype=torch.bool) for _ in range(B)]

    return {
        "vlm_boxes": vlm_boxes,
        "gt_boxes": gt_boxes,
        "element_types": element_types,
        "valid_mask": valid_mask,
        "image_ids": image_ids,
        "image_sizes": image_sizes,
        "gt_present": gt_present,
    }


def _make_empty_batch() -> dict:
    """Build a batch where valid_mask is all False."""
    vlm_boxes = torch.zeros(2, 3, 4)
    gt_boxes = torch.zeros(2, 3, 4)
    element_types = torch.zeros(2, 3, dtype=torch.long)
    valid_mask = torch.zeros(2, 3, dtype=torch.bool)
    image_ids = ["img_empty_0", "img_empty_1"]
    return {
        "vlm_boxes": vlm_boxes,
        "gt_boxes": gt_boxes,
        "element_types": element_types,
        "valid_mask": valid_mask,
        "image_ids": image_ids,
        "image_sizes": torch.tensor([[0, 0], [0, 0]], dtype=torch.float32),
        "gt_present": [torch.zeros(0, dtype=torch.bool) for _ in range(2)],
    }


class MockDataLoader:
    """Iterable that yields a fixed list of batches."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)


# ---------------------------------------------------------------------------
# EvaluationResult
# ---------------------------------------------------------------------------


class TestEvaluationResult:
    def test_defaults(self) -> None:
        er = EvaluationResult()
        assert er.global_metrics.recall == 0.0
        assert er.per_category == {}
        assert er.per_source == {}
        assert er.per_image == []
        assert er.config == {}

    def test_with_metrics(self) -> None:
        mb = MetricsBundle(recall=0.9, precision=0.8, f1=0.85)
        er = EvaluationResult(
            global_metrics=mb,
            per_category={"button": mb},
            per_source={"gui360": mb},
            per_image=[{"image_id": "test", "metrics": mb.to_dict()}],
            config={"iou_threshold": 0.5},
        )
        assert er.global_metrics.recall == 0.9
        assert er.per_category["button"].recall == 0.9
        assert er.per_source["gui360"].precision == 0.8
        assert len(er.per_image) == 1
        assert er.per_image[0]["image_id"] == "test"


# ---------------------------------------------------------------------------
# Evaluator — basic
# ---------------------------------------------------------------------------


class TestEvaluatorBasic:
    def test_constructor_defaults(self) -> None:
        e = Evaluator()
        assert e.iou_threshold == 0.5
        assert e.alignment_tolerance == 0.02
        assert len(e.taxonomy) == 20

    def test_constructor_custom_config(self) -> None:
        config = {
            "iou_threshold": 0.75,
            "alignment_tolerance": 0.01,
            "taxonomy": ["button", "text", "image"],
        }
        e = Evaluator(config=config)
        assert e.iou_threshold == 0.75
        assert e.alignment_tolerance == 0.01
        assert e.taxonomy == ["button", "text", "image"]

    def test_constructor_with_model(self) -> None:
        model = object()
        e = Evaluator(model=model)
        assert e.model is model


# ---------------------------------------------------------------------------
# Evaluator — evaluate
# ---------------------------------------------------------------------------


class TestEvaluatorEvaluate:
    def test_single_batch_no_model(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        assert isinstance(result.global_metrics, MetricsBundle)
        assert 0.0 <= result.global_metrics.recall <= 1.0
        assert result.config["num_samples"] == 2
        # Per-image results
        assert len(result.per_image) == 2
        assert result.per_image[0]["image_id"] == "img_0"

    def test_multiple_batches(self) -> None:
        batches = [
            _make_batch(num_samples=2, num_elems=3),
            _make_batch(num_samples=1, num_elems=5),
        ]
        dl = MockDataLoader(batches)
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        assert result.config["num_samples"] == 3
        assert len(result.per_image) == 3

    def test_per_category_breakdown(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        # element_types are 0..4, so we should have per-category results
        assert len(result.per_category) > 0
        for cat_name, mb in result.per_category.items():
            assert isinstance(mb, MetricsBundle)
            assert 0.0 <= mb.recall <= 1.0

    def test_per_source_breakdown(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        source_map = {"img_0": "gui360", "img_1": "screenspot"}
        evaluator = Evaluator()
        result = evaluator.evaluate(dl, source_map=source_map)

        assert "gui360" in result.per_source
        assert "screenspot" in result.per_source
        for src_name, mb in result.per_source.items():
            assert isinstance(mb, MetricsBundle)

    def test_empty_batch(self) -> None:
        batch = _make_empty_batch()
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        # No valid elements, global metrics should be zero
        assert result.global_metrics.recall == 0.0
        assert result.config["num_samples"] == 0

    def test_with_model_override(self) -> None:
        batch = _make_batch(num_samples=1, num_elems=3)
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl, model=None)
        assert isinstance(result.global_metrics, MetricsBundle)

    def test_no_valid_mask_key(self) -> None:
        """Batch without valid_mask should still work."""
        boxes = torch.rand(1, 4, 4)
        batch = {
            "vlm_boxes": boxes,
            "gt_boxes": boxes,
            "element_types": torch.zeros(1, 4, dtype=torch.long),
            "image_ids": ["img_0"],
        }
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)
        assert result.config["num_samples"] == 1


# ---------------------------------------------------------------------------
# Evaluator — evaluate_model_on_data
# ---------------------------------------------------------------------------


class TestEvaluateModelOnData:
    def test_returns_dict(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        metrics = Evaluator.evaluate_model_on_data(None, dl)
        assert isinstance(metrics, dict)
        assert "recall" in metrics
        assert "f1" in metrics

    def test_all_keys_present(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        metrics = Evaluator.evaluate_model_on_data(None, dl)
        for key in ("recall", "precision", "f1", "position_error",
                     "size_error", "alignment_error"):
            assert key in metrics
            assert isinstance(metrics[key], float)


# ---------------------------------------------------------------------------
# Evaluator — print_report
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_does_not_crash(self, capsys) -> None:
        mb = MetricsBundle(
            recall=0.95, precision=0.85, f1=0.90,
            position_error=0.01, size_error=0.02, alignment_error=0.005,
        )
        er = EvaluationResult(
            global_metrics=mb,
            per_category={
                "button": MetricsBundle(recall=0.9, precision=0.8, f1=0.85),
                "text": MetricsBundle(recall=0.95, precision=0.9, f1=0.92),
            },
            per_source={
                "gui360": MetricsBundle(recall=0.93, precision=0.83, f1=0.88),
            },
            config={"num_samples": 10},
        )
        Evaluator.print_report(er)
        out = capsys.readouterr().out
        assert "Evaluation Report" in out
        assert "Recall:" in out
        assert "F1 Score:" in out
        assert "button" in out
        assert "gui360" in out

    def test_print_report_empty(self, capsys) -> None:
        er = EvaluationResult()
        Evaluator.print_report(er)
        out = capsys.readouterr().out
        assert "Evaluation Report" in out
        assert "0" in out  # num_samples: 0


# ---------------------------------------------------------------------------
# Evaluator — per_image
# ---------------------------------------------------------------------------


class TestPerImage:
    def test_per_image_metrics(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        assert len(result.per_image) == 2
        for entry in result.per_image:
            assert "image_id" in entry
            assert "metrics" in entry
            for key in ("recall", "precision", "f1"):
                assert key in entry["metrics"]

    def test_per_image_ids_match(self) -> None:
        batch = _make_batch(num_samples=2, num_elems=3)
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        ids = [entry["image_id"] for entry in result.per_image]
        assert ids == ["img_0", "img_1"]


# ---------------------------------------------------------------------------
# Evaluator — edge cases
# ---------------------------------------------------------------------------


class TestEvaluatorEdgeCases:
    def test_single_image_single_element(self) -> None:
        """Single image with just one element."""
        batch = {
            "vlm_boxes": torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
            "gt_boxes": torch.tensor([[[0.1, 0.2, 0.3, 0.4]]]),
            "element_types": torch.tensor([[3]]),
            "valid_mask": torch.tensor([[True]]),
            "image_ids": ["only_one"],
            "image_sizes": torch.tensor([[1920, 1080]], dtype=torch.float32),
            "gt_present": [torch.ones(1, dtype=torch.bool)],
        }
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        assert result.global_metrics.recall == pytest.approx(1.0)
        assert result.global_metrics.position_error == pytest.approx(0.0, abs=1e-4)
        assert len(result.per_image) == 1

    def test_disjoint_boxes(self) -> None:
        """Predictions nowhere near ground truth."""
        batch = {
            "vlm_boxes": torch.tensor([[[0.0, 0.0, 0.1, 0.1]]]),
            "gt_boxes": torch.tensor([[[0.9, 0.9, 1.0, 1.0]]]),
            "element_types": torch.tensor([[0]]),
            "valid_mask": torch.tensor([[True]]),
            "image_ids": ["disjoint"],
            "image_sizes": torch.tensor([[1920, 1080]], dtype=torch.float32),
            "gt_present": [torch.ones(1, dtype=torch.bool)],
        }
        dl = MockDataLoader([batch])
        evaluator = Evaluator()
        result = evaluator.evaluate(dl)

        assert result.global_metrics.recall == pytest.approx(0.0)
        assert result.global_metrics.precision == pytest.approx(0.0)
        assert result.global_metrics.f1 == pytest.approx(0.0)

    def test_source_map_unknown(self) -> None:
        """Source map with missing keys should go to 'unknown'."""
        batch = _make_batch(num_samples=2, num_elems=2)
        dl = MockDataLoader([batch])
        source_map = {}  # no matches
        evaluator = Evaluator()
        result = evaluator.evaluate(dl, source_map=source_map)

        assert "unknown" in result.per_source
