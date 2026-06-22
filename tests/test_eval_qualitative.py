"""Tests for qualitative visualization functions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bipartite_gnn_gui.data.ground_truth import GTElement, GroundTruth
from bipartite_gnn_gui.eval.qualitative import (
    plot_correction_comparison,
    plot_correction_grid,
    plot_error_heatmap,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_gt() -> GroundTruth:
    return GroundTruth(
        elements=[
            GTElement(
                element_id="0", bbox=(0.1, 0.2, 0.4, 0.5),
                element_type="button",
            ),
            GTElement(
                element_id="1", bbox=(0.6, 0.6, 0.9, 0.9),
                element_type="text",
            ),
        ],
        image_path="",
        image_width=800,
        image_height=600,
    )


@pytest.fixture
def sample_vlm_pred() -> dict:
    return {
        "image_id": "test",
        "elements": [
            {"element_id": 0, "bbox": [0.12, 0.22, 0.42, 0.52], "label": "button", "confidence": 0.9, "existence_score": 1.0},
            {"element_id": 1, "bbox": [0.58, 0.58, 0.88, 0.88], "label": "text", "confidence": 0.8, "existence_score": 1.0},
        ],
    }


@pytest.fixture
def sample_model_pred() -> dict:
    return {
        "image_id": "test",
        "elements": [
            {"element_id": 0, "bbox": [0.11, 0.21, 0.41, 0.51], "label": "button", "confidence": 0.9, "existence_score": 1.0},
            {"element_id": 1, "bbox": [0.59, 0.59, 0.89, 0.89], "label": "text", "confidence": 0.8, "existence_score": 1.0},
        ],
    }


# ===================================================================
# plot_correction_comparison
# ===================================================================


class TestPlotCorrectionComparison:
    def test_saves_file(self, tmp_dir, sample_gt, sample_vlm_pred, sample_model_pred) -> None:
        save_path = tmp_dir / "comparison.png"
        plot_correction_comparison(sample_gt, sample_vlm_pred, sample_model_pred, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_empty_gt(self, tmp_dir) -> None:
        gt = GroundTruth(elements=[], image_path="", image_width=800, image_height=600)
        vlm_pred = {"elements": []}
        model_pred = {"elements": []}
        save_path = tmp_dir / "empty_gt.png"
        plot_correction_comparison(gt, vlm_pred, model_pred, str(save_path))
        assert save_path.exists()

    def test_handles_empty_vlm(self, tmp_dir, sample_gt) -> None:
        save_path = tmp_dir / "empty_vlm.png"
        plot_correction_comparison(sample_gt, {"elements": []}, {"elements": []}, str(save_path))
        assert save_path.exists()

    def test_handles_empty_model_pred(self, tmp_dir, sample_gt, sample_vlm_pred) -> None:
        save_path = tmp_dir / "empty_model.png"
        plot_correction_comparison(sample_gt, sample_vlm_pred, {"elements": []}, str(save_path))
        assert save_path.exists()

    def test_default_width_height(self, tmp_dir) -> None:
        """Fallback to 800x600 when image_width/image_height are 0."""
        gt = GroundTruth(
            elements=[GTElement(element_id="0", bbox=(0.1, 0.2, 0.3, 0.4), element_type="btn")],
            image_path="",
            image_width=0,
            image_height=0,
        )
        save_path = tmp_dir / "default_dims.png"
        plot_correction_comparison(gt, {"elements": []}, {"elements": []}, str(save_path))
        assert save_path.exists()


# ===================================================================
# plot_error_heatmap
# ===================================================================


class TestPlotErrorHeatmap:
    def test_saves_file(self, tmp_dir) -> None:
        errors = [
            (0.1, 0.2, 0.05),
            (0.3, 0.4, 0.10),
            (0.5, 0.6, 0.15),
            (0.7, 0.8, 0.20),
            (0.9, 0.1, 0.08),
        ]
        save_path = tmp_dir / "heatmap.png"
        plot_error_heatmap(errors, str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_empty_errors(self, tmp_dir) -> None:
        save_path = tmp_dir / "empty_heatmap.png"
        plot_error_heatmap([], str(save_path))
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_handles_single_error(self, tmp_dir) -> None:
        errors = [(0.5, 0.5, 0.1)]
        save_path = tmp_dir / "single_error.png"
        plot_error_heatmap(errors, str(save_path))
        assert save_path.exists()

    def test_custom_grid_size(self, tmp_dir) -> None:
        errors = [(0.1, 0.1, 0.05), (0.9, 0.9, 0.15)]
        save_path = tmp_dir / "custom_grid.png"
        plot_error_heatmap(errors, str(save_path), grid_size=5)
        assert save_path.exists()


# ===================================================================
# plot_correction_grid
# ===================================================================


class TestPlotCorrectionGrid:
    def test_saves_file(self, tmp_dir, sample_gt, sample_vlm_pred, sample_model_pred) -> None:
        save_path = tmp_dir / "grid.png"
        plot_correction_grid(
            [sample_gt], [sample_vlm_pred], [sample_model_pred],
            str(save_path), n_examples=1,
        )
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_multiple_examples(self, tmp_dir, sample_gt, sample_vlm_pred, sample_model_pred) -> None:
        save_path = tmp_dir / "multi_grid.png"
        plot_correction_grid(
            [sample_gt, sample_gt],
            [sample_vlm_pred, sample_vlm_pred],
            [sample_model_pred, sample_model_pred],
            str(save_path), n_examples=2,
        )
        assert save_path.exists()

    def test_handles_empty_inputs(self, tmp_dir) -> None:
        save_path = tmp_dir / "empty_grid.png"
        plot_correction_grid([], [], [], str(save_path), n_examples=4)
        assert save_path.exists()

    def test_clamps_n_examples(self, tmp_dir, sample_gt, sample_vlm_pred, sample_model_pred) -> None:
        """n_examples larger than available data should clamp."""
        save_path = tmp_dir / "clamped_grid.png"
        plot_correction_grid(
            [sample_gt], [sample_vlm_pred], [sample_model_pred],
            str(save_path), n_examples=100,
        )
        assert save_path.exists()

    def test_handles_empty_elements(self, tmp_dir) -> None:
        gt = GroundTruth(elements=[], image_path="", image_width=800, image_height=600)
        save_path = tmp_dir / "empty_elem_grid.png"
        plot_correction_grid(
            [gt], [{"elements": []}], [{"elements": []}],
            str(save_path), n_examples=1,
        )
        assert save_path.exists()
