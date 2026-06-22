"""Tests for baseline evaluators (NoOp, Identity, RandomJitter)."""

from __future__ import annotations

import copy

import pytest
import torch

from bipartite_gnn_gui.eval.baselines import (
    IdentityBaseline,
    NoOpBaseline,
    RandomJitterBaseline,
    _extract_elements,
)


# ===================================================================
# Helper tests
# ===================================================================


class TestExtractElements:
    """Tests for the internal _extract_elements helper."""

    def test_normalises_pixel_coords(self) -> None:
        vlm_json = {
            "elements": [
                {"bbox": [100, 200, 300, 400], "label": "button"},
            ],
            "image_width": 1000,
            "image_height": 1000,
        }
        elems = _extract_elements(vlm_json)
        assert len(elems) == 1
        assert elems[0]["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert elems[0]["label"] == "button"

    def test_passes_through_normalised(self) -> None:
        vlm_json = {
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "text"},
            ],
        }
        elems = _extract_elements(vlm_json)
        assert elems[0]["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_handles_missing_fields(self) -> None:
        vlm_json = {"elements": [{}]}
        elems = _extract_elements(vlm_json)
        # Missing bbox should be skipped
        assert len(elems) == 0

    def test_handles_empty_list(self) -> None:
        assert _extract_elements({"elements": []}) == []

    def test_handles_missing_elements_key(self) -> None:
        assert _extract_elements({}) == []

    def test_skips_non_dict_items(self) -> None:
        vlm_json = {"elements": [None, "string", 42]}
        assert _extract_elements(vlm_json) == []

    def test_clamps_bbox(self) -> None:
        vlm_json = {
            "elements": [
                {"bbox": [-0.1, 1.2, 0.5, 0.6], "label": "btn"},
            ],
        }
        elems = _extract_elements(vlm_json)
        assert elems[0]["bbox"][0] == 0.0
        assert elems[0]["bbox"][1] == 1.0

    def test_output_format(self) -> None:
        vlm_json = {
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button", "confidence": 0.9},
            ],
        }
        elems = _extract_elements(vlm_json)
        assert set(elems[0].keys()) == {
            "element_id", "bbox", "label", "confidence", "existence_score",
        }
        assert elems[0]["confidence"] == 0.9
        assert elems[0]["existence_score"] == 1.0


# ===================================================================
# NoOpBaseline
# ===================================================================


class TestNoOpBaseline:
    def test_preserves_input(self) -> None:
        baseline = NoOpBaseline({}, torch.device("cpu"))
        vlm_json = {
            "image_id": "test001",
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"},
                {"bbox": [0.5, 0.5, 0.8, 0.9], "label": "text", "confidence": 0.8},
            ],
        }
        result = baseline.correct_single(vlm_json)
        assert result["image_id"] == "test001"
        assert len(result["elements"]) == 2
        assert result["elements"][0]["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])
        assert result["elements"][1]["bbox"] == pytest.approx([0.5, 0.5, 0.8, 0.9])
        assert result["model"] == "noop"

    def test_handles_empty_elements(self) -> None:
        baseline = NoOpBaseline({}, torch.device("cpu"))
        result = baseline.correct_single({"image_id": "empty", "elements": []})
        assert result["elements"] == []
        assert result["image_id"] == "empty"

    def test_batch(self) -> None:
        baseline = NoOpBaseline({}, torch.device("cpu"))
        jsons = [
            {"image_id": "a", "elements": []},
            {"image_id": "b", "elements": []},
        ]
        results = baseline.correct_batch(jsons)
        assert len(results) == 2
        assert results[0]["image_id"] == "a"
        assert results[1]["image_id"] == "b"

    def test_output_has_expected_keys(self) -> None:
        baseline = NoOpBaseline({}, torch.device("cpu"))
        result = baseline.correct_single({
            "elements": [{"bbox": [0.0, 0.0, 0.1, 0.1], "label": "btn"}],
        })
        assert "image_id" in result
        assert "elements" in result
        assert "model" in result
        for elem in result["elements"]:
            assert all(k in elem for k in ("element_id", "bbox", "label", "confidence", "existence_score"))


# ===================================================================
# IdentityBaseline
# ===================================================================


class TestIdentityBaseline:
    def test_returns_gt_positions(self) -> None:
        gt_lookup = {
            "img_001": [
                {"bbox": (0.0, 0.0, 0.5, 0.5), "label": "button"},
                {"bbox": (0.6, 0.7, 0.9, 0.95), "label": "text"},
            ],
        }
        baseline = IdentityBaseline({}, torch.device("cpu"), gt_lookup=gt_lookup)
        vlm_json = {
            "image_id": "img_001",
            "elements": [
                {"bbox": [0.05, 0.05, 0.45, 0.45], "label": "btn"},
            ],
        }
        result = baseline.correct_single(vlm_json)
        assert result["image_id"] == "img_001"
        assert len(result["elements"]) == 2
        assert result["elements"][0]["bbox"] == [0.0, 0.0, 0.5, 0.5]
        assert result["elements"][1]["bbox"] == [0.6, 0.7, 0.9, 0.95]
        assert result["model"] == "identity"

    def test_gt_with_element_type_fallback(self) -> None:
        """IdentityBaseline handles GTElement-style dicts with element_type."""
        gt_lookup = {
            "img_x": [
                {"bbox": (0.1, 0.2, 0.3, 0.4), "element_type": "icon"},
            ],
        }
        baseline = IdentityBaseline({}, torch.device("cpu"), gt_lookup=gt_lookup)
        result = baseline.correct_single({"image_id": "img_x", "elements": []})
        assert result["elements"][0]["label"] == "icon"

    def test_fallback_to_vlm_when_gt_missing(self) -> None:
        baseline = IdentityBaseline({}, torch.device("cpu"))
        vlm_json = {
            "image_id": "unknown_img",
            "elements": [{"bbox": [0.1, 0.2, 0.3, 0.4], "label": "btn"}],
        }
        result = baseline.correct_single(vlm_json)
        assert result["elements"][0]["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_set_gt_lookup(self) -> None:
        baseline = IdentityBaseline({}, torch.device("cpu"))
        baseline.set_gt_lookup({
            "img_y": [{"bbox": (0.0, 0.0, 1.0, 1.0), "label": "fullscreen"}],
        })
        result = baseline.correct_single({"image_id": "img_y", "elements": []})
        assert result["elements"][0]["bbox"] == [0.0, 0.0, 1.0, 1.0]

    def test_batch(self) -> None:
        gt_lookup = {
            "a": [{"bbox": (0.0, 0.0, 0.5, 0.5), "label": "btn"}],
            "b": [{"bbox": (0.5, 0.5, 1.0, 1.0), "label": "text"}],
        }
        baseline = IdentityBaseline({}, torch.device("cpu"), gt_lookup=gt_lookup)
        jsons = [
            {"image_id": "a", "elements": []},
            {"image_id": "b", "elements": []},
        ]
        results = baseline.correct_batch(jsons)
        assert len(results) == 2
        assert results[0]["elements"][0]["bbox"] == [0.0, 0.0, 0.5, 0.5]
        assert results[1]["elements"][0]["bbox"] == [0.5, 0.5, 1.0, 1.0]


# ===================================================================
# RandomJitterBaseline
# ===================================================================


class TestRandomJitterBaseline:
    def test_deterministic_with_seed(self) -> None:
        baseline1 = RandomJitterBaseline({}, torch.device("cpu"), seed=42)
        baseline2 = RandomJitterBaseline({}, torch.device("cpu"), seed=42)
        vlm_json = {
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"},
                {"bbox": [0.5, 0.5, 0.8, 0.9], "label": "text"},
            ],
        }
        r1 = baseline1.correct_single(vlm_json)
        r2 = baseline2.correct_single(vlm_json)
        # Same seed = same noise
        for e1, e2 in zip(r1["elements"], r2["elements"]):
            assert e1["bbox"] == pytest.approx(e2["bbox"])

    def test_different_seeds_differ(self) -> None:
        baseline1 = RandomJitterBaseline({}, torch.device("cpu"), seed=1)
        baseline2 = RandomJitterBaseline({}, torch.device("cpu"), seed=2)
        vlm_json = {
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        r1 = baseline1.correct_single(vlm_json)
        r2 = baseline2.correct_single(vlm_json)
        # Extremely unlikely to be identical with different seeds
        assert r1["elements"][0]["bbox"] != r2["elements"][0]["bbox"]

    def test_bbox_stays_in_range(self) -> None:
        baseline = RandomJitterBaseline({}, torch.device("cpu"), seed=99)
        vlm_json = {
            "elements": [
                {"bbox": [0.0, 0.0, 0.01, 0.01], "label": "tiny"},
                {"bbox": [0.99, 0.99, 1.0, 1.0], "label": "corner"},
            ],
        }
        result = baseline.correct_single(vlm_json)
        for elem in result["elements"]:
            for v in elem["bbox"]:
                assert 0.0 <= v <= 1.0, f"bbox value {v} out of [0,1]"

    def test_model_tag(self) -> None:
        baseline = RandomJitterBaseline({}, torch.device("cpu"))
        result = baseline.correct_single({"elements": []})
        assert result["model"] == "random_jitter"

    def test_batch(self) -> None:
        baseline = RandomJitterBaseline({}, torch.device("cpu"), seed=42)
        jsons = [
            {"image_id": "a", "elements": [{"bbox": [0.0, 0.0, 0.1, 0.1], "label": "btn"}]},
            {"image_id": "b", "elements": [{"bbox": [0.2, 0.2, 0.3, 0.3], "label": "text"}]},
        ]
        results = baseline.correct_batch(jsons)
        assert len(results) == 2
        # Different elements -> different noise
        assert results[0]["elements"][0]["bbox"] != results[1]["elements"][0]["bbox"]


# ===================================================================
# Edge cases
# ===================================================================


class TestBaselineEdgeCases:
    def test_noop_with_pixel_coords_normalisation(self) -> None:
        """Verify NoOp normalises pixel coords when image_width/height provided."""
        baseline = NoOpBaseline({}, torch.device("cpu"))
        vlm_json = {
            "elements": [
                {"bbox": [100, 200, 300, 400], "label": "button"},
            ],
            "image_width": 1000,
            "image_height": 1000,
        }
        result = baseline.correct_single(vlm_json)
        assert result["elements"][0]["bbox"] == pytest.approx([0.1, 0.2, 0.3, 0.4])

    def test_random_jitter_same_input_different_calls(self) -> None:
        """Each call to correct_single advances the RNG state."""
        baseline = RandomJitterBaseline({}, torch.device("cpu"), seed=0)
        vlm_json = {
            "elements": [
                {"bbox": [0.5, 0.5, 0.6, 0.6], "label": "btn"},
            ],
        }
        r1 = baseline.correct_single(vlm_json)
        r2 = baseline.correct_single(vlm_json)
        # With deterministic seed, calling twice on same input should
        # NOT give the same result because the RNG advances.
        assert r1["elements"][0]["bbox"] != r2["elements"][0]["bbox"]
