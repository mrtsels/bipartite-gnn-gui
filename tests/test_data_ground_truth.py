"""Tests for ground-truth loading and matching.

Covers GTElement, GroundTruth, format-specific loaders (GUI-360,
ScreenSpot), the factory dispatcher, Hungarian matching, and edge cases.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from torch import Tensor

from bipartite_gnn_gui.data.ground_truth import (
    GTElement,
    GroundTruth,
    GroundTruthParseError,
    _element_to_bbox,
    load_ground_truth,
    load_gui360_annotation,
    load_screenspot_annotation,
    match_predictions_to_ground_truth,
)
from bipartite_gnn_gui.data.vlm_output import VLMOutputElement

# ---------------------------------------------------------------------------
# Inline sample data
# ---------------------------------------------------------------------------

GUI360_SAMPLE: Dict[str, Any] = {
    "image_id": "android_calculator_01",
    "image_width": 1080,
    "image_height": 2340,
    "platform": "android",
    "annotations": [
        {
            "element_id": "calc_btn_7",
            "bbox": [0.10, 0.05, 0.30, 0.12],
            "type": "button",
            "text": "7",
            "attributes": {"clickable": True, "resource_id": "com.example:id/digit_7"},
        },
        {
            "element_id": "calc_display",
            "bbox": [0.05, 0.02, 0.95, 0.08],
            "type": "text",
            "text": "0",
            "attributes": {"is_editable": False},
        },
        {
            "element_id": "empty_text_btn",
            "bbox": [0.40, 0.20, 0.60, 0.28],
            "type": "button",
            "text": "",
            "attributes": {},
        },
    ],
}

SCREENSPOT_SAMPLE: Dict[str, Any] = {
    "image_id": "screenspot_mobile_0001",
    "image_width": 1080,
    "image_height": 1920,
    "group": "mobile",
    "annotations": [
        {
            "element_id": "ss_001_0",
            "bbox": [120, 400, 380, 520],
            "type": "text",
            "text": "Settings",
            "attributes": {"instruction": "Find the Settings menu item"},
        },
        {
            "element_id": "ss_001_1",
            "bbox": [200, 600, 500, 660],
            "type": "button",
            "text": "Save",
            "attributes": {"instruction": "Click the save button"},
        },
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Any, data: Dict[str, Any], name: str = "annotations.json") -> str:
    """Write a dict as JSON to a temporary file and return the path."""
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(path)


def _make_vlm_element(
    bbox: Tuple[float, float, float, float],
    element_type: str = "button",
    element_id: int = 0,
) -> VLMOutputElement:
    return VLMOutputElement(
        element_id=element_id,
        bbox=bbox,
        element_type=element_type,
        text_content=None,
        confidence=1.0,
        attributes={},
        source="test",
    )


def _make_gt_element(
    bbox: Tuple[float, float, float, float],
    element_type: str = "button",
    element_id: str = "gt_0",
) -> GTElement:
    return GTElement(
        element_id=element_id,
        bbox=bbox,
        element_type=element_type,
        text_content=None,
        source_dataset="test",
        metadata={},
    )


# ---------------------------------------------------------------------------
# GTElement
# ---------------------------------------------------------------------------


class TestGTElement:
    def test_defaults(self) -> None:
        """Test that only required fields are needed."""
        elem = GTElement(element_id="e1", bbox=(0.0, 0.0, 1.0, 1.0), element_type="button")
        assert elem.text_content is None
        assert elem.source_dataset == ""
        assert elem.metadata == {}

    def test_construction(self) -> None:
        """Test full construction with all fields."""
        elem = GTElement(
            element_id="e1",
            bbox=(0.1, 0.2, 0.8, 0.9),
            element_type="text",
            text_content="hello",
            source_dataset="gui360",
            metadata={"platform": "android"},
        )
        assert elem.element_id == "e1"
        assert elem.bbox == (0.1, 0.2, 0.8, 0.9)
        assert elem.element_type == "text"
        assert elem.text_content == "hello"
        assert elem.source_dataset == "gui360"
        assert elem.metadata == {"platform": "android"}

    def test_bbox_type(self) -> None:
        """bbox must be a tuple of four floats."""
        elem = GTElement(element_id="e1", bbox=(0.0, 0.0, 1.0, 1.0), element_type="button")
        assert isinstance(elem.bbox, tuple)
        assert len(elem.bbox) == 4
        assert all(isinstance(v, float) for v in elem.bbox)


# ---------------------------------------------------------------------------
# GroundTruth
# ---------------------------------------------------------------------------


class TestGroundTruth:
    def test_defaults(self) -> None:
        gt = GroundTruth()
        assert gt.elements == []
        assert gt.image_path == ""
        assert gt.image_width == 0
        assert gt.image_height == 0
        assert gt.source == ""

    def test_with_elements(self) -> None:
        elem = GTElement(element_id="e1", bbox=(0.0, 0.0, 1.0, 1.0), element_type="button")
        gt = GroundTruth(
            elements=[elem],
            image_path="test.png",
            image_width=100,
            image_height=200,
            source="gui360",
        )
        assert len(gt.elements) == 1
        assert gt.image_path == "test.png"
        assert gt.image_width == 100
        assert gt.image_height == 200
        assert gt.source == "gui360"


# ---------------------------------------------------------------------------
# load_gui360_annotation
# ---------------------------------------------------------------------------


class TestLoadGui360:
    def test_basic(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, GUI360_SAMPLE)
        gt = load_gui360_annotation(path)
        assert gt.source == "gui360"
        assert gt.image_width == 1080
        assert gt.image_height == 2340
        assert gt.image_path == "data/raw/gui360/images/android_calculator_01"
        assert len(gt.elements) == 3

    def test_element_fields(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, GUI360_SAMPLE)
        gt = load_gui360_annotation(path)
        btn = gt.elements[0]
        assert btn.element_id == "calc_btn_7"
        assert btn.bbox == (0.10, 0.05, 0.30, 0.12)
        assert btn.element_type == "button"
        assert btn.text_content == "7"
        assert btn.source_dataset == "gui360"
        assert btn.metadata["platform"] == "android"
        assert btn.metadata["clickable"] is True

    def test_empty_text_becomes_none(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, GUI360_SAMPLE)
        gt = load_gui360_annotation(path)
        empty_elem = gt.elements[2]
        assert empty_elem.element_id == "empty_text_btn"
        assert empty_elem.text_content is None

    def test_empty_annotations(self, tmp_path: Any) -> None:
        data = {
            "image_id": "empty_test",
            "image_width": 100,
            "image_height": 200,
            "platform": "android",
            "annotations": [],
        }
        path = _write_json(tmp_path, data)
        gt = load_gui360_annotation(path)
        assert len(gt.elements) == 0

    def test_missing_annotations_key(self, tmp_path: Any) -> None:
        data = {
            "image_id": "no_ann",
            "image_width": 100,
            "image_height": 200,
            "platform": "ios",
        }
        path = _write_json(tmp_path, data)
        gt = load_gui360_annotation(path)
        assert len(gt.elements) == 0

    def test_type_normalization(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "image_id": "test",
            "image_width": 100,
            "image_height": 200,
            "platform": "web",
            "annotations": [
                {"element_id": "e1", "bbox": [0.1, 0.1, 0.5, 0.5], "type": "Button", "text": ""},
                {"element_id": "e2", "bbox": [0.1, 0.1, 0.5, 0.5], "type": "unknown_xyz", "text": ""},
            ],
        }
        path = _write_json(tmp_path, data)
        gt = load_gui360_annotation(path)
        assert gt.elements[0].element_type == "button"
        assert gt.elements[1].element_type == "other"

    def test_degenerate_bbox_skipped(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "image_id": "test",
            "image_width": 100,
            "image_height": 200,
            "platform": "web",
            "annotations": [
                {"element_id": "good", "bbox": [0.1, 0.1, 0.5, 0.5], "type": "button", "text": ""},
                {"element_id": "bad", "bbox": [0.5, 0.1, 0.1, 0.5], "type": "button", "text": ""},
            ],
        }
        path = _write_json(tmp_path, data)
        gt = load_gui360_annotation(path)
        assert len(gt.elements) == 1
        assert gt.elements[0].element_id == "good"

    def test_image_width_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(GUI360_SAMPLE)
        data["image_width"] = 0
        path = _write_json(tmp_path, data)
        with pytest.raises(GroundTruthParseError, match="image_width must be positive"):
            load_gui360_annotation(path)

    def test_image_height_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(GUI360_SAMPLE)
        data["image_height"] = 0
        path = _write_json(tmp_path, data)
        with pytest.raises(GroundTruthParseError, match="image_height must be positive"):
            load_gui360_annotation(path)


# ---------------------------------------------------------------------------
# load_screenspot_annotation
# ---------------------------------------------------------------------------


class TestLoadScreenspot:
    def test_basic(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, SCREENSPOT_SAMPLE)
        gt = load_screenspot_annotation(path)
        assert gt.source == "screenspot"
        assert gt.image_width == 1080
        assert gt.image_height == 1920
        assert gt.image_path == "data/raw/screenspot/images/screenspot_mobile_0001"
        assert len(gt.elements) == 2

    def test_coordinate_normalization(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, SCREENSPOT_SAMPLE)
        gt = load_screenspot_annotation(path)
        # First element: bbox [120, 400, 380, 520]
        x1, y1, x2, y2 = gt.elements[0].bbox
        assert x1 == pytest.approx(120.0 / 1080.0)
        assert y1 == pytest.approx(400.0 / 1920.0)
        assert x2 == pytest.approx(380.0 / 1080.0)
        assert y2 == pytest.approx(520.0 / 1920.0)
        # All values in [0, 1]
        for v in gt.elements[0].bbox:
            assert 0.0 <= v <= 1.0

    def test_element_fields(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, SCREENSPOT_SAMPLE)
        gt = load_screenspot_annotation(path)
        elem = gt.elements[0]
        assert elem.element_id == "ss_001_0"
        assert elem.element_type == "text"
        assert elem.text_content == "Settings"
        assert elem.source_dataset == "screenspot"
        assert elem.metadata["group"] == "mobile"
        assert elem.metadata["instruction"] == "Find the Settings menu item"

    def test_empty_annotations(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "image_id": "empty_ss",
            "image_width": 100,
            "image_height": 200,
            "group": "desktop",
            "annotations": [],
        }
        path = _write_json(tmp_path, data)
        gt = load_screenspot_annotation(path)
        assert len(gt.elements) == 0

    def test_image_width_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(SCREENSPOT_SAMPLE)
        data["image_width"] = 0
        path = _write_json(tmp_path, data)
        with pytest.raises(GroundTruthParseError, match="image_width must be positive"):
            load_screenspot_annotation(path)

    def test_image_height_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(SCREENSPOT_SAMPLE)
        data["image_height"] = 0
        path = _write_json(tmp_path, data)
        with pytest.raises(GroundTruthParseError, match="image_height must be positive"):
            load_screenspot_annotation(path)

    def test_degenerate_after_normalization_skipped(self, tmp_path: Any) -> None:
        """Bbox with x2 < x1 after normalization should be skipped."""
        data: Dict[str, Any] = {
            "image_id": "test",
            "image_width": 100,
            "image_height": 100,
            "group": "mobile",
            "annotations": [
                {"element_id": "bad", "bbox": [50, 10, 20, 80], "type": "text", "text": ""},
            ],
        }
        path = _write_json(tmp_path, data)
        gt = load_screenspot_annotation(path)
        assert len(gt.elements) == 0


# ---------------------------------------------------------------------------
# Factory: load_ground_truth
# ---------------------------------------------------------------------------


class TestLoadGroundTruth:
    def test_gui360_auto_detect(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, GUI360_SAMPLE, "gui360.json")
        gt = load_ground_truth(path)
        assert gt.source == "gui360"
        assert len(gt.elements) == 3

    def test_screenspot_auto_detect(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, SCREENSPOT_SAMPLE, "screenspot.json")
        gt = load_ground_truth(path)
        assert gt.source == "screenspot"
        assert len(gt.elements) == 2

    def test_explicit_source_gui360(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, GUI360_SAMPLE, "test.json")
        gt = load_ground_truth(path, source="gui360")
        assert gt.source == "gui360"

    def test_explicit_source_screenspot(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, SCREENSPOT_SAMPLE, "test.json")
        gt = load_ground_truth(path, source="screenspot")
        assert gt.source == "screenspot"

    def test_unknown_source_raises(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, GUI360_SAMPLE, "test.json")
        with pytest.raises(GroundTruthParseError, match="Unknown ground-truth source"):
            load_ground_truth(path, source="unknown")

    def test_auto_detect_fails(self, tmp_path: Any) -> None:
        unknown: Dict[str, Any] = {"image_id": "test", "annotations": []}
        path = _write_json(tmp_path, unknown, "unknown.json")
        with pytest.raises(GroundTruthParseError, match="Cannot determine"):
            load_ground_truth(path)


# ---------------------------------------------------------------------------
# match_predictions_to_ground_truth
# ---------------------------------------------------------------------------


class TestMatching:
    def test_perfect_match(self) -> None:
        """One prediction perfectly overlaps one ground truth."""
        preds = [_make_vlm_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button")]
        gts = [_make_gt_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button")]
        matched, fp, fn = match_predictions_to_ground_truth(preds, gts)
        assert len(matched) == 1
        assert matched[0] == (0, 0)
        assert fp == []
        assert fn == []

    def test_partial_match(self) -> None:
        """Two predictions but only one GT — one FP, zero FN."""
        preds = [
            _make_vlm_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button", element_id=0),
            _make_vlm_element(bbox=(0.6, 0.6, 0.9, 0.9), element_type="text", element_id=1),
        ]
        gts = [_make_gt_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button")]
        matched, fp, fn = match_predictions_to_ground_truth(preds, gts)
        # Only one should match (the first pred matches the GT)
        assert len(matched) == 1
        assert len(fp) == 1
        assert len(fn) == 0
        assert 0 in [m[0] for m in matched]  # pred 0 matched
        assert 1 in fp  # pred 1 is FP

    def test_no_match(self) -> None:
        """No overlap between predictions and ground truth."""
        preds = [_make_vlm_element(bbox=(0.0, 0.0, 0.01, 0.01), element_type="button")]
        gts = [_make_gt_element(bbox=(0.9, 0.9, 1.0, 1.0), element_type="button")]
        matched, fp, fn = match_predictions_to_ground_truth(preds, gts, iou_threshold=0.5)
        assert len(matched) == 0
        assert fp == [0]
        assert fn == [0]

    def test_no_predictions(self) -> None:
        """Empty predictions list."""
        preds: List[VLMOutputElement] = []
        gts = [_make_gt_element(bbox=(0.1, 0.1, 0.5, 0.5))]
        matched, fp, fn = match_predictions_to_ground_truth(preds, gts)
        assert matched == []
        assert fp == []
        assert fn == [0]

    def test_no_ground_truth(self) -> None:
        """Empty ground truth list."""
        preds = [_make_vlm_element(bbox=(0.1, 0.1, 0.5, 0.5))]
        gts: List[GTElement] = []
        matched, fp, fn = match_predictions_to_ground_truth(preds, gts)
        assert matched == []
        assert fp == [0]
        assert fn == []

    def test_type_conditioned_blocks_mismatch(self) -> None:
        """With type_conditioned=True, type mismatch prevents matching."""
        preds = [
            _make_vlm_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button", element_id=0),
        ]
        gts = [
            _make_gt_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="text"),
        ]
        matched, fp, fn = match_predictions_to_ground_truth(
            preds, gts, iou_threshold=0.5, type_conditioned=True
        )
        # Perfect IoU overlap, but types differ -> no match
        assert len(matched) == 0
        assert fp == [0]
        assert fn == [0]

    def test_type_conditioned_other_exempt(self) -> None:
        """'other' type is exempt from type_conditioned matching."""
        preds = [
            _make_vlm_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="other", element_id=0),
        ]
        gts = [
            _make_gt_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button"),
        ]
        matched, fp, fn = match_predictions_to_ground_truth(
            preds, gts, iou_threshold=0.5, type_conditioned=True
        )
        # pred type is 'other' -> exempt from type check
        assert len(matched) == 1
        assert fp == []
        assert fn == []

    def test_type_conditioned_allows_match(self) -> None:
        """With type_conditioned=True, matching types still match."""
        preds = [
            _make_vlm_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button", element_id=0),
        ]
        gts = [
            _make_gt_element(bbox=(0.1, 0.1, 0.5, 0.5), element_type="button"),
        ]
        matched, fp, fn = match_predictions_to_ground_truth(
            preds, gts, iou_threshold=0.5, type_conditioned=True
        )
        assert len(matched) == 1
        assert fp == []
        assert fn == []

    def test_hungarian_optimal_assignment(self) -> None:
        """Hungarian should choose the better match over the worse one.

        Two predictions where each overlaps more with one of two GT boxes.
        """
        preds = [
            _make_vlm_element(bbox=(0.0, 0.0, 0.4, 0.4), element_type="button", element_id=0),
            _make_vlm_element(bbox=(0.6, 0.6, 1.0, 1.0), element_type="button", element_id=1),
        ]
        gts = [
            _make_gt_element(bbox=(0.0, 0.0, 0.5, 0.5), element_type="button"),
            _make_gt_element(bbox=(0.5, 0.5, 1.0, 1.0), element_type="button"),
        ]
        matched, fp, fn = match_predictions_to_ground_truth(preds, gts)
        # Each pred should match its closest GT
        assert len(matched) == 2
        assert (0, 0) in matched
        assert (1, 1) in matched
        assert fp == []
        assert fn == []


# ---------------------------------------------------------------------------
# _element_to_bbox helper
# ---------------------------------------------------------------------------


class TestElementToBbox:
    def test_from_vlm_element(self) -> None:
        vlm = _make_vlm_element(bbox=(0.1, 0.2, 0.8, 0.9))
        t = _element_to_bbox(vlm)
        assert isinstance(t, Tensor)
        assert t.shape == (4,)
        assert torch.allclose(t, torch.tensor([0.1, 0.2, 0.8, 0.9]))

    def test_from_gt_element(self) -> None:
        gt = _make_gt_element(bbox=(0.1, 0.2, 0.8, 0.9))
        t = _element_to_bbox(gt)
        assert isinstance(t, Tensor)
        assert t.shape == (4,)
        assert torch.allclose(t, torch.tensor([0.1, 0.2, 0.8, 0.9]))
