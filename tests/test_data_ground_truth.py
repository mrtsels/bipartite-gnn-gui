"""Tests for ground-truth loading and matching.

Covers GTElement, GroundTruth, format-specific loaders (GUI-360,
ScreenSpot), the factory dispatcher, Hungarian matching, and edge cases.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch
from PIL import Image as PILImage
from torch import Tensor

from bipartite_gnn_gui.data.ground_truth import (
    GTElement,
    GroundTruth,
    GroundTruthParseError,
    _element_to_bbox,
    load_ground_truth,
    load_gui360_annotation,
    load_screenspot_annotation,
    load_screenspot_combined,
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


SCREENSPOT_COMBINED_SAMPLE: List[Dict[str, Any]] = [
    {
        "image": "pc_001.png",
        "annotations": [
            {
                "bounding_box": [100, 200, 50, 30],
                "data_type": "icon",
                "objective_reference": "Save button",
                "data_source": "windows",
            },
            {
                "bounding_box": [300, 400, 100, 40],
                "data_type": "text",
                "objective_reference": "Welcome text",
                "data_source": "windows",
            },
            {
                "bounding_box": [500, 100, 40, 20],
                "data_type": "",
                "objective_reference": "",
                "data_source": "",
            },
        ],
    },
    {
        "image": "pc_002.png",
        "annotations": [],
    },
]

#: Creates synthetic PNG files on disk and returns the directory path.
_SCREENSPOT_IMAGE_SIZE = (1920, 1080)


def _create_screenspot_images(tmp_path: Any) -> str:
    """Create synthetic PNG images in a temporary directory.

    Each image has dimensions ``_SCREENSPOT_IMAGE_SIZE``.
    """
    images_dir = tmp_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for entry in SCREENSPOT_COMBINED_SAMPLE:
        img_path = images_dir / entry["image"]
        img = PILImage.new("RGB", _SCREENSPOT_IMAGE_SIZE, color=(128, 128, 128))
        img.save(img_path)
    return str(images_dir)


def _write_combined_json(tmp_path: Any, data: List[Dict[str, Any]], name: str = "ScreenSpot_combined.json") -> str:
    """Write a combined JSON array to a temporary file."""
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(path)


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


# ---------------------------------------------------------------------------
# load_screenspot_combined
# ---------------------------------------------------------------------------


class TestLoadScreenspotCombined:
    def test_basic(self, tmp_path: Any) -> None:
        """Load synthetic combined JSON and verify list of GroundTruth objects."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_groundtruth_fields(self, tmp_path: Any) -> None:
        """Each GroundTruth should have correct fields."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        gt = results[0]
        assert gt.source == "screenspot"
        assert gt.image_width == _SCREENSPOT_IMAGE_SIZE[0]
        assert gt.image_height == _SCREENSPOT_IMAGE_SIZE[1]
        assert gt.image_path.endswith("pc_001.png")
        assert len(gt.elements) == 3

    def test_xywh_to_xyxy_conversion(self, tmp_path: Any) -> None:
        """bounding_box [x,y,w,h] should be converted to normalised xyxy."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        elem = results[0].elements[0]
        # Input: [100, 200, 50, 30] xywh, image=1920x1080
        # Expected xyxy: x1=100/1920, y1=200/1080, x2=150/1920, y2=230/1080
        w_img, h_img = _SCREENSPOT_IMAGE_SIZE
        assert elem.bbox == pytest.approx((
            100.0 / w_img, 200.0 / h_img,
            150.0 / w_img, 230.0 / h_img,
        ))
        # All values in [0, 1]
        for v in elem.bbox:
            assert 0.0 <= v <= 1.0

    def test_field_name_mapping(self, tmp_path: Any) -> None:
        """data_type→type, objective_reference→text, data_source→group."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        elem = results[0].elements[0]
        assert elem.element_type == "icon"  # data_type → type
        assert elem.text_content == "Save button"  # objective_reference → text
        assert elem.metadata["group"] == "windows"  # data_source → group
        assert elem.source_dataset == "screenspot"

    def test_empty_annotations(self, tmp_path: Any) -> None:
        """Entry with empty annotations list should produce zero elements."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        assert results[1].image_path.endswith("pc_002.png")
        assert len(results[1].elements) == 0

    def test_empty_data_type_maps_to_other(self, tmp_path: Any) -> None:
        """Empty data_type should map to 'other'."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        elem = results[0].elements[2]  # empty data_type
        assert elem.element_type == "other"

    def test_empty_text_becomes_none(self, tmp_path: Any) -> None:
        """Empty objective_reference should become None."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        elem = results[0].elements[2]
        assert elem.text_content is None

    def test_missing_annotations_field(self, tmp_path: Any) -> None:
        """Missing 'annotations' key should default to empty list."""
        images_dir = _create_screenspot_images(tmp_path)
        data = [{"image": "pc_001.png"}]  # No annotations key
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results) == 1
        assert len(results[0].elements) == 0

    def test_missing_image_field(self, tmp_path: Any) -> None:
        """Entry with missing 'image' field should be skipped."""
        images_dir = _create_screenspot_images(tmp_path)
        data = [
            SCREENSPOT_COMBINED_SAMPLE[0],
            {"annotations": []},  # no 'image' key
        ]
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results) == 1  # second entry skipped

    def test_invalid_bbox_skipped(self, tmp_path: Any) -> None:
        """Annotation with malformed bounding_box should be skipped."""
        images_dir = _create_screenspot_images(tmp_path)
        data = [{
            "image": "pc_001.png",
            "annotations": [
                {"bounding_box": [100, 200, 50, 30], "data_type": "icon", "data_source": "win"},
                {"bounding_box": [10, 20], "data_type": "text", "data_source": "win"},  # bad len
                {"bounding_box": "not_a_list", "data_type": "text", "data_source": "win"},
            ],
        }]
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results[0].elements) == 1

    def test_degenerate_bbox_skipped(self, tmp_path: Any) -> None:
        """bbox with zero width or height after xywh→xyxy should be skipped."""
        images_dir = _create_screenspot_images(tmp_path)
        data = [{
            "image": "pc_001.png",
            "annotations": [
                {"bounding_box": [100, 200, 0, 30], "data_type": "icon", "data_source": "win"},
                {"bounding_box": [100, 200, 50, 0], "data_type": "icon", "data_source": "win"},
                {"bounding_box": [100, 200, 50, 30], "data_type": "text", "data_source": "win"},
            ],
        }]
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results[0].elements) == 1  # only the valid one

    def test_non_list_json_raises(self, tmp_path: Any) -> None:
        """Non-array JSON should raise GroundTruthParseError."""
        data = {"image": "test.png", "annotations": []}  # dict, not list
        json_path = _write_combined_json(tmp_path, data, "not_combined.json")
        with pytest.raises(GroundTruthParseError, match="JSON array"):
            load_screenspot_combined(json_path, str(tmp_path))

    def test_image_not_found_skips_entry(self, tmp_path: Any) -> None:
        """Missing image file should skip the entire entry."""
        images_dir = _create_screenspot_images(tmp_path)
        data = [
            SCREENSPOT_COMBINED_SAMPLE[0],  # pc_001.png exists
            {"image": "nonexistent.png", "annotations": []},
        ]
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results) == 1  # second entry skipped

    def test_non_dict_entry_skipped(self, tmp_path: Any) -> None:
        """Non-dict entries in the array should be skipped."""
        images_dir = _create_screenspot_images(tmp_path)
        data: List[Any] = [
            SCREENSPOT_COMBINED_SAMPLE[0],
            "not a dict",
            42,
        ]
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results) == 1

    def test_non_dict_annotation_skipped(self, tmp_path: Any) -> None:
        """Non-dict items in annotations list should be skipped."""
        images_dir = _create_screenspot_images(tmp_path)
        data = [{
            "image": "pc_001.png",
            "annotations": [
                {"bounding_box": [100, 200, 50, 30], "data_type": "icon", "data_source": "win"},
                "not a dict",
                123,
            ],
        }]
        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, images_dir)
        assert len(results[0].elements) == 1

    def test_pil_image_size_reading(self, tmp_path: Any) -> None:
        """Verify that PIL reads correct dimensions from the synthetic images."""
        images_dir = _create_screenspot_images(tmp_path)
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        results = load_screenspot_combined(json_path, images_dir)
        for gt in results:
            assert gt.image_width == _SCREENSPOT_IMAGE_SIZE[0]
            assert gt.image_height == _SCREENSPOT_IMAGE_SIZE[1]

    def test_multiple_images(self, tmp_path: Any) -> None:
        """Test with several images having different annotation counts."""
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)
        data = []
        for i in range(5):
            img_name = f"img_{i:03d}.png"
            img = PILImage.new("RGB", (800, 600), color=(i * 40, i * 40, 200))
            img.save(images_dir / img_name)
            annotations = []
            for j in range(i + 1):  # i+1 annotations per image
                annotations.append({
                    "bounding_box": [j * 50, j * 30, 40, 20],
                    "data_type": "button",
                    "objective_reference": f"elem_{j}",
                    "data_source": "test",
                })
            data.append({"image": img_name, "annotations": annotations})

        json_path = _write_combined_json(tmp_path, data)
        results = load_screenspot_combined(json_path, str(images_dir))
        assert len(results) == 5
        for i, gt in enumerate(results):
            assert len(gt.elements) == i + 1
            assert gt.image_width == 800
            assert gt.image_height == 600


# ---------------------------------------------------------------------------
# load_ground_truth combined-format detection
# ---------------------------------------------------------------------------


class TestLoadGroundTruthCombinedDetection:
    def test_combined_format_raises_clear_error(self, tmp_path: Any) -> None:
        """load_ground_truth should raise when given a JSON array."""
        json_path = _write_combined_json(tmp_path, SCREENSPOT_COMBINED_SAMPLE)
        with pytest.raises(GroundTruthParseError, match="combined JSON array"):
            load_ground_truth(json_path)
