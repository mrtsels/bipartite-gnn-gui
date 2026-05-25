"""Tests for VLM output parsing — VLMOutputElement, VLMOutput, normalisation,
and model-specific parsers (Qwen3.5-2B, MiniMax-VL-01)."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from bipartite_gnn_gui.data.vlm_output import (
    ELEMENT_TYPES,
    VLMOutput,
    VLMOutputElement,
    VlmParseError,
    normalize_bbox,
    normalize_element_type,
    parse_minimax_output,
    parse_qwen_output,
)

# ---------------------------------------------------------------------------
# Inline sample data
# ---------------------------------------------------------------------------

QWEN_SAMPLE: Dict[str, Any] = {
    "image_id": "login_screen.png",
    "elements": [
        {
            "bbox_xyxy": [0.25, 0.10, 0.75, 0.18],
            "label": "text",
            "text": "Welcome Back",
            "confidence": 0.99,
        },
        {
            "bbox_xyxy": [0.20, 0.30, 0.80, 0.38],
            "label": "input",
            "text": "Enter username",
            "confidence": 0.95,
        },
        {
            "bbox_xyxy": [0.35, 0.58, 0.65, 0.65],
            "label": "button",
            "text": "Sign In",
            "confidence": 0.97,
        },
    ],
}

MINIMAX_SAMPLE: Dict[str, Any] = {
    "image_id": "modal_example.png",
    "image_width": 1440,
    "image_height": 900,
    "elements": [
        {
            "bbox": [576, 18, 864, 882],
            "category": "modal",
            "confidence": 0.96,
            "text_content": None,
            "attributes": {"backdrop": True},
        },
        {
            "bbox": [820, 36, 848, 64],
            "category": "icon",
            "confidence": 0.88,
            "text_content": "\u2715",
            "attributes": {"role": "close", "clickable": True},
        },
        {
            "bbox": [600, 80, 840, 120],
            "category": "text",
            "confidence": 0.99,
            "text_content": "Delete Confirmation",
            "attributes": {"font_weight": "bold"},
        },
        {
            "bbox": [600, 230, 700, 265],
            "category": "button",
            "confidence": 0.94,
            "text_content": "Cancel",
            "attributes": {"role": "secondary"},
        },
    ],
}


# ---------------------------------------------------------------------------
# VLMOutputElement
# ---------------------------------------------------------------------------


class TestVLMOutputElement:
    def test_minimal_creation(self) -> None:
        elem = VLMOutputElement(
            element_id=0,
            bbox=(0.1, 0.2, 0.3, 0.4),
            element_type="button",
        )
        assert elem.element_id == 0
        assert elem.bbox == (0.1, 0.2, 0.3, 0.4)
        assert elem.element_type == "button"
        assert elem.text_content is None
        assert elem.confidence == 1.0
        assert elem.attributes == {}
        assert elem.source == ""

    def test_full_creation(self) -> None:
        elem = VLMOutputElement(
            element_id=5,
            bbox=(0.0, 0.0, 1.0, 1.0),
            element_type="input",
            text_content="hello",
            confidence=0.85,
            attributes={"placeholder": "type here"},
            source="qwen3.5-2b",
        )
        assert elem.element_id == 5
        assert elem.bbox == (0.0, 0.0, 1.0, 1.0)
        assert elem.element_type == "input"
        assert elem.text_content == "hello"
        assert elem.confidence == 0.85
        assert elem.attributes == {"placeholder": "type here"}
        assert elem.source == "qwen3.5-2b"

    def test_attributes_is_copy(self) -> None:
        """Default factory should produce a fresh dict each time."""
        e1 = VLMOutputElement(element_id=0, bbox=(0, 0, 1, 1), element_type="text")
        e2 = VLMOutputElement(element_id=1, bbox=(0, 0, 1, 1), element_type="text")
        assert e1.attributes is not e2.attributes

    def test_bbox_tuple_immutability(self) -> None:
        """bbox should be a tuple (immutable)."""
        elem = VLMOutputElement(
            element_id=0, bbox=(0.1, 0.2, 0.3, 0.4), element_type="icon"
        )
        assert isinstance(elem.bbox, tuple)


# ---------------------------------------------------------------------------
# VLMOutput
# ---------------------------------------------------------------------------


class TestVLMOutput:
    def test_defaults(self) -> None:
        out = VLMOutput()
        assert out.image_id == ""
        assert out.elements == []
        assert out.model_name == ""
        assert out.image_width == 0
        assert out.image_height == 0
        assert out.timestamp == ""
        assert out.parse_errors == []

    def test_with_elements(self) -> None:
        elem = VLMOutputElement(
            element_id=0, bbox=(0.1, 0.2, 0.3, 0.4), element_type="button"
        )
        out = VLMOutput(
            image_id="test.png",
            elements=[elem],
            model_name="qwen3.5-2b",
            image_width=1920,
            image_height=1080,
            timestamp="2026-05-25T10:00:00Z",
        )
        assert out.image_id == "test.png"
        assert len(out.elements) == 1
        assert out.model_name == "qwen3.5-2b"
        assert out.image_height == 1080

    def test_elements_is_copy(self) -> None:
        """Default factory should produce a fresh list each time."""
        o1 = VLMOutput()
        o2 = VLMOutput()
        assert o1.elements is not o2.elements
        assert o1.parse_errors is not o2.parse_errors


# ---------------------------------------------------------------------------
# normalize_element_type
# ---------------------------------------------------------------------------


class TestNormalizeElementType:
    def test_canonical_lowercase(self) -> None:
        assert normalize_element_type("button") == "button"
        assert normalize_element_type("text") == "text"

    def test_case_insensitive(self) -> None:
        assert normalize_element_type("BUTTON") == "button"
        assert normalize_element_type("Button") == "button"
        assert normalize_element_type("InPuT") == "input"

    def test_aliases(self) -> None:
        assert normalize_element_type("btn") == "button"
        assert normalize_element_type("img") == "image"
        assert normalize_element_type("textbox") == "input"
        assert normalize_element_type("dialog") == "modal"
        assert normalize_element_type("separator") == "divider"

    def test_unknown_maps_to_other(self) -> None:
        assert normalize_element_type("foobar_widget") == "other"
        assert normalize_element_type("") == "other"

    def test_whitespace_handling(self) -> None:
        assert normalize_element_type("  button  ") == "button"

    def test_all_canonical_keys_roundtrip(self) -> None:
        for canonical in ELEMENT_TYPES:
            assert normalize_element_type(canonical) == canonical
            assert normalize_element_type(canonical.upper()) == canonical


# ---------------------------------------------------------------------------
# normalize_bbox
# ---------------------------------------------------------------------------


class TestNormalizeBbox:
    def test_xyxy_already_normalized(self) -> None:
        result = normalize_bbox([0.1, 0.2, 0.8, 0.9], format="xyxy")
        assert result == (0.1, 0.2, 0.8, 0.9)

    def test_xyxy_pixel_to_normalized(self) -> None:
        result = normalize_bbox([100, 200, 800, 900], format="xyxy", img_width=1000, img_height=1000)
        assert result == pytest.approx((0.1, 0.2, 0.8, 0.9), rel=1e-5)

    def test_xywh_to_normalized(self) -> None:
        result = normalize_bbox([0.1, 0.2, 0.3, 0.4], format="xywh")
        # (0.1, 0.2, 0.1+0.3=0.4, 0.2+0.4=0.6)
        assert result == pytest.approx((0.1, 0.2, 0.4, 0.6), rel=1e-5)

    def test_cxcywh_to_normalized(self) -> None:
        result = normalize_bbox([0.5, 0.5, 0.4, 0.4], format="cxcywh")
        # (0.5-0.2=0.3, 0.5-0.2=0.3, 0.5+0.2=0.7, 0.5+0.2=0.7)
        assert result == pytest.approx((0.3, 0.3, 0.7, 0.7), rel=1e-5)

    def test_clamp_values(self) -> None:
        result = normalize_bbox([-0.1, 1.2, 0.5, 0.6], format="xyxy")
        assert result == (0.0, 1.0, 0.5, 0.6)

    def test_pixel_normalization_with_different_dimensions(self) -> None:
        result = normalize_bbox([100, 50, 300, 150], format="xyxy", img_width=400, img_height=200)
        assert result == pytest.approx((0.25, 0.25, 0.75, 0.75), rel=1e-5)

    def test_zero_image_size_no_normalization(self) -> None:
        """When img_width/img_height are 0, no pixel normalization is applied."""
        result = normalize_bbox([0.1, 0.2, 0.3, 0.4], format="xyxy", img_width=0, img_height=0)
        assert result == (0.1, 0.2, 0.3, 0.4)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(VlmParseError, match="bbox must have 4 elements"):
            normalize_bbox([0.1, 0.2, 0.3], format="xyxy")

    def test_empty_list_raises(self) -> None:
        with pytest.raises(VlmParseError, match="bbox must have 4 elements"):
            normalize_bbox([], format="xyxy")

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(VlmParseError, match="Unknown bbox format"):
            normalize_bbox([0, 0, 1, 1], format="invalid")

    def test_xywh_pixel_to_normalized(self) -> None:
        """xywh in pixels, then normalized by image dimensions."""
        result = normalize_bbox([50, 30, 100, 60], format="xywh", img_width=200, img_height=120)
        # xyxy: (50, 30, 150, 90) → normalized: (0.25, 0.25, 0.75, 0.75)
        assert result == pytest.approx((0.25, 0.25, 0.75, 0.75), rel=1e-5)

    def test_cxcywh_pixel_to_normalized(self) -> None:
        """cxcywh in pixels, then normalized by image dimensions."""
        result = normalize_bbox([100, 60, 100, 60], format="cxcywh", img_width=200, img_height=120)
        # xyxy: (50, 30, 150, 90) → normalized: (0.25, 0.25, 0.75, 0.75)
        assert result == pytest.approx((0.25, 0.25, 0.75, 0.75), rel=1e-5)

    def test_default_format_is_xyxy(self) -> None:
        """The default format should be xyxy."""
        result = normalize_bbox([0.1, 0.2, 0.3, 0.4])
        assert result == (0.1, 0.2, 0.3, 0.4)


# ---------------------------------------------------------------------------
# parse_qwen_output
# ---------------------------------------------------------------------------


class TestParseQwenOutput:
    def test_basic_parse(self) -> None:
        result = parse_qwen_output(QWEN_SAMPLE)
        assert result.image_id == "login_screen.png"
        assert result.model_name == "qwen3.5-2b"
        assert len(result.elements) == 3
        assert result.image_width == 0
        assert result.image_height == 0
        assert result.parse_errors == []

    def test_element_fields(self) -> None:
        result = parse_qwen_output(QWEN_SAMPLE)
        elem = result.elements[0]
        assert elem.element_id == 0
        assert elem.bbox == (0.25, 0.10, 0.75, 0.18)
        assert elem.element_type == "text"
        assert elem.text_content == "Welcome Back"
        assert elem.confidence == 0.99
        assert elem.attributes == {}
        assert elem.source == "qwen3.5-2b"

    def test_sequential_ids(self) -> None:
        result = parse_qwen_output(QWEN_SAMPLE)
        for i, elem in enumerate(result.elements):
            assert elem.element_id == i

    def test_missing_confidence_defaults_to_one(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert result.elements[0].confidence == 1.0

    def test_empty_text_becomes_none(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button", "text": ""},
            ],
        }
        result = parse_qwen_output(data)
        assert result.elements[0].text_content is None
        assert "empty text_content" in result.parse_errors[0]

    def test_unknown_type_mapped_to_other(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "weird_type"},
            ],
        }
        result = parse_qwen_output(data)
        assert len(result.elements) == 1
        assert result.elements[0].element_type == "other"
        assert any("unknown type" in e for e in result.parse_errors)

    def test_degenerate_bbox_skipped(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.3, 0.1, 0.1, 0.4], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert len(result.elements) == 0
        assert any("degenerate" in e for e in result.parse_errors)

    def test_missing_bbox_skipped(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert len(result.elements) == 0
        assert any("missing or invalid bbox" in e for e in result.parse_errors)

    def test_missing_label_skipped(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4]},
            ],
        }
        result = parse_qwen_output(data)
        assert len(result.elements) == 0
        assert any("missing or empty label" in e for e in result.parse_errors)

    def test_supports_old_bbox_field_name(self) -> None:
        """Older Qwen versions use 'bbox' instead of 'bbox_xyxy'."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert len(result.elements) == 1
        assert result.elements[0].bbox == (0.1, 0.2, 0.3, 0.4)

    def test_not_a_dict_raises(self) -> None:
        with pytest.raises(VlmParseError, match="Expected dict"):
            parse_qwen_output("not a dict")  # type: ignore[arg-type]

    def test_missing_elements_raises(self) -> None:
        with pytest.raises(VlmParseError, match="Missing or invalid 'elements'"):
            parse_qwen_output({"image_id": "test.png"})

    def test_elements_not_a_list_raises(self) -> None:
        with pytest.raises(VlmParseError, match="Missing or invalid 'elements'"):
            parse_qwen_output({"elements": "not_a_list"})


# ---------------------------------------------------------------------------
# parse_minimax_output
# ---------------------------------------------------------------------------


class TestParseMinimaxOutput:
    def test_basic_parse(self) -> None:
        result = parse_minimax_output(MINIMAX_SAMPLE)
        assert result.image_id == "modal_example.png"
        assert result.model_name == "minimax-vl-01"
        assert len(result.elements) == 4
        assert result.image_width == 1440
        assert result.image_height == 900
        assert result.parse_errors == []

    def test_pixel_normalization(self) -> None:
        result = parse_minimax_output(MINIMAX_SAMPLE)
        # First element: bbox=[576, 18, 864, 882] / (1440, 900)
        elem = result.elements[0]
        expected_x1 = 576.0 / 1440.0
        expected_y1 = 18.0 / 900.0
        expected_x2 = 864.0 / 1440.0
        expected_y2 = 882.0 / 900.0
        assert elem.bbox == pytest.approx(
            (expected_x1, expected_y1, expected_x2, expected_y2), rel=1e-5
        )
        assert elem.element_type == "modal"
        assert elem.text_content is None
        assert elem.confidence == 0.96
        assert elem.attributes == {"backdrop": True}

    def second_element(self) -> None:
        result = parse_minimax_output(MINIMAX_SAMPLE)
        elem = result.elements[1]
        assert elem.element_type == "icon"
        assert elem.text_content == "\u2715"
        assert elem.attributes == {"role": "close", "clickable": True}

    def test_confidence_clamping(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "button",
                    "confidence": 1.5,
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].confidence == 1.0

    def test_null_confidence_default(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "button",
                    "confidence": None,
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].confidence == 1.0

    def test_null_attributes_becomes_empty_dict(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "button",
                    "attributes": None,
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].attributes == {}

    def test_missing_attributes_becomes_empty_dict(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "button",
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].attributes == {}

    def test_compound_category_mapped_by_prefix(self) -> None:
        """MiniMax may output compound types like 'button-primary'."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "button-primary",
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].element_type == "button"

    def test_unknown_category_mapped_to_other(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "nonsense_category",
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].element_type == "other"
        assert any("unknown type" in e for e in result.parse_errors)

    def test_compound_nonsense_prefix_still_other(self) -> None:
        """Compound type with an unrecognized prefix still maps to 'other'."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {
                    "bbox": [10, 10, 50, 50],
                    "category": "foobar-secondary",
                },
            ],
        }
        result = parse_minimax_output(data)
        assert result.elements[0].element_type == "other"

    def test_missing_category_skipped(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 100,
            "image_height": 100,
            "elements": [
                {"bbox": [10, 10, 50, 50]},
            ],
        }
        result = parse_minimax_output(data)
        assert len(result.elements) == 0
        assert any("missing or empty category" in e for e in result.parse_errors)

    def test_zero_image_dimensions_treats_as_normalized(self) -> None:
        """When img_width/img_height are 0, pixel normalization is skipped
        and values are assumed to already be in [0, 1]."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 0,
            "image_height": 0,
            "elements": [
                {
                    "bbox": [0.1, 0.2, 0.3, 0.4],
                    "category": "button",
                },
            ],
        }
        result = parse_minimax_output(data)
        bbox = result.elements[0].bbox
        assert bbox == (0.1, 0.2, 0.3, 0.4)

    def test_not_a_dict_raises(self) -> None:
        with pytest.raises(VlmParseError, match="Expected dict"):
            parse_minimax_output("not a dict")  # type: ignore[arg-type]

    def test_missing_elements_raises(self) -> None:
        with pytest.raises(VlmParseError, match="Missing or invalid 'elements'"):
            parse_minimax_output({"image_id": "test.png"})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_elements_list(self) -> None:
        data: Dict[str, Any] = {"image_id": "empty.png", "elements": []}
        result = parse_qwen_output(data)
        assert len(result.elements) == 0
        assert result.parse_errors == []

    def test_empty_elements_minimax(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "empty.png",
            "image_width": 1920,
            "image_height": 1080,
            "elements": [],
        }
        result = parse_minimax_output(data)
        assert len(result.elements) == 0
        assert result.parse_errors == []

    def test_non_dict_element_in_list(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button"},
                "not a dict",
                {"bbox_xyxy": [0.5, 0.6, 0.7, 0.8], "label": "text"},
            ],
        }
        result = parse_qwen_output(data)
        assert len(result.elements) == 2
        assert any("expected dict" in e for e in result.parse_errors)

    def test_image_dimensions_from_qwen(self) -> None:
        """Qwen may optionally include image dimensions."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "image_width": 800,
            "image_height": 600,
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert result.image_width == 800
        assert result.image_height == 600

    def test_vlm_parse_error_is_value_error(self) -> None:
        """VlmParseError should be a subclass of ValueError."""
        assert issubclass(VlmParseError, ValueError)

    def test_vlm_parse_error_message(self) -> None:
        with pytest.raises(VlmParseError) as exc_info:
            raise VlmParseError("test error")
        assert str(exc_info.value) == "test error"

    def test_qwen_timestamp_passthrough(self) -> None:
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "timestamp": "2026-05-25T12:00:00Z",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert result.timestamp == "2026-05-25T12:00:00Z"

    def test_bbox_out_of_range_clamped(self) -> None:
        """Bbox values slightly outside [0, 1] should be clamped."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [-0.05, 0.0, 1.02, 0.5], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        bbox = result.elements[0].bbox
        assert bbox[0] == 0.0
        assert bbox[2] == 1.0

    def test_missing_image_id_defaults_empty(self) -> None:
        data: Dict[str, Any] = {
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button"},
            ],
        }
        result = parse_qwen_output(data)
        assert result.image_id == ""


# ---------------------------------------------------------------------------
# Integration / realistic scenarios
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_qwen_login_form(self) -> None:
        """Parse a realistic login form (Example 1 from the design doc)."""
        result = parse_qwen_output(QWEN_SAMPLE)
        assert len(result.elements) == 3
        # First element: text "Welcome Back"
        assert result.elements[0].element_type == "text"
        assert result.elements[0].text_content == "Welcome Back"
        assert result.elements[0].bbox == pytest.approx((0.25, 0.10, 0.75, 0.18), rel=1e-5)
        # Second element: input "Enter username"
        assert result.elements[1].element_type == "input"
        assert result.elements[1].text_content == "Enter username"
        # Third element: button "Sign In"
        assert result.elements[2].element_type == "button"
        assert result.elements[2].text_content == "Sign In"
        assert result.elements[2].confidence == 0.97

    def test_minimax_modal(self) -> None:
        """Parse a realistic modal with multiple element types."""
        result = parse_minimax_output(MINIMAX_SAMPLE)
        assert len(result.elements) == 4
        # modal
        assert result.elements[0].element_type == "modal"
        assert result.elements[0].confidence == 0.96
        assert result.elements[0].attributes == {"backdrop": True}
        # icon (close)
        assert result.elements[1].element_type == "icon"
        assert result.elements[1].text_content == "\u2715"
        # text
        assert result.elements[2].element_type == "text"
        assert result.elements[2].text_content == "Delete Confirmation"
        # button (Cancel)
        assert result.elements[3].element_type == "button"
        assert result.elements[3].text_content == "Cancel"

    def test_parse_errors_accumulate(self) -> None:
        """Multiple parse issues should all be recorded."""
        data: Dict[str, Any] = {
            "image_id": "messy.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button"},
                {"bbox_xyxy": [0.5, 0.5, 0.4, 0.6], "label": "text"},
                {"label": "button"},
                {"bbox_xyxy": [0.7, 0.8, 0.9, 0.95], "label": "garbage_type"},
            ],
        }
        result = parse_qwen_output(data)
        # Valid elements: index 0 (button), index 3 (other for garbage_type)
        # Invalid: index 1 (degenerate), index 2 (missing bbox)
        assert len(result.elements) == 2
        assert len(result.parse_errors) >= 3

    def test_confidence_out_of_range_clamped(self) -> None:
        """Confidence values outside [0, 1] should be clamped."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "button", "confidence": -0.5},
            ],
        }
        result = parse_qwen_output(data)
        assert result.elements[0].confidence == 0.0
        data["elements"][0]["confidence"] = 2.5
        result = parse_qwen_output(data)
        assert result.elements[0].confidence == 1.0

    def test_mixed_case_labels(self) -> None:
        """Labels in any case should be normalized."""
        data: Dict[str, Any] = {
            "image_id": "test.png",
            "elements": [
                {"bbox_xyxy": [0.1, 0.2, 0.3, 0.4], "label": "BUTTON"},
                {"bbox_xyxy": [0.5, 0.6, 0.7, 0.8], "label": "CheckBox"},
            ],
        }
        result = parse_qwen_output(data)
        assert result.elements[0].element_type == "button"
        assert result.elements[1].element_type == "checkbox"
