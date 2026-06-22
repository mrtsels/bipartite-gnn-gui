"""Tests for the RICO data loader module.

Covers bounds string parsing, Android class-to-type mapping,
leaf node extraction, View Hierarchy and Semantic Annotation
parsers, image ID extraction, and directory loading.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from bipartite_gnn_gui.data.ground_truth import GroundTruthParseError
from bipartite_gnn_gui.data.rico_loader import (
    _find_leaf_nodes,
    get_rico_image_id,
    load_rico_directory,
    parse_rico_bounds,
    parse_rico_semantic,
    parse_rico_view_hierarchy,
    rico_class_to_type,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_path: Any, data: Dict[str, Any], name: str = "test.json") -> str:
    """Write a dict as JSON to a temporary file and return the path."""
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    return str(path)


# ---------------------------------------------------------------------------
# Sample RICO View Hierarchy JSON
# ---------------------------------------------------------------------------

_RICO_VH_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.MainActivity",
    "screen_id": "screenshot_001",
    "screen_width": 1440,
    "screen_height": 2560,
    "root": {
        "bounds": "[0,0][1440,2560]",
        "class": "android.widget.FrameLayout",
        "children": [
            {
                "bounds": "[0,0][1440,2560]",
                "class": "android.widget.LinearLayout",
                "children": [
                    {
                        "bounds": "[50,100][200,300]",
                        "class": "android.widget.Button",
                        "text": "Submit",
                        "content-desc": "",
                        "clickable": True,
                        "visibility": "visible",
                    },
                    {
                        "bounds": "[300,100][500,300]",
                        "class": "android.widget.TextView",
                        "text": "Welcome",
                        "content-desc": "Welcome text",
                        "clickable": False,
                        "visibility": "visible",
                    },
                    {
                        "bounds": "[600,100][800,300]",
                        "class": "android.widget.EditText",
                        "text": "",
                        "content-desc": "Search input",
                        "clickable": True,
                        "visibility": "visible",
                    },
                ],
            },
        ],
    },
}

# View Hierarchy with invisible nodes and zero-area bboxes
_RICO_VH_FILTER_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.Filter",
    "screen_id": "screenshot_filter",
    "screen_width": 720,
    "screen_height": 1280,
    "root": {
        "bounds": "[0,0][720,1280]",
        "class": "android.widget.FrameLayout",
        "children": [
            {
                "bounds": "[10,10][100,100]",
                "class": "android.widget.Button",
                "clickable": True,
                "visibility": "visible",
            },
            {
                "bounds": "[110,10][200,100]",
                "class": "android.widget.Button",
                "clickable": True,
                "visibility": "invisible",
            },
            {
                "bounds": "[10,110][100,200]",
                "class": "android.widget.TextView",
                "text": "Gone node",
                "clickable": False,
                "visibility": "gone",
            },
            {
                "bounds": "[50,210][50,310]",
                "class": "android.widget.ImageView",
                "visibility": "visible",
            },
        ],
    },
}

# View Hierarchy with deep nesting
_RICO_VH_DEEP_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.Deep",
    "screen_id": "screenshot_deep",
    "screen_width": 1080,
    "screen_height": 1920,
    "root": {
        "bounds": "[0,0][1080,1920]",
        "class": "android.widget.FrameLayout",
        "children": [
            {
                "bounds": "[0,0][1080,1920]",
                "class": "android.widget.LinearLayout",
                "children": [
                    {
                        "bounds": "[0,0][1080,200]",
                        "class": "android.widget.RelativeLayout",
                        "children": [
                            {
                                "bounds": "[0,0][1080,100]",
                                "class": "android.widget.LinearLayout",
                                "children": [
                                    {
                                        "bounds": "[10,10][100,90]",
                                        "class": "android.widget.ImageView",
                                        "visibility": "visible",
                                    },
                                ],
                            },
                            {
                                "bounds": "[0,100][1080,200]",
                                "class": "android.widget.FrameLayout",
                                "children": [
                                    {
                                        "bounds": "[500,110][580,190]",
                                        "class": "android.widget.Button",
                                        "text": "OK",
                                        "visibility": "visible",
                                    },
                                ],
                            },
                        ],
                    },
                ],
            },
        ],
    },
}

# Empty View Hierarchy (no visible leaf nodes after filtering)
_RICO_VH_EMPTY_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.Empty",
    "screen_id": "screenshot_empty",
    "screen_width": 1080,
    "screen_height": 1920,
    "root": {
        "bounds": "[0,0][1080,1920]",
        "class": "android.widget.FrameLayout",
        "children": [
            {
                "bounds": "[0,0][1080,1920]",
                "class": "android.widget.LinearLayout",
                "children": [
                    {
                        "bounds": "[0,0][1080,1920]",
                        "class": "android.widget.ScrollView",
                        "visibility": "gone",
                        "children": [],
                    },
                ],
            },
        ],
    },
}

# Semantic Annotation sample
_RICO_SEMANTIC_SAMPLE: Dict[str, Any] = {
    "screen_id": "screenshot_sem",
    "screen_width": 1440,
    "screen_height": 2560,
    "annotations": [
        {
            "element_id": "sem_0",
            "bbox": [50, 100, 200, 300],
            "class": "android.widget.Button",
            "text": "Submit",
            "clickable": True,
        },
        {
            "element_id": "sem_1",
            "bbox": [300, 100, 500, 300],
            "class": "android.widget.TextView",
            "text": "Label",
            "clickable": False,
        },
    ],
}


# ===================================================================
# parse_rico_bounds
# ===================================================================


class TestParseRicoBounds:
    def test_standard_format(self) -> None:
        result = parse_rico_bounds("[0,0][100,200]")
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_with_whitespace(self) -> None:
        result = parse_rico_bounds("  [50,100][200,300]  ")
        assert result == (50.0, 100.0, 200.0, 300.0)

    def test_negative_values(self) -> None:
        result = parse_rico_bounds("[-10,-5][100,200]")
        assert result == (-10.0, -5.0, 100.0, 200.0)

    def test_large_values(self) -> None:
        result = parse_rico_bounds("[0,0][1440,2560]")
        assert result == (0.0, 0.0, 1440.0, 2560.0)

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(GroundTruthParseError, match="Cannot parse"):
            parse_rico_bounds("not a bounds string")

    def test_partial_match_raises(self) -> None:
        with pytest.raises(GroundTruthParseError, match="Cannot parse"):
            parse_rico_bounds("[0,0]")


# ===================================================================
# rico_class_to_type
# ===================================================================


class TestRicoClassToType:
    # Exact matches
    def test_Button(self) -> None:
        assert rico_class_to_type("android.widget.Button") == "button"

    def test_ImageButton(self) -> None:
        assert rico_class_to_type("android.widget.ImageButton") == "icon"

    def test_ImageView(self) -> None:
        assert rico_class_to_type("android.widget.ImageView") == "image"

    def test_TextView(self) -> None:
        assert rico_class_to_type("android.widget.TextView") == "text"

    def test_EditText(self) -> None:
        assert rico_class_to_type("android.widget.EditText") == "input"

    def test_CheckBox(self) -> None:
        assert rico_class_to_type("android.widget.CheckBox") == "icon"

    def test_Switch(self) -> None:
        assert rico_class_to_type("android.widget.Switch") == "icon"

    def test_Spinner(self) -> None:
        assert rico_class_to_type("android.widget.Spinner") == "icon"

    def test_ProgressBar(self) -> None:
        assert rico_class_to_type("android.widget.ProgressBar") == "icon"

    def test_WebView(self) -> None:
        assert rico_class_to_type("android.webkit.WebView") == "container"

    def test_ListView(self) -> None:
        assert rico_class_to_type("android.widget.ListView") == "list"

    def test_ScrollView(self) -> None:
        assert rico_class_to_type("android.widget.ScrollView") == "container"

    # Suffix-based fallbacks (non-standard package prefixes)
    def test_custom_Button(self) -> None:
        assert rico_class_to_type("com.example.MyButton") == "button"

    def test_custom_ImageView(self) -> None:
        assert rico_class_to_type("com.example.MyImageView") == "image"

    def test_custom_EditText(self) -> None:
        assert rico_class_to_type("myapp.widgets.EditText") == "input"

    def test_custom_Switch(self) -> None:
        assert rico_class_to_type("org.lib.Switch") == "icon"

    def test_custom_ListView(self) -> None:
        assert rico_class_to_type("com.test.ListView") == "list"

    # Unknown classes
    def test_unknown_class_returns_other(self) -> None:
        assert rico_class_to_type("android.widget.RelativeLayout") == "other"

    def test_unknown_class_returns_other2(self) -> None:
        assert rico_class_to_type("some.random.Class") == "other"

    def test_empty_string_returns_other(self) -> None:
        assert rico_class_to_type("") == "other"


# ===================================================================
# _find_leaf_nodes
# ===================================================================


class TestFindLeafNodes:
    def test_single_node_no_children(self) -> None:
        node = {"bounds": "[0,0][10,10]", "class": "android.widget.Button"}
        leaves = _find_leaf_nodes(node)
        assert len(leaves) == 1
        assert leaves[0] is node

    def test_single_node_empty_children(self) -> None:
        node: Dict[str, Any] = {
            "bounds": "[0,0][10,10]",
            "class": "android.widget.Button",
            "children": [],
        }
        leaves = _find_leaf_nodes(node)
        assert len(leaves) == 1

    def test_single_node_none_children(self) -> None:
        node: Dict[str, Any] = {
            "bounds": "[0,0][10,10]",
            "class": "android.widget.Button",
            "children": None,
        }
        leaves = _find_leaf_nodes(node)
        assert len(leaves) == 1

    def test_shallow_nesting(self) -> None:
        root: Dict[str, Any] = {
            "bounds": "[0,0][100,200]",
            "class": "android.widget.FrameLayout",
            "children": [
                {"bounds": "[10,10][50,50]", "class": "android.widget.Button"},
                {"bounds": "[60,10][90,50]", "class": "android.widget.TextView"},
            ],
        }
        leaves = _find_leaf_nodes(root)
        assert len(leaves) == 2

    def test_deep_nesting(self) -> None:
        root: Dict[str, Any] = {
            "bounds": "[0,0][100,200]",
            "class": "android.widget.FrameLayout",
            "children": [
                {
                    "bounds": "[0,0][100,200]",
                    "class": "android.widget.LinearLayout",
                    "children": [
                        {
                            "bounds": "[0,0][100,100]",
                            "class": "android.widget.RelativeLayout",
                            "children": [
                                {
                                    "bounds": "[10,10][50,50]",
                                    "class": "android.widget.Button",
                                },
                            ],
                        },
                        {
                            "bounds": "[0,100][100,200]",
                            "class": "android.widget.FrameLayout",
                            "children": [
                                {
                                    "bounds": "[60,110][90,190]",
                                    "class": "android.widget.TextView",
                                },
                            ],
                        },
                    ],
                },
            ],
        }
        leaves = _find_leaf_nodes(root)
        assert len(leaves) == 2

    def test_empty_root_children(self) -> None:
        root: Dict[str, Any] = {
            "bounds": "[0,0][100,200]",
            "class": "android.widget.FrameLayout",
            "children": [],
        }
        leaves = _find_leaf_nodes(root)
        assert len(leaves) == 1  # root itself is a leaf

    def test_ignores_non_dict_children(self) -> None:
        root: Dict[str, Any] = {
            "bounds": "[0,0][100,200]",
            "class": "android.widget.FrameLayout",
            "children": [
                "not a dict",
                {"bounds": "[10,10][50,50]", "class": "android.widget.Button"},
            ],
        }
        leaves = _find_leaf_nodes(root)
        assert len(leaves) == 1


# ===================================================================
# get_rico_image_id
# ===================================================================


class TestGetRicoImageId:
    def test_basic(self) -> None:
        vh_json = {"screen_id": "screenshot_001"}
        assert get_rico_image_id(vh_json) == "screenshot_001.png"

    def test_missing_screen_id(self) -> None:
        vh_json: Dict[str, Any] = {}
        assert get_rico_image_id(vh_json) == ".png"


# ===================================================================
# parse_rico_view_hierarchy
# ===================================================================


class TestParseViewHierarchy:
    def test_basic(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "screenshot_001.json")
        images_dir = str(tmp_path)
        gt = parse_rico_view_hierarchy(path, images_dir)
        assert gt.source == "rico"
        assert gt.image_width == 1440
        assert gt.image_height == 2560
        assert gt.image_path == str(tmp_path / "screenshot_001.png")
        assert len(gt.elements) == 3

    def test_element_fields(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "screenshot_001.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))

        # First element: Button "Submit"
        btn = gt.elements[0]
        assert btn.element_id == "screenshot_001_0"
        assert btn.element_type == "button"
        assert btn.text_content == "Submit"
        assert btn.source_dataset == "rico"
        assert btn.metadata["class"] == "android.widget.Button"
        assert btn.metadata["clickable"] is True

        # Second element: TextView "Welcome"
        tv = gt.elements[1]
        assert tv.element_id == "screenshot_001_1"
        assert tv.element_type == "text"
        assert tv.text_content == "Welcome"

        # Third element: EditText with empty text, falls back to content-desc
        et = gt.elements[2]
        assert et.element_id == "screenshot_001_2"
        assert et.element_type == "input"
        assert et.text_content == "Search input"

    def test_coordinate_normalization(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "screenshot_001.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))

        # First element: bounds "[50,100][200,300]" on 1440x2560
        x1, y1, x2, y2 = gt.elements[0].bbox
        assert x1 == pytest.approx(50.0 / 1440.0)
        assert y1 == pytest.approx(100.0 / 2560.0)
        assert x2 == pytest.approx(200.0 / 1440.0)
        assert y2 == pytest.approx(300.0 / 2560.0)

        # All values in [0, 1]
        for v in gt.elements[0].bbox:
            assert 0.0 <= v <= 1.0

    def test_all_bboxes_in_range(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "screenshot_001.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        for elem in gt.elements:
            for v in elem.bbox:
                assert 0.0 <= v <= 1.0

    def test_visibility_filtering(self, tmp_path: Any) -> None:
        """Only 'visible' nodes should be included."""
        path = _write_json(tmp_path, _RICO_VH_FILTER_SAMPLE, "sfilter.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        # Sample has: 1 visible Button, 1 invisible Button, 1 gone TextView, 1 zero-area ImageView
        # Should only get the visible Button
        assert len(gt.elements) == 1
        assert gt.elements[0].element_type == "button"

    def test_zero_area_bbox_filtered(self, tmp_path: Any) -> None:
        """Nodes with x2 <= x1 or y2 <= y1 should be skipped."""
        zero_area_sample: Dict[str, Any] = {
            "activity_name": "test",
            "screen_id": "zero",
            "screen_width": 100,
            "screen_height": 100,
            "root": {
                "bounds": "[0,0][100,100]",
                "class": "android.widget.FrameLayout",
                "children": [
                    {
                        "bounds": "[10,10][20,20]",
                        "class": "android.widget.Button",
                        "visibility": "visible",
                    },
                    {
                        "bounds": "[50,10][50,50]",
                        "class": "android.widget.Button",
                        "visibility": "visible",
                    },
                    {
                        "bounds": "[10,50][20,50]",
                        "class": "android.widget.Button",
                        "visibility": "visible",
                    },
                ],
            },
        }
        path = _write_json(tmp_path, zero_area_sample, "zero.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        # Only the first element (non-zero area) should be included
        assert len(gt.elements) == 1

    def test_empty_view_hierarchy(self, tmp_path: Any) -> None:
        """An empty View Hierarchy (no leaf nodes after filtering) returns empty GroundTruth."""
        path = _write_json(tmp_path, _RICO_VH_EMPTY_SAMPLE, "empty.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 0
        assert gt.source == "rico"
        assert gt.image_width == 1080
        assert gt.image_height == 1920

    def test_single_leaf_node(self, tmp_path: Any) -> None:
        """View Hierarchy with a single leaf node."""
        single_sample: Dict[str, Any] = {
            "activity_name": "test",
            "screen_id": "single",
            "screen_width": 100,
            "screen_height": 100,
            "root": {
                "bounds": "[0,0][100,100]",
                "class": "android.widget.FrameLayout",
                "children": [
                    {
                        "bounds": "[10,10][50,50]",
                        "class": "android.widget.Button",
                        "text": "Click",
                        "visibility": "visible",
                    },
                ],
            },
        }
        path = _write_json(tmp_path, single_sample, "single.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].element_type == "button"
        assert gt.elements[0].text_content == "Click"

    def test_deep_nesting(self, tmp_path: Any) -> None:
        """Deeply nested View Hierarchy should extract all leaf nodes."""
        path = _write_json(tmp_path, _RICO_VH_DEEP_SAMPLE, "deep.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 2
        # Check both leaf nodes were extracted
        types = {e.element_type for e in gt.elements}
        assert types == {"image", "button"}

    def test_screen_width_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(_RICO_VH_SAMPLE)
        data["screen_width"] = 0
        path = _write_json(tmp_path, data, "bad.json")
        with pytest.raises(GroundTruthParseError, match="screen_width must be positive"):
            parse_rico_view_hierarchy(path, str(tmp_path))

    def test_screen_height_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(_RICO_VH_SAMPLE)
        data["screen_height"] = 0
        path = _write_json(tmp_path, data, "bad.json")
        with pytest.raises(GroundTruthParseError, match="screen_height must be positive"):
            parse_rico_view_hierarchy(path, str(tmp_path))

    def test_missing_root_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "screen_id": "test",
            "screen_width": 100,
            "screen_height": 100,
        }
        path = _write_json(tmp_path, data, "noroot.json")
        with pytest.raises(GroundTruthParseError, match="Missing or invalid 'root'"):
            parse_rico_view_hierarchy(path, str(tmp_path))

    def test_content_desc_fallback(self, tmp_path: Any) -> None:
        """When text is empty, content-desc should be used."""
        sample: Dict[str, Any] = {
            "activity_name": "test",
            "screen_id": "desc_test",
            "screen_width": 100,
            "screen_height": 100,
            "root": {
                "bounds": "[0,0][100,100]",
                "class": "android.widget.FrameLayout",
                "children": [
                    {
                        "bounds": "[10,10][50,50]",
                        "class": "android.widget.Button",
                        "text": "",
                        "content-desc": "Navigate back",
                        "visibility": "visible",
                    },
                ],
            },
        }
        path = _write_json(tmp_path, sample, "desc.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].text_content == "Navigate back"

    def test_both_text_and_content_desc_empty(self, tmp_path: Any) -> None:
        """When both text and content-desc are empty, text_content is None."""
        sample: Dict[str, Any] = {
            "activity_name": "test",
            "screen_id": "no_text",
            "screen_width": 100,
            "screen_height": 100,
            "root": {
                "bounds": "[0,0][100,100]",
                "class": "android.widget.FrameLayout",
                "children": [
                    {
                        "bounds": "[10,10][50,50]",
                        "class": "android.widget.Button",
                        "text": "",
                        "content-desc": "",
                        "visibility": "visible",
                    },
                ],
            },
        }
        path = _write_json(tmp_path, sample, "notext.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].text_content is None

    def test_root_as_leaf(self, tmp_path: Any) -> None:
        """When root has no children, it is treated as a leaf."""
        sample: Dict[str, Any] = {
            "activity_name": "test",
            "screen_id": "root_leaf",
            "screen_width": 100,
            "screen_height": 100,
            "root": {
                "bounds": "[10,10][50,50]",
                "class": "android.widget.Button",
                "text": "Lone",
                "visibility": "visible",
            },
        }
        path = _write_json(tmp_path, sample, "root_leaf.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].element_type == "button"
        assert gt.elements[0].text_content == "Lone"


# ===================================================================
# parse_rico_semantic
# ===================================================================


class TestParseRicoSemantic:
    def test_basic(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path)
        assert gt.source == "rico"
        assert gt.image_width == 1440
        assert gt.image_height == 2560
        assert len(gt.elements) == 2

    def test_element_fields(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path)

        elem0 = gt.elements[0]
        assert elem0.element_id == "sem_0"
        assert elem0.element_type == "button"
        assert elem0.text_content == "Submit"
        assert elem0.source_dataset == "rico"
        assert elem0.metadata["class"] == "android.widget.Button"
        assert elem0.metadata["clickable"] is True

    def test_coordinate_normalization(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path)

        # First element: bbox [50, 100, 200, 300] on 1440x2560
        x1, y1, x2, y2 = gt.elements[0].bbox
        assert x1 == pytest.approx(50.0 / 1440.0)
        assert y1 == pytest.approx(100.0 / 2560.0)
        assert x2 == pytest.approx(200.0 / 1440.0)
        assert y2 == pytest.approx(300.0 / 2560.0)

        for v in gt.elements[0].bbox:
            assert 0.0 <= v <= 1.0

    def test_empty_annotations(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "screen_id": "empty_sem",
            "screen_width": 100,
            "screen_height": 100,
            "annotations": [],
        }
        path = _write_json(tmp_path, data, "empty_sem.json")
        gt = parse_rico_semantic(path)
        assert len(gt.elements) == 0

    def test_degenerate_bbox_skipped(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "screen_id": "degen_sem",
            "screen_width": 100,
            "screen_height": 100,
            "annotations": [
                {"element_id": "good", "bbox": [10, 10, 50, 50], "class": "android.widget.Button"},
                {"element_id": "bad_x", "bbox": [50, 10, 30, 50], "class": "android.widget.Button"},
            ],
        }
        path = _write_json(tmp_path, data, "degen_sem.json")
        gt = parse_rico_semantic(path)
        assert len(gt.elements) == 1
        assert gt.elements[0].element_id == "good"

    def test_screen_width_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = dict(_RICO_SEMANTIC_SAMPLE)
        data["screen_width"] = 0
        path = _write_json(tmp_path, data, "bad_sem.json")
        with pytest.raises(GroundTruthParseError, match="screen_width must be positive"):
            parse_rico_semantic(path)

    def test_default_element_id(self, tmp_path: Any) -> None:
        """Annotation without element_id should get a generated id."""
        data: Dict[str, Any] = {
            "screen_id": "gen_id",
            "screen_width": 100,
            "screen_height": 100,
            "annotations": [
                {"bbox": [10, 10, 50, 50], "class": "android.widget.Button"},
            ],
        }
        path = _write_json(tmp_path, data, "gen_id.json")
        gt = parse_rico_semantic(path)
        assert len(gt.elements) == 1
        assert gt.elements[0].element_id == "gen_id_0"


# ===================================================================
# load_rico_directory
# ===================================================================


class TestLoadRicoDirectory:
    def test_loads_all_json_files(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "com.example.app"
        app_dir.mkdir()

        # Create two JSON files
        with (app_dir / "screenshot_001.json").open("w") as f:
            json.dump(_RICO_VH_SAMPLE, f)
        with (app_dir / "screenshot_002.json").open("w") as f:
            data = dict(_RICO_VH_SAMPLE)
            data["screen_id"] = "screenshot_002"
            data["root"] = {
                "bounds": "[0,0][100,100]",
                "class": "android.widget.Button",
                "text": "OK",
                "visibility": "visible",
            }
            json.dump(data, f)

        results = load_rico_directory(app_dir)
        assert len(results) == 2
        assert results[0].image_path == str(app_dir / "screenshot_001.png")
        assert results[1].image_path == str(app_dir / "screenshot_002.png")
        assert results[0].source == "rico"
        assert results[1].source == "rico"

    def test_directory_not_found_raises(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            load_rico_directory(tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "com.empty.app"
        app_dir.mkdir()
        results = load_rico_directory(app_dir)
        assert len(results) == 0

    def test_skips_non_json_files(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "com.mixed.app"
        app_dir.mkdir()

        with (app_dir / "screenshot_001.json").open("w") as f:
            json.dump(_RICO_VH_SAMPLE, f)
        (app_dir / "screenshot_001.png").touch()  # Should be ignored
        (app_dir / "readme.txt").touch()  # Should be ignored

        results = load_rico_directory(app_dir)
        assert len(results) == 1

    def test_handles_parse_error_gracefully(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "com.bad.app"
        app_dir.mkdir()

        # Create one valid and one invalid JSON
        with (app_dir / "screenshot_001.json").open("w") as f:
            json.dump(_RICO_VH_SAMPLE, f)
        (app_dir / "bad.json").write_text("not valid json")

        results = load_rico_directory(app_dir)
        # Only the valid one should be loaded
        assert len(results) == 1
        assert results[0].image_path == str(app_dir / "screenshot_001.png")
