"""Tests for the RICO data loader module.

Covers bounds parsing (list + string), Android class-to-type mapping,
component-label-to-type mapping, leaf node extraction, View Hierarchy
and Semantic Annotation parsers, image ID extraction, and directory loading.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from bipartite_gnn_gui.data.ground_truth import GroundTruthParseError
from bipartite_gnn_gui.data.rico_loader import (
    _find_leaf_nodes,
    component_label_to_type,
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
# Sample RICO View Hierarchy JSON — actual on-disk format
# ---------------------------------------------------------------------------

# View Hierarchy: root is at data["activity"]["root"]
# bounds are lists, content-desc is a list, text key may be missing
_RICO_VH_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.MainActivity",
    "activity": {
        "root": {
            "bounds": [0, 0, 1440, 2560],
            "class": "android.widget.FrameLayout",
            "visibility": "visible",
            "visible-to-user": True,
            "children": [
                {
                    "bounds": [0, 0, 1440, 2560],
                    "class": "android.widget.LinearLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [50, 100, 200, 300],
                            "class": "android.widget.Button",
                            "text": "Submit",
                            "content-desc": [None],
                            "clickable": True,
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                        {
                            "bounds": [300, 100, 500, 300],
                            "class": "android.widget.TextView",
                            "text": "Welcome",
                            "content-desc": ["Welcome text"],
                            "clickable": False,
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                        {
                            "bounds": [600, 100, 800, 300],
                            "class": "android.widget.EditText",
                            "content-desc": ["Search input"],
                            "clickable": True,
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                    ],
                },
            ],
        }
    },
}

# View Hierarchy with invisible nodes, zero-area bboxes, and visible-to-user
_RICO_VH_FILTER_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.Filter",
    "activity": {
        "root": {
            "bounds": [0, 0, 720, 1280],
            "class": "android.widget.FrameLayout",
            "visibility": "visible",
            "visible-to-user": True,
            "children": [
                {
                    "bounds": [10, 10, 100, 100],
                    "class": "android.widget.Button",
                    "clickable": True,
                    "visibility": "visible",
                    "visible-to-user": True,
                },
                {
                    "bounds": [110, 10, 200, 100],
                    "class": "android.widget.Button",
                    "clickable": True,
                    "visibility": "invisible",
                    "visible-to-user": False,
                },
                {
                    "bounds": [10, 110, 100, 200],
                    "class": "android.widget.TextView",
                    "text": "Gone node",
                    "clickable": False,
                    "visibility": "gone",
                    "visible-to-user": False,
                },
                {
                    "bounds": [50, 210, 50, 310],
                    "class": "android.widget.ImageView",
                    "visibility": "visible",
                    "visible-to-user": True,
                },
            ],
        }
    },
}

# View Hierarchy with deep nesting
_RICO_VH_DEEP_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.Deep",
    "activity": {
        "root": {
            "bounds": [0, 0, 1080, 1920],
            "class": "android.widget.FrameLayout",
            "visibility": "visible",
            "visible-to-user": True,
            "children": [
                {
                    "bounds": [0, 0, 1080, 1920],
                    "class": "android.widget.LinearLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [0, 0, 1080, 200],
                            "class": "android.widget.RelativeLayout",
                            "visibility": "visible",
                            "visible-to-user": True,
                            "children": [
                                {
                                    "bounds": [0, 0, 1080, 100],
                                    "class": "android.widget.LinearLayout",
                                    "visibility": "visible",
                                    "visible-to-user": True,
                                    "children": [
                                        {
                                            "bounds": [10, 10, 100, 90],
                                            "class": "android.widget.ImageView",
                                            "visibility": "visible",
                                            "visible-to-user": True,
                                        },
                                    ],
                                },
                                {
                                    "bounds": [0, 100, 1080, 200],
                                    "class": "android.widget.FrameLayout",
                                    "visibility": "visible",
                                    "visible-to-user": True,
                                    "children": [
                                        {
                                            "bounds": [500, 110, 580, 190],
                                            "class": "android.widget.Button",
                                            "text": "OK",
                                            "visibility": "visible",
                                            "visible-to-user": True,
                                        },
                                    ],
                                },
                            ],
                        },
                    ],
                },
            ],
        }
    },
}

# Empty View Hierarchy (no visible leaf nodes after filtering)
_RICO_VH_EMPTY_SAMPLE: Dict[str, Any] = {
    "activity_name": "com.example.Empty",
    "activity": {
        "root": {
            "bounds": [0, 0, 1080, 1920],
            "class": "android.widget.FrameLayout",
            "visibility": "visible",
            "visible-to-user": True,
            "children": [
                {
                    "bounds": [0, 0, 1080, 1920],
                    "class": "android.widget.LinearLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [0, 0, 1080, 1920],
                            "class": "android.widget.ScrollView",
                            "visibility": "gone",
                            "visible-to-user": False,
                            "children": [],
                        },
                    ],
                },
            ],
        }
    },
}

# Semantic Annotation sample — top-level IS the tree with componentLabel
_RICO_SEMANTIC_SAMPLE: Dict[str, Any] = {
    "class": "com.android.internal.policy.PhoneWindow$DecorView",
    "bounds": [0, 0, 1440, 2560],
    "children": [
        {
            "class": "android.widget.Button",
            "bounds": [50, 100, 200, 300],
            "text": "Submit",
            "componentLabel": "Button",
            "clickable": True,
        },
        {
            "class": "android.widget.TextView",
            "bounds": [300, 100, 500, 300],
            "text": "Label",
            "componentLabel": "Text",
            "clickable": False,
        },
        {
            "class": "android.widget.ImageView",
            "bounds": [600, 200, 700, 300],
            "componentLabel": "Icon",
            "clickable": False,
        },
    ],
}

# Semantic Annotation with Drawer and Toolbar (container types)
_RICO_SEMANTIC_COMPLEX: Dict[str, Any] = {
    "class": "android.widget.FrameLayout",
    "bounds": [0, 0, 1440, 2560],
    "children": [
        {
            "class": "android.widget.LinearLayout",
            "bounds": [0, 0, 1440, 2560],
            "componentLabel": "Drawer",
            "children": [
                {
                    "class": "android.widget.TextView",
                    "bounds": [0, 100, 300, 150],
                    "text": "Menu Item",
                    "componentLabel": "Text",
                },
            ],
        },
        {
            "class": "android.widget.LinearLayout",
            "bounds": [0, 0, 1440, 200],
            "componentLabel": "Toolbar",
            "children": [
                {
                    "class": "android.widget.TextView",
                    "bounds": [500, 50, 940, 150],
                    "text": "Page Title",
                    "componentLabel": "Text",
                },
            ],
        },
    ],
}


# ===================================================================
# parse_rico_bounds — list AND string formats
# ===================================================================


class TestParseRicoBounds:
    # --- list format (actual on-disk format) -------------------------------
    def test_list_format(self) -> None:
        result = parse_rico_bounds([0, 0, 100, 200])
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_list_format_large(self) -> None:
        result = parse_rico_bounds([0, 0, 1440, 2560])
        assert result == (0.0, 0.0, 1440.0, 2560.0)

    def test_list_format_negative(self) -> None:
        result = parse_rico_bounds([-10, -5, 100, 200])
        assert result == (-10.0, -5.0, 100.0, 200.0)

    def test_list_format_float(self) -> None:
        result = parse_rico_bounds([10.5, 20.3, 100, 200])
        assert result == (10.5, 20.3, 100.0, 200.0)

    def test_list_wrong_length_raises(self) -> None:
        with pytest.raises(GroundTruthParseError, match="Expected 4-element"):
            parse_rico_bounds([1, 2, 3])

    # --- string format (backward compatibility) ----------------------------
    def test_string_format(self) -> None:
        result = parse_rico_bounds("[0,0][100,200]")
        assert result == (0.0, 0.0, 100.0, 200.0)

    def test_string_with_whitespace(self) -> None:
        result = parse_rico_bounds("  [50,100][200,300]  ")
        assert result == (50.0, 100.0, 200.0, 300.0)

    def test_string_negative_values(self) -> None:
        result = parse_rico_bounds("[-10,-5][100,200]")
        assert result == (-10.0, -5.0, 100.0, 200.0)

    def test_string_invalid_format_raises(self) -> None:
        with pytest.raises(GroundTruthParseError, match="Cannot parse"):
            parse_rico_bounds("not a bounds string")

    def test_string_partial_match_raises(self) -> None:
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

    # Suffix-based fallbacks
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
# component_label_to_type
# ===================================================================


class TestComponentLabelToType:
    def test_Icon(self) -> None:
        assert component_label_to_type("Icon") == "icon"

    def test_Text(self) -> None:
        assert component_label_to_type("Text") == "text"

    def test_Input(self) -> None:
        assert component_label_to_type("Input") == "input"

    def test_Drawer(self) -> None:
        assert component_label_to_type("Drawer") == "container"

    def test_Image(self) -> None:
        assert component_label_to_type("Image") == "image"

    def test_Button(self) -> None:
        assert component_label_to_type("Button") == "button"

    def test_List(self) -> None:
        assert component_label_to_type("List") == "list"

    def test_Checkbox(self) -> None:
        assert component_label_to_type("Checkbox") == "icon"

    def test_Switch(self) -> None:
        assert component_label_to_type("Switch") == "icon"

    def test_OnOff(self) -> None:
        assert component_label_to_type("On/Off") == "icon"

    def test_RadioButton(self) -> None:
        assert component_label_to_type("Radio Button") == "icon"

    def test_TextButton(self) -> None:
        assert component_label_to_type("Text Button") == "button"

    def test_Toolbar(self) -> None:
        assert component_label_to_type("Toolbar") == "container"

    def test_unknown_returns_other(self) -> None:
        assert component_label_to_type("UnknownThing") == "other"

    def test_empty_string_returns_other(self) -> None:
        assert component_label_to_type("") == "other"


# ===================================================================
# _find_leaf_nodes
# ===================================================================


class TestFindLeafNodes:
    def test_single_node_no_children(self) -> None:
        node = {"bounds": [0, 0, 10, 10], "class": "android.widget.Button"}
        leaves = _find_leaf_nodes(node)
        assert len(leaves) == 1
        assert leaves[0] is node

    def test_single_node_empty_children(self) -> None:
        node: Dict[str, Any] = {
            "bounds": [0, 0, 10, 10],
            "class": "android.widget.Button",
            "children": [],
        }
        leaves = _find_leaf_nodes(node)
        assert len(leaves) == 1

    def test_single_node_none_children(self) -> None:
        node: Dict[str, Any] = {
            "bounds": [0, 0, 10, 10],
            "class": "android.widget.Button",
            "children": None,
        }
        leaves = _find_leaf_nodes(node)
        assert len(leaves) == 1

    def test_shallow_nesting(self) -> None:
        root: Dict[str, Any] = {
            "bounds": [0, 0, 100, 200],
            "class": "android.widget.FrameLayout",
            "children": [
                {"bounds": [10, 10, 50, 50], "class": "android.widget.Button"},
                {"bounds": [60, 10, 90, 50], "class": "android.widget.TextView"},
            ],
        }
        leaves = _find_leaf_nodes(root)
        assert len(leaves) == 2

    def test_deep_nesting(self) -> None:
        root: Dict[str, Any] = {
            "bounds": [0, 0, 100, 200],
            "class": "android.widget.FrameLayout",
            "children": [
                {
                    "bounds": [0, 0, 100, 200],
                    "class": "android.widget.LinearLayout",
                    "children": [
                        {
                            "bounds": [0, 0, 100, 100],
                            "class": "android.widget.RelativeLayout",
                            "children": [
                                {"bounds": [10, 10, 50, 50], "class": "android.widget.Button"},
                            ],
                        },
                        {
                            "bounds": [0, 100, 100, 200],
                            "class": "android.widget.FrameLayout",
                            "children": [
                                {"bounds": [60, 110, 90, 190], "class": "android.widget.TextView"},
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
            "bounds": [0, 0, 100, 200],
            "class": "android.widget.FrameLayout",
            "children": [],
        }
        leaves = _find_leaf_nodes(root)
        assert len(leaves) == 1  # root itself is a leaf

    def test_ignores_non_dict_children(self) -> None:
        root: Dict[str, Any] = {
            "bounds": [0, 0, 100, 200],
            "class": "android.widget.FrameLayout",
            "children": [
                "not a dict",
                {"bounds": [10, 10, 50, 50], "class": "android.widget.Button"},
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
        assert get_rico_image_id(vh_json) == "screenshot_001.jpg"

    def test_missing_screen_id(self) -> None:
        vh_json: Dict[str, Any] = {}
        assert get_rico_image_id(vh_json) == ".jpg"


# ===================================================================
# parse_rico_view_hierarchy — actual on-disk format
# ===================================================================


class TestParseViewHierarchy:
    def test_basic(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "10101.json")
        images_dir = str(tmp_path)
        gt = parse_rico_view_hierarchy(path, images_dir)
        assert gt.source == "rico"
        assert gt.image_width == 1440
        assert gt.image_height == 2560
        # image path uses .jpg by default
        assert gt.image_path == str(tmp_path / "10101.jpg")
        assert len(gt.elements) == 3

    def test_element_fields(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "10101.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))

        # First element: Button "Submit"
        btn = gt.elements[0]
        assert btn.element_id == "10101_0"
        assert btn.element_type == "button"
        assert btn.text_content == "Submit"
        assert btn.source_dataset == "rico"
        assert btn.metadata["class"] == "android.widget.Button"
        assert btn.metadata["clickable"] is True

        # Second element: TextView "Welcome"
        tv = gt.elements[1]
        assert tv.element_id == "10101_1"
        assert tv.element_type == "text"
        assert tv.text_content == "Welcome"

        # Third element: EditText with no text, falls back to content-desc list
        et = gt.elements[2]
        assert et.element_id == "10101_2"
        assert et.element_type == "input"
        assert et.text_content == "Search input"

    def test_coordinate_normalization(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "10101.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))

        # First element: bounds [50,100,200,300] on 1440x2560
        x1, y1, x2, y2 = gt.elements[0].bbox
        assert x1 == pytest.approx(50.0 / 1440.0)
        assert y1 == pytest.approx(100.0 / 2560.0)
        assert x2 == pytest.approx(200.0 / 1440.0)
        assert y2 == pytest.approx(300.0 / 2560.0)

        for v in gt.elements[0].bbox:
            assert 0.0 <= v <= 1.0

    def test_all_bboxes_in_range(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "10101.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        for elem in gt.elements:
            for v in elem.bbox:
                assert 0.0 <= v <= 1.0

    def test_visibility_filtering(self, tmp_path: Any) -> None:
        """Only nodes with visibility='visible' AND visible-to-user=True."""
        path = _write_json(tmp_path, _RICO_VH_FILTER_SAMPLE, "filter.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        # 1 visible button, 1 invisible button, 1 gone textview, 1 zero-area image
        assert len(gt.elements) == 1
        assert gt.elements[0].element_type == "button"

    def test_zero_area_bbox_filtered(self, tmp_path: Any) -> None:
        """Nodes with x2 <= x1 or y2 <= y1 should be skipped."""
        zero_area_sample: Dict[str, Any] = {
            "activity_name": "test",
            "activity": {
                "root": {
                    "bounds": [0, 0, 100, 100],
                    "class": "android.widget.FrameLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [10, 10, 20, 20],
                            "class": "android.widget.Button",
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                        {
                            "bounds": [50, 10, 50, 50],
                            "class": "android.widget.Button",
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                        {
                            "bounds": [10, 50, 20, 50],
                            "class": "android.widget.Button",
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                    ],
                }
            },
        }
        path = _write_json(tmp_path, zero_area_sample, "zero.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1

    def test_empty_view_hierarchy(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_EMPTY_SAMPLE, "empty.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 0
        assert gt.source == "rico"
        assert gt.image_width == 1080
        assert gt.image_height == 1920

    def test_single_leaf_node(self, tmp_path: Any) -> None:
        single_sample: Dict[str, Any] = {
            "activity_name": "test",
            "activity": {
                "root": {
                    "bounds": [0, 0, 100, 100],
                    "class": "android.widget.FrameLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [10, 10, 50, 50],
                            "class": "android.widget.Button",
                            "text": "Click",
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                    ],
                }
            },
        }
        path = _write_json(tmp_path, single_sample, "single.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].element_type == "button"
        assert gt.elements[0].text_content == "Click"

    def test_deep_nesting(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_VH_DEEP_SAMPLE, "deep.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 2
        types = {e.element_type for e in gt.elements}
        assert types == {"image", "button"}

    def test_root_bounds_missing_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "activity": {
                "root": {
                    "class": "android.widget.FrameLayout",
                    "children": [],
                }
            }
        }
        path = _write_json(tmp_path, data, "bad.json")
        with pytest.raises(GroundTruthParseError, match="Cannot derive screen"):
            parse_rico_view_hierarchy(path, str(tmp_path))

    def test_root_bounds_zero_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "activity": {
                "root": {
                    "bounds": [0, 0, 0, 0],
                    "class": "android.widget.FrameLayout",
                    "children": [],
                }
            }
        }
        path = _write_json(tmp_path, data, "zero_size.json")
        with pytest.raises(GroundTruthParseError, match="must be positive"):
            parse_rico_view_hierarchy(path, str(tmp_path))

    def test_missing_root_raises(self, tmp_path: Any) -> None:
        data: Dict[str, Any] = {
            "activity_name": "test",
            # no "activity" or "root" key
        }
        path = _write_json(tmp_path, data, "noroot.json")
        with pytest.raises(GroundTruthParseError, match="Cannot locate root"):
            parse_rico_view_hierarchy(path, str(tmp_path))

    def test_content_desc_fallback_from_list(self, tmp_path: Any) -> None:
        """content-desc as list [None, 'Navigate back'] → use first non-None."""
        sample: Dict[str, Any] = {
            "activity": {
                "root": {
                    "bounds": [0, 0, 100, 100],
                    "class": "android.widget.FrameLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [10, 10, 50, 50],
                            "class": "android.widget.Button",
                            "text": None,
                            "content-desc": [None, "Navigate back"],
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                    ],
                }
            }
        }
        path = _write_json(tmp_path, sample, "desc.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].text_content == "Navigate back"

    def test_both_text_and_content_desc_empty(self, tmp_path: Any) -> None:
        """When text is missing and content-desc is [None], text_content is None."""
        sample: Dict[str, Any] = {
            "activity": {
                "root": {
                    "bounds": [0, 0, 100, 100],
                    "class": "android.widget.FrameLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [10, 10, 50, 50],
                            "class": "android.widget.Button",
                            "content-desc": [None],
                            "visibility": "visible",
                            "visible-to-user": True,
                        },
                    ],
                }
            }
        }
        path = _write_json(tmp_path, sample, "notext.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].text_content is None

    def test_root_as_leaf(self, tmp_path: Any) -> None:
        """When root has no children, it is treated as a leaf."""
        sample: Dict[str, Any] = {
            "activity": {
                "root": {
                    "bounds": [10, 10, 50, 50],
                    "class": "android.widget.Button",
                    "text": "Lone",
                    "visibility": "visible",
                    "visible-to-user": True,
                }
            }
        }
        path = _write_json(tmp_path, sample, "root_leaf.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 1
        assert gt.elements[0].element_type == "button"
        assert gt.elements[0].text_content == "Lone"

    # ---- visible-to-user filtering ----------------------------------------
    def test_visible_to_user_false_filtered(self, tmp_path: Any) -> None:
        """Node with visibility='visible' but visible-to-user=False → skipped."""
        sample: Dict[str, Any] = {
            "activity": {
                "root": {
                    "bounds": [0, 0, 100, 100],
                    "class": "android.widget.FrameLayout",
                    "visibility": "visible",
                    "visible-to-user": True,
                    "children": [
                        {
                            "bounds": [10, 10, 50, 50],
                            "class": "android.widget.Button",
                            "visibility": "visible",
                            "visible-to-user": False,
                        },
                    ],
                }
            }
        }
        path = _write_json(tmp_path, sample, "vtuf.json")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert len(gt.elements) == 0

    # ---- image path tests -------------------------------------------------
    def test_image_path_png_found(self, tmp_path: Any) -> None:
        """When .png exists and .jpg doesn't, use .png."""
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "10101.json")
        # Create a .png file but no .jpg
        (tmp_path / "10101.png").write_text("fake png")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert gt.image_path == str(tmp_path / "10101.png")

    def test_image_path_jpg_preferred(self, tmp_path: Any) -> None:
        """When both .jpg and .png exist, .jpg is preferred."""
        path = _write_json(tmp_path, _RICO_VH_SAMPLE, "10101.json")
        (tmp_path / "10101.jpg").write_text("fake jpg")
        (tmp_path / "10101.png").write_text("fake png")
        gt = parse_rico_view_hierarchy(path, str(tmp_path))
        assert gt.image_path == str(tmp_path / "10101.jpg")


# ===================================================================
# parse_rico_semantic — uses same tree format
# ===================================================================


class TestParseRicoSemantic:
    def test_basic(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path, str(tmp_path))
        assert gt.source == "rico"
        assert gt.image_width == 1440
        assert gt.image_height == 2560
        assert len(gt.elements) == 3

    def test_element_fields(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path, str(tmp_path))

        elem0 = gt.elements[0]
        assert elem0.element_id == "sem_0"
        assert elem0.element_type == "button"  # from componentLabel "Button"
        assert elem0.text_content == "Submit"
        assert elem0.source_dataset == "rico"
        assert elem0.metadata["class"] == "android.widget.Button"
        assert elem0.metadata["clickable"] is True
        assert elem0.metadata["componentLabel"] == "Button"

    def test_component_label_type_mapping(self, tmp_path: Any) -> None:
        """componentLabel takes priority over class name for type."""
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path, str(tmp_path))

        types = {e.element_type for e in gt.elements}
        assert types == {"button", "text", "icon"}

    def test_coordinate_normalization(self, tmp_path: Any) -> None:
        path = _write_json(tmp_path, _RICO_SEMANTIC_SAMPLE, "sem.json")
        gt = parse_rico_semantic(path, str(tmp_path))

        x1, y1, x2, y2 = gt.elements[0].bbox
        assert x1 == pytest.approx(50.0 / 1440.0)
        assert y1 == pytest.approx(100.0 / 2560.0)
        assert x2 == pytest.approx(200.0 / 1440.0)
        assert y2 == pytest.approx(300.0 / 2560.0)
        for v in gt.elements[0].bbox:
            assert 0.0 <= v <= 1.0

    def test_empty_semantic(self, tmp_path: Any) -> None:
        """Top-level node with no children → treated as leaf if visible."""
        data: Dict[str, Any] = {
            "class": "android.widget.FrameLayout",
            "bounds": [0, 0, 100, 100],
            "children": [],
        }
        path = _write_json(tmp_path, data, "empty_sem.json")
        gt = parse_rico_semantic(path, str(tmp_path))
        assert len(gt.elements) == 1

    def test_nested_container_with_leaves(self, tmp_path: Any) -> None:
        """Semantic tree with Drawer/Toolbar containers → only leaves extracted."""
        path = _write_json(tmp_path, _RICO_SEMANTIC_COMPLEX, "complex.json")
        gt = parse_rico_semantic(path, str(tmp_path))
        # Only leaf nodes (Text inside Drawer, Text inside Toolbar) = 2 leaves
        assert len(gt.elements) == 2
        text_contents = {e.text_content for e in gt.elements}
        assert text_contents == {"Menu Item", "Page Title"}
        # Both should be type "text" from componentLabel
        for e in gt.elements:
            assert e.element_type == "text"

    def test_nonexistent_root_raises(self, tmp_path: Any) -> None:
        """Dict without 'children' or 'class' at top level."""
        data: Dict[str, Any] = {"some": "data"}
        path = _write_json(tmp_path, data, "bad_sem.json")
        with pytest.raises(GroundTruthParseError, match="Cannot locate root"):
            parse_rico_semantic(path, str(tmp_path))


# ===================================================================
# load_rico_directory
# ===================================================================


class TestLoadRicoDirectory:
    def test_loads_all_json_files(self, tmp_path: Any) -> None:
        """Flat directory with multiple JSON files."""
        app_dir = tmp_path / "combined"
        app_dir.mkdir()

        with (app_dir / "10001.json").open("w") as f:
            json.dump(_RICO_VH_SAMPLE, f)
        with (app_dir / "10002.json").open("w") as f:
            data: Dict[str, Any] = {
                "activity": {
                    "root": {
                        "bounds": [0, 0, 100, 100],
                        "class": "android.widget.FrameLayout",
                        "visibility": "visible",
                        "visible-to-user": True,
                        "children": [
                            {
                                "bounds": [10, 10, 50, 50],
                                "class": "android.widget.Button",
                                "text": "OK",
                                "visibility": "visible",
                                "visible-to-user": True,
                            },
                        ],
                    }
                }
            }
            json.dump(data, f)

        results = load_rico_directory(app_dir)
        assert len(results) == 2
        assert results[0].image_path == str(app_dir / "10001.jpg")
        assert results[1].image_path == str(app_dir / "10002.jpg")
        assert results[0].source == "rico"
        assert results[1].source == "rico"

    def test_directory_not_found_raises(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            load_rico_directory(tmp_path / "nonexistent")

    def test_empty_directory(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "empty_dir"
        app_dir.mkdir()
        results = load_rico_directory(app_dir)
        assert len(results) == 0

    def test_skips_non_json_files(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "mixed_dir"
        app_dir.mkdir()

        with (app_dir / "screenshot.json").open("w") as f:
            json.dump(_RICO_VH_SAMPLE, f)
        (app_dir / "screenshot.jpg").touch()
        (app_dir / "readme.txt").touch()

        results = load_rico_directory(app_dir)
        assert len(results) == 1

    def test_handles_parse_error_gracefully(self, tmp_path: Any) -> None:
        app_dir = tmp_path / "bad_dir"
        app_dir.mkdir()

        with (app_dir / "good.json").open("w") as f:
            json.dump(_RICO_VH_SAMPLE, f)
        (app_dir / "bad.json").write_text("not valid json")

        results = load_rico_directory(app_dir)
        assert len(results) == 1
