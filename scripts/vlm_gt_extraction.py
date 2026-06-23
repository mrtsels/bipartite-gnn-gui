#!/usr/bin/env python3
"""VLM-compatible element extraction from RICO View Hierarchy.

The standard extract_elements only returns leaf nodes where
visible-to-user=True. But many RICO apps incorrectly label visible
UI elements as visible-to-user=False. This module provides a more
permissive extraction that better matches what a VLM actually sees.
"""

from __future__ import annotations

from typing import List

from bipartite_gnn_gui.graph.schema import ElementNode


# Classes that represent actual UI widgets (not just layout containers)
UI_WIDGET_CLASSES = {
    "TextView", "EditText", "Button", "ImageButton", "ImageView",
    "CheckBox", "RadioButton", "Switch", "ToggleButton", "Spinner",
    "ProgressBar", "SeekBar", "RatingBar", "WebView", "VideoView",
    "DatePicker", "TimePicker", "CalendarView", "Chronometer",
    "TextClock", "ImageSwitcher", "TextSwitcher", "ViewSwitcher",
    "AdapterViewFlipper", "StackView", "TabHost", "TabWidget",
    "AutoCompleteTextView", "MultiAutoCompleteTextView",
    "CheckedTextView", "CompoundButton", "RadioGroup",
    "SearchView", "ZoomButton", "ZoomControls",
    "NumberPicker", "DialerFilter", "TwoLineListItem",
    "TextViewCustomFont", "EditTextCustomFont", "ImageButtonCustomFont",
    "ButtonCustomFont", "CheckBoxCustomFont",
}


def _short_class(cls: str) -> str:
    """Extract short class name from full qualified name."""
    return cls.rsplit(".", 1)[-1]


def extract_elements_vlm(root: dict) -> list[ElementNode]:
    """Extract UI elements from RICO view hierarchy with VLM-compatible filtering.

    More permissive than the standard extract_elements:
    - Includes elements regardless of visible-to-user flag
    - Only filters out visibility=gone and invisible
    - Includes non-leaf nodes if they have a meaningful widget class name

    Args:
        root: Root node of the View Hierarchy.

    Returns:
        List of ElementNode objects.
    """
    elements: list[ElementNode] = []

    def walk(node: dict, depth: int = 0):
        if depth > 50:
            return
        
        cls = node.get("class", "")
        if not cls:
            return
        
        short = _short_class(cls)
        children = node.get("children")
        is_leaf = not isinstance(children, list) or len(children) == 0
        
        # Skip elements with visibility="gone"
        vis = node.get("visibility", "visible")
        if vis == "gone":
            return
        
        bounds = node.get("bounds", [0, 0, 0, 0])
        if len(bounds) != 4:
            return
        
        x1, y1, x2, y2 = bounds
        if x2 <= x1 or y2 <= y1:
            return
        
        # Include element if:
        is_widget = short in UI_WIDGET_CLASSES
        is_container = not is_leaf and not is_widget
        
        # Always include leaf widgets AND non-leaf widgets
        if is_widget or is_leaf:
            label = _rico_class_to_label_vlm(cls)
            elements.append(ElementNode(
                bbox=[x1, y1, x2, y2],
                label=label,
                confidence=1.0,
                element_id=f"elem_{len(elements)}",
            ))
        
        # Continue traversal for containers
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    walk(child, depth + 1)

    walk(root)
    return elements


def _rico_class_to_label_vlm(cls: str) -> str:
    """Map Android class name to canonical type (same as run_experiment)."""
    short = cls.rsplit(".", 1)[-1]
    mapping = {
        "Button": "button",
        "ImageButton": "icon",
        "ImageView": "image",
        "TextView": "text",
        "EditText": "input",
        "CheckBox": "checkbox",
        "Switch": "switch",
        "Spinner": "icon",
        "ProgressBar": "icon",
        "WebView": "container",
        "ListView": "list",
        "ScrollView": "container",
        "TabWidget": "tab",
        "RadioButton": "radio",
        "SeekBar": "slider",
        "TextViewCustomFont": "text",
        "EditTextCustomFont": "input",
        "ImageButtonCustomFont": "icon",
        "ButtonCustomFont": "button",
        "CheckBoxCustomFont": "checkbox",
    }
    for suffix, label in mapping.items():
        if short.endswith(suffix):
            return label
    return "other"
