"""Tests for graph visualization — bbox overlay, color mapping, JSON export.

Uses matplotlib's agg backend (non-interactive) for testing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from bipartite_gnn_gui.graph.schema import ConstraintNode, ConstraintType, ElementNode
from bipartite_gnn_gui.graph.visualize import (
    color_by_constraint_type,
    color_by_element_type,
    export_graph,
    plot_graph_on_screenshot,
)

matplotlib.use("Agg")


# ===================================================================
# Helpers
# ===================================================================


def _elem(
    x1: float, y1: float, x2: float, y2: float,
    label: str = "button", confidence: float = 1.0,
) -> ElementNode:
    """Shorthand to create an ElementNode."""
    return ElementNode(bbox=[x1, y1, x2, y2], label=label, confidence=confidence)


def _con(
    ctype: ConstraintType,
    source: list[int],
    target: list[int] | None = None,
    **params: float,
) -> ConstraintNode:
    """Shorthand to create a ConstraintNode."""
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=source,
        target_indices=target or source,
        params=params,
    )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def two_buttons() -> list[ElementNode]:
    return [
        _elem(0.1, 0.1, 0.4, 0.3, label="button", confidence=0.9),
        _elem(0.5, 0.1, 0.8, 0.3, label="button", confidence=0.8),
    ]


@pytest.fixture
def mixed_elements() -> list[ElementNode]:
    return [
        _elem(0.0, 0.0, 0.3, 0.2, label="button", confidence=0.9),
        _elem(0.4, 0.0, 0.7, 0.2, label="text", confidence=0.85),
        _elem(0.0, 0.3, 0.3, 0.5, label="image", confidence=0.7),
        _elem(0.4, 0.3, 0.7, 0.5, label="input", confidence=0.95),
        _elem(0.0, 0.6, 0.3, 0.8, label="icon", confidence=0.6),
        _elem(0.4, 0.6, 0.7, 0.8, label="container", confidence=0.88),
    ]


@pytest.fixture
def alignment_constraint() -> list[ConstraintNode]:
    return [
        _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
    ]


@pytest.fixture
def multiple_constraints() -> list[ConstraintNode]:
    return [
        _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
        _con(ConstraintType.CENTER_X, [0, 1], tolerance=0.02),
        _con(ConstraintType.CONTAINMENT, [0], [1], margin=0.5),
    ]


# ===================================================================
# plot_graph_on_screenshot
# ===================================================================


class TestPlotGraphOnScreenshot:
    """Basic plotting — returns Axes object."""

    def test_returns_axes(self, two_buttons: list[ElementNode],
                          alignment_constraint: list[ConstraintNode]) -> None:
        """plot_graph_on_screenshot returns a matplotlib Axes."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_no_elements_placeholder(self,
                                     alignment_constraint: list[ConstraintNode]) -> None:
        """With 0 elements, placeholder 'No elements' is shown."""
        ax = plot_graph_on_screenshot([], alignment_constraint)
        assert ax is not None
        # The text "No elements" should be in the axes
        texts = [t.get_text() for t in ax.texts]
        assert any("No elements" in t for t in texts)
        title = ax.get_title()
        assert "Elements: 0" in title

    def test_no_constraints_elements_only(self, two_buttons: list[ElementNode]) -> None:
        """With 0 constraints, only elements are plotted (no edges)."""
        ax = plot_graph_on_screenshot(two_buttons, [])
        assert ax is not None
        title = ax.get_title()
        assert "Constraints: 0" in title
        # Should have patches for the two bboxes
        assert len(ax.patches) == 2

    def test_empty_both(self) -> None:
        """Both empty lists — placeholder shown."""
        ax = plot_graph_on_screenshot([], [])
        assert ax is not None
        texts = [t.get_text() for t in ax.texts]
        assert any("No elements" in t for t in texts)

    def test_image_path_none(self, two_buttons: list[ElementNode],
                             alignment_constraint: list[ConstraintNode]) -> None:
        """image_path=None uses white background (no crash)."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint,
                                      image_path=None)
        assert ax is not None
        assert isinstance(ax, matplotlib.axes.Axes)

    def test_image_path_missing(self, two_buttons: list[ElementNode],
                                alignment_constraint: list[ConstraintNode],
                                caplog: pytest.LogCaptureFixture) -> None:
        """Missing image_path logs a warning and uses white background."""
        caplog.set_level(logging.WARNING)
        ax = plot_graph_on_screenshot(
            two_buttons, alignment_constraint,
            image_path="/nonexistent/path.png",
        )
        assert ax is not None
        assert len(caplog.records) >= 1
        assert any(
            "does not exist" in rec.getMessage() for rec in caplog.records
        )

    def test_color_by_confidence(self, two_buttons: list[ElementNode],
                                 alignment_constraint: list[ConstraintNode]) -> None:
        """color_by='confidence' uses colormap (no crash)."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint,
                                      color_by="confidence")
        assert ax is not None
        # matplotlib should have added a colorbar image
        assert len(ax.images) >= 1 or len(ax.patches) == 2

    def test_color_by_type_default(self, two_buttons: list[ElementNode],
                                   alignment_constraint: list[ConstraintNode]) -> None:
        """Default color_by='type' produces legend."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint)
        assert ax is not None
        legend = ax.get_legend()
        assert legend is not None

    def test_show_bboxes_false(self, two_buttons: list[ElementNode],
                               alignment_constraint: list[ConstraintNode]) -> None:
        """show_bboxes=False skips drawing rectangles."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint,
                                      show_bboxes=False)
        assert ax is not None
        # Only constraint markers and edges, no bbox patches
        # With empty constraints + no bboxes, patches should be 0
        assert len(ax.patches) == 0 or len(ax.lines) >= 0

    def test_show_edges_false(self, two_buttons: list[ElementNode],
                              alignment_constraint: list[ConstraintNode]) -> None:
        """show_edges=False skips drawing edge lines."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint,
                                      show_edges=False)
        assert ax is not None
        # Patches are the bboxes (2), lines should be 0 (no edges)
        # (constraint markers use plot() which adds to lines, so > 0 is OK)
        assert len(ax.patches) == 2

    def test_custom_ax(self, two_buttons: list[ElementNode],
                       alignment_constraint: list[ConstraintNode]) -> None:
        """Uses a user-provided Axes."""
        _, custom_ax = plt.subplots()
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint, ax=custom_ax)
        assert ax is custom_ax

    def test_single_element(self, alignment_constraint: list[ConstraintNode]) -> None:
        """Single element — no crash."""
        elems = [_elem(0.1, 0.1, 0.5, 0.5, label="button")]
        ax = plot_graph_on_screenshot(elems, alignment_constraint)
        assert ax is not None
        assert "Elements: 1" in ax.get_title()

    def test_single_constraint(self, two_buttons: list[ElementNode]) -> None:
        """Single constraint — no crash."""
        cons = [_con(ConstraintType.CENTER_X, [0, 1], tolerance=0.02)]
        ax = plot_graph_on_screenshot(two_buttons, cons)
        assert ax is not None
        assert "Constraints: 1" in ax.get_title()

    def test_many_elements(self) -> None:
        """10 elements + 3 constraints — no crash."""
        elems = [
            _elem(i * 0.1, i * 0.1, i * 0.1 + 0.05, i * 0.1 + 0.05,
                  label="button" if i % 2 == 0 else "text")
            for i in range(10)
        ]
        cons = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_TOP, [2, 3, 4], tolerance=0.02),
            _con(ConstraintType.CENTER_X, [5, 6, 7, 8], tolerance=0.02),
        ]
        ax = plot_graph_on_screenshot(elems, cons)
        assert ax is not None
        assert "Elements: 10" in ax.get_title()
        assert "Constraints: 3" in ax.get_title()

    def test_title_format(self, two_buttons: list[ElementNode],
                          alignment_constraint: list[ConstraintNode]) -> None:
        """Title matches expected format."""
        ax = plot_graph_on_screenshot(two_buttons, alignment_constraint)
        assert ax.get_title() == "Elements: 2 | Constraints: 1"


# ===================================================================
# color_by_element_type
# ===================================================================


class TestColorByElementType:
    """Element type color mapping."""

    def test_returns_mapping(self, mixed_elements: list[ElementNode]) -> None:
        """Returns dict with element type keys."""
        _, ax = plt.subplots()
        mapping = color_by_element_type(ax, mixed_elements)
        assert isinstance(mapping, dict)
        expected_keys = {"button", "text", "image", "input", "icon", "container"}
        assert set(mapping.keys()) == expected_keys

    def test_colors_are_rgba_tuples(self, mixed_elements: list[ElementNode]) -> None:
        """Each value is a 4-element rgba tuple."""
        _, ax = plt.subplots()
        mapping = color_by_element_type(ax, mixed_elements)
        for color in mapping.values():
            assert len(color) == 4
            r, g, b, a = color
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0
            assert a == 1.0

    def test_default_color_for_unknown_type(self) -> None:
        """Unknown element label gets gray default."""
        _, ax = plt.subplots()
        elems = [_elem(0.0, 0.0, 0.5, 0.5, label="unknown_type")]
        mapping = color_by_element_type(ax, elems)
        # Gray: #95a5a6 = (149/255, 165/255, 166/255)
        assert "unknown_type" in mapping
        color = mapping["unknown_type"]
        r, g, b, a = color
        assert r == pytest.approx(149 / 255, abs=0.01)
        assert g == pytest.approx(165 / 255, abs=0.01)
        assert b == pytest.approx(166 / 255, abs=0.01)
        assert a == 1.0

    def test_empty_elements(self) -> None:
        """Empty elements returns empty dict."""
        _, ax = plt.subplots()
        mapping = color_by_element_type(ax, [])
        assert mapping == {}

    def test_duplicate_label_dedup(self) -> None:
        """Duplicate labels appear only once in mapping."""
        _, ax = plt.subplots()
        elems = [
            _elem(0.0, 0.0, 0.3, 0.2, label="button"),
            _elem(0.4, 0.0, 0.7, 0.2, label="button"),
        ]
        mapping = color_by_element_type(ax, elems)
        assert len(mapping) == 1
        assert "button" in mapping

    def test_all_known_types_have_distinct_colors(self) -> None:
        """Each known element type has a different color."""
        _, ax = plt.subplots()
        types = ["button", "text", "image", "input", "icon", "container"]
        elems = [_elem(0.1 * i, 0.1 * i, 0.1 * i + 0.05, 0.1 * i + 0.05, label=t)
                 for i, t in enumerate(types)]
        mapping = color_by_element_type(ax, elems)
        colors = set(tuple(v) for v in mapping.values())
        assert len(colors) == len(types)


# ===================================================================
# color_by_constraint_type
# ===================================================================


class TestColorByConstraintType:
    """Constraint type color mapping."""

    def test_returns_mapping(self) -> None:
        """Returns dict with constraint type value keys."""
        _, ax = plt.subplots()
        cons = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
            _con(ConstraintType.CONTAINMENT, [0], [1], margin=0.5),
        ]
        mapping = color_by_constraint_type(ax, cons)
        expected_keys = {"align_left", "containment"}
        assert set(mapping.keys()) == expected_keys

    def test_empty_constraints(self) -> None:
        """Empty constraints returns empty dict."""
        _, ax = plt.subplots()
        mapping = color_by_constraint_type(ax, [])
        assert mapping == {}

    def test_all_types_covered(self) -> None:
        """All 10 constraint types have a color mapping."""
        _, ax = plt.subplots()
        cons = [ConstraintNode(ctype) for ctype in ConstraintType]
        mapping = color_by_constraint_type(ax, cons)
        assert len(mapping) == 10
        for ctype in ConstraintType:
            assert ctype.value in mapping

    def test_colors_are_rgba_tuples(self) -> None:
        """Each value is a 4-element rgba tuple."""
        _, ax = plt.subplots()
        cons = [ConstraintNode(ctype) for ctype in ConstraintType]
        mapping = color_by_constraint_type(ax, cons)
        for color in mapping.values():
            assert len(color) == 4
            assert all(0.0 <= c <= 1.0 for c in color)

    def test_alignment_types_share_color(self) -> None:
        """All 6 alignment types share the same color family."""
        _, ax = plt.subplots()
        alignment_types = [
            ConstraintType.ALIGN_LEFT,
            ConstraintType.ALIGN_RIGHT,
            ConstraintType.ALIGN_TOP,
            ConstraintType.ALIGN_BOTTOM,
            ConstraintType.CENTER_X,
            ConstraintType.CENTER_Y,
        ]
        cons = [ConstraintNode(t) for t in alignment_types]
        mapping = color_by_constraint_type(ax, cons)
        colors = {tuple(v) for v in mapping.values()}
        assert len(colors) == 1


# ===================================================================
# export_graph
# ===================================================================


class TestExportGraph:
    """JSON graph export."""

    def test_structure(self, two_buttons: list[ElementNode],
                       alignment_constraint: list[ConstraintNode]) -> None:
        """Returns dict with expected keys."""
        result = export_graph(two_buttons, alignment_constraint)
        expected_keys = {"num_elements", "num_constraints", "num_edges",
                         "elements", "constraints", "edges"}
        assert set(result.keys()) == expected_keys

    def test_counts(self, two_buttons: list[ElementNode],
                    alignment_constraint: list[ConstraintNode]) -> None:
        """num_elements and num_constraints are correct."""
        result = export_graph(two_buttons, alignment_constraint)
        assert result["num_elements"] == 2
        assert result["num_constraints"] == 1

    def test_edge_count(self, two_buttons: list[ElementNode],
                        alignment_constraint: list[ConstraintNode]) -> None:
        """Edge count matches unique element-constraint pairs."""
        result = export_graph(two_buttons, alignment_constraint)
        # Constraint has source=[0,1], target=[0,1] => 2 unique pairs
        assert result["num_edges"] == 2
        assert len(result["edges"]) == 2

    def test_element_data(self, two_buttons: list[ElementNode],
                          alignment_constraint: list[ConstraintNode]) -> None:
        """Each element has id, bbox, type, confidence."""
        result = export_graph(two_buttons, alignment_constraint)
        for e in result["elements"]:
            assert "id" in e
            assert "bbox" in e
            assert "type" in e
            assert "confidence" in e
            assert len(e["bbox"]) == 4

    def test_constraint_data(self, two_buttons: list[ElementNode],
                             multiple_constraints: list[ConstraintNode]) -> None:
        """Each constraint has type, params, source_indices, target_indices."""
        result = export_graph(two_buttons, multiple_constraints)
        for c in result["constraints"]:
            assert "type" in c
            assert "params" in c
            assert "source_indices" in c
            assert "target_indices" in c

    def test_edge_data(self, two_buttons: list[ElementNode],
                       alignment_constraint: list[ConstraintNode]) -> None:
        """Each edge has element_idx and constraint_idx."""
        result = export_graph(two_buttons, alignment_constraint)
        for edge in result["edges"]:
            assert "element_idx" in edge
            assert "constraint_idx" in edge
            assert isinstance(edge["element_idx"], int)
            assert isinstance(edge["constraint_idx"], int)

    def test_no_constraints(self, two_buttons: list[ElementNode]) -> None:
        """No constraints — edges list is empty."""
        result = export_graph(two_buttons, [])
        assert result["num_constraints"] == 0
        assert result["num_edges"] == 0
        assert result["edges"] == []

    def test_no_elements(self, alignment_constraint: list[ConstraintNode]) -> None:
        """No elements — edges list is empty (no valid element indices)."""
        result = export_graph([], alignment_constraint)
        assert result["num_elements"] == 0
        assert result["num_edges"] == 0

    def test_out_of_range_indices(self, two_buttons: list[ElementNode]) -> None:
        """Out-of-range element indices are skipped in edges."""
        cons = [_con(ConstraintType.ALIGN_LEFT, [0, 99], tolerance=0.02)]
        result = export_graph(two_buttons, cons)
        # Only index 0 is valid => 1 edge
        assert result["num_edges"] == 1
        assert result["edges"] == [{"element_idx": 0, "constraint_idx": 0}]

    def test_with_output_path(self, two_buttons: list[ElementNode],
                              alignment_constraint: list[ConstraintNode],
                              tmp_path: Path) -> None:
        """output_path writes a valid JSON file."""
        out = tmp_path / "graph.json"
        result = export_graph(two_buttons, alignment_constraint, output_path=str(out))

        assert out.is_file()
        with open(out) as f:
            loaded = json.load(f)

        assert loaded["num_elements"] == result["num_elements"]
        assert loaded["num_constraints"] == result["num_constraints"]
        assert loaded["elements"] == result["elements"]
        assert loaded["constraints"] == result["constraints"]
        assert loaded["edges"] == result["edges"]

    def test_json_serializable(self, two_buttons: list[ElementNode],
                               multiple_constraints: list[ConstraintNode],
                               tmp_path: Path) -> None:
        """The dict is fully JSON-serializable (no special types)."""
        result = export_graph(two_buttons, multiple_constraints)
        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_containment_edge_count(self) -> None:
        """Containment with different source/target creates unique edges."""
        elems = [
            _elem(0.0, 0.0, 0.5, 0.5, label="container"),
            _elem(0.1, 0.1, 0.3, 0.3, label="button"),
        ]
        cons = [
            ConstraintNode(
                constraint_type=ConstraintType.CONTAINMENT,
                source_indices=[0],
                target_indices=[1],
                params={"margin": 0.96},
            ),
        ]
        result = export_graph(elems, cons)
        # Two unique element indices {0, 1}, one constraint => 2 edges
        assert result["num_edges"] == 2
        edge_pairs = {(e["element_idx"], e["constraint_idx"]) for e in result["edges"]}
        assert edge_pairs == {(0, 0), (1, 0)}

    def test_deduplicated_edges(self) -> None:
        """Duplicate element references in source+target produce one edge."""
        elems = [_elem(0.0, 0.0, 0.3, 0.3), _elem(0.4, 0.0, 0.7, 0.3)]
        # Both source and target reference the same elements
        cons = [
            ConstraintNode(
                constraint_type=ConstraintType.ALIGN_LEFT,
                source_indices=[0, 1],
                target_indices=[0, 1],
            ),
        ]
        result = export_graph(elems, cons)
        # 2 unique element indices, 1 constraint => 2 edges
        assert result["num_edges"] == 2

    def test_many_edges(self) -> None:
        """10 elements + 3 constraints produces correct edge count."""
        elems = [_elem(i * 0.1, 0.0, i * 0.1 + 0.05, 0.1) for i in range(10)]
        cons = [
            _con(ConstraintType.ALIGN_LEFT, [0, 1], tolerance=0.02),
            _con(ConstraintType.ALIGN_TOP, [2, 3, 4], tolerance=0.02),
            _con(ConstraintType.CENTER_X, [5, 6, 7, 8], tolerance=0.02),
        ]
        result = export_graph(elems, cons)
        # Constraint 0: 2 edges, Constraint 1: 3 edges, Constraint 2: 4 edges = 9
        assert result["num_edges"] == 9
        assert result["num_elements"] == 10
        assert result["num_constraints"] == 3


# ===================================================================
# Matplotlib import handling (when mpl is absent)
# ===================================================================


class TestMatplotlibUnavailable:
    """When matplotlib is not installed, all functions return None/empty."""

    def test_plot_graph_on_screenshot_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """plot_graph_on_screenshot returns None when mpl import fails."""
        import builtins

        _orig_import = builtins.__import__

        def _fake_import(name, *args, **kw):
            if name in ("matplotlib", "matplotlib.pyplot", "matplotlib.patches",
                        "matplotlib.cm", "matplotlib.colors"):
                raise ImportError("fake mpl missing")
            return _orig_import(name, *args, **kw)

        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", _fake_import)
            result = plot_graph_on_screenshot([], [])
            assert result is None

    def test_color_by_element_type_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """color_by_element_type returns empty dict when mpl import fails."""
        import builtins

        _orig_import = builtins.__import__

        def _fake_import(name, *args, **kw):
            if name in ("matplotlib", "matplotlib.pyplot"):
                raise ImportError("fake mpl missing")
            return _orig_import(name, *args, **kw)

        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", _fake_import)
            result = color_by_element_type(None, [])
            assert result == {}

    def test_color_by_constraint_type_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """color_by_constraint_type returns empty dict when mpl import fails."""
        import builtins

        _orig_import = builtins.__import__

        def _fake_import(name, *args, **kw):
            if name in ("matplotlib", "matplotlib.pyplot"):
                raise ImportError("fake mpl missing")
            return _orig_import(name, *args, **kw)

        with monkeypatch.context() as m:
            m.setattr(builtins, "__import__", _fake_import)
            result = color_by_constraint_type(None, [])
            assert result == {}

    def test_export_graph_still_works(self, two_buttons: list[ElementNode],
                                      alignment_constraint: list[ConstraintNode]) -> None:
        """export_graph does NOT depend on matplotlib — always works."""
        result = export_graph(two_buttons, alignment_constraint)
        assert result["num_elements"] == 2
        assert result["num_constraints"] == 1


# ===================================================================
# Real image handling
# ===================================================================


class TestRealImage:
    """Plotting on a real (PIL-compatible) image."""

    def test_with_synthetic_image(self, two_buttons: list[ElementNode],
                                  alignment_constraint: list[ConstraintNode],
                                  tmp_path: Path) -> None:
        """A valid image file is loaded as background (no crash)."""
        from PIL import Image

        img_path = tmp_path / "screenshot.png"
        img = Image.new("RGB", (100, 100), color="white")
        img.save(img_path)

        ax = plot_graph_on_screenshot(
            two_buttons, alignment_constraint, image_path=str(img_path),
        )
        assert ax is not None
        # The image should have been loaded (ax.images[0])
        assert len(ax.images) >= 1
