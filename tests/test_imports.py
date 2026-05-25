"""Smoke tests for package import resolution."""

from __future__ import annotations


def test_package_imports() -> None:
    import bipartite_gnn_gui
    from bipartite_gnn_gui import data, eval, graph, model, utils

    assert bipartite_gnn_gui.__version__ == "0.1.0"
    assert hasattr(data, "GUIDataset")
    assert hasattr(graph, "BipartiteGraphBuilder")
    assert hasattr(model, "BipartiteGNNCorrector")
    assert hasattr(eval, "Evaluator")
    assert hasattr(utils, "Config")
