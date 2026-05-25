"""
Test suite for bipartite-gnn-gui.

Organized to mirror the source package structure:
    tests/data/     — Tests for data loading, preprocessing, and datasets.
    tests/graph/    — Tests for bipartite graph construction and constraints.
    tests/model/    — Tests for encoder, heads, loss, and training loop.
    tests/eval/     — Tests for evaluation metrics and evaluators.
    tests/utils/    — Tests for configuration, logging, and helpers.

Run all tests with:
    pytest tests/ -v

Run a specific test module:
    pytest tests/test_graph_builder.py -v
"""
