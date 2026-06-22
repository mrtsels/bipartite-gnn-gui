"""Tests for the hyperparameter sweep pipeline.

Tests cover:
1. run_experiment returns correctly shaped dict
2. sweep.py creates a results.json file
3. results.json correctly accumulates entries across runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =========================================================================
# Test 1: run_experiment returns the correct dict shape
# =========================================================================


class TestRunExperimentReturnsDict:
    """Verify that run_experiment returns a dict with expected keys."""

    EXPECTED_KEYS = {
        "best_val_loss",
        "final_train_loss",
        "recall",
        "precision",
        "f1",
        "position_error",
        "size_error",
        "noop_recall",
        "noop_precision",
        "noop_f1",
        "noop_position_error",
    }

    def test_return_shape(self):
        """Mock the full pipeline and verify dict keys."""
        from scripts.run_experiment import run_experiment, ExperimentConfig
        from bipartite_gnn_gui.graph.schema import ElementNode

        # Build mock with all needed patches
        patches = [
            patch("scripts.run_experiment.BipartiteGraphBuilder"),
            patch("scripts.run_experiment.Trainer"),
            patch("scripts.run_experiment.BipartiteGNNCorrector"),
            patch("scripts.run_experiment.Path.is_dir", return_value=True),
        ]

        # Create mock metrics
        mock_metrics = MagicMock()
        mock_metrics.recall = 0.85
        mock_metrics.precision = 0.42
        mock_metrics.f1 = 0.56
        mock_metrics.position_error = 0.12
        mock_metrics.size_error = 0.08

        mock_noop = MagicMock()
        mock_noop.recall = 0.99
        mock_noop.precision = 0.50
        mock_noop.f1 = 0.66
        mock_noop.position_error = 0.28

        with patch("scripts.run_experiment.evaluate_model",
                   return_value=(mock_metrics, mock_noop)):
            mock_parse = MagicMock()
            mock_parse.return_value = {
                "root": {"bounds": [0, 0, 1920, 1080]},
                "width": 1920,
                "height": 1080,
            }
            with patch("scripts.run_experiment.parse_rico_vh", mock_parse):
                with patch("scripts.run_experiment.extract_elements",
                           return_value=[
                               ElementNode(bbox=[0,0,100,100], label="text", confidence=1.0),
                               ElementNode(bbox=[50,50,150,150], label="button", confidence=1.0),
                           ]):
                    with patch("scripts.run_experiment.build_graph"):
                        with patch("scripts.run_experiment.Path.glob") as mock_glob:
                            fake_paths = [MagicMock(spec=Path) for _ in range(5)]
                            for p in fake_paths:
                                p.__str__.return_value = "/fake/rico/0.json"
                            mock_glob.return_value = fake_paths

                            cfg = ExperimentConfig()
                            cfg.n_samples = 5
                            cfg.rico_dir = "/fake/rico"
                            cfg.checkpoint_dir = "/tmp/fake_checkpoints"
                            cfg.epochs = 2

                            result = run_experiment(cfg)

        assert isinstance(result, dict), "run_experiment should return a dict"
        if "error" not in result:
            for key in self.EXPECTED_KEYS:
                assert key in result, f"Missing expected key: {key}"
            assert isinstance(result["best_val_loss"], float)
            assert isinstance(result["recall"], float)
            assert 0 <= result["recall"] <= 1.0
            assert isinstance(result["noop_recall"], float)


# =========================================================================
# Test 2: sweep creates results file
# =========================================================================


class TestSweepCreatesResultsFile:
    """Verify that sweep.py produces experiments/results.json."""

    def test_sweep_creates_results_json(self, tmp_path: Path):
        """Run sweep with 2 fast configs and check results.json exists."""
        test_configs = [
            ("test_a", 64, 1e-3, 0.08, 2),
            ("test_b", 128, 1e-3, 0.12, 2),
        ]

        mock_result = {
            "best_val_loss": 0.042,
            "final_train_loss": 0.038,
            "recall": 0.85,
            "precision": 0.42,
            "f1": 0.56,
            "position_error": 0.12,
            "size_error": 0.08,
            "noop_recall": 0.99,
            "noop_precision": 0.50,
            "noop_f1": 0.66,
            "noop_position_error": 0.28,
        }

        with patch("experiments.sweep.CONFIGS", test_configs), \
             patch("experiments.sweep.RESULTS_FILE",
                   tmp_path / "results.json"), \
             patch("experiments.sweep.CHECKPOINT_BASE",
                   tmp_path / "checkpoints"), \
             patch("experiments.sweep.run_experiment",
                   return_value=mock_result):

            from experiments.sweep import main as sweep_main
            sweep_main()

        results_file = tmp_path / "results.json"
        assert results_file.exists(), \
            f"results.json should exist at {results_file}"
        with open(results_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2, f"Expected 2 entries, got {len(data)}"
        assert data[0]["name"] == test_configs[0][0]
        assert data[1]["name"] == test_configs[1][0]


# =========================================================================
# Test 3: results.json append behavior
# =========================================================================


class TestResultsJsonAppend:
    """Verify that results.json correctly appends across multiple runs."""

    def test_append_accumulates_entries(self, tmp_path: Path):
        """Run sweep twice and verify 4 total entries."""
        results_file = tmp_path / "results.json"

        test_configs = [
            ("test_a", 64, 1e-3, 0.08, 2),
            ("test_b", 128, 1e-3, 0.12, 2),
        ]

        mock_result = {
            "best_val_loss": 0.042,
            "final_train_loss": 0.038,
            "recall": 0.85,
            "precision": 0.42,
            "f1": 0.56,
            "position_error": 0.12,
            "size_error": 0.08,
            "noop_recall": 0.99,
            "noop_precision": 0.50,
            "noop_f1": 0.66,
            "noop_position_error": 0.28,
        }

        # Run sweep twice
        for _ in range(2):
            with patch("experiments.sweep.CONFIGS", test_configs), \
                 patch("experiments.sweep.RESULTS_FILE", results_file), \
                 patch("experiments.sweep.CHECKPOINT_BASE",
                       tmp_path / "checkpoints"), \
                 patch("experiments.sweep.run_experiment",
                       return_value=mock_result):

                from experiments.sweep import main as sweep_main
                sweep_main()

        # Check results
        with open(results_file) as f:
            data = json.load(f)
        assert len(data) == 4, \
            f"Expected 4 entries (2 runs × 2 configs), got {len(data)}"

        # Verify structure of each entry
        for entry in data:
            assert "timestamp" in entry
            assert "name" in entry
            assert "config" in entry
            assert "results" in entry
            assert "hidden_dim" in entry["config"]
            assert "lr" in entry["config"]
            assert "noise_scale" in entry["config"]
            assert "best_val_loss" in entry["results"]

    def test_results_file_created_if_not_exists(self, tmp_path: Path):
        """Verify results.json is created if it doesn't already exist."""
        results_file = tmp_path / "results.json"
        assert not results_file.exists()

        test_configs = [("test_only", 64, 1e-3, 0.08, 2)]
        mock_result = {
            "best_val_loss": 0.05,
            "final_train_loss": 0.04,
            "recall": 0.8,
            "precision": 0.4,
            "f1": 0.53,
            "position_error": 0.15,
            "size_error": 0.10,
            "noop_recall": 0.95,
            "noop_precision": 0.45,
            "noop_f1": 0.61,
            "noop_position_error": 0.30,
        }

        with patch("experiments.sweep.CONFIGS", test_configs), \
             patch("experiments.sweep.RESULTS_FILE", results_file), \
             patch("experiments.sweep.CHECKPOINT_BASE",
                   tmp_path / "checkpoints"), \
             patch("experiments.sweep.run_experiment",
                   return_value=mock_result):

            from experiments.sweep import main as sweep_main
            sweep_main()

        assert results_file.exists()
        with open(results_file) as f:
            data = json.load(f)
        assert len(data) == 1


# =========================================================================
# Run tests
# =========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
