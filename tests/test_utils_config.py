"""Tests for config system — DataConfig, ModelConfig, TrainingConfig, Config,
load_config, save_config, validate_config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

from bipartite_gnn_gui.utils.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    save_config,
    validate_config,
)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


class TestDataConfig:
    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.raw_dir == "data/raw"
        assert cfg.processed_dir == "data/processed"
        assert cfg.dataset_names == ["gui360", "screenspot"]
        assert cfg.val_split == 0.1
        assert cfg.test_split == 0.1

    def test_custom_values(self) -> None:
        cfg = DataConfig(
            raw_dir="custom/raw",
            processed_dir="custom/processed",
            dataset_names=["my_dataset"],
            val_split=0.2,
            test_split=0.3,
        )
        assert cfg.raw_dir == "custom/raw"
        assert cfg.processed_dir == "custom/processed"
        assert cfg.dataset_names == ["my_dataset"]
        assert cfg.val_split == 0.2
        assert cfg.test_split == 0.3

    def test_dataset_names_is_copy(self) -> None:
        """Default factory should give a fresh list each time."""
        cfg1 = DataConfig()
        cfg2 = DataConfig()
        assert cfg1.dataset_names is not cfg2.dataset_names


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_defaults(self) -> None:
        cfg = ModelConfig()
        assert cfg.hidden_dim == 128
        assert cfg.n_layers == 2
        assert cfg.dropout == 0.1
        assert cfg.encoder_type == "bipartite_graphsage"
        assert cfg.head_dims == {"coord": 4, "violation": 1, "existence": 1}

    def test_custom_values(self) -> None:
        cfg = ModelConfig(
            hidden_dim=256,
            n_layers=3,
            dropout=0.5,
            encoder_type="gat",
            head_dims={"coord": 2, "violation": 1},
        )
        assert cfg.hidden_dim == 256
        assert cfg.n_layers == 3
        assert cfg.dropout == 0.5
        assert cfg.encoder_type == "gat"
        assert cfg.head_dims == {"coord": 2, "violation": 1}

    def test_head_dims_is_copy(self) -> None:
        cfg1 = ModelConfig()
        cfg2 = ModelConfig()
        assert cfg1.head_dims is not cfg2.head_dims


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lr == 0.001
        assert cfg.epochs == 10
        assert cfg.batch_size == 8
        assert cfg.seed == 42
        assert cfg.weight_decay == 0.0
        assert cfg.warmup_steps == 0
        assert cfg.grad_clip == 1.0
        assert cfg.amp is False

    def test_custom_values(self) -> None:
        cfg = TrainingConfig(
            lr=0.01,
            epochs=50,
            batch_size=32,
            seed=7,
            weight_decay=1e-4,
            warmup_steps=100,
            grad_clip=5.0,
            amp=True,
        )
        assert cfg.lr == 0.01
        assert cfg.epochs == 50
        assert cfg.batch_size == 32
        assert cfg.seed == 7
        assert cfg.weight_decay == 1e-4
        assert cfg.warmup_steps == 100
        assert cfg.grad_clip == 5.0
        assert cfg.amp is True


# ---------------------------------------------------------------------------
# Config (top-level)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_defaults(self) -> None:
        cfg = Config()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)
        # Verify nested defaults
        assert cfg.data.raw_dir == "data/raw"
        assert cfg.model.hidden_dim == 128
        assert cfg.training.lr == 0.001

    def test_custom_nested(self) -> None:
        cfg = Config(
            data=DataConfig(raw_dir="custom/raw"),
            model=ModelConfig(hidden_dim=512),
            training=TrainingConfig(epochs=100),
        )
        assert cfg.data.raw_dir == "custom/raw"
        assert cfg.model.hidden_dim == 512
        assert cfg.training.epochs == 100
        # Other fields should remain at defaults
        assert cfg.training.lr == 0.001

    def test_to_dict(self) -> None:
        cfg = Config()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "data" in d
        assert "model" in d
        assert "training" in d
        assert d["data"]["raw_dir"] == "data/raw"
        assert d["model"]["hidden_dim"] == 128
        assert d["training"]["lr"] == 0.001

    def test_to_dict_custom(self) -> None:
        cfg = Config(
            data=DataConfig(raw_dir="other/raw"),
            model=ModelConfig(n_layers=5),
            training=TrainingConfig(amp=True),
        )
        d = cfg.to_dict()
        assert d["data"]["raw_dir"] == "other/raw"
        assert d["model"]["n_layers"] == 5
        assert d["training"]["amp"] is True


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_none_returns_defaults(self) -> None:
        cfg = load_config(None)
        assert cfg.data.raw_dir == "data/raw"
        assert cfg.model.hidden_dim == 128
        assert cfg.training.lr == 0.001

    def test_load_full_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "full.yaml"
        payload = {
            "data": {
                "raw_dir": "custom/raw",
                "processed_dir": "custom/processed",
                "dataset_names": ["ds1", "ds2"],
                "val_split": 0.15,
                "test_split": 0.15,
            },
            "model": {
                "hidden_dim": 64,
                "n_layers": 4,
                "dropout": 0.2,
                "encoder_type": "sage",
                "head_dims": {"coord": 2},
            },
            "training": {
                "lr": 0.0005,
                "epochs": 20,
                "batch_size": 16,
                "seed": 123,
                "weight_decay": 1e-5,
                "warmup_steps": 50,
                "grad_clip": 0.5,
                "amp": True,
            },
        }
        with open(path, "w") as f:
            yaml.safe_dump(payload, f)

        cfg = load_config(path)
        assert cfg.data.raw_dir == "custom/raw"
        assert cfg.data.processed_dir == "custom/processed"
        assert cfg.data.dataset_names == ["ds1", "ds2"]
        assert cfg.data.val_split == 0.15
        assert cfg.model.hidden_dim == 64
        assert cfg.model.n_layers == 4
        assert cfg.model.head_dims == {"coord": 2}
        assert cfg.training.lr == 0.0005
        assert cfg.training.epochs == 20
        assert cfg.training.amp is True

    def test_load_partial_yaml(self, tmp_path: Path) -> None:
        """Partial YAML — only override a few fields, rest should be defaults."""
        path = tmp_path / "partial.yaml"
        payload = {
            "data": {"raw_dir": "partial/raw"},
            "training": {"lr": 0.01},
        }
        with open(path, "w") as f:
            yaml.safe_dump(payload, f)

        cfg = load_config(path)
        # Overridden fields
        assert cfg.data.raw_dir == "partial/raw"
        assert cfg.training.lr == 0.01
        # Default fields
        assert cfg.data.processed_dir == "data/processed"
        assert cfg.model.hidden_dim == 128
        assert cfg.training.epochs == 10

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        """Empty YAML file should yield all defaults."""
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert cfg.data.raw_dir == "data/raw"
        assert cfg.model.hidden_dim == 128
        assert cfg.training.lr == 0.001

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(path)

    def test_load_with_path_string(self, tmp_path: Path) -> None:
        """load_config should accept a string path."""
        path = tmp_path / "str.yaml"
        payload = {"data": {"val_split": 0.25}}
        with open(path, "w") as f:
            yaml.safe_dump(payload, f)

        cfg = load_config(str(path))
        assert cfg.data.val_split == 0.25

    def test_load_with_path_object(self, tmp_path: Path) -> None:
        """load_config should also accept a Path object."""
        path = tmp_path / "pathobj.yaml"
        payload = {"model": {"n_layers": 6}}
        with open(path, "w") as f:
            yaml.safe_dump(payload, f)

        cfg = load_config(path)
        assert cfg.model.n_layers == 6

    def test_load_and_modify(self, tmp_path: Path) -> None:
        """Load defaults, modify, and verify in-memory mutation works."""
        cfg = load_config(None)
        cfg.data.raw_dir = "modified/raw"
        cfg.model.hidden_dim = 999
        assert cfg.data.raw_dir == "modified/raw"
        assert cfg.model.hidden_dim == 999


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_save(self, tmp_path: Path) -> None:
        cfg = Config(
            data=DataConfig(raw_dir="save/raw"),
            model=ModelConfig(hidden_dim=256),
            training=TrainingConfig(epochs=5),
        )
        path = tmp_path / "saved.yaml"
        save_config(cfg, path)
        assert path.exists()

        # Verify file content is well-formed YAML
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded["data"]["raw_dir"] == "save/raw"
        assert loaded["model"]["hidden_dim"] == 256
        assert loaded["training"]["epochs"] == 5

    def test_round_trip(self, tmp_path: Path) -> None:
        """Save a config, reload it, and verify equivalence."""
        original = Config(
            data=DataConfig(raw_dir="rt/raw", dataset_names=["a", "b"]),
            model=ModelConfig(hidden_dim=64, head_dims={"coord": 2, "violation": 1}),
            training=TrainingConfig(lr=0.01, amp=True),
        )
        path = tmp_path / "roundtrip.yaml"
        save_config(original, path)
        reloaded = load_config(path)

        assert reloaded.data.raw_dir == original.data.raw_dir
        assert reloaded.data.dataset_names == original.data.dataset_names
        assert reloaded.model.hidden_dim == original.model.hidden_dim
        assert reloaded.model.head_dims == original.model.head_dims
        assert reloaded.training.lr == original.training.lr
        assert reloaded.training.amp == original.training.amp


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_from_config_object(self) -> None:
        cfg = Config(data=DataConfig(raw_dir="v/raw"))
        result = validate_config(cfg)
        assert result is cfg  # Should return the same object

    def test_from_dict(self) -> None:
        data: Dict[str, Any] = {
            "data": {"raw_dir": "dict/raw"},
            "model": {"hidden_dim": 512},
            "training": {"lr": 0.1},
        }
        cfg = validate_config(data)
        assert isinstance(cfg, Config)
        assert cfg.data.raw_dir == "dict/raw"
        assert cfg.model.hidden_dim == 512
        assert cfg.training.lr == 0.1

    def test_from_empty_dict(self) -> None:
        """Empty dict should produce all defaults."""
        cfg = validate_config({})
        assert isinstance(cfg, Config)
        assert cfg.data.raw_dir == "data/raw"
        assert cfg.model.hidden_dim == 128
        assert cfg.training.lr == 0.001

    def test_from_none_like(self) -> None:
        """None should produce all defaults via _coerce_config."""
        cfg = validate_config(None)  # type: ignore[arg-type]
        assert isinstance(cfg, Config)
        assert cfg.data.raw_dir == "data/raw"
        assert cfg.model.hidden_dim == 128

    def test_partial_dict(self) -> None:
        """Dict with only one section should get defaults for others."""
        cfg = validate_config({"data": {"raw_dir": "partial/raw"}})
        assert cfg.data.raw_dir == "partial/raw"
        assert cfg.model.hidden_dim == 128
        assert cfg.training.lr == 0.001


# ---------------------------------------------------------------------------
# Integration: modify → save → reload lifecycle
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_modify_save_reload(self, tmp_path: Path) -> None:
        """Simulate the full lifecycle: load defaults -> modify -> save -> reload."""
        # 1. Load defaults
        cfg = load_config(None)
        assert cfg.data.raw_dir == "data/raw"

        # 2. Modify
        cfg.data.raw_dir = "experiments/run1/raw"
        cfg.model.hidden_dim = 256
        cfg.training.epochs = 20

        # 3. Save
        path = tmp_path / "experiment.yaml"
        save_config(cfg, path)

        # 4. Reload
        cfg2 = load_config(path)

        # 5. Verify
        assert cfg2.data.raw_dir == "experiments/run1/raw"
        assert cfg2.model.hidden_dim == 256
        assert cfg2.training.epochs == 20
        # Unmodified defaults preserved
        assert cfg2.data.val_split == 0.1
        assert cfg2.model.dropout == 0.1
        assert cfg2.training.lr == 0.001

    def test_create_overwrite_reload(self, tmp_path: Path) -> None:
        """Save multiple configs to the same file (overwrite), verify last wins."""
        path = tmp_path / "overwrite.yaml"

        cfg1 = Config(data=DataConfig(raw_dir="first"))
        save_config(cfg1, path)
        assert load_config(path).data.raw_dir == "first"

        cfg2 = Config(data=DataConfig(raw_dir="second"))
        save_config(cfg2, path)
        assert load_config(path).data.raw_dir == "second"
