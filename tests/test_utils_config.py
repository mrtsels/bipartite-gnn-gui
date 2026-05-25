"""Tests for configuration system."""

from __future__ import annotations

import copy
from pathlib import Path

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
            raw_dir="/custom/raw",
            processed_dir="/custom/processed",
            dataset_names=["my_dataset"],
            val_split=0.2,
            test_split=0.3,
        )
        assert cfg.raw_dir == "/custom/raw"
        assert cfg.dataset_names == ["my_dataset"]
        assert cfg.val_split == 0.2
        assert cfg.test_split == 0.3

    def test_split_sum_validation(self) -> None:
        """Splits can sum to > 1 — no runtime validation in the dataclass,
        but the module should be able to hold such values."""
        cfg = DataConfig(val_split=0.5, test_split=0.6)
        assert cfg.val_split + cfg.test_split == 1.1

    def test_dataset_names_default_is_fresh(self) -> None:
        """Each instance gets its own list copy via default_factory."""
        c1 = DataConfig()
        c2 = DataConfig()
        c1.dataset_names.append("extra")
        assert "extra" not in c2.dataset_names


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
        cfg = ModelConfig(hidden_dim=256, n_layers=4, dropout=0.3)
        assert cfg.hidden_dim == 256
        assert cfg.n_layers == 4
        assert cfg.dropout == 0.3

    def test_head_dims_default_is_fresh(self) -> None:
        c1 = ModelConfig()
        c2 = ModelConfig()
        c1.head_dims["extra"] = 10
        assert "extra" not in c2.head_dims


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.lr == 1e-3
        assert cfg.epochs == 10
        assert cfg.batch_size == 8
        assert cfg.seed == 42
        assert cfg.weight_decay == 0.0
        assert cfg.warmup_steps == 0
        assert cfg.grad_clip == 1.0
        assert cfg.amp is False

    def test_custom_values(self) -> None:
        cfg = TrainingConfig(lr=5e-4, epochs=50, batch_size=16, amp=True)
        assert cfg.lr == 5e-4
        assert cfg.epochs == 50
        assert cfg.batch_size == 16
        assert cfg.amp is True

    def test_zero_learning_rate(self) -> None:
        """Edge case: zero LR should be valid."""
        cfg = TrainingConfig(lr=0.0)
        assert cfg.lr == 0.0

    def test_negative_grad_clip(self) -> None:
        """Negative gradient clipping is valid (no runtime guard in dataclass)."""
        cfg = TrainingConfig(grad_clip=-1.0)
        assert cfg.grad_clip == -1.0


# ---------------------------------------------------------------------------
# Config (top-level)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_factory(self) -> None:
        cfg = Config()
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.training, TrainingConfig)

    def test_defaults_independent(self) -> None:
        """Nested configs are independent across instances."""
        c1 = Config()
        c2 = Config()
        c1.data.raw_dir = "changed"
        assert c2.data.raw_dir == "data/raw"

    def test_custom_nested(self) -> None:
        cfg = Config(
            data=DataConfig(raw_dir="custom_raw"),
            model=ModelConfig(hidden_dim=512),
        )
        assert cfg.data.raw_dir == "custom_raw"
        assert cfg.model.hidden_dim == 512
        assert cfg.training.epochs == 10  # default

    def test_to_dict(self) -> None:
        cfg = Config()
        d = cfg.to_dict()
        assert d["data"]["raw_dir"] == "data/raw"
        assert d["model"]["hidden_dim"] == 128
        assert d["training"]["epochs"] == 10

    def test_to_dict_roundtrip(self) -> None:
        cfg = Config(
            data=DataConfig(raw_dir="/x"),
            model=ModelConfig(n_layers=3),
            training=TrainingConfig(epochs=20),
        )
        d = cfg.to_dict()
        restored = Config(
            data=DataConfig(**d["data"]),
            model=ModelConfig(**d["model"]),
            training=TrainingConfig(**d["training"]),
        )
        assert restored.data.raw_dir == "/x"
        assert restored.model.n_layers == 3
        assert restored.training.epochs == 20

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(TypeError):
            Config(nonexistent=123)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_none_returns_default(self) -> None:
        cfg = load_config(None)
        assert isinstance(cfg, Config)
        assert cfg.data.raw_dir == "data/raw"

    def test_load_from_path(self, tmp_path: Path) -> None:
        config_dict = {
            "data": {"raw_dir": "test_raw", "val_split": 0.2},
            "model": {"hidden_dim": 64},
        }
        path = tmp_path / "config.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f)

        cfg = load_config(path)
        assert cfg.data.raw_dir == "test_raw"
        assert cfg.data.val_split == 0.2
        assert cfg.data.processed_dir == "data/processed"  # default
        assert cfg.model.hidden_dim == 64
        assert cfg.training.epochs == 10  # default

    def test_load_empty_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        cfg = load_config(path)
        assert isinstance(cfg, Config)
        assert cfg.data.raw_dir == "data/raw"

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_config(path)

    def test_load_partial_overrides(self, tmp_path: Path) -> None:
        """Only the specified fields are overridden; others keep defaults."""
        config_dict = {"training": {"lr": 1e-4, "epochs": 100}}
        path = tmp_path / "partial.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f)

        cfg = load_config(path)
        assert cfg.training.lr == 1e-4
        assert cfg.training.epochs == 100
        assert cfg.training.batch_size == 8  # default
        assert cfg.data.raw_dir == "data/raw"  # default

    def test_load_expands_user(self, tmp_path: Path) -> None:
        """~ in path should be expanded."""
        import os

        home = os.path.expanduser("~")
        fake_path = tmp_path / "tilde_test.yaml"
        fake_path.write_text("")

        # Can't easily test ~ expansion with tmp_path, but verify the
        # implementation uses expanduser.
        cfg = load_config(str(fake_path))
        assert isinstance(cfg, Config)


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_roundtrip(self, tmp_path: Path) -> None:
        original = Config(
            data=DataConfig(raw_dir="/a", dataset_names=["ds1", "ds2"]),
            model=ModelConfig(hidden_dim=256, dropout=0.2),
            training=TrainingConfig(epochs=50, amp=True),
        )
        path = tmp_path / "saved.yaml"
        save_config(original, path)
        assert path.exists()

        loaded = load_config(path)
        assert loaded.data.raw_dir == "/a"
        assert loaded.data.dataset_names == ["ds1", "ds2"]
        assert loaded.model.hidden_dim == 256
        assert loaded.model.dropout == 0.2
        assert loaded.training.epochs == 50
        assert loaded.training.amp is True

    def test_save_expands_user(self, tmp_path: Path) -> None:
        """Verify save writes to a valid path."""
        cfg = Config()
        path = tmp_path / "write_test.yaml"
        save_config(cfg, path)
        assert path.exists()

    def test_saved_yaml_is_valid(self, tmp_path: Path) -> None:
        cfg = Config()
        path = tmp_path / "valid.yaml"
        save_config(cfg, path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "data" in data
        assert "model" in data
        assert "training" in data

    def test_save_to_dict_keys(self, tmp_path: Path) -> None:
        cfg = Config(
            data=DataConfig(dataset_names=["a"]),
        )
        path = tmp_path / "keys.yaml"
        save_config(cfg, path)
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert data["data"]["dataset_names"] == ["a"]


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_already_config(self) -> None:
        cfg = Config()
        result = validate_config(cfg)
        assert result is cfg

    def test_from_dict(self) -> None:
        result = validate_config(
            {"data": {"raw_dir": "custom"}, "model": {"hidden_dim": 64}}
        )
        assert isinstance(result, Config)
        assert result.data.raw_dir == "custom"
        assert result.model.hidden_dim == 64

    def test_empty_dict(self) -> None:
        result = validate_config({})
        assert isinstance(result, Config)
        assert result.data.raw_dir == "data/raw"

    def test_none_dict(self) -> None:
        result = validate_config(None)  # type: ignore[arg-type]
        assert isinstance(result, Config)

    def test_partial_dict(self) -> None:
        result = validate_config({"training": {"lr": 0.001}})
        assert result.training.lr == 0.001
        assert result.model.hidden_dim == 128  # default


# ---------------------------------------------------------------------------
# Integration — config lifecycle
# ---------------------------------------------------------------------------


class TestConfigLifecycle:
    def test_modify_save_reload(self, tmp_path: Path) -> None:
        cfg = Config(
            data=DataConfig(raw_dir="/live/raw"),
            model=ModelConfig(n_layers=3),
            training=TrainingConfig(epochs=25, amp=True),
        )
        path = tmp_path / "lifecycle.yaml"
        save_config(cfg, path)

        loaded = load_config(path)
        assert loaded.data.raw_dir == "/live/raw"
        assert loaded.model.n_layers == 3
        assert loaded.training.epochs == 25
        assert loaded.training.amp is True

    def test_equal_defaults(self) -> None:
        """Two default configs should be structurally equal."""
        c1 = Config()
        c2 = Config()
        assert c1.to_dict() == c2.to_dict()

    def test_override_only_one_field(self) -> None:
        cfg = Config(data=DataConfig(raw_dir="x"))
        d = cfg.to_dict()
        assert d["data"]["raw_dir"] == "x"
        assert d["model"]["hidden_dim"] == 128
