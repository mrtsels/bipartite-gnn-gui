"""Lightweight configuration objects and YAML loading."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


@dataclass
class DataConfig:
    """Data-related configuration."""

    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset_names: list[str] = field(default_factory=lambda: ["gui360", "screenspot"])
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class ModelConfig:
    """Model hyperparameters."""

    hidden_dim: int = 128
    n_layers: int = 2
    dropout: float = 0.1
    encoder_type: str = "bipartite_graphsage"
    head_dims: dict[str, int] = field(default_factory=lambda: {"coord": 4, "violation": 1, "existence": 1})


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 8
    seed: int = 42
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip: float = 1.0
    amp: bool = False


@dataclass
class Config:
    """Top-level experiment configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a plain dictionary."""

        return asdict(self)


def _coerce_config(data: Mapping[str, Any] | None) -> Config:
    payload = dict(data or {})
    return Config(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
    )


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML or return defaults."""

    if path is None:
        return Config()

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return _coerce_config(yaml.safe_load(handle))


def validate_config(config: Config | Mapping[str, Any]) -> Config:
    """Validate and normalize a config object."""

    if isinstance(config, Config):
        return config
    return _coerce_config(config)


def save_config(config: Config, path: str | Path) -> None:
    """Save configuration to YAML."""

    with Path(path).expanduser().open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
