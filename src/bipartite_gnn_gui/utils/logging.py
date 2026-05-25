"""Logging helpers for experiments and scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional


def setup_logger(name: str = "bipartite_gnn_gui", level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Create or configure a logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(Path(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "bipartite_gnn_gui") -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name)


class MetricsLogger:
    """Base class for metric logging backends."""

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        raise NotImplementedError


class NoopMetricsLogger(MetricsLogger):
    """Metrics logger that intentionally does nothing."""

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        return None


class WandbMetricsLogger(MetricsLogger):
    """Optional Weights & Biases logger."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._available = False

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        return None


class TensorboardMetricsLogger(MetricsLogger):
    """Optional TensorBoard logger."""

    def __init__(self, *_: Any, **__: Any) -> None:
        self._available = False

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        return None
