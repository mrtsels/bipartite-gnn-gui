"""Logging helpers for experiments and scripts."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional wandb backend
# ---------------------------------------------------------------------------

try:
    import wandb

    _wandb_available = True
except ImportError:
    _wandb_available = False

# ---------------------------------------------------------------------------
# Optional TensorBoard backend
# ---------------------------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter

    _tensorboard_available = True
except ImportError:
    _tensorboard_available = False


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


class MetricsLogger(ABC):
    """Base class for metric logging backends."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics.

        Args:
            metrics: Metric name-to-value mapping.
            step: Optional global step / epoch number.
        """

    @abstractmethod
    def finish(self) -> None:
        """Flush and clean up the logging backend."""


class NoopMetricsLogger(MetricsLogger):
    """Metrics logger that intentionally does nothing."""

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbMetricsLogger(MetricsLogger):
    """Optional Weights & Biases logger.

    When the ``wandb`` package is installed, ``__init__`` calls
    ``wandb.init()`` and ``log_metrics`` forwards to ``wandb.log()``.
    When unavailable, all methods are no-ops and ``available`` returns
    ``False``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._available = _wandb_available
        self._run = None
        if _wandb_available:
            self._run = wandb.init(*args, **kwargs)

    @property
    def available(self) -> bool:
        """Whether the wandb backend is installed."""
        return self._available

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


class TensorboardMetricsLogger(MetricsLogger):
    """Optional TensorBoard logger.

    When ``torch.utils.tensorboard`` is available, ``__init__`` creates
    a ``SummaryWriter`` and ``log_metrics`` writes each metric via
    ``add_scalar``.  When unavailable, all methods are no-ops and
    ``available`` returns ``False``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._available = _tensorboard_available
        self._writer = None
        if _tensorboard_available:
            self._writer = SummaryWriter(*args, **kwargs)

    @property
    def available(self) -> bool:
        """Whether the TensorBoard backend is installed."""
        return self._available

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if self._writer is not None:
            for key, value in metrics.items():
                self._writer.add_scalar(key, value, global_step=step or 0)

    def finish(self) -> None:
        if self._writer is not None:
            self._writer.close()
