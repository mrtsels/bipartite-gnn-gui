"""Logging helpers for experiments and scripts.

Provides structured console logging and optional experiment tracking
backends (WandB, TensorBoard) with graceful fallback when extras
are not installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Console logging
# ---------------------------------------------------------------------------

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "bipartite_gnn_gui",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """Create or configure a logger.

    Args:
        name: Logger name.
        level: Logging level (e.g. ``logging.DEBUG``).
        log_file: Optional path to a log file. When provided, output
            is written both to stderr and the file.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(Path(log_file))
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "bipartite_gnn_gui") -> logging.Logger:
    """Return a module logger.

    Args:
        name: Logger name.

    Returns:
        :class:`logging.Logger` instance (may be pre-configured by
        :func:`setup_logger`).
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Metrics logging backends
# ---------------------------------------------------------------------------


class MetricsLogger(ABC):
    """Abstract base class for metric logging backends."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dictionary of metrics.

        Args:
            metrics: Metric name → scalar value mapping.
            step: Optional global training step.
        """

    def finish(self) -> None:
        """Clean up resources (e.g. close file handles, flush buffers)."""


class NoopMetricsLogger(MetricsLogger):
    """Metrics logger that intentionally does nothing.

    Use as a drop-in default when no experiment tracking backend is
    configured.
    """

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        return None


class WandbMetricsLogger(MetricsLogger):
    """Optional Weights & Biases logger.

    Gracefully degrades to a no-op when ``wandb`` is not installed.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        try:
            import wandb  # noqa: F811

            self._run = wandb.init(*args, **kwargs)  # type: ignore[misc]
            self._available = True
        except (ImportError, Exception):
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the WandB backend was successfully initialised."""
        return self._available

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if not self._available:
            return None
        import wandb

        wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self._available:
            import wandb

            wandb.finish()


class TensorboardMetricsLogger(MetricsLogger):
    """Optional TensorBoard logger.

    Gracefully degrades to a no-op when ``torch.utils.tensorboard`` is
    not available.
    """

    def __init__(self, log_dir: str | Path = "runs", **kwargs: Any) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir), **kwargs)
            self._available = True
        except (ImportError, Exception):
            self._available = False

    @property
    def available(self) -> bool:
        """Whether the TensorBoard backend was successfully initialised."""
        return self._available

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        if not self._available:
            return None
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, global_step=step or 0)

    def finish(self) -> None:
        if self._available:
            self._writer.close()
