"""Tests for logging helpers."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pytest

from bipartite_gnn_gui.utils.logging import (
    MetricsLogger,
    NoopMetricsLogger,
    TensorboardMetricsLogger,
    WandbMetricsLogger,
    get_logger,
    setup_logger,
)


def _clean_loggers(*names: str) -> None:
    """Remove all handlers from named loggers to keep tests isolated."""
    for name in names:
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# setup_logger
# ---------------------------------------------------------------------------


class TestSetupLogger:
    """setup_logger creates and configures loggers."""

    def test_basic(self) -> None:
        name = "test_basic"
        _clean_loggers(name)
        logger = setup_logger(name)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1

    def test_debug_level(self) -> None:
        name = "test_debug"
        _clean_loggers(name)
        logger = setup_logger(name, level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_with_file(self, tmp_path: Path) -> None:
        name = "test_file"
        log_file = tmp_path / "test.log"
        _clean_loggers(name)
        logger = setup_logger(name, log_file=log_file)
        logger.info("hello world")
        assert log_file.exists()
        content = log_file.read_text()
        assert "hello world" in content

    def test_no_duplicate_stream_handlers(self) -> None:
        name = "test_no_dup"
        _clean_loggers(name)
        setup_logger(name)
        n_handlers_before = len(logging.getLogger(name).handlers)
        setup_logger(name)
        logger = logging.getLogger(name)
        assert len(logger.handlers) == n_handlers_before

    def test_file_append(self, tmp_path: Path) -> None:
        name = "test_append"
        log_file = tmp_path / "append.log"
        _clean_loggers(name)
        logger = setup_logger(name, log_file=log_file)
        logger.info("first")
        logger.info("second")
        content = log_file.read_text()
        assert content.count("first") == 1
        assert content.count("second") == 1

    def test_formatted_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        name = "test_format"
        _clean_loggers(name)
        logger = setup_logger(name, level=logging.DEBUG)
        logger.info("format check")
        captured = capsys.readouterr()
        # Format: "%(asctime)s %(levelname)s %(name)s: %(message)s"
        # StreamHandler defaults to sys.stderr
        assert "format check" in captured.err
        assert "INFO" in captured.err
        assert name in captured.err
        # Verify timestamp-like prefix (digits at start)
        assert re.search(r"\d{4}-\d{2}-\d{2}", captured.err)


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """get_logger returns configured logger instances."""

    def test_returns_logger(self) -> None:
        logger = get_logger("test_get")
        assert isinstance(logger, logging.Logger)

    def test_default_name(self) -> None:
        logger = get_logger()
        assert logger.name == "bipartite_gnn_gui"

    def test_same_instance(self) -> None:
        a = get_logger("test_same")
        b = get_logger("test_same")
        assert a is b

    def test_picks_up_setup_config(self) -> None:
        name = "test_setup_get"
        _clean_loggers(name)
        setup_logger(name, level=logging.WARNING)
        logger = get_logger(name)
        assert logger.level == logging.WARNING


# ---------------------------------------------------------------------------
# MetricsLogger ABC
# ---------------------------------------------------------------------------


class TestMetricsLoggerABC:
    """MetricsLogger is an abstract base class."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            MetricsLogger()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class MinimalLogger(MetricsLogger):
            def log_metrics(self, metrics: dict, step: int | None = None) -> None:
                pass

            def finish(self) -> None:
                pass

        instance = MinimalLogger()
        assert isinstance(instance, MetricsLogger)
        assert isinstance(instance, ABC)


# ---------------------------------------------------------------------------
# NoopMetricsLogger
# ---------------------------------------------------------------------------


class TestNoopMetricsLogger:
    """NoopMetricsLogger silently accepts all calls."""

    def test_log_metrics_does_not_raise(self) -> None:
        logger = NoopMetricsLogger()
        logger.log_metrics({"loss": 0.5, "acc": 0.9})
        logger.log_metrics({})
        logger.log_metrics({"lr": 1e-4}, step=10)

    def test_finish_does_not_raise(self) -> None:
        logger = NoopMetricsLogger()
        logger.finish()


# ---------------------------------------------------------------------------
# WandbMetricsLogger
# ---------------------------------------------------------------------------


class TestWandbMetricsLogger:
    """WandbMetricsLogger gracefully handles missing wandb package."""

    def test_fallback(self) -> None:
        logger = WandbMetricsLogger()
        assert not logger.available
        logger.log_metrics({"loss": 0.5})
        logger.finish()

    def test_multiple_metrics(self) -> None:
        logger = WandbMetricsLogger()
        logger.log_metrics({"loss": 0.1, "acc": 0.95, "lr": 1e-4})
        logger.finish()

    def test_empty_metrics(self) -> None:
        logger = WandbMetricsLogger()
        logger.log_metrics({})
        logger.finish()

    def test_kwargs_accepted(self) -> None:
        logger = WandbMetricsLogger(project="test", config={"lr": 0.01})
        assert not logger.available
        logger.finish()


# ---------------------------------------------------------------------------
# TensorboardMetricsLogger
# ---------------------------------------------------------------------------


class TestTensorboardMetricsLogger:
    """TensorboardMetricsLogger gracefully handles missing tensorboard package."""

    def test_fallback(self) -> None:
        logger = TensorboardMetricsLogger()
        assert not logger.available
        logger.log_metrics({"loss": 0.5})
        logger.finish()

    def test_multiple_metrics(self) -> None:
        logger = TensorboardMetricsLogger()
        logger.log_metrics({"loss": 0.1, "acc": 0.95, "lr": 1e-4})
        logger.finish()

    def test_empty_metrics(self) -> None:
        logger = TensorboardMetricsLogger()
        logger.log_metrics({})
        logger.finish()

    def test_custom_log_dir(self, tmp_path: Path) -> None:
        log_dir = tmp_path / "tb_logs"
        logger = TensorboardMetricsLogger(log_dir=str(log_dir))
        assert not logger.available
        logger.finish()


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestIntegration:
    """End-to-end scenarios combining multiple logging components."""

    def test_console_logger_lifecycle(self, capsys: pytest.CaptureFixture[str]) -> None:
        name = "int_console"
        _clean_loggers(name)
        logger = setup_logger(name, level=logging.DEBUG)
        logger.info("lifecycle check")
        captured = capsys.readouterr()
        # StreamHandler defaults to sys.stderr
        assert "lifecycle check" in captured.err

    def test_file_logger_content(self, tmp_path: Path) -> None:
        name = "int_file"
        log_file = tmp_path / "integration.log"
        _clean_loggers(name)
        logger = setup_logger(name, log_file=log_file)
        logger.info("integration test message")
        content = log_file.read_text()
        assert "integration test message" in content
        assert "INFO" in content
        assert name in content

    def test_multiple_loggers_to_separate_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "alpha.log"
        f2 = tmp_path / "beta.log"

        _clean_loggers("alpha", "beta")

        logger_a = setup_logger("alpha", log_file=f1)
        logger_b = setup_logger("beta", log_file=f2)

        logger_a.info("from alpha")
        logger_b.info("from beta")

        content_a = f1.read_text()
        content_b = f2.read_text()

        assert "from alpha" in content_a
        assert "from beta" not in content_a
        assert "from beta" in content_b
        assert "from alpha" not in content_b

    def test_all_backends_satisfy_interface(self) -> None:
        """Every backend implements log_metrics and finish without raising."""
        backends: list[MetricsLogger] = [
            NoopMetricsLogger(),
            WandbMetricsLogger(),
            TensorboardMetricsLogger(),
        ]
        for backend in backends:
            backend.log_metrics({"loss": 0.5})
            backend.finish()
