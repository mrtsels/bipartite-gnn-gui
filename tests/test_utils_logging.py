"""Tests for logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pytest

from bipartite_gnn_gui.utils.logging import (
    MetricsLogger,
    NoopMetricsLogger,
    TensorboardMetricsLogger,
    WandbMetricsLogger,
    get_logger,
    setup_logger,
)


# ---------------------------------------------------------------------------
# setup_logger / get_logger
# ---------------------------------------------------------------------------


class TestSetupLogger:
    def test_basic_logger(self) -> None:
        logger = setup_logger("test_basic")
        assert logger.name == "test_basic"
        assert logger.level == logging.INFO
        assert logger.propagate is False

    def test_logger_level_debug(self) -> None:
        logger = setup_logger("test_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_with_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "test.log"
        logger = setup_logger("test_file", log_file=str(log_file))
        logger.info("hello")
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "hello" in content

    def test_logger_no_duplicate_handlers(self) -> None:
        logger = setup_logger("test_duplicate")
        original_handler_count = len(logger.handlers)
        logger = setup_logger("test_duplicate")
        assert len(logger.handlers) == original_handler_count

    def test_file_handler_append(self, tmp_path: Path) -> None:
        log_file = tmp_path / "append.log"
        logger1 = setup_logger("test_append1", log_file=str(log_file))
        logger1.info("first")
        logger2 = setup_logger("test_append2", log_file=str(log_file))
        logger2.info("second")
        # Use a separate logger instance to read the file
        content = log_file.read_text(encoding="utf-8")
        assert "first" in content
        assert "second" in content

    def test_logger_formatted_output(self, tmp_path: Path) -> None:
        log_file = tmp_path / "format.log"
        logger = setup_logger("test_format", log_file=str(log_file))
        logger.info("format check")
        content = log_file.read_text(encoding="utf-8")
        # The default format contains timestamp, level, name, message
        assert "INFO" in content
        assert "test_format" in content
        assert "format check" in content
        assert " " in content  # timestamp has spaces


class TestGetLogger:
    def test_get_logger_returns_logger(self) -> None:
        logger = get_logger("test_get")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_get"

    def test_get_logger_default_name(self) -> None:
        logger = get_logger()
        assert logger.name == "bipartite_gnn_gui"

    def test_get_logger_returns_same_instance(self) -> None:
        """get_logger should return the same instance for the same name."""
        l1 = get_logger("test_same")
        l2 = get_logger("test_same")
        assert l1 is l2

    def test_get_logger_picks_up_setup(self) -> None:
        """Logger configured via setup_logger should be returned by get_logger."""
        setup_logger("test_pickup", level=logging.DEBUG)
        logger = get_logger("test_pickup")
        assert logger.level == logging.DEBUG


# ---------------------------------------------------------------------------
# MetricsLogger ABC
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    def test_abc_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            MetricsLogger()  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        class ConcreteLogger(MetricsLogger):
            def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
                self.last = (metrics, step)

        cl = ConcreteLogger()
        cl.log_metrics({"loss": 1.0}, step=10)
        assert cl.last == ({"loss": 1.0}, 10)


# ---------------------------------------------------------------------------
# NoopMetricsLogger
# ---------------------------------------------------------------------------


class TestNoopMetricsLogger:
    def test_log_metrics(self) -> None:
        logger = NoopMetricsLogger()
        # Should not raise
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.log_metrics({})
        logger.log_metrics({"a": 1, "b": 2}, step=100)

    def test_finish(self) -> None:
        logger = NoopMetricsLogger()
        logger.finish()  # Should not raise


# ---------------------------------------------------------------------------
# WandbMetricsLogger (no wandb installed → graceful fallback)
# ---------------------------------------------------------------------------


class TestWandbMetricsLogger:
    def test_fallback_when_wandb_missing(self) -> None:
        """When wandb is not installed, the logger degrades gracefully."""
        logger = WandbMetricsLogger()
        assert logger.available is False
        # Should not raise
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.finish()

    def test_multiple_metrics(self) -> None:
        logger = WandbMetricsLogger()
        logger.log_metrics({"a": 1, "b": 2, "c": 3})
        logger.log_metrics({"a": 2}, step=10)
        # No crash — graceful fallback

    def test_empty_metrics(self) -> None:
        logger = WandbMetricsLogger()
        logger.log_metrics({})
        logger.finish()

    def test_init_kwargs_accepted(self) -> None:
        """Extra kwargs should be accepted (forwarded to wandb.init)."""
        logger = WandbMetricsLogger(project="test", config={"lr": 0.01})
        assert logger.available is False


# ---------------------------------------------------------------------------
# TensorboardMetricsLogger (no tensorboard installed → graceful fallback)
# ---------------------------------------------------------------------------


class TestTensorboardMetricsLogger:
    def test_fallback_when_tensorboard_missing(self) -> None:
        logger = TensorboardMetricsLogger()
        assert logger.available is False
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.finish()

    def test_multiple_metrics(self) -> None:
        logger = TensorboardMetricsLogger()
        logger.log_metrics({"a": 1, "b": 2})
        logger.log_metrics({"a": 2}, step=10)

    def test_empty_metrics(self) -> None:
        logger = TensorboardMetricsLogger()
        logger.log_metrics({})

    def test_custom_log_dir(self, tmp_path: Path) -> None:
        """Custom log_dir should be accepted even when TensorBoard unavailable."""
        logger = TensorboardMetricsLogger(log_dir=str(tmp_path / "tb_logs"))
        assert logger.available is False


# ---------------------------------------------------------------------------
# Integration — logging lifecycle
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_console_logger_lifecycle(self) -> None:
        logger = setup_logger("test_lifecycle")
        logger.info("start")
        logger.debug("not shown")  # default level is INFO
        logger.warning("warning")
        logger.error("error")

    def test_file_logger_content(self, tmp_path: Path) -> None:
        log_file = tmp_path / "integration.log"
        logger = setup_logger("test_integration_file", log_file=str(log_file))
        msgs = ["first message", "second message", "third message"]
        for m in msgs:
            logger.info(m)
        content = log_file.read_text(encoding="utf-8")
        for m in msgs:
            assert m in content

    def test_multiple_loggers_separate_files(self, tmp_path: Path) -> None:
        f1 = tmp_path / "run1.log"
        f2 = tmp_path / "run2.log"
        l1 = setup_logger("test_multi1", log_file=str(f1))
        l2 = setup_logger("test_multi2", log_file=str(f2))
        l1.info("from logger 1")
        l2.info("from logger 2")
        assert "from logger 1" in f1.read_text(encoding="utf-8")
        assert "from logger 2" in f2.read_text(encoding="utf-8")

    def test_all_metric_loggers_accept_same_interface(self) -> None:
        """All backends satisfy the MetricsLogger interface."""
        loggers: list[MetricsLogger] = [
            NoopMetricsLogger(),
            WandbMetricsLogger(),
            TensorboardMetricsLogger(),
        ]
        for logger in loggers:
            logger.log_metrics({"loss": 1.0}, step=0)
            logger.log_metrics({"accuracy": 0.95})
            logger.finish()
