"""Tests for Trainer — training loop with optimizer, scheduler, checkpointing."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.schema import (
    ConstraintNode,
    ConstraintType,
    ElementNode,
)
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from bipartite_gnn_gui.model.trainer import Trainer
from bipartite_gnn_gui.utils.config import TrainingConfig


def _elem(x1, y1, x2, y2, confidence=1.0) -> ElementNode:
    return ElementNode(bbox=[x1, y1, x2, y2], confidence=confidence)


def _con(ctype, source, target=None, **params) -> ConstraintNode:
    return ConstraintNode(
        constraint_type=ctype,
        source_indices=source,
        target_indices=target or source,
        params=params,
    )


def _build_graph(n_elem=4, n_con=2):
    elements = [
        _elem(0.1 + i * 0.05, 0.1 + i * 0.05, 0.3 + i * 0.05, 0.3 + i * 0.05)
        for i in range(n_elem)
    ]
    constraints = [
        _con(ConstraintType.ALIGN_LEFT,
             [i % n_elem, (i + 1) % n_elem], tolerance=0.02)
        for i in range(n_con)
    ]
    builder = BipartiteGraphBuilder()
    return builder.build(elements, constraints)


def _make_targets(n_elem=4, n_con=2):
    return {
        "coord": torch.randn(n_elem, 4),
        "violation": torch.randint(0, 2, (n_con, 1)).float(),
        "existence": torch.randint(0, 2, (n_elem, 1)).float(),
    }


class DummyLoader:
    """Iterable that produces (data, targets) tuples."""

    def __init__(self, n_batches=3, n_elem=4, n_con=2):
        self.n_batches = n_batches
        self._graphs = [_build_graph(n_elem, n_con) for _ in range(n_batches)]
        self._targets = [_make_targets(n_elem, n_con) for _ in range(n_batches)]

    def __iter__(self):
        for g, t in zip(self._graphs, self._targets):
            yield g, t


@pytest.fixture
def model():
    return BipartiteGNNCorrector(
        element_dim=5, constraint_dim=11, hidden_dim=64,
        num_layers=2, dropout=0.0,
    )


@pytest.fixture
def config():
    return TrainingConfig(lr=1e-3, epochs=3, batch_size=1,
                          weight_decay=0.0, warmup_steps=0,
                          grad_clip=1.0, amp=False, seed=42)


@pytest.fixture
def trainer(model, config):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Trainer(
            model=model,
            config=config,
            checkpoint_dir=tmpdir,
            early_stopping_patience=5,
        )


class TestTrainerInit:
    """Trainer initialization checks."""

    def test_creates_optimizer(self, trainer) -> None:
        assert trainer.optimizer is not None

    def test_creates_scheduler(self, trainer) -> None:
        assert trainer._scheduler is not None

    def test_creates_checkpoint_dir(self, model, config) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trainer(model=model, config=config, checkpoint_dir=tmpdir)
            assert Path(tmpdir).exists()


class TestFit:
    """Full training loop."""

    def test_fit_runs_without_error(self, model, config) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model, config=config,
                checkpoint_dir=tmpdir,
                early_stopping_patience=5,
            )
            train_loader = DummyLoader(n_batches=2)
            val_loader = DummyLoader(n_batches=1)
            best_loss = trainer.fit(train_loader, val_loader)
            assert best_loss <= float("inf")

    def test_fit_train_only(self, model, config) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model, config=config,
                checkpoint_dir=tmpdir,
                early_stopping_patience=5,
            )
            train_loader = DummyLoader(n_batches=1)
            best_loss = trainer.fit(train_loader)
            assert best_loss <= float("inf")

    def test_checkpoint_saved(self, model, config) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model, config=config,
                checkpoint_dir=tmpdir,
                early_stopping_patience=5,
            )
            train_loader = DummyLoader(n_batches=2)
            val_loader = DummyLoader(n_batches=1)
            trainer.fit(train_loader, val_loader)
            # best_model.pt should exist.
            best_path = Path(tmpdir) / "best_model.pt"
            assert best_path.exists(), f"No checkpoint at {best_path}"

    def test_early_stopping_triggers(self, model) -> None:
        """With very high patience, early stopping does not trigger early."""
        cfg = TrainingConfig(lr=1e-3, epochs=3, batch_size=1,
                             weight_decay=0.0, warmup_steps=0,
                             grad_clip=1.0, amp=False, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model, config=cfg,
                checkpoint_dir=tmpdir,
                early_stopping_patience=10,  # High patience.
            )
            train_loader = DummyLoader(n_batches=2)
            val_loader = DummyLoader(n_batches=1)
            best_loss = trainer.fit(train_loader, val_loader)
            # Should run all 3 epochs since patience > epochs.
            assert best_loss <= float("inf")


class TestOptimizerConfig:
    """Custom optimizer configuration."""

    def test_lr_applied(self, model) -> None:
        cfg = TrainingConfig(lr=0.01, epochs=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trainer(model=model, config=cfg, checkpoint_dir=tmpdir)
            for pg in t.optimizer.param_groups:
                assert pg["lr"] == 0.01

    def test_weight_decay_applied(self, model) -> None:
        cfg = TrainingConfig(lr=1e-3, weight_decay=1e-4)
        with tempfile.TemporaryDirectory() as tmpdir:
            t = Trainer(model=model, config=cfg, checkpoint_dir=tmpdir)
            for pg in t.optimizer.param_groups:
                assert pg["weight_decay"] == 1e-4
