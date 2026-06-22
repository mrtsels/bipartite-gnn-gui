"""Full training orchestrator with AMP, early stopping, and checkpointing.

Provides a ``Trainer`` class that manages the full training lifecycle:
epoch loops, optimizer/scheduler setup, mixed precision, gradient clipping,
validation, checkpointing, and early stopping.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import nn, torch

    AdamW = None  # type: ignore[assignment]
    CosineAnnealingLR = None  # type: ignore[assignment]
    LambdaLR = None  # type: ignore[assignment]

from bipartite_gnn_gui.utils.bbox import xyxy_to_xywh
from bipartite_gnn_gui.utils.config import TrainingConfig
from bipartite_gnn_gui.utils.logging import MetricsLogger, NoopMetricsLogger

from .losses import CombinedLoss
from .model import BipartiteGNNCorrector

logger = logging.getLogger(__name__)


def _get_device(device: torch.device | None = None) -> torch.device:
    """Return the target device, auto-detecting if not provided."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _linear_warmup(
    current_step: int, warmup_steps: int, peak_lr: float,
) -> float:
    """Linear warmup factor from 0 to 1."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, current_step / warmup_steps)


class Trainer:
    """Full training orchestrator for ``BipartiteGNNCorrector``.

    Manages:
        - Epoch training/validation loops.
        - AdamW optimizer with configurable learning rate and weight decay.
        - Cosine annealing scheduler with optional linear warmup.
        - Gradient clipping.
        - Automatic mixed precision (AMP) on CUDA.
        - Early stopping with patience.
        - Best-model checkpointing.
        - Metrics logging.

    Args:
        model: The ``BipartiteGNNCorrector`` to train.
        config: Training configuration (lr, epochs, grad_clip, etc.).
        device: Target device (auto-detected if ``None``).
        metrics_logger: Optional metrics logger (default no-op).
        checkpoint_dir: Directory for saving checkpoints.
        early_stopping_patience: Epochs without improvement before stopping.
        min_delta: Minimum loss improvement to count as progress.
    """

    def __init__(
        self,
        model: BipartiteGNNCorrector,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
        metrics_logger: MetricsLogger | None = None,
        checkpoint_dir: str | Path = "./checkpoints",
        early_stopping_patience: int = 10,
        min_delta: float = 1e-4,
    ) -> None:
        self.model = model
        self.config = config or TrainingConfig()
        self.device = _get_device(device)
        self.metrics_logger = metrics_logger or NoopMetricsLogger()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta

        self.model.to(self.device)

        # Optimizer.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Scheduler: cosine annealing with optional warmup.
        self._warmup_steps = self.config.warmup_steps
        self._scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr * 1e-3,
        )
        self._current_step = 0

        # AMP scaler.
        self._scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)  # type: ignore[attr-defined]

        # Early stopping state.
        self.best_val_loss = float("inf")
        self._patience_counter = 0
        self._best_epoch = 0

        # Checkpoint directory.
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
    ) -> float:
        """Run the full training loop.

        Args:
            train_loader: Iterable yielding ``(data, targets)`` tuples per batch.
            val_loader: Optional validation data loader.

        Returns:
            Best validation loss achieved.
        """
        logger.info(
            "Starting training: epochs=%d, lr=%.1e, device=%s, amp=%s",
            self.config.epochs,
            self.config.lr,
            self.device,
            self.config.amp,
        )

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch(train_loader, epoch)
            self.metrics_logger.log_metrics({"train_loss": train_loss}, step=epoch)
            logger.info(
                "Epoch %d/%d — train_loss: %.6f", epoch, self.config.epochs, train_loss
            )

            if val_loader is not None:
                val_loss = self._validate(val_loader)
                self.metrics_logger.log_metrics({"val_loss": val_loss}, step=epoch)
                logger.info(
                    "Epoch %d/%d — val_loss: %.6f", epoch, self.config.epochs, val_loss
                )

                improved = val_loss < self.best_val_loss - self.min_delta
                if improved:
                    self.best_val_loss = val_loss
                    self._best_epoch = epoch
                    self._patience_counter = 0
                    self._save_checkpoint(epoch, val_loss, is_best=True)
                    logger.info(
                        "Epoch %d — new best val_loss: %.6f", epoch, val_loss
                    )
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.early_stopping_patience:
                        logger.info(
                            "Early stopping at epoch %d (no improvement for %d epochs)",
                            epoch,
                            self._patience_counter,
                        )
                        break

                # Step scheduler after validation.
                self._scheduler.step()
            else:
                self._scheduler.step()

        self.metrics_logger.finish()
        return self.best_val_loss

    def _train_epoch(self, loader: Any, epoch: int) -> float:
        """Train for one epoch, return average loss."""
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                data, targets = batch
            else:
                data = batch
                targets = {}

            data = self._to_device(data)
            targets = self._to_device(targets)

            # Warmup scaling.
            self._current_step += 1
            warmup_factor = _linear_warmup(
                self._current_step, self._warmup_steps, self.config.lr
            )
            if self._current_step <= self._warmup_steps:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.config.lr * warmup_factor

            # Extract alignment loss inputs from HeteroData.
            original_bboxes = None
            edge_index = None
            try:
                original_bboxes = xyxy_to_xywh(data["element"].x[:, :4])
            except Exception:
                pass
            try:
                edge_index = data["element", "to", "constraint"].edge_index
            except Exception:
                pass

            with torch.cuda.amp.autocast(enabled=self.config.amp):  # type: ignore[attr-defined]
                predictions = self.model(data)
                loss = self.model.compute_loss(
                    predictions, targets,
                    original_bboxes=original_bboxes,
                    edge_index=edge_index,
                )

            self.optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            if self.config.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )
            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.detach()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss.item() if isinstance(avg_loss, torch.Tensor) else float(avg_loss)  # type: ignore[return-value]

    def _validate(self, loader: Any) -> float:
        """Evaluate model on the validation set."""
        self.model.eval()
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    data, targets = batch
                else:
                    data = batch
                    targets = {}

                data = self._to_device(data)
                targets = self._to_device(targets)

                # Extract alignment loss inputs from HeteroData.
                original_bboxes = None
                edge_index = None
                try:
                    original_bboxes = xyxy_to_xywh(data["element"].x[:, :4])
                except Exception:
                    pass
                try:
                    edge_index = data["element", "to", "constraint"].edge_index
                except Exception:
                    pass

                with torch.cuda.amp.autocast(enabled=self.config.amp):  # type: ignore[attr-defined]
                    predictions = self.model(data)
                    loss = self.model.compute_loss(
                        predictions, targets,
                        original_bboxes=original_bboxes,
                        edge_index=edge_index,
                    )

                total_loss += loss.detach()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss.item() if isinstance(avg_loss, torch.Tensor) else float(avg_loss)  # type: ignore[return-value]

    def _to_device(self, obj: Any) -> Any:
        """Recursively move tensors in a nested structure to ``self.device``."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(v) for v in obj)
        # Handle PyG HeteroData/Data objects (have a .to() method).
        if hasattr(obj, "to"):
            return obj.to(self.device)
        return obj

    def _save_checkpoint(
        self, epoch: int, val_loss: float, is_best: bool = False,
    ) -> None:
        """Save model checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        if is_best:
            path = self.checkpoint_dir / "best_model.pt"
        else:
            path = self.checkpoint_dir / f"model_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, path)
        logger.debug("Checkpoint saved: %s", path)
