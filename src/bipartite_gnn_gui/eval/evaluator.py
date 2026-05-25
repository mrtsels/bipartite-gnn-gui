"""Dataset evaluator for model outputs."""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import compute_all_metrics


@dataclass
class EvaluationResult:
    """Simple wrapper for evaluation output."""

    metrics: dict[str, float]


class Evaluator:
    """Evaluate predictions against targets."""

    def evaluate(self, prediction_boxes, target_boxes) -> EvaluationResult:
        return EvaluationResult(metrics=compute_all_metrics(prediction_boxes, target_boxes))
