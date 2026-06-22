"""
Evaluation module — Metrics and evaluators for GUI structure correction.

Defines the core metrics for evaluating GUI element parsing quality:

    PositionError     — ‖(x̂, ŷ) − (x, y)‖₂  (Euclidean distance of top-left corner).
    SizeError         — ‖(ŵ, ĥ) − (w, h)‖₂   (Euclidean distance of width & height).
    AlignmentError    — Deviation from expected alignment groups.
    ElementRecall     — Fraction of ground-truth elements correctly matched (IoU > 0.5).
    ElementPrecision  — Fraction of predicted elements matching a ground-truth element.
    IoU               — Intersection-over-Union for matched element pairs.

Submodules:
    metrics      — Individual metric implementations.
    evaluator    — Evaluator class that computes all metrics over a dataset.
    baselines    — Baseline correction methods for comparison.
    qualitative  — Visualization tools for qualitative analysis.
"""

from .metrics import (
    AlignmentError,
    ElementPrecision,
    ElementRecall,
    F1Score,
    MetricsBundle,
    PositionError,
    SizeError,
    compute_all_metrics,
    compute_iou,
)
from .evaluator import Evaluator, EvaluationResult
from .baselines import BaselineNoCorrection, BaselineRuleBased, BaselineMLPOnly
from .qualitative import (
    side_by_side_plot,
    plot_error_heatmap,
    plot_category_breakdown,
)

__all__ = [
    "AlignmentError",
    "ElementPrecision",
    "ElementRecall",
    "F1Score",
    "MetricsBundle",
    "PositionError",
    "SizeError",
    "compute_all_metrics",
    "compute_iou",
    "Evaluator",
    "EvaluationResult",
    "BaselineNoCorrection",
    "BaselineRuleBased",
    "BaselineMLPOnly",
    "side_by_side_plot",
    "plot_error_heatmap",
    "plot_category_breakdown",
]
