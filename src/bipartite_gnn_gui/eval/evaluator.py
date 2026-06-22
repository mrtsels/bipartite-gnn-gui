"""Dataset evaluator for model outputs.

Provides a ``Evaluator`` class that computes the full metrics pipeline over
a trained model and dataset, aggregating results into a structured
``EvaluationResult`` with global, per-category, and per-source breakdowns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover - optional dependency fallback
    from bipartite_gnn_gui._compat import Tensor, torch

from .metrics import (
    AlignmentError,
    ElementPrecision,
    ElementRecall,
    F1Score,
    MetricsBundle,
    PositionError,
    SizeError,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Structured container for evaluation output.

    Attributes:
        global_metrics: Overall metrics (recall, precision, f1, errors).
        per_category: Metrics broken down by element type label.
        per_source: Metrics broken down by dataset source
            (``"gui360"``, ``"screenspot"``, ``"rico"``).
        per_image: Per-image metric bundles for fine-grained analysis.
        config: Snapshot of eval configuration for reproducibility.
    """

    global_metrics: MetricsBundle = field(default_factory=MetricsBundle)
    per_category: Dict[str, MetricsBundle] = field(default_factory=dict)
    per_source: Dict[str, MetricsBundle] = field(default_factory=dict)
    per_image: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Compute evaluation metrics over a dataset with optional model inference.

    The evaluator iterates over a DataLoader, optionally runs a model
    to correct predictions, and aggregates metrics across all samples.

    Args:
        model: Trained model for inference (optional — pass ``None`` to
            evaluate raw VLM predictions directly).
        dataset: The dataset being evaluated (kept for reference, not
            iterated directly – use a DataLoader).
        config: Evaluation configuration dict (ioU threshold, tolerances,
            taxonomy, etc.).
        device: Target device (auto-detected if ``None``).
    """

    def __init__(
        self,
        model: Any = None,
        dataset: Any = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.config = config or {}
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # IoU and alignment settings
        self.iou_threshold = float(self.config.get("iou_threshold", 0.5))
        self.alignment_tolerance = float(self.config.get("alignment_tolerance", 0.02))

        # Taxonomy for per-category breakdown
        self.taxonomy: List[str] = list(
            self.config.get(
                "taxonomy",
                [
                    "button", "text", "image", "input", "icon",
                    "container", "card", "checkbox", "radio", "slider",
                    "switch", "label", "tab", "menu", "divider",
                    "list", "modal", "toast", "banner", "other",
                ],
            )
        )

    # ------------------------------------------------------------------
    # Main evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader: Any,
        model: Any = None,
        source_map: Optional[Dict[str, str]] = None,
    ) -> EvaluationResult:
        """Evaluate over all batches in *dataloader*.

        Args:
            dataloader: Iterable yielding batches.  Each batch is a dict
                with at least ``"vlm_boxes"``, ``"gt_boxes"``,
                ``"element_types"``, ``"valid_mask"``, ``"image_ids"``.
            model: Override the model instance (defaults to ``self.model``).
            source_map: Optional ``{image_id: source_name}`` mapping for
                per-source metric breakdown.

        Returns:
            ``EvaluationResult`` with aggregated metrics.
        """
        model = model if model is not None else self.model

        # Accumulators
        all_pred: List[Tensor] = []
        all_gt: List[Tensor] = []
        all_types: List[Tensor] = []
        all_image_ids: List[tuple] = []

        for batch in dataloader:
            pred_boxes = batch["vlm_boxes"]
            gt_boxes = batch["gt_boxes"]
            valid = batch.get("valid_mask")
            elem_types = batch.get("element_types")
            image_ids = batch.get("image_ids", [])

            # Optionally run model inference (placeholder — model takes
            # HeteroData graphs, not raw box tensors)
            if model is not None:
                pred_boxes = self._run_model(model, batch, pred_boxes)

            # Extract valid boxes per image
            B = pred_boxes.shape[0] if pred_boxes.dim() >= 2 else 1
            for b in range(B):
                v_mask = valid[b] if valid is not None else torch.ones(
                    pred_boxes.shape[1] if pred_boxes.dim() >= 2 else 1,
                    dtype=torch.bool,
                )
                if valid is not None and v_mask.sum() == 0:
                    continue

                p = pred_boxes[b][v_mask] if pred_boxes.dim() >= 2 else pred_boxes
                g = gt_boxes[b][v_mask] if gt_boxes.dim() >= 2 else gt_boxes
                t = (
                    elem_types[b][v_mask]
                    if elem_types is not None and elem_types.dim() >= 1
                    else None
                )
                img_id = image_ids[b] if b < len(image_ids) else f"img_{b}"

                all_pred.append(p)
                all_gt.append(g)
                if t is not None:
                    all_types.append(t)
                all_image_ids.append((img_id, len(all_pred) - 1))

        # Aggregate
        result = EvaluationResult(
            config={
                "iou_threshold": self.iou_threshold,
                "alignment_tolerance": self.alignment_tolerance,
                "num_samples": len(all_pred),
            },
        )

        # Global metrics — pool all boxes
        if all_pred:
            cat_pred = torch.cat(all_pred, dim=0)
            cat_gt = torch.cat(all_gt, dim=0)
            result.global_metrics = compute_all_metrics(
                cat_pred, cat_gt,
                iou_threshold=self.iou_threshold,
                alignment_tolerance=self.alignment_tolerance,
            )
        else:
            result.global_metrics = MetricsBundle()

        # Per-category breakdown
        if all_types:
            result.per_category = self._compute_per_category(
                all_pred, all_gt, all_types,
            )

        # Per-source breakdown
        if source_map is not None:
            result.per_source = self._compute_per_source(
                all_pred, all_gt, all_image_ids, source_map,
            )

        # Per-image results
        result.per_image = self._compute_per_image(
            all_pred, all_gt, all_image_ids,
        )

        return result

    # ------------------------------------------------------------------
    # Static convenience method
    # ------------------------------------------------------------------

    @staticmethod
    def evaluate_model_on_data(model: Any, dataloader: Any) -> Dict[str, float]:
        """Evaluate *model* on *dataloader* and return global metrics dict.

        Args:
            model: Trained model to evaluate.
            dataloader: DataLoader yielding batches.

        Returns:
            Dict of metric name → float value.
        """
        evaluator = Evaluator(model=model)
        result = evaluator.evaluate(dataloader)
        return result.global_metrics.to_dict()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(result: EvaluationResult) -> None:
        """Pretty-print an evaluation result to stdout.

        Args:
            result: ``EvaluationResult`` to display.
        """
        gm = result.global_metrics
        sep = "─" * 44

        print("\n" + sep)
        print("  Evaluation Report")
        print(sep)
        print(f"  Samples evaluated: {result.config.get('num_samples', 0)}")
        print()
        print("  ── Global Metrics ──")
        print(f"    Recall:        {gm.recall:.4f}")
        print(f"    Precision:     {gm.precision:.4f}")
        print(f"    F1 Score:      {gm.f1:.4f}")
        print(f"    Position Err:  {gm.position_error:.4f}")
        print(f"    Size Err:      {gm.size_error:.4f}")
        print(f"    Alignment Err: {gm.alignment_error:.4f}")

        if result.per_category:
            print()
            print("  ── Per Category ──")
            header = f"    {'Category':<14s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'PosErr':>8s}"
            print(header)
            print("    " + "─" * 48)
            for cat_name, mb in sorted(result.per_category.items()):
                print(
                    f"    {cat_name:<14s} "
                    f"{mb.recall:8.4f} {mb.precision:8.4f} "
                    f"{mb.f1:8.4f} {mb.position_error:8.4f}"
                )

        if result.per_source:
            print()
            print("  ── Per Source ──")
            header = f"    {'Source':<14s} {'Recall':>8s} {'Prec':>8s} {'F1':>8s} {'PosErr':>8s}"
            print(header)
            print("    " + "─" * 48)
            for src_name, mb in sorted(result.per_source.items()):
                print(
                    f"    {src_name:<14s} "
                    f"{mb.recall:8.4f} {mb.precision:8.4f} "
                    f"{mb.f1:8.4f} {mb.position_error:8.4f}"
                )

        print(sep + "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_model(
        self, model: Any, batch: Dict[str, Any], pred_boxes: Tensor,
    ) -> Tensor:
        """Run model inference on a batch, returning corrected boxes.

        Placeholder: the model expects ``HeteroData`` graphs, not raw
        box tensors.  Returns *pred_boxes* unchanged unless overridden.
        Subclasses or advanced users may override this to integrate the
        full ``InferencePipeline``.
        """
        return pred_boxes

    def _compute_per_category(
        self,
        all_pred: List[Tensor],
        all_gt: List[Tensor],
        all_types: List[Tensor],
    ) -> Dict[str, MetricsBundle]:
        """Compute metrics broken down by element type.

        Each element type is mapped to a label via *self.taxonomy*.
        """
        # Group indices by element type
        type_indices: Dict[int, List[tuple]] = {}
        for i, types in enumerate(all_types):
            for j in range(types.shape[0]):
                t = int(types[j].item())
                if t < 0:
                    continue  # padding
                type_indices.setdefault(t, []).append((i, j))

        result: Dict[str, MetricsBundle] = {}
        for type_idx, idx_pairs in sorted(type_indices.items()):
            label = (
                self.taxonomy[type_idx]
                if type_idx < len(self.taxonomy)
                else f"type_{type_idx}"
            )

            # Collect boxes for this type
            pred_boxes_list = [all_pred[i][j:j+1] for i, j in idx_pairs]
            gt_boxes_list = [all_gt[i][j:j+1] for i, j in idx_pairs]

            if not pred_boxes_list:
                result[label] = MetricsBundle()
                continue

            cat_pred = torch.cat(pred_boxes_list, dim=0)
            cat_gt = torch.cat(gt_boxes_list, dim=0)
            result[label] = compute_all_metrics(
                cat_pred, cat_gt,
                iou_threshold=self.iou_threshold,
                alignment_tolerance=self.alignment_tolerance,
            )

        return result

    def _compute_per_source(
        self,
        all_pred: List[Tensor],
        all_gt: List[Tensor],
        all_image_ids: List[tuple],
        source_map: Dict[str, str],
    ) -> Dict[str, MetricsBundle]:
        """Compute metrics broken down by dataset source."""
        # Group indices by source
        source_indices: Dict[str, List[int]] = {}
        for img_id, idx in all_image_ids:
            src = source_map.get(img_id, "unknown")
            source_indices.setdefault(src, []).append(idx)

        result: Dict[str, MetricsBundle] = {}
        for src, indices in sorted(source_indices.items()):
            src_pred = torch.cat([all_pred[i] for i in indices], dim=0)
            src_gt = torch.cat([all_gt[i] for i in indices], dim=0)
            result[src] = compute_all_metrics(
                src_pred, src_gt,
                iou_threshold=self.iou_threshold,
                alignment_tolerance=self.alignment_tolerance,
            )

        return result

    def _compute_per_image(
        self,
        all_pred: List[Tensor],
        all_gt: List[Tensor],
        all_image_ids: List[tuple],
    ) -> List[Dict[str, Any]]:
        """Compute per-image metrics."""
        results: List[Dict[str, Any]] = []
        for img_id, idx in all_image_ids:
            mb = compute_all_metrics(
                all_pred[idx], all_gt[idx],
                iou_threshold=self.iou_threshold,
                alignment_tolerance=self.alignment_tolerance,
            )
            results.append({
                "image_id": img_id,
                "metrics": mb.to_dict(),
            })
        return results
