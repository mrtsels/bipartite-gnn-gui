"""Qualitative analysis visualizations for GUI structure correction.

Provides plotting utilities for:
- Side-by-side correction comparison (GT vs VLM vs corrected).
- Error heatmaps showing spatial distribution of errors.
- Grid of correction comparisons for paper figures.

All plots use a consistent colour scheme:
- **Green** — ground truth
- **Red**   — VLM prediction
- **Blue**  — corrected (GNN / baseline)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")  # Non-interactive backend (headless)

from bipartite_gnn_gui.data.ground_truth import GroundTruth
from bipartite_gnn_gui.utils.bbox import compute_iou

try:
    from PIL import Image as PILImage

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    PILImage = None  # type: ignore[assignment]
    _HAS_PIL = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COLOUR_GT = "#2ca02c"  # green
_COLOUR_VLM = "#d62728"  # red
_COLOUR_CORRECTED = "#1f77b4"  # blue


def _denormalise_bbox(
    bbox: Tuple[float, float, float, float]
    | List[float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    """Convert normalized ``(x1, y1, x2, y2)`` to pixel coordinates."""
    x1, y1, x2, y2 = map(float, bbox)
    return (x1 * width, y1 * height, x2 * width, y2 * height)


def _load_image(image_path: str, width: int, height: int) -> Any:
    """Load an image, returning an ndarray, or fall back to white canvas."""
    if _HAS_PIL and image_path:
        try:
            img = PILImage.open(image_path).convert("RGB")
            img = img.resize((width, height))
            return np.asarray(img)
        except Exception as exc:
            logger.warning("Could not load image %s: %s", image_path, exc)
    # Fallback: white canvas
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def _draw_bbox(
    ax: Any,
    bbox: Tuple[float, float, float, float] | List[float],
    colour: str,
    label: str = "",
    linewidth: float = 1.5,
) -> None:
    """Draw a single bounding box on an axis.

    Args:
        ax: Matplotlib axis.
        bbox: ``(x1, y1, x2, y2)`` in pixel coordinates.
        colour: Line colour.
        label: Optional legend label.
        linewidth: Line width in points.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    rect = FancyBboxPatch(
        (x1, y1),
        w,
        h,
        linewidth=linewidth,
        edgecolor=colour,
        facecolor="none",
        label=label,
        boxstyle="square,pad=0",
    )
    ax.add_patch(rect)


def _compute_iou_between_dicts(
    pred_elements: List[Dict[str, Any]],
    gt_elements: List[Dict[str, Any]],
) -> float:
    """Compute mean IoU between matched prediction and ground-truth elements.

    Uses Hungarian matching.  Returns 0.0 if no valid matches exist.
    """
    if not pred_elements or not gt_elements:
        return 0.0

    try:
        import torch
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return 0.0

    # Build tensors
    pred_boxes = torch.tensor(
        [e["bbox"] for e in pred_elements], dtype=torch.float32
    )
    gt_boxes = torch.tensor(
        [e["bbox"] for e in gt_elements], dtype=torch.float32
    )

    iou_matrix = compute_iou(pred_boxes, gt_boxes)  # (M, N)

    # Hungarian matching with 0.5 IoU threshold
    cost = 1.0 - iou_matrix
    cost[iou_matrix < 0.5] = float("inf")

    if not torch.isfinite(cost).any():
        return 0.0

    row_idx, col_idx = linear_sum_assignment(cost.numpy())
    matched_ious = [iou_matrix[i, j].item() for i, j in zip(row_idx, col_idx)
                    if iou_matrix[i, j] >= 0.5]

    if not matched_ious:
        return 0.0
    return float(np.mean(matched_ious))


# ===================================================================
# Public API
# ===================================================================


def plot_correction_comparison(
    gt: GroundTruth,
    vlm_pred: Dict[str, Any],
    model_pred: Dict[str, Any],
    save_path: str,
) -> None:
    """Plot a side-by-side comparison of GT, VLM, and corrected bboxes.

    Draws all three sets of bounding boxes overlaid on the screenshot
    image (or a white canvas if the image is unavailable).

    Args:
        gt: Ground-truth annotations (``GroundTruth`` object).
        vlm_pred: Raw VLM prediction dict with an ``"elements"`` list.
        model_pred: Corrected output dict (from model or baseline) with
            an ``"elements"`` list.
        save_path: Path to save the figure (e.g. ``"comparison.png"``).
    """
    img_w = gt.image_width or 800
    img_h = gt.image_height or 600
    image = _load_image(gt.image_path, img_w, img_h)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image)
    ax.set_title("Correction Comparison", fontsize=14)

    # Draw GT bboxes (green)
    for elem in gt.elements:
        bbox_px = _denormalise_bbox(elem.bbox, img_w, img_h)
        _draw_bbox(ax, bbox_px, _COLOUR_GT, label="Ground Truth" if gt.elements[0] is elem else "")

    # Draw VLM bboxes (red)
    vlm_elems = vlm_pred.get("elements", [])
    for elem in vlm_elems:
        bbox = elem.get("bbox", [0, 0, 0, 0])
        bbox_px = _denormalise_bbox(bbox, img_w, img_h)
        _draw_bbox(ax, bbox_px, _COLOUR_VLM, label="VLM Prediction" if vlm_elems[0] is elem else "")

    # Draw corrected bboxes (blue)
    model_elems = model_pred.get("elements", [])
    for elem in model_elems:
        bbox = elem.get("bbox", [0, 0, 0, 0])
        bbox_px = _denormalise_bbox(bbox, img_w, img_h)
        _draw_bbox(ax, bbox_px, _COLOUR_CORRECTED, label="Corrected" if model_elems[0] is elem else "")

    # Compute and display mean IoU
    gt_dicts = [
        {"bbox": e.bbox} for e in gt.elements
    ]
    iou_vlm = _compute_iou_between_dicts(vlm_elems, gt_dicts)
    iou_model = _compute_iou_between_dicts(model_elems, gt_dicts)

    info_text = (
        f"Mean IoU (VLM vs GT): {iou_vlm:.3f}\n"
        f"Mean IoU (Corrected vs GT): {iou_model:.3f}"
    )
    ax.text(
        0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    if gt.elements or vlm_elems or model_elems:
        ax.legend(loc="lower right", fontsize=8)
    ax.axis("off")

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_heatmap(
    errors: List[Tuple[float, float, float]],
    save_path: str,
    grid_size: int = 10,
) -> None:
    """Plot a heatmap of position errors across the screen.

    Bins element centres into a ``grid_size × grid_size`` grid and
    shows the mean position error per cell.

    Args:
        errors: List of ``(x_center, y_center, error)`` tuples, where
            coordinates are in normalized ``[0, 1]`` and the error is a
            positive scalar (e.g. Euclidean distance).
        save_path: Path to save the figure (e.g. ``"heatmap.png"``).
        grid_size: Number of bins per axis (default 10).
    """
    if not errors:
        # Empty — save a blank figure
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.text(0.5, 0.5, "No error data", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Bin centres and errors
    centres = np.array([(e[0], e[1]) for e in errors])
    error_vals = np.array([e[2] for e in errors])

    # Accumulate per bin
    grid_sum = np.zeros((grid_size, grid_size))
    grid_count = np.zeros((grid_size, grid_size))

    for (cx, cy), err in zip(centres, error_vals):
        ix = int(np.clip(cx * grid_size, 0, grid_size - 1))
        iy = int(np.clip(cy * grid_size, 0, grid_size - 1))
        # y axis is flipped in image coords
        iy = grid_size - 1 - iy
        grid_sum[iy, ix] += err
        grid_count[iy, ix] += 1

    grid_mean = np.divide(
        grid_sum, grid_count,
        out=np.zeros_like(grid_sum),
        where=grid_count > 0,
    )

    # Mask empty bins
    grid_mean = np.ma.masked_where(grid_count == 0, grid_mean)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        grid_mean, cmap="YlOrRd", interpolation="nearest",
        extent=[0, 1, 0, 1], aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax, label="Mean Position Error")

    ax.set_xlabel("Normalized x")
    ax.set_ylabel("Normalized y")
    ax.set_title("Position Error Heatmap")

    # Overlay count of elements per bin
    for i in range(grid_size):
        for j in range(grid_size):
            if grid_count[i, j] > 0:
                x_centre = (j + 0.5) / grid_size
                y_centre = (grid_size - 1 - i + 0.5) / grid_size
                ax.text(
                    x_centre, y_centre,
                    f"{grid_count[i, j]:.0f}",
                    ha="center", va="center",
                    fontsize=6, color="black",
                )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_correction_grid(
    gts: Sequence[GroundTruth],
    vlm_preds: Sequence[Dict[str, Any]],
    model_preds: Sequence[Dict[str, Any]],
    save_path: str,
    n_examples: int = 16,
) -> None:
    """Plot a grid of correction comparisons for multiple examples.

    Each row has 4 columns: **screenshot thumbnail**, **GT bboxes**,
    **VLM bboxes**, and **corrected bboxes**.  Useful for paper figures.

    Args:
        gts: Sequence of ``GroundTruth`` objects (one per example).
        vlm_preds: Sequence of VLM prediction dicts.
        model_preds: Sequence of corrected output dicts.
        save_path: Path to save the figure (e.g. ``"correction_grid.png"``).
        n_examples: Number of examples to show (default 16).  Clamped to
            the minimum length of the input sequences.
    """
    n = min(n_examples, len(gts), len(vlm_preds), len(model_preds))
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No examples to display", ha="center", va="center", fontsize=14)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # Layout: n rows × 4 columns
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(cols * 3, n * 2.5), squeeze=False)

    for row in range(n):
        gt = gts[row]
        vlm_pred = vlm_preds[row]
        model_pred = model_preds[row]

        img_w = gt.image_width or 800
        img_h = gt.image_height or 600
        image = _load_image(gt.image_path, img_w, img_h)

        # Column 0: Screenshot thumbnail
        ax_img = axes[row, 0]
        ax_img.imshow(image)
        ax_img.set_title(f"Screenshot {row + 1}", fontsize=9)
        ax_img.axis("off")

        # Column 1: GT bboxes on screenshot
        ax_gt = axes[row, 1]
        ax_gt.imshow(image)
        ax_gt.set_title("Ground Truth", fontsize=9)
        for elem in gt.elements:
            bbox_px = _denormalise_bbox(elem.bbox, img_w, img_h)
            _draw_bbox(ax_gt, bbox_px, _COLOUR_GT, linewidth=1.2)
        ax_gt.axis("off")

        # Column 2: VLM bboxes on screenshot
        ax_vlm = axes[row, 2]
        ax_vlm.imshow(image)
        ax_vlm.set_title("VLM Prediction", fontsize=9)
        for elem in vlm_pred.get("elements", []):
            bbox = elem.get("bbox", [0, 0, 0, 0])
            bbox_px = _denormalise_bbox(bbox, img_w, img_h)
            _draw_bbox(ax_vlm, bbox_px, _COLOUR_VLM, linewidth=1.2)
        ax_vlm.axis("off")

        # Column 3: Corrected bboxes on screenshot
        ax_corr = axes[row, 3]
        ax_corr.imshow(image)
        ax_corr.set_title("Corrected", fontsize=9)
        for elem in model_pred.get("elements", []):
            bbox = elem.get("bbox", [0, 0, 0, 0])
            bbox_px = _denormalise_bbox(bbox, img_w, img_h)
            _draw_bbox(ax_corr, bbox_px, _COLOUR_CORRECTED, linewidth=1.2)
        ax_corr.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
