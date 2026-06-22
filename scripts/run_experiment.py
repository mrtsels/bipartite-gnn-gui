#!/usr/bin/env python3
"""Standardized training pipeline for bipartite-gnn-gui on RICO data.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --n 100 --epochs 10 --hidden 64
    python scripts/run_experiment.py --config configs/experiment.yaml --lr 0.0001

Parses RICO View Hierarchy JSONs, builds bipartite graphs with simulated
VLM noise, trains a BipartiteGNNCorrector via the existing Trainer class,
evaluates with compute_all_metrics and NoOp baseline comparison, and
saves checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bipartite_gnn_gui.eval.metrics import MetricsBundle, compute_all_metrics
from bipartite_gnn_gui.eval.evaluator import Evaluator
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from bipartite_gnn_gui.model.trainer import Trainer
from bipartite_gnn_gui.utils.config import TrainingConfig

logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Flat experiment configuration loaded from YAML + CLI overrides."""

    # Data
    n_samples: int = 500
    noise_scale: float = 0.12
    val_split: float = 0.2
    seed: int = 42
    rico_dir: str = "data/rico_local/combined"
    cache_dir: str = "data/rico_cache"
    vlm_dir: str = ""  # dir with real VLM predictions; empty = simulated noise

    # Model
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    # Training
    epochs: int = 50
    lr: float = 0.001
    batch_size: int = 8
    weight_decay: float = 1e-5
    warmup_steps: int = 50
    grad_clip: float = 1.0
    amp: bool = False
    early_stopping_patience: int = 10

    # Evaluation
    iou_threshold: float = 0.5
    alignment_tolerance: float = 0.02

    # Logging
    checkpoint_dir: str = "checkpoints/experiment"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str) -> ExperimentConfig:
        """Load config from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls._from_dict(raw or {})

    @classmethod
    def _from_dict(cls, d: dict) -> ExperimentConfig:
        """Nested-dict to flat config, with defaults for missing keys."""
        cfg = cls()
        sections = {
            "data": ["n_samples", "noise_scale", "val_split", "seed", "rico_dir", "cache_dir"],
            "model": ["hidden_dim", "num_layers", "dropout"],
            "training": ["epochs", "lr", "batch_size", "weight_decay", "warmup_steps",
                         "grad_clip", "amp", "early_stopping_patience"],
            "evaluation": ["iou_threshold", "alignment_tolerance"],
            "logging": ["checkpoint_dir", "log_level"],
        }
        for section, keys in sections.items():
            sec = d.get(section, {})
            if isinstance(sec, dict):
                for k in keys:
                    if k in sec:
                        setattr(cfg, k, sec[k])
        return cfg

    def update_from_args(self, args: argparse.Namespace) -> None:
        """Overwrite fields from parsed CLI arguments."""
        overrides = {
            "n_samples": args.n,
            "epochs": args.epochs,
            "hidden_dim": args.hidden,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "amp": args.amp,
            "val_split": args.val_split,
            "noise_scale": args.noise_scale,
            "vlm_dir": args.vlm_dir,
            "checkpoint_dir": args.checkpoint_dir,
        }
        for key, value in overrides.items():
            if value is not None:
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# RICO parsing helpers
# ---------------------------------------------------------------------------


def parse_rico_vh(path: str | Path) -> dict | None:
    """Load and validate a RICO View Hierarchy JSON.

    Returns a dict with keys ``root``, ``width``, ``height``,
    or ``None`` if the file is invalid.
    """
    try:
        with open(path) as f:
            raw = json.load(f)
        activity = raw.get("activity", {})
        root = activity.get("root")
        if not root:
            root = raw.get("root")
        if not root:
            return None
        bounds = root.get("bounds", [0, 0, 0, 0])
        if len(bounds) != 4 or bounds[2] <= 0 or bounds[3] <= 0:
            return None
        return {"root": root, "width": bounds[2], "height": bounds[3]}
    except Exception:
        return None


def rico_class_to_label(cls: str) -> str:
    """Map Android class name to canonical type."""
    short = cls.rsplit(".", 1)[-1]
    mapping = {
        "Button": "button",
        "ImageButton": "icon",
        "ImageView": "image",
        "TextView": "text",
        "EditText": "input",
        "CheckBox": "checkbox",
        "Switch": "switch",
        "Spinner": "icon",
        "ProgressBar": "icon",
        "WebView": "container",
        "ListView": "list",
        "ScrollView": "container",
        "TabWidget": "tab",
        "RadioButton": "radio",
        "SeekBar": "slider",
    }
    for suffix, label in mapping.items():
        if short.endswith(suffix):
            return label
    return "other"


def load_vlm_predictions(path: str | Path) -> list[ElementNode]:
    """Load real VLM predictions from a JSON file and return ElementNodes.

    The JSON must follow the Qwen3-VL output format produced by
    scripts/generate_vlm_predictions.py:
      {"elements": [{"bbox_xyxy": [x1,y1,x2,y2], "label": "button", ...}, ...]}
    or the legacy Qwen3.5-2B format:
      [{"bbox_xyxy": [x1,y1,x2,y2], "label": "button", ...}, ...]

    Args:
        path: Path to prediction JSON.

    Returns:
        List of ElementNode objects. Empty list if file is missing or malformed.
    """
    import json

    try:
        with open(path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, OSError):
        return []

    # Handle both formats: {"elements": [...]} and plain [...]
    if isinstance(raw, dict):
        elements_raw = raw.get("elements", [])
    elif isinstance(raw, list):
        elements_raw = raw
    else:
        return []

    nodes: list[ElementNode] = []
    for item in elements_raw:
        if not isinstance(item, dict):
            continue
        # Support bbox_xyxy (Qwen3-VL) and bbox (Qwen3.5-2B)
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = map(float, bbox)
        except (ValueError, TypeError):
            continue
        label = _normalize_label(item.get("label", "other"))
        nodes.append(ElementNode(
            bbox=[x1, y1, x2, y2],
            label=label,
            confidence=float(item.get("confidence", 1.0)),
        ))
    return nodes


_LABEL_ALIASES = {
    "btn": "button", "img": "image", "glyph": "icon",
    "textbox": "input", "search": "input", "textarea": "input", "textfield": "input",
    "div": "container", "section": "container", "frame": "container", "panel": "container",
    "check": "checkbox", "radiobutton": "radio", "range": "slider",
    "toggle": "switch", "dropdown": "menu", "nav": "menu",
    "separator": "divider", "hr": "divider",
    "dialog": "modal", "overlay": "modal",
    "snackbar": "toast", "notification": "toast",
    "announcement": "banner", "alertbar": "banner",
}


def _normalize_label(label: str) -> str:
    """Map VLM label to canonical type."""
    key = label.strip().lower()
    return _LABEL_ALIASES.get(key, key)


def extract_elements(root: dict) -> list[ElementNode]:
    """Extract visible leaf element nodes from a RICO View Hierarchy tree.

    Args:
        root: Root node of the view hierarchy.

    Returns:
        List of ``ElementNode`` objects for visible leaf elements.
    """
    elements: list[ElementNode] = []

    def walk(node: dict, depth: int = 0):
        if depth > 50:
            return
        children = node.get("children")
        is_leaf = not isinstance(children, list) or len(children) == 0

        if is_leaf:
            vis = node.get("visibility", "visible")
            if vis != "visible":
                return
            v2u = node.get("visible-to-user", True)
            if v2u is False:
                return
            bounds = node.get("bounds", [0, 0, 0, 0])
            if len(bounds) != 4:
                return
            x1, y1, x2, y2 = bounds
            if x2 <= x1 or y2 <= y1:
                return
            cls = node.get("class", "")
            if not cls:
                return
            label = rico_class_to_label(cls)
            text = node.get("text") or ""
            if not text:
                cd = node.get("content-desc", [None])
                if isinstance(cd, list) and cd[0] is not None:
                    text = str(cd[0])
            elements.append(
                ElementNode(
                    bbox=[x1, y1, x2, y2],
                    label=label,
                    confidence=1.0,
                    element_id=f"elem_{len(elements)}",
                    features={"text_len": len(str(text))},
                )
            )
        else:
            for child in children:
                if isinstance(child, dict):
                    walk(child, depth + 1)

    walk(root)
    return elements


def normalize_bbox(elem: ElementNode, img_w: int, img_h: int) -> ElementNode:
    """Convert pixel bbox to normalized [0, 1]."""
    x1, y1, x2, y2 = elem.bbox
    return ElementNode(
        bbox=[x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h],
        label=elem.label,
        confidence=elem.confidence,
        element_id=elem.element_id,
        features=elem.features,
    )


def make_noisy_vlm(
    gt_elements: list[ElementNode], noise_scale: float = 0.12
) -> list[ElementNode]:
    """Create simulated VLM predictions by adding Gaussian noise to GT bboxes.

    Args:
        gt_elements: Ground-truth element nodes with normalized bboxes.
        noise_scale: Standard deviation of the Gaussian noise (applied
            proportionally to element width/height).

    Returns:
        Noisy element nodes simulating VLM predictions.
    """
    vlm_nodes: list[ElementNode] = []
    for i, elem in enumerate(gt_elements):
        noise = torch.randn(4) * noise_scale
        x1, y1, x2, y2 = elem.bbox
        w, h = x2 - x1, y2 - y1
        nx1 = max(0.0, min(1.0, x1 + noise[0].item() * w))
        ny1 = max(0.0, min(1.0, y1 + noise[1].item() * h))
        nx2 = max(0.0, min(1.0, x2 + noise[2].item() * w))
        ny2 = max(0.0, min(1.0, y2 + noise[3].item() * h))
        vlm_nodes.append(
            ElementNode(
                bbox=[nx1, ny1, nx2, ny2],
                label=elem.label,
                confidence=max(0.5, 1.0 - abs(noise.mean().item())),
                element_id=f"vlm_{i}",
            )
        )
    return vlm_nodes


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


def build_graph(
    gt_elements: list[ElementNode],
    noise_scale: float,
    builder: BipartiteGraphBuilder,
    vlm_elements: list[ElementNode] | None = None,
) -> Tuple[Any, Dict[str, Tensor]] | None:
    """Build a (HeteroData, targets) pair from ground-truth elements.

    Simulates noisy VLM predictions, extracts heuristic constraints,
    constructs the bipartite graph, and assembles the target dict.

    Args:
        gt_elements: Normalised ground-truth element nodes.
        noise_scale: Noise scale for VLM simulation.
        builder: Graph builder instance.

    Returns:
        ``(hetero_data, targets)`` tuple, or ``None`` if the sample
        is degenerate (< 2 elements or 0 constraints).
    """
    if len(gt_elements) < 2:
        return None

    # Use real VLM predictions when provided, otherwise simulate noise
    if vlm_elements is None:
        vlm_nodes = make_noisy_vlm(gt_elements, noise_scale=noise_scale)
    else:
        vlm_nodes = vlm_elements

    # Extract constraints from GT structure
    constraints = extract_all_constraints(gt_elements)
    if len(constraints) == 0:
        return None

    # Build HeteroData graph
    hetero_data = builder.build(vlm_elements, constraints)

    # Build targets
    N = len(gt_elements)
    N_con = len(constraints)
    gt_boxes = torch.tensor(
        [e.bbox for e in gt_elements], dtype=torch.float32
    )
    vlm_boxes = torch.tensor(
        [e.bbox for e in vlm_nodes], dtype=torch.float32
    )

    # Convert xyxy → cxcywh for delta computation
    def _cxcywh(b):
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)

    gt_xywh = _cxcywh(gt_boxes)
    vlm_xywh = _cxcywh(vlm_boxes)
    delta = gt_xywh - vlm_xywh

    targets: Dict[str, Tensor] = {
        "coord": delta,           # (N, 4) model predicts Δcx, Δcy, Δw, Δh
        "gt_boxes": gt_boxes,     # (N, 4) raw GT xyxy for evaluation
        "existence": torch.ones(N, 1, dtype=torch.float32),
        "violation": torch.zeros(N_con, 1, dtype=torch.float32),
    }

    return hetero_data, targets


# ---------------------------------------------------------------------------
# Simple graph dataset
# ---------------------------------------------------------------------------


class GraphListDataset(Dataset):
    """Thin wrapper around a list of ``(HeteroData, targets)`` pairs.

    Each item is returned as-is (no collation needed when using
    ``batch_size=None`` in the DataLoader).
    """

    def __init__(self, items: List[Tuple[Any, Dict[str, Tensor]]]) -> None:
        self.items = items

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Tensor]]:
        return self.items[idx]

    def __len__(self) -> int:
        return len(self.items)

    @classmethod
    def from_graphs(
        cls, graphs: List[Tuple[Any, Any]]
    ) -> GraphListDataset:
        """Build from a list of ``(vlm_data, gt_data)`` tuples (legacy format).

        Converts legacy-style GT data to the target dict format expected
        by ``BipartiteGNNCorrector.compute_loss``.

        Args:
            graphs: List of ``(vlm_hetero_data, gt_hetero_data)`` tuples.

        Returns:
            New ``GraphListDataset`` with properly structured targets.
        """
        items: List[Tuple[Any, Dict[str, Tensor]]] = []
        for vlm_data, gt_data in graphs:
            N = gt_data["element"].x.shape[0]
            N_con = gt_data["constraint"].x.shape[0]
            targets = {
                "coord": gt_data["element"].x[:, :4],
                "existence": torch.ones(N, 1, dtype=torch.float32),
                "violation": torch.zeros(N_con, 1, dtype=torch.float32),
            }
            items.append((vlm_data, targets))
        return cls(items)


# ---------------------------------------------------------------------------
# Per-epoch reporting
# ---------------------------------------------------------------------------


def print_epoch_report(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float | None,
    lr: float,
    best_val: float,
    val_metrics: MetricsBundle | None = None,
    noop_metrics: MetricsBundle | None = None,
) -> None:
    """Print a clean per-epoch report to stdout."""
    sep = "─" * 62
    print()
    print(sep)
    print(f"  Epoch {epoch:3d}/{total_epochs}  |  LR: {lr:.2e}")
    print(f"  Train Loss: {train_loss:.6f}", end="")
    if val_loss is not None:
        print(f"  |  Val Loss: {val_loss:.6f}  |  Best: {best_val:.6f}")
    else:
        print()
    if val_metrics is not None:
        print(
            f"  Recall: {val_metrics.recall:.4f}  "
            f"Precision: {val_metrics.precision:.4f}  "
            f"F1: {val_metrics.f1:.4f}"
        )
        print(
            f"  Position Error: {val_metrics.position_error:.4f}  "
            f"Size Error: {val_metrics.size_error:.4f}  "
            f"Alignment Error: {val_metrics.alignment_error:.4f}"
        )
    if noop_metrics is not None:
        print(
            f"  NoOp baseline — Recall: {noop_metrics.recall:.4f}  "
            f"PosErr: {noop_metrics.position_error:.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _apply_deltas_to_vlm_boxes(
    vlm_boxes_xyxy: Tensor, deltas_xywh: Tensor
) -> Tensor:
    """Apply predicted deltas (in xywh) to VLM boxes (in xyxy) and return corrected xyxy.

    The model predicts deltas in ``(Δcx, Δcy, Δw, Δh)`` format.
    VLM boxes are in ``(x1, y1, x2, y2)`` pixel/normalized format.
    We convert VLM boxes to xywh, apply the delta, clamp, and convert back.

    Args:
        vlm_boxes_xyxy: ``(N, 4)`` VLM input boxes in xyxy format.
        deltas_xywh: ``(N, 4)`` predicted deltas in xywh format.

    Returns:
        ``(N, 4)`` corrected boxes in xyxy format, clamped to [0, 1].
    """
    N = min(vlm_boxes_xyxy.size(0), deltas_xywh.size(0))
    vlm = vlm_boxes_xyxy[:N]
    delta = deltas_xywh[:N]

    # Convert xyxy → xywh
    x1, y1, x2, y2 = vlm[:, 0], vlm[:, 1], vlm[:, 2], vlm[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    xywh = torch.stack([cx, cy, w, h], dim=-1)

    # Apply delta
    corrected_xywh = xywh + delta

    # Clamp width/height to be non-negative
    corrected_xywh[:, 2] = corrected_xywh[:, 2].clamp(min=1e-6)
    corrected_xywh[:, 3] = corrected_xywh[:, 3].clamp(min=1e-6)

    # Convert xywh → xyxy
    ccx, ccy, cw, ch = (
        corrected_xywh[:, 0],
        corrected_xywh[:, 1],
        corrected_xywh[:, 2],
        corrected_xywh[:, 3],
    )
    corrected_xyxy = torch.stack(
        [ccx - cw / 2.0, ccy - ch / 2.0, ccx + cw / 2.0, ccy + ch / 2.0], dim=-1
    )

    # Clamp to [0, 1]
    corrected_xyxy = corrected_xyxy.clamp(0.0, 1.0)
    return corrected_xyxy


def evaluate_model(
    model: BipartiteGNNCorrector,
    val_dataset: GraphListDataset,
    device: torch.device,
    iou_threshold: float = 0.5,
    alignment_tolerance: float = 0.02,
) -> Tuple[MetricsBundle, MetricsBundle]:
    """Evaluate model and NoOp baseline on the validation set.

    The model's coordinate head predicts deltas in xywh format relative
    to the VLM input boxes.  This function applies those deltas to obtain
    corrected boxes in xyxy, then compares them to ground-truth boxes.

    Args:
        model: Trained model.
        val_dataset: Validation dataset.
        device: Target device.
        iou_threshold: IoU threshold for recall/precision.
        alignment_tolerance: Tolerance for alignment detection.

    Returns:
        ``(model_metrics, noop_metrics)`` tuple.
    """
    model.eval()
    all_corrected_boxes: List[Tensor] = []
    all_gt_boxes: List[Tensor] = []
    all_vlm_boxes: List[Tensor] = []

    with torch.no_grad():
        for hetero_data, targets in val_dataset:
            hetero_data = hetero_data.to(device)
            outputs = model(hetero_data)
            if "coord" in outputs:
                # Get original VLM boxes from the graph's element node features
                vlm_boxes_xyxy = hetero_data["element"].x[:, :4].cpu()
                gt_boxes = targets.get("gt_boxes", targets["coord"])
                deltas = outputs["coord"].cpu()

                # Apply deltas to get corrected boxes
                corrected_boxes = _apply_deltas_to_vlm_boxes(vlm_boxes_xyxy, deltas)
                all_corrected_boxes.append(corrected_boxes)
                all_gt_boxes.append(gt_boxes)
                all_vlm_boxes.append(vlm_boxes_xyxy)

    if not all_corrected_boxes:
        return MetricsBundle(), MetricsBundle()

    # Model metrics: corrected boxes vs GT boxes
    pred_coords = torch.cat(all_corrected_boxes, dim=0)
    gt_coords = torch.cat(all_gt_boxes, dim=0)

    model_metrics = compute_all_metrics(
        pred_coords, gt_coords,
        iou_threshold=iou_threshold,
        alignment_tolerance=alignment_tolerance,
    )

    # NoOp baseline: VLM input boxes (uncorrected) vs GT boxes
    if all_vlm_boxes:
        noop_pred = torch.cat(all_vlm_boxes, dim=0)
        noop_metrics = compute_all_metrics(
            noop_pred, gt_coords,
            iou_threshold=iou_threshold,
            alignment_tolerance=alignment_tolerance,
        )
    else:
        noop_metrics = MetricsBundle()

    return model_metrics, noop_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def create_parser() -> argparse.ArgumentParser:
    """Build argument parser with experiment options."""
    parser = argparse.ArgumentParser(
        description="Run a standardized bipartite-gnn-gui training experiment."
    )
    parser.add_argument("--n", type=int, default=None,
                        help="Number of RICO JSONs to use")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs")
    parser.add_argument("--hidden", type=int, default=None,
                        help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (graphs per step)")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="Enable automatic mixed precision")
    parser.add_argument("--val-split", type=float, default=None,
                        help="Validation fraction")
    parser.add_argument("--noise-scale", type=float, default=None,
                        help="Noise scale for VLM simulation")
    parser.add_argument("--vlm-dir", type=str, default=None,
                        help="Directory with real VLM predictions (replaces simulated noise)")
    parser.add_argument("--config", type=str, default="",
                        help="Path to YAML config file")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Checkpoint output directory")
    return parser


def setup_logging(level: str) -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run a full experiment with the given configuration and return results.

    Args:
        cfg: Fully populated ExperimentConfig instance.

    Returns:
        Dict with keys: best_val_loss, final_train_loss, recall, precision,
        f1, position_error, size_error, noop_recall, noop_precision,
        noop_f1, noop_position_error.
    """
    # Set seeds
    torch.manual_seed(cfg.seed)

    logger.info("=" * 55)
    logger.info("BIPARTITE GNN — RICO Training Experiment")
    logger.info("=" * 55)
    logger.info("Device: %s", DEVICE)
    logger.info("Config: n=%d, epochs=%d, hidden=%d, lr=%.1e, noise=%.3f",
                cfg.n_samples, cfg.epochs, cfg.hidden_dim, cfg.lr, cfg.noise_scale)
    logger.info("Checkpoints: %s", cfg.checkpoint_dir)

    # ── 1. Discover RICO JSONs ──
    rico_dir = Path(cfg.rico_dir)
    if not rico_dir.is_dir():
        logger.error("RICO data directory not found: %s", rico_dir)
        return {"error": f"RICO data directory not found: {rico_dir}"}

    all_jsons = sorted(rico_dir.glob("*.json"))
    n_use = cfg.n_samples if cfg.n_samples > 0 else len(all_jsons)
    n_use = min(n_use, len(all_jsons))
    jsons = all_jsons[:n_use]
    logger.info("Found %d RICO JSONs, using %d", len(all_jsons), n_use)

    # ── 2. Parse and build graphs ──
    builder = BipartiteGraphBuilder()
    all_graphs: List[Tuple[Any, Dict[str, Tensor]]] = []

    t0 = time.time()
    n_skipped = 0
    for idx, path in enumerate(jsons):
        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            logger.info("  Parsed %d/%d (%d skipped, %.1f files/s)",
                        idx, n_use, n_skipped, rate)

        parsed = parse_rico_vh(path)
        if parsed is None:
            n_skipped += 1
            continue

        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])

        # Normalize and filter
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        # Load real VLM predictions if available
        vlm_nodes: list[ElementNode] | None = None
        if cfg.vlm_dir:
            vlm_path = Path(cfg.vlm_dir) / f"{path.stem}.json"
            if vlm_path.exists():
                vlm_nodes = load_vlm_predictions(vlm_path)

        result = build_graph(gt_elements, cfg.noise_scale, builder,
                             vlm_elements=vlm_nodes)
        if result is None:
            n_skipped += 1
            continue

        all_graphs.append(result)

    dt = time.time() - t0
    logger.info("Parsed %d valid graphs (%d skipped) in %.1fs (%.2f files/s)",
                len(all_graphs), n_skipped, dt,
                len(all_graphs) / max(dt, 0.001))

    if len(all_graphs) < 2:
        logger.error("Need at least 2 valid graphs, got %d", len(all_graphs))
        return {"error": f"Need at least 2 valid graphs, got {len(all_graphs)}"}

    # Show stats
    n_elems = [g[0]["element"].x.shape[0] for g in all_graphs]
    n_cons = [g[0]["constraint"].x.shape[0] for g in all_graphs]
    n_edges = [g[0]["element", "to", "constraint"].edge_index.shape[1]
               for g in all_graphs]
    import statistics
    logger.info(
        "Graph stats: %.1f±%.1f elem, %.1f±%.1f con, %.1f±%.1f edges",
        sum(n_elems) / len(n_elems),
        statistics.stdev(n_elems) if len(n_elems) > 1 else 0.0,
        sum(n_cons) / len(n_cons),
        statistics.stdev(n_cons) if len(n_cons) > 1 else 0.0,
        sum(n_edges) / len(n_edges),
        statistics.stdev(n_edges) if len(n_edges) > 1 else 0.0,
    )

    # ── 3. Train/val split ──
    split_idx = int(len(all_graphs) * (1.0 - cfg.val_split))
    train_items = all_graphs[:split_idx]
    val_items = all_graphs[split_idx:]

    train_dataset = GraphListDataset(train_items)
    val_dataset = GraphListDataset(val_items)

    # DataLoader with batch_size=None yields individual (data, targets) tuples
    train_loader = DataLoader(
        train_dataset, batch_size=None, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=None, shuffle=False
    )

    logger.info("Split: %d train / %d val", len(train_dataset), len(val_dataset))

    # ── 4. Build model ──
    model = BipartiteGNNCorrector(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d params, hidden=%d, layers=%d",
                n_params, cfg.hidden_dim, cfg.num_layers)

    # ── 5. Configure and run Trainer ──
    train_cfg = TrainingConfig(
        lr=cfg.lr,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        grad_clip=cfg.grad_clip,
        amp=cfg.amp,
        seed=cfg.seed,
    )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    trainer = Trainer(
        model=model,
        config=train_cfg,
        device=DEVICE,
        checkpoint_dir=checkpoint_dir,
        early_stopping_patience=cfg.early_stopping_patience,
        min_delta=1e-4,
    )

    logger.info("=" * 55)
    logger.info("Starting training (%d epochs)...", cfg.epochs)
    logger.info("=" * 55)

    best_val_loss = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
    )

    final_train_loss = getattr(trainer, 'last_train_loss', float('nan'))
    logger.info("Training complete. Best val loss: %.6f", best_val_loss)

    # ── 6. Final evaluation ──
    logger.info("=" * 55)
    logger.info("Final Evaluation")
    logger.info("=" * 55)

    model_metrics, noop_metrics = evaluate_model(
        model, val_dataset, DEVICE,
        iou_threshold=cfg.iou_threshold,
        alignment_tolerance=cfg.alignment_tolerance,
    )

    print()
    print("=" * 55)
    print("  GNN Correction Results (validation set)")
    print("=" * 55)
    print(f"  Recall:          {model_metrics.recall:.4f}")
    print(f"  Precision:       {model_metrics.precision:.4f}")
    print(f"  F1 Score:        {model_metrics.f1:.4f}")
    print(f"  Position Error:  {model_metrics.position_error:.4f}")
    print(f"  Size Error:      {model_metrics.size_error:.4f}")
    print(f"  Alignment Error: {model_metrics.alignment_error:.4f}")
    print()
    print("  NoOp Baseline (uncorrected VLM inputs)")
    print("=" * 55)
    print(f"  Recall:          {noop_metrics.recall:.4f}")
    print(f"  Precision:       {noop_metrics.precision:.4f}")
    print(f"  F1 Score:        {noop_metrics.f1:.4f}")
    print(f"  Position Error:  {noop_metrics.position_error:.4f}")
    print(f"  Size Error:      {noop_metrics.size_error:.4f}")
    print(f"  Alignment Error: {noop_metrics.alignment_error:.4f}")

    # Improvement over NoOp
    if noop_metrics.position_error > 0:
        pos_improvement = (
            (noop_metrics.position_error - model_metrics.position_error)
            / noop_metrics.position_error * 100.0
        )
        print()
        print(f"  Position error improvement: {pos_improvement:+.1f}%")
    print("=" * 55)

    # ── 7. Save final checkpoint ──
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = {
        "epoch": cfg.epochs,
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "val_metrics": model_metrics.to_dict(),
        "noop_metrics": noop_metrics.to_dict(),
        "best_val_loss": best_val_loss,
    }
    final_path = checkpoint_dir / "final_model.pt"
    torch.save(final_checkpoint, final_path)
    logger.info("Final checkpoint saved: %s", final_path)

    print()
    print("=" * 55)
    logger.info("EXPERIMENT COMPLETE ✅")
    logger.info("%d graphs / %d epochs / %d params",
                len(all_graphs), cfg.epochs, n_params)
    print("=" * 55)

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": final_train_loss,
        "recall": model_metrics.recall,
        "precision": model_metrics.precision,
        "f1": model_metrics.f1,
        "position_error": model_metrics.position_error,
        "size_error": model_metrics.size_error,
        "noop_recall": noop_metrics.recall,
        "noop_precision": noop_metrics.precision,
        "noop_f1": noop_metrics.f1,
        "noop_position_error": noop_metrics.position_error,
    }


def main() -> bool:
    parser = create_parser()
    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = ExperimentConfig.from_yaml(args.config)
        logger.info("Loaded config from %s", args.config)
    else:
        cfg = ExperimentConfig()
    cfg.update_from_args(args)

    # Apply log level
    setup_logging(cfg.log_level)

    results = run_experiment(cfg)
    if "error" in results:
        logger.error(results["error"])
        return False
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
