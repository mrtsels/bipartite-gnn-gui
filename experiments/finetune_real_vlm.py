#!/usr/bin/env python3
"""Fine-tune the GNN on REAL RICO VLM data (not synthetic dropping).

Compares completion metrics before/after fine-tuning on the same 41 validation images.

Usage:
    python experiments/finetune_real_vlm.py

Output:
    - checkpoints/violation_detection/real_vlm_finetuned.pt
    - Printed before/after comparison table
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import (
    DEVICE,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
    _normalize_label,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VLM_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/vlm_predictions/rico_qwen_flash")
RICO_DIR = Path("/Users/minimx/bipartite-gnn-gui/data/rico_local/combined")
CHECKPOINT_PATH = Path(
    "/Users/minimx/bipartite-gnn-gui/checkpoints/violation_detection/best_model.pt"
)
OUTPUT_CKPT = Path(
    "checkpoints/violation_detection/real_vlm_finetuned.pt"
)

# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def center_distance(box_a, box_b):
    """Euclidean distance between box centres (normalised coords)."""
    cx_a = (box_a[0] + box_a[2]) / 2.0
    cy_a = (box_a[1] + box_a[3]) / 2.0
    cx_b = (box_b[0] + box_b[2]) / 2.0
    cy_b = (box_b[1] + box_b[3]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def hungarian_match(
    pred_elems: list[ElementNode],
    gt_elems: list[ElementNode],
    threshold: float = 0.1,
):
    """Match predicted elements to GT using centre-distance Hungarian.

    Returns:
        matched: list of (pred_idx, gt_idx) pairs.
        fp_indices: list of unmatched pred indices.
        fn_indices: list of unmatched gt indices.
    """
    M, N = len(pred_elems), len(gt_elems)
    if M == 0 or N == 0:
        return [], list(range(M)), list(range(N))
    INF = 1e9
    cost = torch.full((M, N), INF, dtype=torch.float32)
    for i, pe in enumerate(pred_elems):
        for j, ge in enumerate(gt_elems):
            d = center_distance(pe.bbox, ge.bbox)
            if d <= threshold:
                cost[i, j] = d
    has_finite = torch.isfinite(cost).any()
    if not has_finite:
        return [], list(range(M)), list(range(N))
    row_ind, col_ind = linear_sum_assignment(cost.numpy())
    matched, matched_rows, matched_cols = [], set(), set()
    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < INF / 2:
            matched.append((int(i), int(j)))
            matched_rows.add(int(i))
            matched_cols.add(int(j))
    fp = [i for i in range(M) if i not in matched_rows]
    fn = [j for j in range(N) if j not in matched_cols]
    return matched, fp, fn


def compute_metrics(matched, fp, fn, total_pred, total_gt):
    """Compute Precision, Recall, F1."""
    tp = len(matched)
    precision = tp / max(total_pred, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "tp": tp,
        "fp": len(fp),
        "fn": len(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": total_pred,
        "n_gt": total_gt,
    }


def compute_iou(box1, box2, eps=1e-8):
    """IoU of two xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter + eps
    return inter / union


def nms(bboxes, scores, iou_threshold=0.5):
    """Greedy NMS on a list of xyxy bboxes."""
    if len(bboxes) == 0:
        return []
    indices = list(range(len(bboxes)))
    indices.sort(key=lambda i: scores[i], reverse=True)
    keep = []
    while indices:
        i = indices.pop(0)
        keep.append(i)
        to_remove = [j for j in indices if compute_iou(bboxes[i], bboxes[j]) > iou_threshold]
        for j in to_remove:
            indices.remove(j)
    return keep


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_vlm_elements(vlm_path: Path) -> list[ElementNode] | None:
    """Load VLM predictions from JSON and return normalised ElementNodes."""
    try:
        vlm_data = json.loads(vlm_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    raw_elems = vlm_data.get("elements", [])
    img_w = vlm_data.get("image_width", 1)
    img_h = vlm_data.get("image_height", 1)
    if img_w <= 0 or img_h <= 0:
        return None

    elements: list[ElementNode] = []
    for item in raw_elems:
        bbox = item.get("bbox_xyxy") or item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        x1, x2 = x1 / img_w, x2 / img_w
        y1, y2 = y1 / img_h, y2 / img_h
        if x2 <= x1 or y2 <= y1:
            continue
        label = _normalize_label(item.get("label", "other"))
        elements.append(ElementNode(
            bbox=[x1, y1, x2, y2],
            label=label,
            confidence=1.0,  # placeholder; will be overridden if available
        ))
    return elements


def load_gt_elements(gt_path: Path, min_elems: int = 1) -> list[ElementNode] | None:
    """Load RICO GT and return normalised ElementNodes."""
    if not gt_path.exists():
        return None
    parsed = parse_rico_vh(gt_path)
    if parsed is None:
        return None
    rico_w, rico_h = parsed["width"], parsed["height"]
    gt_raw = extract_elements(parsed["root"])
    gt_elems = [normalize_bbox(e, rico_w, rico_h) for e in gt_raw]
    gt_elems = [e for e in gt_elems if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]]
    if len(gt_elems) < min_elems:
        return None
    return gt_elems


# ---------------------------------------------------------------------------
# Graph building for training
# ---------------------------------------------------------------------------


def build_vlm_training_graph(
    vlm_elems: list[ElementNode],
    matched_pairs: list[tuple[int, int]],
    builder: BipartiteGraphBuilder,
) -> tuple | None:
    """Build constraint graph from VLM elements with training targets.

    Targets:
        - existence: 1 for matched (TP), 0 for unmatched (FP)
        - violation: 0 for all constraints (no synthetic dropping)
        - coord: zeros (no coordinate refinement targets without GT mapping)

    Returns:
        (hetero_data, targets_dict) or None if no constraints.
    """
    if len(vlm_elems) < 2:
        return None

    matched_set = set(i for i, _ in matched_pairs)
    existence_labels = [1.0 if i in matched_set else 0.0 for i in range(len(vlm_elems))]

    constraints = extract_all_constraints(vlm_elems)
    if len(constraints) == 0:
        return None

    data = builder.build(vlm_elems, constraints)
    if data is None:
        return None

    targets = {
        "existence": torch.tensor(existence_labels, dtype=torch.float32).view(-1, 1),
        "coord": torch.zeros((len(vlm_elems), 4), dtype=torch.float32),
        "violation": torch.zeros((len(constraints), 1), dtype=torch.float32),
    }
    return data, targets


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RealVLMDataset(Dataset):
    """Dataset that builds VLM-graphs with real FP/FN labels at init time."""

    def __init__(self, vlm_files: list[Path], builder: BipartiteGraphBuilder):
        self.samples: list[tuple] = []  # (data, targets)
        self.image_ids: list[str] = []
        self.vlm_elements_list: list[list[ElementNode]] = []
        self.matched_counts: list[int] = []

        for vlm_path in vlm_files:
            vlm_elems = load_vlm_elements(vlm_path)
            if vlm_elems is None or len(vlm_elems) < 2:
                continue

            gt_elems = load_gt_elements(RICO_DIR / f"{vlm_path.stem}.json")
            if gt_elems is None or len(gt_elems) < 1:
                continue

            matched, fp_idx, fn_idx = hungarian_match(vlm_elems, gt_elems, threshold=0.1)

            result = build_vlm_training_graph(vlm_elems, matched, builder)
            if result is None:
                continue

            data, targets = result
            self.samples.append((data, targets))
            self.image_ids.append(vlm_path.stem)
            self.vlm_elements_list.append(vlm_elems)
            self.matched_counts.append(len(matched))

        logger.info(
            "Dataset: %d samples (%d total VLM elements, %d matched TP)",
            len(self.samples),
            sum(len(e) for e in self.vlm_elements_list),
            sum(self.matched_counts),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# GNN completion pipeline (for evaluation)
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_gnn_pipeline(
    model: BipartiteGNNCorrector,
    vlm_elems: list[ElementNode],
    builder: BipartiteGraphBuilder,
    violation_threshold: float = 0.5,
) -> list[ElementNode]:
    """Run GNN correction on VLM-detected elements.

    1. Build constraint graph from VLM elements
    2. Model inference → violation scores + proposals
    3. For violated constraints, propose new elements at predicted bbox
    4. NMS-deduplicate proposals
    5. Return all VLM elements + proposals (=corrected set)
    """
    if len(vlm_elems) < 3:
        return list(vlm_elems)

    constraints = extract_all_constraints(vlm_elems)
    if len(constraints) == 0:
        return list(vlm_elems)

    data = builder.build(vlm_elems, constraints)
    if data is None:
        return list(vlm_elems)

    data_gpu = data.to(DEVICE)
    pred = model(data_gpu)

    n_con = len(constraints)

    violation = pred.get("violation", torch.zeros(n_con, 1, device=DEVICE)).cpu()
    proposals_raw = pred.get("proposal")

    proposed_elems: list[ElementNode] = []
    if proposals_raw is not None and violation is not None:
        # Apply sigmoid to get [0,1] scores
        violation_scores = torch.sigmoid(violation.view(-1))

        violated_mask = violation_scores > violation_threshold
        violated_indices = violated_mask.nonzero(as_tuple=False).view(-1).tolist()

        proposal_bboxes: list[list[float]] = []
        proposal_scores: list[float] = []
        for vi in violated_indices:
            bbox = proposals_raw[vi].cpu().tolist()
            x1, y1, x2, y2 = bbox
            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            proposal_bboxes.append([x1, y1, x2, y2])
            proposal_scores.append(float(violation_scores[vi].item()))

        keep_indices = nms(proposal_bboxes, proposal_scores, iou_threshold=0.5)
        for ki in keep_indices:
            bbox = proposal_bboxes[ki]
            proposed_elems.append(ElementNode(
                bbox=bbox,
                label="other",
                confidence=proposal_scores[ki],
            ))

    corrected = list(vlm_elems) + proposed_elems
    return corrected


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def load_model(ckpt_path: str | Path = CHECKPOINT_PATH,
               hidden_dim: int = 128) -> BipartiteGNNCorrector:
    """Load a BipartiteGNNCorrector from checkpoint, handling state dict quirks."""
    state = torch.load(str(ckpt_path), map_location="cpu")
    model = BipartiteGNNCorrector(
        hidden_dim=hidden_dim, dropout=0.1,
    )
    sd = state.get("model_state_dict", state)
    # Handle module. prefix
    sd_clean = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd_clean, strict=False)
    logger.info("Model loaded from %s (strict=False)", ckpt_path)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_completion(
    model: BipartiteGNNCorrector,
    vlm_files: list[Path],
    builder: BipartiteGraphBuilder,
    label: str = "Model",
) -> dict:
    """Run completion pipeline on a set of images and return aggregated metrics.

    Returns dict with precision, recall, f1, tp, fp, fn, n_pred, n_gt,
    and per-image averages.
    """
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    total_n_pred, total_n_gt = 0, 0
    n_images = 0
    skipped = 0
    n_proposals = 0
    per_image_list = []

    for vlm_path in vlm_files:
        vlm_elems = load_vlm_elements(vlm_path)
        if vlm_elems is None or len(vlm_elems) < 1:
            skipped += 1
            continue

        gt_elems = load_gt_elements(RICO_DIR / f"{vlm_path.stem}.json")
        if gt_elems is None or len(gt_elems) < 1:
            skipped += 1
            continue

        # Run GNN pipeline
        corrected = run_gnn_pipeline(model, vlm_elems, builder)
        n_proposals_this = len(corrected) - len(vlm_elems)
        n_proposals += n_proposals_this

        if len(corrected) == 0:
            corrected = list(vlm_elems)

        matched, fp_idx, fn_idx = hungarian_match(corrected, gt_elems, threshold=0.1)
        met = compute_metrics(matched, fp_idx, fn_idx, len(corrected), len(gt_elems))

        total_tp += met["tp"]
        total_fp += met["fp"]
        total_fn += met["fn"]
        total_n_pred += met["n_pred"]
        total_n_gt += met["n_gt"]
        per_image_list.append(met)
        n_images += 1

    if n_images == 0:
        logger.warning("No images evaluated for %s!", label)
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_images": 0}

    # Aggregated (pooled) metrics
    agg_prec = total_tp / max(total_n_pred, 1)
    agg_rec = total_tp / max(total_n_gt, 1)
    agg_f1 = 2 * agg_prec * agg_rec / max(agg_prec + agg_rec, 1e-8)

    # Per-image averages
    avg_prec = sum(m["precision"] for m in per_image_list) / n_images
    avg_rec = sum(m["recall"] for m in per_image_list) / n_images
    avg_f1 = sum(m["f1"] for m in per_image_list) / n_images

    logger.info(
        "%s: %d images, Prec=%.4f Rec=%.4f F1=%.4f (pooled)",
        label, n_images, agg_prec, agg_rec, agg_f1,
    )

    return {
        "precision": agg_prec,
        "recall": agg_rec,
        "f1": agg_f1,
        "precision_avg": avg_prec,
        "recall_avg": avg_rec,
        "f1_avg": avg_f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_pred": total_n_pred,
        "n_gt": total_n_gt,
        "n_images": n_images,
        "skipped": skipped,
        "n_proposals": n_proposals,
    }


# ---------------------------------------------------------------------------
# GNN violation accuracy & proposal MSE (on-graph metrics)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_gnn_on_graph(
    model: BipartiteGNNCorrector,
    dataset: RealVLMDataset,
) -> dict:
    """Evaluate GNN violation/existence accuracy on the graph-level task.

    Returns dict with violation_acc, existence_acc, and proposal_mse.
    """
    model.eval()
    viol_correct, viol_total = 0, 0
    exist_correct, exist_total = 0, 0
    mse_sum, mse_count = 0.0, 0

    for idx in range(len(dataset)):
        data, targets = dataset[idx]
        data_gpu = data.to(DEVICE)

        pred = model(data_gpu)

        # Violation accuracy (all targets are 0, predictions should be near 0)
        if "violation" in pred and targets["violation"].numel() > 0:
            v_pred = torch.sigmoid(pred["violation"].cpu().view(-1))
            v_tgt = targets["violation"].view(-1)
            viol_correct += ((v_pred > 0.5) == (v_tgt > 0.5)).sum().item()
            viol_total += v_tgt.numel()

        # Existence accuracy (matched=1, FP=0)
        if "existence" in pred and targets["existence"].numel() > 0:
            e_pred = torch.sigmoid(pred["existence"].cpu().view(-1))
            e_tgt = targets["existence"].view(-1)
            exist_correct += ((e_pred > 0.5) == (e_tgt > 0.5)).sum().item()
            exist_total += e_tgt.numel()

        # Proposal MSE (on all constraints)
        if "proposal" in pred:
            prop = pred["proposal"].cpu()
            mse = torch.nn.functional.mse_loss(prop, torch.zeros_like(prop))
            mse_sum += mse.item() * prop.numel()
            mse_count += prop.numel()

    results = {
        "violation_acc": viol_correct / max(viol_total, 1),
        "existence_acc": exist_correct / max(exist_total, 1),
        "proposal_mse": mse_sum / max(mse_count, 1),
        "viol_total": viol_total,
        "exist_total": exist_total,
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    logger.info("=" * 75)
    logger.info("FINE-TUNE GNN ON REAL RICO VLM DATA")
    logger.info("=" * 75)

    device = torch.device("cpu")
    builder = BipartiteGraphBuilder()

    # ------------------------------------------------------------------
    # 1. Load all VLM files and split
    # ------------------------------------------------------------------
    vlm_files = sorted(VLM_DIR.glob("*.json"))
    logger.info("VLM predictions: %d files", len(vlm_files))

    # Shuffle deterministically for reproducibility
    rng = torch.Generator()
    rng.manual_seed(42)
    indices = torch.randperm(len(vlm_files), generator=rng).tolist()
    vlm_files_sorted = [vlm_files[i] for i in indices]

    split = int(len(vlm_files_sorted) * 0.8)
    train_files = vlm_files_sorted[:split]
    val_files = vlm_files_sorted[split:]

    logger.info("Train: %d files, Val: %d files", len(train_files), len(val_files))

    # ------------------------------------------------------------------
    # 2. Build datasets
    # ------------------------------------------------------------------
    train_ds = RealVLMDataset(train_files, builder)
    val_ds = RealVLMDataset(val_files, builder)

    if len(train_ds) < 2:
        logger.error("Training dataset too small (%d samples)", len(train_ds))
        return
    if len(val_ds) < 1:
        logger.error("Validation dataset too small (%d samples)", len(val_ds))
        return

    def collate_fn(batch):
        data_list, target_list = zip(*batch)
        return list(data_list), list(target_list)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # ------------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------------
    if not CHECKPOINT_PATH.exists():
        logger.error("Checkpoint not found: %s", CHECKPOINT_PATH)
        return

    model = load_model(CHECKPOINT_PATH, hidden_dim=128).to(device)

    # Set loss weights
    model.loss_fn.coord_weight = 1.0
    model.loss_fn.violation_weight = 1.0
    model.loss_fn.existence_weight = 1.0
    model.loss_fn.alignment_weight = 0.0  # disable alignment loss

    # ------------------------------------------------------------------
    # 4. Before fine-tuning: evaluate on val set
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("BEFORE FINE-TUNING — Evaluating baseline model")
    logger.info("=" * 75)

    before_graph = eval_gnn_on_graph(model, val_ds)
    before_comp = evaluate_completion(model, val_files, builder, label="Baseline (before)")

    # ------------------------------------------------------------------
    # 5. Fine-tune
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 30

    logger.info("\n" + "=" * 75)
    logger.info("FINE-TUNING: %d epochs, lr=1e-4", num_epochs)
    logger.info("%s | %s | %s | %s | %s",
                "Epoch", "TrainLoss", "ValLoss", "ValViolAcc", "Time")
    logger.info("-" * 75)

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for data_list, target_list in train_loader:
            for d, t in zip(data_list, target_list):
                d = d.to(device)
                optimizer.zero_grad()
                loss = model.compute_loss(model(d), t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_viol_accs = []

        with torch.no_grad():
            for data_list, target_list in val_loader:
                for d, t in zip(data_list, target_list):
                    d = d.to(device)
                    pred = model(d)
                    val_loss += model.compute_loss(pred, t).item()

                    if "violation" in pred and t["violation"].numel() > 0:
                        v_pred = torch.sigmoid(pred["violation"].cpu().view(-1))
                        v_tgt = t["violation"].view(-1)
                        acc = ((v_pred > 0.5) == (v_tgt > 0.5)).float().mean().item()
                        val_viol_accs.append(acc)

        train_loss /= max(len(train_ds), 1)
        val_loss /= max(len(val_ds), 1)
        mean_viol_acc = sum(val_viol_accs) / max(len(val_viol_accs), 1)

        logger.info("%6d | %.5f | %.5f | %.4f | %.1fs",
                    epoch, train_loss, val_loss, mean_viol_acc, time.time() - t0)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save checkpoint
            OUTPUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss},
                       str(OUTPUT_CKPT))
            logger.info("  → Saved best checkpoint (epoch %d, val_loss=%.5f)", epoch, val_loss)

    logger.info("\nFine-tuning complete. Best val loss: %.5f (epoch %d)", best_val_loss, best_epoch)

    # ------------------------------------------------------------------
    # 6. After fine-tuning: re-load best model and evaluate
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("AFTER FINE-TUNING — Evaluating fine-tuned model")
    logger.info("=" * 75)

    # Reload best checkpoint
    if OUTPUT_CKPT.exists():
        state = torch.load(str(OUTPUT_CKPT), map_location="cpu")
        sd = state.get("model_state_dict", state)
        model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        logger.info("Reloaded best checkpoint from %s", OUTPUT_CKPT)

    after_graph = eval_gnn_on_graph(model, val_ds)
    after_comp = evaluate_completion(model, val_files, builder, label="Fine-tuned (after)")

    # ------------------------------------------------------------------
    # 7. Report before/after comparison
    # ------------------------------------------------------------------
    print()
    print("=" * 75)
    print("  REAL VLM FINE-TUNE — BEFORE vs AFTER COMPLETION METRICS")
    print("=" * 75)
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")
    print(f"  Val images: {before_comp["n_images"]}")
    print()

    def delta_str(b, a):
        d = a - b
        return f"{d:+7.4f}"

    header = f"  {"":30s} {"Before (baseline)":>18s} {"After (fine-tuned)":>18s} {"Δ":>10s}"
    sep = f"  {"─" * 30:30s} {"─" * 18:>18s} {"─" * 18:>18s} {"─" * 10:>10s}"
    print(header)
    print(sep)
    print(f"  {"GNN Violation Acc":30s} {before_graph["violation_acc"]:>10.4f}   {after_graph["violation_acc"]:>10.4f}   {delta_str(before_graph["violation_acc"], after_graph["violation_acc"]):>10s}")
    print(f"  {"GNN Existence Acc":30s} {before_graph["existence_acc"]:>10.4f}   {after_graph["existence_acc"]:>10.4f}   {delta_str(before_graph["existence_acc"], after_graph["existence_acc"]):>10s}")
    print(f"  {"GNN Proposal MSE":30s} {before_graph["proposal_mse"]:>10.4f}   {after_graph["proposal_mse"]:>10.4f}   {delta_str(before_graph["proposal_mse"], after_graph["proposal_mse"]):>10s}")
    print()
    print(f"  {"Completion F1 (pooled)":30s} {before_comp["f1"]:>10.4f}   {after_comp["f1"]:>10.4f}   {delta_str(before_comp["f1"], after_comp["f1"]):>10s}")
    print(f"  {"Precision (pooled)":30s} {before_comp["precision"]:>10.4f}   {after_comp["precision"]:>10.4f}   {delta_str(before_comp["precision"], after_comp["precision"]):>10s}")
    print(f"  {"Recall (pooled)":30s} {before_comp["recall"]:>10.4f}   {after_comp["recall"]:>10.4f}   {delta_str(before_comp["recall"], after_comp["recall"]):>10s}")
    print()
    print(f"  {"Per-image F1 (avg)":30s} {before_comp["f1_avg"]:>10.4f}   {after_comp["f1_avg"]:>10.4f}   {delta_str(before_comp["f1_avg"], after_comp["f1_avg"]):>10s}")
    print(f"  {"Per-image Prec (avg)":30s} {before_comp["precision_avg"]:>10.4f}   {after_comp["precision_avg"]:>10.4f}   {delta_str(before_comp["precision_avg"], after_comp["precision_avg"]):>10s}")
    print(f"  {"Per-image Rec (avg)":30s} {before_comp["recall_avg"]:>10.4f}   {after_comp["recall_avg"]:>10.4f}   {delta_str(before_comp["recall_avg"], after_comp["recall_avg"]):>10s}")
    print()
    print(f"  {"TP count":30s} {before_comp["tp"]:>10d}   {after_comp["tp"]:>10d}")
    print(f"  {"FP count":30s} {before_comp["fp"]:>10d}   {after_comp["fp"]:>10d}")
    print(f"  {"FN count":30s} {before_comp["fn"]:>10d}   {after_comp["fn"]:>10d}")
    print(f"  {"Proposals (val total)":30s} {"—":>10s}   {after_comp["n_proposals"]:>10d}")
    print()
    print(f"  {"Total val GT elements":30s} {before_comp["n_gt"]:>10d}   {after_comp["n_gt"]:>10d}")
    print(f"  {"Total val pred elements":30s} {before_comp["n_pred"]:>10d}   {after_comp["n_pred"]:>10d}")
    print("=" * 75)

    d_f1 = after_comp["f1"] - before_comp["f1"]
    d_prec = after_comp["precision"] - before_comp["precision"]
    d_rec = after_comp["recall"] - before_comp["recall"]
    print(f"\n  Summary: F1 {before_comp['f1']:.4f} → {after_comp['f1']:.4f} ({d_f1:+.4f})")
    print(f"           Precision {before_comp["precision"]:.4f} → {after_comp["precision"]:.4f} ({d_prec:+.4f})")
    print(f"           Recall    {before_comp["recall"]:.4f} → {after_comp["recall"]:.4f} ({d_rec:+.4f})")
    print(f"           Violation Acc {before_graph["violation_acc"]:.4f} → {after_graph["violation_acc"]:.4f}")
    print("=" * 75)

    # Save results
    results = {
        "checkpoint": str(CHECKPOINT_PATH),
        "output_checkpoint": str(OUTPUT_CKPT),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "val_images": before_comp["n_images"],
        "before": {
            "graph": {k: v for k, v in before_graph.items() if isinstance(v, (int, float))},
            "completion": {k: v for k, v in before_comp.items() if isinstance(v, (int, float))},
        },
        "after": {
            "graph": {k: v for k, v in after_graph.items() if isinstance(v, (int, float))},
            "completion": {k: v for k, v in after_comp.items() if isinstance(v, (int, float))},
        },
        "delta": {
            "violation_acc": after_graph["violation_acc"] - before_graph["violation_acc"],
            "existence_acc": after_graph["existence_acc"] - before_graph["existence_acc"],
            "proposal_mse": after_graph["proposal_mse"] - before_graph["proposal_mse"],
            "f1": d_f1,
            "precision": d_prec,
            "recall": d_rec,
        },
    }
    out_path = Path("experiments/vlm_completion/finetune_real_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
