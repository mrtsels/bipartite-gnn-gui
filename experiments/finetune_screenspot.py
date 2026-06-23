#!/usr/bin/env python3
"""Fine-tune GNN on ScreenSpot VLM predictions.

Step 1: Load ScreenSpot VLM predictions as pseudo-GT element sets
Step 2: Use train_violation.py's build_violation_graph() for proper self-supervised training
Step 3: Evaluate zero-shot on ScreenSpot before/after fine-tuning
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector

BUILDER = BipartiteGraphBuilder()
from scripts.train_violation import build_violation_graph

logger = logging.getLogger(__name__)


def load_vlm_elements(vlm_dir: str, max_n: int = 500) -> List[List[ElementNode]]:
    """Load ScreenSpot VLM predictions as pseudo-GT element sets."""
    vlm_path = Path(vlm_dir)
    files = sorted(vlm_path.glob("*.json"))[:max_n]
    all_elems: List[List[ElementNode]] = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        w, h = data.get("image_width", 1440), data.get("image_height", 2560)
        elems = []
        for el in data.get("elements", []):
            bbox = el.get("bbox_xyxy", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1) / w
            y1 = max(0, y1) / h
            x2 = min(w, x2) / w
            y2 = min(h, y2) / h
            if x2 <= x1 or y2 <= y1:
                continue
            label = el.get("label", "text")
            elems.append(ElementNode(bbox=[x1, y1, x2, y2], confidence=1.0, label=label))
        if len(elems) >= 3:
            all_elems.append(elems)
    logger.info(f"Loaded {len(all_elems)} ScreenSpot layouts ({sum(len(e) for e in all_elems)} elements)")
    return all_elems


class VLMViolationDataset(Dataset):
    """Build violation graphs on the fly from VLM-predicted layouts."""

    def __init__(self, elements_list: List[List[ElementNode]], drop_ratio: float = 0.4):
        self.elements_list = elements_list
        self.drop_ratio = drop_ratio

    def __len__(self) -> int:
        return len(self.elements_list)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        result = build_violation_graph(
            self.elements_list[idx], builder=BUILDER, drop_ratio=self.drop_ratio, seed=idx
        )
        if result is None:
            # Fallback: retry with different seed
            result = build_violation_graph(
                self.elements_list[idx], builder=BUILDER, drop_ratio=self.drop_ratio, seed=idx + 1000
            )
            if result is None:
                raise ValueError(f"No valid graph for item {idx}")
        return result[0], result[1]


def load_model(ckpt_path: str = "checkpoints/violation_detection/best_model.pt",
               hidden_dim: int = 128) -> BipartiteGNNCorrector:
    state = torch.load(ckpt_path, map_location="cpu")
    model = BipartiteGNNCorrector(
        hidden_dim=hidden_dim, dropout=0.1,
        coord_weight=0.0, existence_weight=0.0,
    )
    sd = state.get("model_state_dict", state)
    model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
    return model


def evaluate(model: BipartiteGNNCorrector,
             elems_list: List[List[ElementNode]],
             drop_ratio: float = 0.4,
             max_n: int = 100) -> float:
    """Evaluate violation detection accuracy on a held-out set."""
    model.eval()
    accs = []
    for idx in range(min(max_n, len(elems_list))):
        result = build_violation_graph(
            elems_list[idx], builder=BUILDER, drop_ratio=drop_ratio, seed=999 + idx
        )
        if result is None:
            continue
        data, targets = result
        with torch.no_grad():
            preds = model(data)
        if "violation" in preds and targets["violation"].numel() > 0:
            acc = ((preds["violation"].view(-1) > 0.5) ==
                   (targets["violation"].view(-1) > 0.5)).float().mean().item()
            accs.append(acc)
    return sum(accs) / len(accs) if accs else 0.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    device = torch.device("cpu")

    # Load ScreenSpot VLM layouts
    vlm_dir = "data/vlm_predictions/screenspot_qwen_flash"
    elements_list = load_vlm_elements(vlm_dir, max_n=500)
    if len(elements_list) < 10:
        logger.error(f"Only {len(elements_list)} layouts")
        return

    # Split
    split = int(len(elements_list) * 0.8)
    train_elems = elements_list[:split]
    val_elems = elements_list[split:]

    train_ds = VLMViolationDataset(train_elems, drop_ratio=0.4)
    val_ds = VLMViolationDataset(val_elems, drop_ratio=0.4)

    def collate(batch):
        d, t = zip(*batch)
        return list(d), list(t)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate)

    # Pre-finetuning eval
    model = load_model().to(device)
    logger.info("=== Pre Fine-Tuning ===")
    pre_acc = evaluate(model, val_elems, drop_ratio=0.4, max_n=50)
    logger.info(f"Val acc on ScreenSpot: {pre_acc:.4f}")

    # Fine-tune
    optim = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    logger.info(f"\n=== Fine-Tuning: {len(train_ds)} train, {len(val_ds)} val ===")
    logger.info(f"{'Epoch':>6} | {'TrainLoss':>9} | {'ValLoss':>7} | {'ValAcc':>6} |  Time")

    best_val = float("inf")
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for data_list, target_list in train_loader:
            for d, t in zip(data_list, target_list):
                d = d.to(device)
                optim.zero_grad()
                loss = model.compute_loss(model(d), t)
                loss.backward()
                optim.step()
                train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_accs = []
        with torch.no_grad():
            for data_list, target_list in val_loader:
                for d, t in zip(data_list, target_list):
                    d = d.to(device)
                    preds = model(d)
                    val_loss += model.compute_loss(preds, t).item()
                    if "violation" in preds and len(preds["violation"]) > 0:
                        acc = ((preds["violation"].view(-1) > 0.5) ==
                               (t["violation"].view(-1) > 0.5)).float().mean().item()
                        val_accs.append(acc)

        val_loss /= len(val_ds)
        train_loss /= len(train_ds)
        mean_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0
        logger.info(f"{epoch:>6} | {train_loss:.5f} | {val_loss:.5f} | {mean_acc:.4f} | {time.time()-t0:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "val_loss": val_loss},
                       "checkpoints/violation_detection/screenspot_finetuned.pt")

    logger.info(f"\nFine-tuning complete. Best val loss: {best_val:.5f}")

    # Post-finetuning eval on ScreenSpot
    logger.info("\n=== Post Fine-Tuning ===")
    post_acc = evaluate(model, val_elems, drop_ratio=0.4, max_n=50)
    logger.info(f"Pre:  {pre_acc:.4f}")
    logger.info(f"Post: {post_acc:.4f}")
    logger.info(f"Gain: {post_acc - pre_acc:+.4f}")


if __name__ == "__main__":
    main()
