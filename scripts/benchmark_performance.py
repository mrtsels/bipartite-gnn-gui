#!/usr/bin/env python3
"""Phase 6 — Performance benchmarks for the bipartite GNN project.

Measures 4 key metrics:
  6.1 Data loading throughput     — ms/img, images/s  (200 RICO JSONs)
  6.2 Graph building scaling      — ms vs element count (10, 50, 100, 500)
  6.3 Training throughput         — steps/s  (50 graphs, 3 epochs, hidden=64)
  6.4 Inference latency           — p50/p95/p99 ms  (100 graphs, eval forward)

Usage:
    python scripts/benchmark_performance.py

Output:
    Prints a formatted report to stdout.
"""

from __future__ import annotations

import sys
import time
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project imports (following the pattern from scripts/train_violation.py)
# ---------------------------------------------------------------------------
from bipartite_gnn_gui.graph.builder import BipartiteGraphBuilder
from bipartite_gnn_gui.graph.constraints import extract_all_constraints
from bipartite_gnn_gui.graph.schema import ElementNode
from bipartite_gnn_gui.model.model import BipartiteGNNCorrector
from scripts.run_experiment import (
    DEVICE,
    GraphListDataset,
    extract_elements,
    normalize_bbox,
    parse_rico_vh,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_RICO_DIR = Path(_PROJECT_ROOT / "data" / "rico_local" / "combined")
SEED = 42
torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"RICO data: {_RICO_DIR}")
print()


# ===================================================================
# 6.1 — Data loading throughput
# ===================================================================
def benchmark_data_loading(n_samples: int = 200) -> Dict[str, float]:
    """Measure time to load + parse + build graphs from *n_samples* RICO JSONs.

    Pipeline: parse_rico_vh → extract_elements → normalize_bbox →
    filter degenerates → extract_all_constraints → BipartiteGraphBuilder.build()

    Returns dict with avg_ms, images_per_sec, n_success, n_skipped.
    """
    print("─" * 62)
    print("  6.1 — Data loading throughput")
    print("─" * 62)

    all_jsons = sorted(_RICO_DIR.glob("*.json"))
    if len(all_jsons) > n_samples:
        all_jsons = all_jsons[:n_samples]

    print(f"  JSON files: {len(all_jsons)}")
    n_success = 0
    n_skipped = 0
    times: List[float] = []

    builder = BipartiteGraphBuilder()

    t_start = time.perf_counter()
    for path in all_jsons:
        t0 = time.perf_counter()

        parsed = parse_rico_vh(path)
        if parsed is None:
            n_skipped += 1
            continue

        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        if len(gt_elements) < 2:
            n_skipped += 1
            continue

        constraints = extract_all_constraints(gt_elements)
        if len(constraints) == 0:
            n_skipped += 1
            continue

        try:
            _ = builder.build(gt_elements, constraints)
        except Exception:
            n_skipped += 1
            continue

        dt = (time.perf_counter() - t0) * 1000  # ms
        times.append(dt)
        n_success += 1

    t_total = time.perf_counter() - t_start

    avg_ms = statistics.mean(times) if times else 0.0
    images_per_sec = n_success / t_total if t_total > 0 else 0.0

    print(f"  Successfully built:  {n_success} graphs")
    print(f"  Skipped:             {n_skipped}")
    print(f"  Total time:          {t_total:.2f}s")
    print(f"  Average per image:   {avg_ms:.1f} ms")
    print(f"  Throughput:          {images_per_sec:.1f} images/s")
    print()

    return {
        "avg_ms": avg_ms,
        "images_per_sec": images_per_sec,
        "n_success": n_success,
        "n_skipped": n_skipped,
    }


# ===================================================================
# 6.2 — Graph-building scaling
# ===================================================================
def benchmark_graph_scaling() -> Dict[int, float]:
    """Measure BipartiteGraphBuilder.build() time for synthetic element sets.

    Creates synthetic ElementNode lists of sizes [10, 50, 100, 500] and
    times just the builder.build() call.

    Returns dict: element_count → avg_ms.
    """
    print("─" * 62)
    print("  6.2 — Graph-building scaling")
    print("─" * 62)

    sizes = [10, 50, 100, 500]
    results: Dict[int, float] = {}
    builder = BipartiteGraphBuilder()

    for n_elem in sizes:
        # Create synthetic elements with random bboxes
        elements = [
            ElementNode(
                bbox=[
                    float(torch.rand(1).item() * 0.8),
                    float(torch.rand(1).item() * 0.8),
                    float(torch.rand(1).item() * 0.1 + 0.8),
                    float(torch.rand(1).item() * 0.1 + 0.8),
                ],
                label="button",
                confidence=1.0,
            )
            for _ in range(n_elem)
        ]

        constraints = extract_all_constraints(elements)
        if len(constraints) == 0:
            # Force at least one constraint — group all elements
            # into an alignment constraint so builder has work to do
            from bipartite_gnn_gui.graph.schema import ConstraintNode, ConstraintType

            constraints = [
                ConstraintNode(
                    constraint_type=ConstraintType.ALIGN_LEFT,
                    source_indices=list(range(n_elem)),
                    target_indices=list(range(n_elem)),
                )
            ]

        # Warm-up
        _ = builder.build(elements, constraints)

        # Timed runs
        n_runs = max(1, 200 // max(n_elem, 1))
        times: List[float] = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = builder.build(elements, constraints)
            dt = (time.perf_counter() - t0) * 1000  # ms
            times.append(dt)

        avg_ms = statistics.mean(times)
        results[n_elem] = avg_ms
        print(f"  {n_elem:4d} elements  →  {avg_ms:.3f} ms  (over {n_runs} runs)")

    # Print scaling ratios
    if 10 in results:
        base = results[10]
        for n, t in sorted(results.items()):
            ratio = t / base if base > 0 else 0.0
            print(f"    └─ scaling factor vs n=10: {ratio:.2f}x")
    print()

    return results


# ===================================================================
# 6.3 — Training throughput
# ===================================================================
def benchmark_training_throughput(n_graphs: int = 50, n_epochs: int = 3) -> Dict[str, float]:
    """Measure training steps/s.

    Loads *n_graphs* RICO graphs, creates a BipartiteGNNCorrector(hidden_dim=64),
    and trains for *n_epochs* epochs.

    Returns dict with steps_per_second, total_steps, total_time.
    """
    print("─" * 62)
    print("  6.3 — Training throughput")
    print("─" * 62)
    print(f"  Graphs: {n_graphs}  |  Epochs: {n_epochs}  |  hidden_dim: 64")

    # Build graphs
    all_jsons = sorted(_RICO_DIR.glob("*.json"))[:n_graphs]
    builder = BipartiteGraphBuilder()
    graphs: List[Tuple[Any, Dict[str, Tensor]]] = []
    n_skipped = 0

    for path in all_jsons:
        parsed = parse_rico_vh(path)
        if parsed is None:
            n_skipped += 1
            continue

        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        if len(gt_elements) < 2:
            n_skipped += 1
            continue

        constraints = extract_all_constraints(gt_elements)
        if len(constraints) == 0:
            n_skipped += 1
            continue

        try:
            hetero_data = builder.build(gt_elements, constraints)
        except Exception:
            n_skipped += 1
            continue

        N = len(gt_elements)
        N_con = len(constraints)
        gt_boxes = torch.tensor(
            [e.bbox for e in gt_elements], dtype=torch.float32
        )
        targets = {
            "coord": torch.zeros(N, 4, dtype=torch.float32),
            "gt_boxes": gt_boxes,
            "existence": torch.ones(N, 1, dtype=torch.float32),
            "violation": torch.zeros(N_con, 1, dtype=torch.float32),
        }
        graphs.append((hetero_data, targets))

    if len(graphs) == 0:
        print("  ERROR: No valid graphs built!")
        return {"steps_per_second": 0.0, "total_steps": 0, "total_time": 0.0}

    print(f"  Built {len(graphs)} graphs ({n_skipped} skipped)")

    # Create dataset, model, optimizer
    dataset = GraphListDataset(graphs)
    loader = DataLoader(dataset, batch_size=None, shuffle=True)

    model = BipartiteGNNCorrector(hidden_dim=64, dropout=0.1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Training loop
    model.train()
    total_steps = 0
    t_start = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        for data, targets_batch in loader:
            data = data.to(DEVICE)
            targets_batch = {k: v.to(DEVICE) for k, v in targets_batch.items()}

            optimizer.zero_grad()
            predictions = model(data)
            loss = model.compute_loss(predictions, targets_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_steps += 1

    t_total = time.perf_counter() - t_start
    steps_per_sec = total_steps / t_total if t_total > 0 else 0.0

    print(f"  Total steps:       {total_steps}")
    print(f"  Total time:        {t_total:.2f}s")
    print(f"  Steps/second:      {steps_per_sec:.2f}")
    print()

    return {
        "steps_per_second": steps_per_sec,
        "total_steps": total_steps,
        "total_time": t_total,
    }


# ===================================================================
# 6.4 — Inference latency
# ===================================================================
def benchmark_inference_latency(n_graphs: int = 100) -> Dict[str, float]:
    """Measure inference latency percentiles.

    Loads *n_graphs* RICO graphs and runs model.eval() forward passes.

    Returns dict with p50_ms, p95_ms, p99_ms.
    """
    print("─" * 62)
    print("  6.4 — Inference latency")
    print("─" * 62)
    print(f"  Graphs: {n_graphs}")

    # Build graphs
    all_jsons = sorted(_RICO_DIR.glob("*.json"))[:n_graphs]
    builder = BipartiteGraphBuilder()
    graph_data_list: List[Any] = []
    n_skipped = 0

    for path in all_jsons:
        parsed = parse_rico_vh(path)
        if parsed is None:
            n_skipped += 1
            continue

        img_w, img_h = parsed["width"], parsed["height"]
        gt_raw = extract_elements(parsed["root"])
        gt_elements = [normalize_bbox(e, img_w, img_h) for e in gt_raw]
        gt_elements = [
            e for e in gt_elements
            if e.bbox[2] > e.bbox[0] and e.bbox[3] > e.bbox[1]
        ]

        if len(gt_elements) < 2:
            n_skipped += 1
            continue

        constraints = extract_all_constraints(gt_elements)
        if len(constraints) == 0:
            n_skipped += 1
            continue

        try:
            hetero_data = builder.build(gt_elements, constraints)
            graph_data_list.append(hetero_data)
        except Exception:
            n_skipped += 1
            continue

    if len(graph_data_list) == 0:
        print("  ERROR: No valid graphs built!")
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    print(f"  Built {len(graph_data_list)} graphs ({n_skipped} skipped)")

    # Model
    model = BipartiteGNNCorrector(hidden_dim=64, dropout=0.1).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            _ = model(graph_data_list[0].to(DEVICE))

    # Timed forward passes
    latencies: List[float] = []
    with torch.no_grad():
        for data in graph_data_list:
            data = data.to(DEVICE)
            t0 = time.perf_counter()
            _ = model(data)
            dt = (time.perf_counter() - t0) * 1000  # ms
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            latencies.append(dt)

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p50 = sorted_lat[int(n * 0.50)]
    p95 = sorted_lat[int(n * 0.95)]
    p99 = sorted_lat[int(n * 0.99)]

    print(f"  p50 latency:  {p50:.2f} ms")
    print(f"  p95 latency:  {p95:.2f} ms")
    print(f"  p99 latency:  {p99:.2f} ms")
    print(f"  Mean latency: {statistics.mean(latencies):.2f} ms")
    print()

    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99}


# ===================================================================
# Main entry point
# ===================================================================
def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       PHASE 6 — PERFORMANCE BENCHMARKS                 ║")
    print("║       Bipartite GNN for GUI Structure Correction       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    t_all = time.perf_counter()

    # 6.1
    r1 = benchmark_data_loading(n_samples=200)

    # 6.2
    r2 = benchmark_graph_scaling()

    # 6.3
    r3 = benchmark_training_throughput(n_graphs=50, n_epochs=3)

    # 6.4
    r4 = benchmark_inference_latency(n_graphs=100)

    total_time = time.perf_counter() - t_all

    # ── Final report ──────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  PHASE 6 BENCHMARK REPORT — SUMMARY")
    print("=" * 62)
    print()
    print(f"  Total benchmark time:          {total_time:.1f}s")
    print()

    print("  6.1 Data Loading Throughput")
    print(f"    Average per image:            {r1['avg_ms']:.1f} ms")
    print(f"    Throughput:                   {r1['images_per_sec']:.1f} images/s")
    print(f"    Successful builds:            {r1['n_success']}")
    print(f"    Skipped:                      {r1['n_skipped']}")
    print()

    print("  6.2 Graph-Building Scaling")
    for n_elem in sorted(r2.keys()):
        print(f"    {n_elem:4d} elements:            {r2[n_elem]:.3f} ms")
    if 10 in r2:
        base = r2[10]
        for n, t in sorted(r2.items()):
            ratio = t / base if base > 0 else 0.0
            print(f"      └─ vs n=10:               {ratio:.2f}x")
    print()

    print("  6.3 Training Throughput")
    print(f"    Steps/second:                 {r3['steps_per_second']:.2f}")
    print(f"    Total steps (3 epochs):       {r3['total_steps']}")
    print(f"    Total training time:          {r3['total_time']:.2f}s")
    print()

    print("  6.4 Inference Latency")
    print(f"    p50:                          {r4['p50_ms']:.2f} ms")
    print(f"    p95:                          {r4['p95_ms']:.2f} ms")
    print(f"    p99:                          {r4['p99_ms']:.2f} ms")
    print()

    print("─" * 62)
    print("  BENCHMARK COMPLETE")
    print("─" * 62)


if __name__ == "__main__":
    main()
