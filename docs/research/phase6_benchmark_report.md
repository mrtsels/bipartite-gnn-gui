# Phase 6: Performance Benchmark Report

> **Date:** 2026-06-23
> **Script:** `scripts/benchmark_performance.py`
> **Results:** `experiments/benchmarks/performance_results.json`

## 6.1 Data Loading Throughput

**Method:** Load 200 RICO JSONs, parse view hierarchy, extract elements,
build constraint graph, construct HeteroData. Time per image.

| Metric | Value |
|---|---|
| Images processed | 198/200 |
| Avg time per image | **2.1 ms** |
| Throughput | **467 images/s** |

Bottleneck is disk I/O (SSD). Constraint extraction + graph construction
are negligible at typical element counts (~22 elements/image).

## 6.2 Graph Building Scaling

**Method:** Synthetic layouts with 10/50/100/500 grid elements.
Measure `BipartiteGraphBuilder.build()` wall time.

| Elements | Constraints | Avg Time |
|---|---|---|
| 10 | ~25 | **0.201 ms** |
| 50 | ~180 | **2.365 ms** |
| 100 | ~450 | **6.832 ms** |
| 500 | ~2,800 | **255.23 ms** |

Scales O(N²) due to pairwise containment and alignment extraction.
At realistic sizes (~22 elements), graph building is ~5ms — negligible
compared to VLM inference (~2s).

## 6.3 Training Throughput

**Method:** 50 RICO graphs → `BipartiteGNNCorrector(hidden_dim=64)` → 3 epochs → measure forward+backward+optimize step time.

| Metric | Value |
|---|---|
| Graphs | 50 |
| Steps | 150 |
| Total time | 0.42s |
| **Throughput** | **357 steps/s** |

No GPU used — CPU-only on M3 MacBook Pro. Training 2,000 samples × 50 epochs
completes in under 5 minutes. Model is lightweight (57K parameters).

## 6.4 Inference Latency

**Method:** 100 RICO graphs → model.eval() → `torch.no_grad()` forward only.
Measure wall clock per inference.

| Metric | Latency |
|---|---|
| **p50** | **0.53 ms** |
| p95 | 0.96 ms |
| p99 | 1.11 ms |
| avg | 0.59 ms |
| min | 0.31 ms |
| max | 1.38 ms |

## Summary

The GNN pipeline is **not the performance bottleneck** in any scenario:

| Component | Latency | Bottleneck? |
|---|---|---|
| VLM inference | ~2s (Qwen3-Flash) | **Yes — VLM** |
| Graph building | ~5ms (typical), 255ms (500 elem) | No |
| GNN inference | **0.5ms** | No |
| GNN training | 357 steps/s | No |

**"Humble brag" takeaway:** GNN inference under 1ms — you could process
2,000 images per second if the VLM could keep up.
