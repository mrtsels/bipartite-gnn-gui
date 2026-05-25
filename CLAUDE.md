# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project: **Heterogeneous Bipartite GNN for GUI Structure Error Correction**. A post-correction framework that refines noisy GUI element predictions from lightweight VLMs (Qwen3.5-2B, MiniMax-VL-01) using a heterogeneous bipartite GraphSAGE model.

## Current State

The repository is an **initial scaffold**: all module APIs are designed with docstrings and import definitions in `__init__.py` files, but no actual implementations exist. The task list in `TASK.md` defines four phases (data pipeline → graph construction → GNN model → evaluation) with ~25 tasks to implement.

## Build & Test Commands

```bash
# Install in dev mode
pip install -e .

# Install with dev/test dependencies
pip install -e ".[dev,test]"

# Run all tests
pytest tests/ -v

# Run a specific test module
pytest tests/test_graph_builder.py -v

# Run tests with coverage
pytest tests/ --cov=bipartite_gnn_gui -v
```

## Architecture

```
VLM JSON output → Bipartite Graph (Element × Constraint nodes) → GraphSAGE → Δ𝐱 predictions → Corrected JSON
```

### Package Layout

- `src/bipartite_gnn_gui/data/` — VLM output parsing, ground-truth loading (GUI-360°, ScreenSpot), coordinate normalization, feature extraction, PyTorch Dataset/DataLoader.
- `src/bipartite_gnn_gui/graph/` — Graph schema (element/constraint node types), constraint extraction (alignment, containment, spacing, grid), `HeteroData` builder, visualization, augmentation.
- `src/bipartite_gnn_gui/model/` — Heterogeneous GraphSAGE encoder, three prediction heads (coordinate refinement, violation, existence), combined loss function, training loop, inference pipeline.
- `src/bipartite_gnn_gui/eval/` — Metrics (PositionError, SizeError, AlignmentError, ElementRecall, Precision, IoU), Evaluator class, baselines (no-correction, rule-based, MLP-only), qualitative visualization.
- `src/bipartite_gnn_gui/utils/` — YAML config system, structured logging, helpers (seeding, bbox transforms, IoU).

### Key Technical Details

- **Dependencies**: PyTorch ≥2.1.0, PyTorch Geometric ≥2.4.0, transformers, numpy, pillow.
- **Python**: ≥3.10 required.
- **Build system**: setuptools via `pyproject.toml`, package found under `src/`.
- **Testing**: pytest configured with `testpaths = ["tests"]`, file pattern `test_*.py`.
- **Datasets**: GUI-360° (~50K elements, ~3.5K screenshots) and ScreenSpot (~30K elements, ~5K screenshots) — data expected under `data/raw/`, `data/processed/`.
- **Graph**: Heterogeneous bipartite graph `G = (V_e ∪ V_c, E)` using PyG's `HeteroData`. Two-layer message passing: element → constraint → element.

### Implementation Order (from TASK.md)

The project follows four sequential phases. Each phase should be completed before moving to the next:
1. **Phase 1** — Data pipeline: `vlm_output.py`, `ground_truth.py`, `preprocess.py`, `dataset.py`, `config.py`, `logging.py`
2. **Phase 2** — Graph construction: `schema.py`, `constraints.py`, `builder.py`, `visualize.py`, `augment.py`
3. **Phase 3** — GNN model: `encoder.py`, `heads.py`, `model.py`, `losses.py`, `trainer.py`, `inference.py`
4. **Phase 4** — Evaluation: `metrics.py`, `evaluator.py`, `baselines.py`, `qualitative.py`, experiments
