# AGENTS.md

Instructions for Codex. This is a research project — readability and reproducibility matter more than shipping speed.

---

## Behavioral Guidelines

These frame *how* to work. The sections below frame *what* to build.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that *your* changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

### 5. Ship Incrementally: Commit After Every Change

**One change = one commit. Commit, push, sync immediately.**

After every single change (editing a file, adding a test, fixing a bug, etc.):
1. `git add` the changed files
2. `git commit -m "<area>: <description>"`
3. `git push origin main`

Rules:
- Never batch multiple changes into one commit.
- Push to `main` directly (no feature branches for single-change commits).
- If the user asks you to open a PR, use a branch. Otherwise, push to main.

---

## Project

**Heterogeneous Bipartite GNN for GUI Structure Error Correction.** Post-correction framework that refines noisy GUI element predictions from lightweight VLMs (Qwen3.5-2B, MiniMax-VL-01) using a heterogeneous bipartite GraphSAGE model.

## Current State

Initial scaffold — module APIs designed with docstrings and `__init__.py` exports, no implementations yet. ~25 tasks across 4 phases in `TASK.md`. Implement sequentially: data → graph → model → eval.

## Commands

```bash
pip install -e .                          # dev install
pip install -e ".[dev,test]"              # with test deps
pytest tests/ -v                          # all tests
pytest tests/test_graph_builder.py -v     # single module
pytest tests/ --cov=bipartite_gnn_gui -v  # coverage
python -c "from torch_geometric import seed_everything; seed_everything(42)"  # seeding check
```

## Architecture

```
VLM JSON → Bipartite Graph (Element × Constraint) → GraphSAGE → Δ𝐱 → Corrected JSON
```

### Package Layout

| Path | Responsibility |
|---|---|
| `src/bipartite_gnn_gui/data/` | VLM parsing, ground-truth loading, normalization, features, Dataset/DataLoader |
| `src/bipartite_gnn_gui/graph/` | Schema, constraint extraction, HeteroData builder, viz, augmentation |
| `src/bipartite_gnn_gui/model/` | Hetero GraphSAGE encoder, 3 prediction heads, loss, trainer, inference |
| `src/bipartite_gnn_gui/eval/` | Metrics (PositionError, AlignmentError, Recall, Precision, IoU), evaluator, baselines, qual viz |
| `src/bipartite_gnn_gui/utils/` | YAML config, structured logging, seeding, bbox transforms, IoU |

### Key Details

- Python ≥3.10, PyTorch ≥2.1, PyG ≥2.4, setuptools under `src/`.
- Tests: pytest. Files match `test_*.py` under `tests/`.
- Graph: heterogeneous bipartite `G = (Vₑ ∪ V_c, E)`, PyG `HeteroData`. Two-layer message passing: element → constraint → element.
- Datasets: GUI-360° (~50K elements, ~3.5K screenshots), ScreenSpot (~30K elements, ~5K screenshots). Raw at `data/raw/`, processed at `data/processed/`.

## Code Style (PyTorch Research Patterns)

- **Imports**: standard library → third-party → local. One import per line (not grouped). Within torch: `import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`.
- **Type annotations**: required on all public function signatures. Use `from typing import ...`. For tensors: `Tensor` from torch. For optional: `Optional[Tensor] = None`.
- **Device handling**: accept `device` as kwarg with default `None` (auto-detect via `torch.cuda.is_available()`). Never hardcode `"cuda"` or `"cpu"`.
- **nn.Module style**: `__init__` defines layers, `forward` defines computation. Always call `super().__init__()`. Use `self.register_buffer()` for non-parameter tensors (not `self.xyz = tensor`).
- **Docstrings**: Google-style. Args/Returns/Raises sections. One-liner summary line + blank line + detail if needed.
  ```python
  def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
      """Compute pairwise IoU between two sets of bounding boxes.

      Args:
          box1: (N, 4) tensor of [x1, y1, x2, y2].
          box2: (M, 4) tensor of [x1, y1, x2, y2].

      Returns:
          (N, M) tensor of IoU values.
      """
  ```
- **Prefer clarity**: explicit loops over clever vectorization when readability suffers. Favor named intermediate variables. No one-liner comprehensions nested more than 2 levels.
- **No notebooks in src/**: notebooks go in `experiments/` or `notebooks/`.

## Anti-Patterns (Don't Do These)

- **Don't use wildcard imports**: `from torch import *` or `from module import *` — explicit names only.
- **Don't mutate function arguments** (especially list/dict params). Return new objects.
- **Don't hardcode paths** — use the YAML config system in `utils/config.py`.
- **Don't use `try/except: pass`** — log the exception or re-raise. Bare except catches `KeyboardInterrupt`.
- **Don't use `torch.Tensor()` constructor** — use `torch.tensor()`, `torch.zeros()`, `torch.randn()` etc. `torch.Tensor()` uses global defaults silently.
- **Don't use `dict` for HeteroData** — use PyG's `HeteroData` API (`data['node_type'].x`, `data['edge_type'].edge_index`).
- **Don't define classes in `__init__.py`** — keep them as re-export only. Classes go in their own module files.
- **Don't leave unused imports, dead code, or `print()` statements** in committed code. Use the logger.
- **Don't put training logic in model definitions** — trainer goes in `trainer.py`, model goes in `model.py`.

## Reproducibility

- **Seed everything** at the start of every training run:
  ```python
  from torch_geometric import seed_everything
  from utils.helpers import set_deterministic
  seed_everything(cfg.seed)
  ```
- **Log all hyperparameters** to a structured file (YAML or JSON) alongside each run's outputs. Include: seed, learning rate, hidden dim, num layers, optimizer, weight decay, dataset split ratios, date.
- **Set `torch.use_deterministic_algorithms(True)`** when benchmarking. Note: this can slow things — acceptable during eval/metrics collection.
- **Pin `requirements.txt`** (or `pyproject.toml` extras) with minimum versions. No unpinned `pip install SomePackage`.

## File Naming & Imports

- **Module files**: snake_case (e.g., `vlm_output.py`, `ground_truth.py`, `constraint_extraction.py`).
- **Classes**: PascalCase. One primary class per file unless tightly coupled.
- **Functions/variables**: snake_case.
- **Private helpers**: prefix with `_` (e.g., `_normalize_coords`).
- **Import style within the package**:
  ```python
  # Good — import from module path
  from bipartite_gnn_gui.data.dataset import GUIDataset
  from bipartite_gnn_gui.graph.builder import HeteroGraphBuilder

  # Also acceptable for deep nesting
  from bipartite_gnn_gui.model import GraphSAGEEncoder  # re-exported in __init__
  ```
- **Test files**: mirror the module path. `test_data_vlm.py` tests `data/vlm_output.py`.

## Commit Message Style

```
<area>: <brief imperative sentence, lowercase, no period>

<optional body — why, not what. Bullet points for multiple reasons.>
```

Areas: `data`, `graph`, `model`, `eval`, `utils`, `test`, `docs`, `infra`.

Good:
```
data: add iou-based bipartite matching for ground-truth alignment
model: clamp delta predictions to [-0.5, 0.5] to prevent blowup
graph: cache constraint matrices to speed up dataloading
```

Bad:
```
fixed some bugs                    # too vague, no area
Update code                        # tells nothing
This commit refactors the way that we handle...  # too wordy
```

## Implementation Order (from TASK.md)

Phase-by-phase, no skipping:
1. **Phase 1** — Data: `vlm_output.py`, `ground_truth.py`, `preprocess.py`, `dataset.py`, `config.py`, `logging.py`
2. **Phase 2** — Graph: `schema.py`, `constraints.py`, `builder.py`, `visualize.py`, `augment.py`
3. **Phase 3** — Model: `encoder.py`, `heads.py`, `model.py`, `losses.py`, `trainer.py`, `inference.py`
4. **Phase 4** — Eval: `metrics.py`, `evaluator.py`, `baselines.py`, `qualitative.py`, experiments
