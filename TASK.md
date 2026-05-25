# Task List — Bipartite-GNN-GUI

> Phase-based development plan for Heterogeneous Bipartite GNN for GUI Structure Error Correction.

---

## Phase 1: Data Pipeline & Infrastructure

**Goal:** Set up data loading, preprocessing, and project infrastructure.

### Tasks

- [ ] **1.1** Implement VLM output loader (`data/vlm_output.py`)
  - Parse JSON output from Qwen3.5-2B and MiniMax-VL-01 into a unified `VLMOutput` dataclass.
  - Normalize coordinate formats (absolute vs. relative, different origin conventions).
  - Handle missing/optional fields with sensible defaults.

- [ ] **1.2** Implement ground-truth loader (`data/ground_truth.py`)
  - Load ground-truth annotations from GUI-360° and ScreenSpot datasets.
  - Match ground-truth elements to VLM-predicted elements via IoU bipartite matching.
  - Filter unmatched elements (false positives / false negatives).

- [ ] **1.3** Implement data preprocessing (`data/preprocess.py`)
  - Coordinate normalization (scale to [0, 1] relative to screenshot dimensions).
  - Feature extraction for each element: type embedding, spatial features (x, y, w, h), visual crop features (optional).
  - Train/validation/test splitting.

- [ ] **1.4** Implement dataset classes (`data/dataset.py`)
  - `GUIDataset` — PyTorch `Dataset` that yields processed samples.
  - `GUIDataModule` — PyTorch Lightning or custom DataLoader factory.
  - Support batching with collation of variable-size element sets.

- [ ] **1.5** Implement configuration (`utils/config.py`)
  - YAML-based configuration system for model hyperparameters, training settings, and dataset paths.
  - Config validation with `pydantic` or dataclass-based schema.

- [ ] **1.6** Set up logging & experiment tracking (`utils/logging.py`)
  - Structured logging to console and file.
  - Metric logging compatible with WandB / TensorBoard.

### Deliverables

- [ ] Data loaders for both datasets with unified interface.
- [ ] Config system with example YAML file.
- [ ] Smoketest that loads a sample, runs preprocessing, and verifies output shape.

---

## Phase 2: Bipartite Graph Construction

**Goal:** Build heterogeneous bipartite graphs from VLM JSON output.

### Tasks

- [ ] **2.1** Define graph schema (`graph/schema.py`)
  - `ElementNode` features: type one-hot, spatial features (x, y, w, h), confidence score.
  - `ConstraintNode` types: `ALIGN_LEFT`, `ALIGN_RIGHT`, `ALIGN_TOP`, `ALIGN_BOTTOM`, `CENTER_X`, `CENTER_Y`, `SAME_SIZE`, `SPACING`, `CONTAINMENT`, `GRID`.
  - Edge features: spatial distance, relative position, intersection-over-union.

- [ ] **2.2** Implement constraint extraction (`graph/constraints.py`)
  - Extract spatial constraints from ground-truth layouts (for training).
  - Propose heuristic constraints from VLM predictions (for inference).
  - Constraint types:
    - **Alignment constraints**: elements sharing the same left/right/top/bottom edge.
    - **Containment constraints**: parent-child container relationships.
    - **Spacing constraints**: consistent gaps between adjacent elements.
    - **Grid constraints**: elements arranged in row/column patterns.

- [ ] **2.3** Implement bipartite graph builder (`graph/builder.py`)
  - Take element list + constraint list → PyG `HeteroData` object.
  - Create `(element, belongs_to, constraint)` edges.
  - Add reverse edges `(constraint, affects, element)` for bidirectional message passing.
  - Assign initial node features as tensors.

- [ ] **2.4** Implement graph visualization (`graph/visualize.py`)
  - Plot the bipartite graph overlaid on the screenshot.
  - Color-code element nodes by type and constraint nodes by constraint type.
  - Export to PNG / SVG for qualitative analysis.

- [ ] **2.5** Implement graph augmentation (`graph/augment.py`)
  - Random node dropout (simulate VLM omissions).
  - Coordinate jitter (simulate VLM misalignment).
  - Constraint perturbation (add/remove random constraints).

### Deliverables

- [ ] Graph builder that converts VLM JSON → `HeteroData` object.
- [ ] Constraint extraction from both ground truth and heuristics.
- [ ] Visualization tool for qualitative inspection.
- [ ] Unit tests verifying graph construction on synthetic data.

---

## Phase 3: GNN Model (GraphSAGE)

**Goal:** Implement GraphSAGE-based correction model with violation prediction and coordinate refinement.

### Tasks

- [ ] **3.1** Implement GraphSAGE encoder (`model/encoder.py`)
  - Heterogeneous GraphSAGE using `to_hetero()` or custom `HeteroConv`.
  - Two layers of SAGEConv with ReLU activation and dropout.
  - Message passing: element → constraint → element (bipartite flow).
  - Output: refined element embeddings.

- [ ] **3.2** Implement refinement heads (`model/heads.py`)
  - `CoordinateRefinementHead`: MLP mapping element embeddings → Δ𝐱ᵢ = (Δx, Δy, Δw, Δh).
  - `ViolationPredictionHead`: MLP mapping constraint embeddings → binary violation score (is this constraint violated?).
  - `ExistencePredictionHead`: MLP predicting element existence probability (for detecting omissions).

- [ ] **3.3** Implement the full model (`model/model.py`)
  - `BipartiteGNNCorrector` — end-to-end model combining encoder + all heads.
  - Forward pass: `(x_dict, edge_index_dict) → (Δcoords, violations, existence)`.
  - Support for batched graphs via PyG `Batch` or dummy node padding.

- [ ] **3.4** Implement loss functions (`model/losses.py`)
  - `ℒ_coord = SmoothL1(Δ𝐱_pred, Δ𝐱_target)` — coordinate refinement loss.
  - `ℒ_violation = BCE(violation_pred, violation_gt)` — constraint violation loss.
  - `ℒ_alignment = ∑_{(i,j)∈A} ‖Δ𝐱_i − Δ𝐱_j‖²` — alignment consistency regularizer.
  - `ℒ_existence = BCE(existence_pred, existence_gt)` — element existence loss.
  - Combined: `ℒ = ℒ_coord + λ₁ℒ_violation + λ₂ℒ_alignment + λ₃ℒ_existence`.

- [ ] **3.5** Implement training loop (`model/trainer.py`)
  - Training and validation loops with epoch-level metrics.
  - Learning rate scheduling (cosine annealing with warmup).
  - Gradient clipping, early stopping, and model checkpointing.
  - Mixed-precision training (AMP) support.

- [ ] **3.6** Implement inference pipeline (`model/inference.py`)
  - `correct_layout(vlm_json, screenshot) → corrected_json`.
  - End-to-end: VLM JSON → graph → GNN forward → apply Δ𝐱 → output corrected JSON.
  - Batch inference support for processing multiple screenshots.

### Deliverables

- [ ] GraphSAGE encoder with bipartite message passing.
- [ ] Coordinate refinement + violation prediction heads.
- [ ] Training loop with loss combination and scheduling.
- [ ] End-to-end inference pipeline.
- [ ] Unit tests for forward pass shape correctness.

---

## Phase 4: Evaluation & Experiments

**Goal:** Evaluate the model on benchmark datasets and baselines.

### Tasks

- [ ] **4.1** Implement evaluation metrics (`eval/metrics.py`)
  - `PositionError`: `‖(x̂, ŷ) − (x, y)‖₂` averaged over all elements.
  - `SizeError`: `‖(ŵ, ĥ) − (w, h)‖₂` averaged over all elements.
  - `AlignmentError`: alignment group deviation (see Metrics table in README).
  - `ElementRecall`: fraction of ground-truth elements correctly detected (IoU > 0.5).
  - `ElementPrecision`: fraction of predicted elements matching a ground-truth element.
  - `IoU`: standard Intersection-over-Union for matched element pairs.

- [ ] **4.2** Implement evaluator (`eval/evaluator.py`)
  - `Evaluator` class that runs all metrics over a dataset.
  - Per-category breakdown (button, text, image, input, etc.).
  - Statistical significance testing (paired bootstrap or Wilcoxon).

- [ ] **4.3** Implement baselines (`eval/baselines.py`)
  - Baseline 1: VLM output (no correction).
  - Baseline 2: Rule-based correction (NMS, snap-to-grid, margin-based adjustment).
  - Baseline 3: Fine-tuned VLM (if compute permits).
  - Baseline 4: MLP-only correction (no graph structure).

- [ ] **4.4** Run experiments and analysis (`experiments/`)
  - Experiment 1: Ablation on constraint types (which constraints are most helpful?).
  - Experiment 2: Sensitivity to graph construction hyperparameters.
  - Experiment 3: Robustness to varying VLM noise levels.
  - Experiment 4: Generalization across datasets (train on GUI-360°, eval on ScreenSpot and vice versa).

- [ ] **4.5** Qualitative analysis (`eval/qualitative.py`)
  - Side-by-side visualization: ground truth vs. VLM output vs. corrected output.
  - Case studies: best improvements, failure modes, edge cases.
  - Attention/interaction pattern analysis (which constraints influence which elements?).

- [ ] **4.6** Report results (`experiments/report.py`)
  - Generate LaTeX tables and matplotlib figures.
  - Summary statistics and key findings.
  - Export results to JSON/CSV for downstream analysis.

### Deliverables

- [ ] Comprehensive evaluation suite with all defined metrics.
- [ ] Baseline comparisons demonstrating improvement over VLM-only output.
- [ ] Ablation studies showing contribution of each component.
- [ ] Qualitative visualizations for paper figures.
- [ ] Final results table with statistical significance.

---

## Milestones

| Milestone | Deadline | Description |
|-----------|----------|-------------|
| **M1** | — | Phase 1 complete: data loaders working on both datasets. |
| **M2** | — | Phase 2 complete: graph construction verified on synthetic and real data. |
| **M3** | — | Phase 3 complete: model training converges on validation set. |
| **M4** | — | Phase 4 complete: all evaluation metrics and baselines reported. |
