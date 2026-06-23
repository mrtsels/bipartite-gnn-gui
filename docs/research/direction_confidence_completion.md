# Research Directions: Beyond Coordinate Correction

> Post-Phase 4 experimental summary and forward-looking research agenda.
> Two directions: **confidence scoring** (practical) and **structural completion** (academic).

---

## Background: What We Learned

After completing the full pipeline (data → graph → model → training → evaluation →
VLM integration), we ran 5 classes of experiments. Key findings:

| Experiment | Noise Source | VLM Quality | GNN vs NoOp |
|---|---|---|---|
| Simulated Gaussian | `make_noisy_vlm()` | pos_err≈0.04 | ❌ never beats NoOp |
| Qwen3-VL Plus (cloud) | real VLM, 15 elem/img | pos_err≈0.013 | ❌ no room to improve |
| Qwen3-VL Flash (cloud) | real VLM, 15 elem/img | pos_err≈0.0001 | ❌ sub-pixel accuracy |
| LLaVA-7B (local) | real VLM, 3.6 elem/img | pos_err≈0.004 | ❌ too few elements |
| Existence mode | reweighted loss | — | ❌ No false positives to filter |

**Root cause**: A **Goldilocks problem**.

```
Moondream (1 elem)  ──  too weak: graph has no structure
LLaVA (3.6 elem)      ──  too sparse: only 3.6/24 elements, can't build rich constraints
[ Ideal: 8-12 elem/img with noticeable positional errors ]
Qwen3 Flash (15 elem) ──  too accurate: pos_err < 0.0001, nothing to correct
Qwen3 Plus (15 elem)  ──  same problem
```

**The fundamental architectural limitation**: our GNN takes N elements in → outputs N
elements out. It can *refine* existing detections but cannot *generate* new ones.
Building a better coordinate corrector is *necessary but insufficient* — the real
value lies elsewhere.

---

## Direction 1: Constraint-Aware Confidence Scoring

### Concept

Instead of correcting coordinates, the GNN predicts a **confidence score**
for each VLM detection: *"How reliable is this detection, given its spatial context?"*

```
VLM predictions ──→ [Bipartite Constraint Graph] ──→ GNN predicts confidence scores
                                                      ↓
                                          Downstream: filter, rank, or weight
```

### Why This Works

- VLMs have **type-dependent reliability**: buttons are well-localized, icons drift
- Spatial context provides **independent evidence**: if *all* elements in a row are
  left-aligned except one, that one is likely wrong
- Two-hop message passing (`element → constraint → element`) is perfectly suited:
  a constraint node aggregates information from *all* participant elements and
  broadcasts alignment status back

### Training Signal

```python
target_confidence[i] = exp(-‖Δpos_i‖ / σ)   # sigmoid decay with positional error
```

Where `Δpos_i = ‖VLM_center_i - GT_center_i‖`. Elements with large positional errors
get low confidence targets; accurate ones get high targets. The GNN learns to
*approximate ground-truth accuracy from graph structure alone*.

### Evaluation

- **AUROC** of confidence prediction (does GNN confidence correlate with actual error?)
- **Precision@K**: keep top-K elements by GNN confidence — how clean is the output?
- **Filtered metrics**: apply confidence threshold, re-compute recall/precision/F1

### Practical Value

- **Immediately usable** with any VLM: no change to detection pipeline
- **Qwen3-VL friendly**: even a near-perfect VLM has variable reliability;
  GNN learns which 5% of detections to distrust
- **Downstream**: feed filtered elements to OCR, layout analysis, HTML generation

---

## Direction 2: Structural Completeness — Element Infilling from Constraints

### Concept

A **paradigm shift** from coordinate correction to element detection.
The GNN learns: *"Which elements are missing, and where should they be?"*

### Core Insight

A partial layout creates **incomplete constraints**:

```
Full layout:   [icon A] ──align_left── [text B]
                                         ↓
Partial (weak VLM):  [icon A] ──align_left── (MISSING)
                          ↑
                    The constraint exists, but only one end is connected
```

These "dangling edges" in the bipartite graph are structural signatures of
**what should exist but doesn't**. The GNN can learn to recognize and propose
the missing element's type, position, and size.

### Validation Experiment

> **Status: ✅ DONE** — `scripts/train_violation.py`

Training: 500 RICO samples, 40% elements removed per sample. GNN predicts
which constraints are violated (binary classification via `violation_head`).

| Metric | Value | Baseline (random) |
|---|---|---|
| Accuracy | **91.2%** | ~68% |
| Val loss | 0.193 | 0.693 |
| Training | Stable, no overfitting | — |

This validates the core hypothesis: **the constraint graph encodes structural
completeness information that the GNN can decode**. Random dropping of 40%
of elements creates detectable constraint violations — the GNN learns to
identify incomplete constraints at 91% accuracy.

### Next Step: Element Proposal from Violated Constraints

Given a violated constraint node embedding (with a known "gap" — the position
of the missing element can be triangulated from surviving participants), we
can add a **proposal head** that predicts the missing element's bbox and type.
This is the remaining implementation for Phase 4.9.4-4.9.5.

### Architecture Extension

Extend the existing three-head model with a fourth head:

```python
class ElementProposalHead(nn.Module):
    """
    Input:  constraint_node_embedding (hidden_dim)
    Output: [Δx, Δy, Δw, Δh, element_type_logits, proposal_confidence]
    
    Only active for constraint nodes with < 2 incident element edges.
    """
```

Training strategy:

```
Stage 1: Pretrain on Complete Layouts
  - Full RICO GT layouts → build graphs → train encoder
  - Task: predict element types from constraint context (self-supervised proxy)

Stage 2: Finetune with Synthetic Masking
  - RICO GT 24 elements → randomly drop 60% → "weak VLM simulation"
  - Build constraint graph from survivors
  - GNN predicts: which constraints are violated? where are the missing elements?
  - Loss: MSE(proposed_position, GT_position) + cross_entropy(proposed_type, GT_type)
```

### Experiment Design (Clean, Self-Contained)

```
Dataset:       RICO 66K screenshots (no VLM needed!)
Input:         GT layout with 60% elements randomly removed
Graph:         build from survivors only
Target:        predict missing elements from violated constraints
Baseline:      random proposal, nearest-neighbor from training set
Metric:        recall@N, average IoU of proposed elements

Advantage:     fully self-supervised, completely reproducible,
               needs zero VLM predictions
```

### Expected Strengths

- **Spatial reasoning**: adjacent elements have correlated missing patterns
- **Type propagation**: if a "Settings" icon exists, a "Back" button should be nearby
- **Multi-constraint integration**: missing element position triangulated from
  alignment + spacing + containment constraints simultaneously

### Challenging Cases (Interesting Failure Modes)

- **Dense toolbars** with 8+ tiny icons: constraint graph is too dense, GNN can't
  distinguish individual elements from the cluster
- **Free-form layouts** (maps, drawings): constraints are weak or absent;
  GNN has nothing to reason from
- **Modal overlays**: elements appear and disappear depending on state;
  GNN needs temporal context to distinguish missing from hidden

---

## Joint Roadmap

```
Phase 1 (2 weeks): Direction 1 — Confidence Scoring
  ├── Modify GNN head to output confidence scores
  ├── Train confidence target from positional error
  ├── Evaluate on Qwen3-VL and LLaVA predictions
  └── Publish as notebook + results

Phase 2 (3 weeks): Direction 2 — Structural Completion
  ├── Implement ElementProposalHead
  ├── Create synthetic-masking data pipeline
  ├── Pretrain on complete layouts
  ├── Finetune on masked layouts
  ├── Evaluate recall@N, IoU
  └── Write up findings

Phase 3 (optional): Joint System
  └── Confidence filter + element proposals → demo pipeline
```

---

## Implementation Notes

### Files to Create / Modify

| File | Purpose |
|---|---|
| `src/bipartite_gnn_gui/model/confidence_head.py` | New head for confidence scoring |
| `src/bipartite_gnn_gui/model/proposal_head.py` | New head for element proposals |
| `src/bipartite_gnn_gui/data/masking.py` | Synthetic element masking pipeline |
| `scripts/train_confidence.py` | Training script for Direction 1 |
| `scripts/train_completion.py` | Training script for Direction 2 |
| `docs/research/direction_confidence_completion.md` | This document |

### Testing Strategy

```
tests/test_confidence_head.py     # Unit tests for new head
tests/test_proposal_head.py       # Unit tests for proposal head
tests/test_masking.py             # Unit tests for masking logic
```

---

## Related Work & Positioning

| Approach | Relationship | Our Difference |
|---|---|---|
| Object detection (YOLO/DETR) | VLM = detector | GNN post-processes, doesn't compete |
| LayoutGAN / LayoutTransformer | generative layout | GNN uses constraints, not adversarial |
| Graph completion (link prediction) | analogous | bipartite constraint graph is novel domain |
| VLM-native structured output | Qwen3 function calling | GNN is model-agnostic; any VLM output works |

---

## Out of Scope (for now)

- **Full generative novel-element proposal**: too expensive; needs layout-level generation
- **End-to-end trainable VLM+GNN**: VLM gradients needed; expensive
- **Multi-modal (image + graph) fusion**: requires screenshot encoder; heavier compute
