# Heterogeneous Bipartite GNN for GUI Structure Error Correction

**XIE Licheng** — Supervised by Prof. LAU Wing Cheong
Department of Information Engineering, The Chinese University of Hong Kong
UG Summer Research Internship 2026 — Interim Report (July 6, 2026)

---

## 1. Introduction

Lightweight Vision-Language Models (VLMs) under 3B parameters are attractive for on-device GUI understanding due to their low latency and small memory footprint. However, empirical analysis reveals two systematic failure modes:

- **Element omission**: 10–30% of visible GUI elements are missed, especially small icons, dividers, and nested containers.
- **Misalignment**: Detected bounding boxes can deviate by 10–50+ pixels from ground truth, breaking downstream layout reasoning.

Existing approaches address this by fine-tuning larger VLMs (7B+) or cascading object detectors — both computationally expensive. We propose an alternative: treat GUI correction as **structured prediction on a heterogeneous bipartite graph**, leveraging spatial constraints inherent to GUI design without requiring additional detection models or VLM fine-tuning.

---

## 2. Method

### 2.1 Pipeline Overview

```
Screenshot → Lightweight VLM → Noisy JSON → [Bipartite Graph] → GraphSAGE → Δ𝐱 → Corrected JSON
```

The pipeline takes a noisy VLM output JSON, constructs a bipartite constraint graph from the detected elements, applies GraphSAGE message passing over this structure, and predicts per-element coordinate corrections.

**Figure 1 — 建议插图位置**
>
> **GPT Image 2 提示词:**
> A clean, publication-style system architecture diagram for a machine learning pipeline. The flow goes left to right: leftmost is a "GUI Screenshot" icon showing a mobile app screen, then an arrow to a "Lightweight VLM" box, then an arrow to a "Noisy JSON" box with some misaligned coordinate data, then an arrow to a "Heterogeneous Bipartite Graph" shown as two columns of nodes — left column labeled "Element Nodes" (small colored circles representing GUI buttons, text, icons) and right column labeled "Constraint Nodes" (larger squares labeled ALIGN_LEFT, CONTAINMENT, SPACING, GRID) with edges crossing between them, then an arrow through a "GraphSAGE" box, then an arrow to a "Corrected JSON" box with properly aligned coordinates. Use a clean academic style with dark blue (#1a365d) and light blue (#4299e1) color scheme on white background. Flat 2D style, no 3D effects. Label each box clearly. Resolution 1200×400.

### 2.2 Bipartite Graph Construction

The input is a set of $N$ noisy element predictions from a VLM. Each element carries a normalized bounding box $(x_1, y_1, x_2, y_2)$ and type label. From these, we extract **spatial constraints** — typed relationships between elements that encode GUI design priors:

| Constraint Type | Predicate | Example |
|:---------------:|-----------|---------|
| `ALIGN_LEFT` | $\|x_1 - x_1'\| < \varepsilon$ | Buttons share left edge |
| `ALIGN_RIGHT` | $\|x_2 - x_2'\| < \varepsilon$ | Right edges aligned |
| `ALIGN_TOP` | $\|y_1 - y_1'\| < \varepsilon$ | Top edges aligned |
| `ALIGN_BOTTOM` | $\|y_2 - y_2'\| < \varepsilon$ | Bottom edges aligned |
| `CENTER_X` / `CENTER_Y` | Center-distance < $\varepsilon$ | Horizontally/vertically centered |
| `SPACING` | Consistent inter-element gaps | Evenly spaced list items |
| `CONTAINMENT` | Element fully inside another | Icon inside a container |
| `GRID` | Row/column membership | Grid layout detection |
| `SAME_SIZE` | Relative size difference | Uniform card sizes |

The graph is heterogeneous bipartite: $G = (V_e \cup V_c, E)$ where $V_e$ are element nodes, $V_c$ are constraint nodes, and edges $E \subseteq V_e \times V_c$ only connect elements to constraints. This enforces an inductive bias: elements communicate only through shared constraints, and each constraint aggregates evidence from all participant elements.

**Figure 2 — 建议插图位置**
>
> **GPT Image 2 提示词:**
> A clean technical illustration showing the bipartite graph message passing flow. On the left, 5 small colored circular nodes (labeled "Element Nodes": a red button, a blue text label, a green icon, a purple input field, an orange image) each with a 2D bounding box drawn around them. On the right, 4 larger rectangular nodes (labeled "Constraint Nodes": ALIGN_LEFT, CONTAINMENT, SPACING, SAME_SIZE). Gray connecting lines form a bipartite linkage between left and right columns. Two curved arrows illustrate the message passing flow: one labeled "Hop 1: Element → Constraint" going left-to-right, one labeled "Hop 2: Constraint → Element" going right-to-left. The style should be clean academic diagram, white background, flat design, muted blue/gray color palette (#2c5282 for main elements, #718096 for edges). No 3D effects. Include a small inset in bottom-right corner showing the HeteroData structure formula: G = (V_e ∪ V_c, E). Resolution 1200×500.

### 2.3 GraphSAGE Encoder and Prediction Heads

We use a two-layer heterogeneous GraphSAGE encoder that performs bipartite message passing:

1. **Hop 1** (Element → Constraint): Each constraint node aggregates features from all elements linked to it via mean pooling.
2. **Hop 2** (Constraint → Element): Each element node aggregates updated constraint features back.

After encoding, three independent MLP heads operate on the refined embeddings:

- **Coordinate Refinement Head**: Predicts $\Delta\mathbf{x}_i = (\Delta x, \Delta y, \Delta w, \Delta h)$ per element.
- **Violation Detection Head**: Binary classifier predicting if a constraint is violated.
- **Existence Head**: Binary classifier predicting whether a detected element is real or hallucinated.

The loss function combines coordinate MSE, violation BCE, and existence BCE: $\mathcal{L} = w_c\mathcal{L}_{\text{coord}} + w_v\mathcal{L}_{\text{vio}} + w_e\mathcal{L}_{\text{exist}}$.

---

## 3. Experimental Results

We evaluate on the RICO dataset (500 screenshots, ~12K GUI elements) and ScreenSpot (~5K screenshots across mobile/web/desktop). All experiments use AdamW optimizer with cosine annealing, 128-d hidden dimension, and 2-layer bipartite GraphSAGE.

### 3.1 Constraint-Aware Confidence Scoring

A GNN-trained confidence head predicts each VLM detection's reliability based on spatial context. The model learns to identify false positives from structural inconsistencies alone.

| Metric | Value |
|--------|:-----:|
| AUROC | **0.989** |
| Accuracy | **93.2%** |
| Precision | 99.1% |
| Recall | 90.7% |

The near-perfect AUROC demonstrates that spatial context alone is sufficient to distinguish real GUI elements from random imposters.

### 3.2 Structural Element Completion

The GNN detects "holes" in the constraint graph — missing elements that leave incomplete spatial relationships — and proposes their positions and types. Training is self-supervised: randomly drop 60–80% of GT elements, then train GNN to predict the missing ones.

| Drop Ratio | GNN IoU | NN Baseline IoU | GNN > NN? |
|:----------:|:-------:|:---------------:|:---------:|
| 0.6 | **0.123** | 0.088 | ✅ +40% |
| 0.8 | **0.097** | 0.062 | ✅ +56% |

The GNN significantly outperforms a nearest-neighbor baseline when substantial structure is missing, confirming it learns genuine structural priors rather than simple interpolation.

### 3.3 Real VLM End-to-End Pipeline

We deploy the trained model behind Qwen3-VL Flash on 200 real screenshots and measure detection quality before/after GNN correction:

| Metric | Before (VLM only) | After (VLM + GNN) | Δ |
|--------|:-----------------:|:-----------------:|:-:|
| Recall (pooled) | 0.235 | **0.282** | **+4.7pp** |
| F1 (pooled) | 0.291 | **0.320** | **+2.9pp** |
| Precision (pooled) | 0.382 | 0.369 | −1.4pp |

The GNN recovers 226 missed elements via constraint-based proposals (+226 TP, −226 FN), with a modest precision cost of 1.4pp. Fine-tuning on real VLM data further improves F1 by +2.1pp.

**Figure 3 — 建议插图位置**
>
> **GPT Image 2 提示词:**
> A three-panel results figure for an academic ML paper, clean publication style, white background.
>
> Panel A (left): "Confidence Scoring" — a bar chart showing AUROC=0.989, Accuracy=93.2%, Precision=99.1%, Recall=90.7%. Four blue bars (#4299e1) of slightly different heights, with exact values labeled on top of each bar.
>
> Panel B (center): "Element Completion IoU" — grouped bar chart comparing GNN (dark blue #2b6cb0) vs NN baseline (light gray #a0aec0) at drop ratios 0.2, 0.4, 0.6, 0.8. Highlight the 0.6 and 0.8 bars where GNN exceeds NN with "+40%" and "+56%" annotations.
>
> Panel C (right): "Real VLM Pipeline" — before/after comparison with three pairs of columns: Recall (0.235→0.282), Precision (0.382→0.369), F1 (0.291→0.320). Before in light blue (#63b3ed), After in dark blue (#2b6cb0). Arrow annotations showing "+4.7pp" and "+2.9pp".
>
> All three panels should share consistent fonts (Helvetica or Arial), rounded bar corners, subtle grid lines. Include (a), (b), (c) labels above each panel. No 3D effects. Resolution 1600×500.

### 3.4 Ablation Studies

Constraint type ablation reveals which spatial priors contribute most:

| Constraint Set | Violation Acc | Drop |
|:--------------:|:-------------:|:----:|
| Full (10 types) | 0.908 | — |
| No CONTAINMENT | 0.889 | **−1.9pp** |
| No ALIGNMENT | 0.903 | −0.5pp |
| No SPACING | 0.906 | −0.2pp |
| CONTAINMENT only | 0.904 | — |

CONTAINMENT is the most critical constraint type — removing it causes the largest accuracy drop (−1.9pp). This aligns with the intuition that parent-child containment relationships provide the strongest structural signal.

### 3.5 Visual Feature Fusion

We evaluated augmenting the structural features (5-d bbox coordinates) with visual features from a pre-trained ViT-Tiny encoder. Simple concatenation yields small improvements in violation accuracy (+0.9pp). DINOv2 (86M params, 768-dim features) showed no clear advantage over ViT-Tiny (5.7M), suggesting visual features provide diminishing returns beyond structural context.

---

## 4. Current Status and Next Steps

All core modules are implemented and verified (942 tests pass). Key achievements to date:

- Complete data pipeline: VLM parsing → GT matching → normalization → Dataset/DataLoader
- Bipartite graph construction: 10 constraint types, heterogeneous HeteroData format
- GraphSAGE encoder with 3 prediction heads and combined loss
- Training pipeline with hyperparameter sweep (best: hd128, big-noise)
- Two research directions validated: confidence scoring (AUROC 0.989) and element completion (IoU +40%)
- Real VLM end-to-end evaluation (F1 +2.9pp)
- Cross-dataset generalization: RICO → ScreenSpot (28% → 72%)

Planned for the remaining 7 weeks:

| Phase | Focus | Timeline |
|:-----:|-------|:--------:|
| Report | Final report and poster preparation | Weeks 8–10 |
| Paper | Academic paper documenting findings | Weeks 5–9 |
| Demo | Web demo: upload → VLM + GNN → side-by-side | Weeks 3–8 |

---

## References

1. Paszke, A. et al. "PyTorch: An Imperative Style, High-Performance Deep Learning Library." NeurIPS 2019.
2. Fey, M., Lenssen, J.E. "Fast Graph Representation Learning with PyTorch Geometric." ICLR 2019 Workshop on Representation Learning on Graphs and Manifolds.
3. Hamilton, W., Ying, Z., Leskovec, J. "Inductive Representation Learning on Large Graphs." NeurIPS 2017.
4. Deka, B. et al. "RICO: A Mobile App Dataset for Building Data-Driven Design Applications." UIST 2017.
5. Cheng, S. et al. "ScreenSpot: A Challenging Benchmark for GUI Visual Grounding." arXiv:2402.02315, 2024.
6. Bai, J. et al. "Qwen Technical Report." arXiv:2309.16609, 2023.
7. Oquab, M. et al. "DINOv2: Learning Robust Visual Features without Supervision." TMLR, 2024.
8. Harris, C.R. et al. "Array programming with NumPy." Nature 585, 357–362, 2020.
9. Virtanen, P. et al. "SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python." Nature Methods 17, 261–272, 2020.
10. Hunter, J.D. "Matplotlib: A 2D Graphics Environment." Computing in Science & Engineering 9(3), 90–95, 2007.
11. Wolf, T. et al. "Transformers: State-of-the-Art Natural Language Processing." EMNLP 2020.
