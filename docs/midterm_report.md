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

### 2.1 Mathematical Foundation

#### 2.1.1 Heterogeneous Bipartite Graph

We formalize GUI structure as a heterogeneous bipartite graph. Let a screenshot yield $N$ detected elements $\mathcal{E} = \{e_i\}_{i=1}^N$, each with a normalized bounding box $\mathbf{b}_i = (x_1^{(i)}, y_1^{(i)}, x_2^{(i)}, y_2^{(i)}) \in [0,1]^4$ and a type label $t_i \in \mathcal{T}$ where $\mathcal{T}$ is the set of GUI element types (button, text, icon, etc.).

From these elements, we extract $M$ spatial constraints $\mathcal{C} = \{c_j\}_{j=1}^M$. Each constraint $c_j$ is defined by a type $\tau_j \in \{\texttt{ALIGN\_LEFT}, \texttt{CONTAINMENT}, \dots\}$, a source index set $S_j \subseteq \{1,\dots,N\}$, and a target index set $T_j \subseteq \{1,\dots,N\}$. A constraint exists when its predicate $P_{\tau}(\mathbf{b}_{S_j}, \mathbf{b}_{T_j})$ evaluates to true under a tolerance $\varepsilon$.

The heterogeneous bipartite graph is then:

$$G = (\mathcal{V}, \mathcal{E}_{\text{edge}}, \phi, \psi)$$

Where:
- $\mathcal{V} = \mathcal{E} \cup \mathcal{C}$ is the node set, partitioned into $V_e$ (element nodes, $|V_e| = N$) and $V_c$ (constraint nodes, $|V_c| = M$).
- $\phi: \mathcal{V} \to \{0,1\}$ maps each node to its type (0 for element, 1 for constraint).
- $\mathcal{E}_{\text{edge}}$ is the edge set. Edges only cross partitions: $\mathcal{E}_{\text{edge}} \subseteq V_e \times V_c$. An edge $(e_i, c_j)$ exists iff $i \in S_j \cup T_j$. There are no element-element or constraint-constraint edges.
- $\psi: \mathcal{E}_{\text{edge}} \to \mathbb{R}$ assigns each edge a weight based on the normalized distance between the element and the constraint's subspace.

#### 2.1.2 Bipartite Message Passing

Message passing on this graph proceeds in two alternating hops, each implemented as a SAGEConv layer (Hamilton et al., 2017).

**Hop 1 — Element to Constraint.** Each constraint node $c_j$ aggregates features from its incident elements:

$$\mathbf{h}_{c_j}^{(1)} = \sigma\!\left( \mathbf{W}_1 \cdot \text{MEAN}\!\left(\left\{\mathbf{h}_{e_i}^{(0)} : (e_i, c_j) \in \mathcal{E}_{\text{edge}}\right\}\right) + \mathbf{b}_1 \right)$$

where $\mathbf{h}_{e_i}^{(0)} \in \mathbb{R}^{d_0}$ is the initial element feature vector (concatenation of bbox and one-hot type), and $\sigma$ is ReLU.

**Hop 2 — Constraint to Element.** Each element node $e_i$ aggregates the updated constraint representations back:

$$\mathbf{h}_{e_i}^{(2)} = \sigma\!\left( \mathbf{W}_2 \cdot \text{MEAN}\!\left(\left\{\mathbf{h}_{c_j}^{(1)} : (e_i, c_j) \in \mathcal{E}_{\text{edge}}\right\}\right) + \mathbf{b}_2 \right)$$

This two-hop design enforces a strong inductive bias: elements communicate only through shared spatial relationships. If two elements participate in the same constraint (e.g., both are left-aligned), their information is fused at the constraint node and propagated back. Elements that share no constraints never exchange messages.

#### 2.1.3 Initial Node Features

Each element's initial feature vector $\mathbf{h}_{e_i}^{(0)}$ is 5-dimensional: the four normalized bbox coordinates plus a normalized area $a_i = (x_2 - x_1)(y_2 - y_1)$. An optional visual feature $\mathbf{v}_i \in \mathbb{R}^{d_v}$ from a frozen ViT encoder (192-d vit_tiny or 768-d DINOv2) can be concatenated to form $\mathbf{h}_{e_i}^{(0)} \in \mathbb{R}^{5 + d_v}$.

Constraint node features $\mathbf{h}_{c_j}^{(0)}$ embed the constraint type $\tau_j$ as a one-hot vector (10-d) and encode spatial statistics of the participant elements (mean pairwise distance, mean containment overlap ratio, alignment residual).

#### 2.1.4 Multi-Task Learning Objective

After message passing, three prediction heads operate on the refined element embeddings $\{\mathbf{h}_{e_i}^{(2)}\}$:

**Coordinate refinement.** An MLP predicts a per-element delta vector:

$$\Delta\mathbf{x}_i = \text{MLP}_{\text{coord}}(\mathbf{h}_{e_i}^{(2)})$$

optimized by smooth L1 loss:

$$\mathcal{L}_{\text{coord}} = \frac{1}{N} \sum_{i=1}^{N} \text{smooth}_{L_1}(\Delta\mathbf{x}_i - \Delta\mathbf{x}_i^*)$$

where $\Delta\mathbf{x}_i^*$ is the ground-truth correction.

**Violation detection.** A binary classifier predicts whether constraint $c_j$ is violated:

$$\hat{v}_j = \sigma(\text{MLP}_{\text{vio}}(\mathbf{h}_{c_j}^{(1)}))$$

$$\mathcal{L}_{\text{vio}} = -\frac{1}{M} \sum_{j=1}^{M} \left[ v_j^* \log \hat{v}_j + (1 - v_j^*) \log(1 - \hat{v}_j) \right]$$

where $v_j^* \in \{0, 1\}$ indicates whether the constraint is genuinely violated in the GT layout.

**Existence scoring.** A binary classifier predicts whether detected element $e_i$ is genuine (TP) or hallucinated (FP):

$$\hat{e}_i = \sigma(\text{MLP}_{\text{exist}}(\mathbf{h}_{e_i}^{(2)}))$$

$$\mathcal{L}_{\text{exist}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ e_i^* \log \hat{e}_i + (1 - e_i^*) \log(1 - \hat{e}_i) \right]$$

**Combined loss.** The total objective is a weighted sum:

$$\mathcal{L} = w_c \mathcal{L}_{\text{coord}} + w_v \mathcal{L}_{\text{vio}} + w_e \mathcal{L}_{\text{exist}}$$

with default weights $w_c = 1.0$, $w_v = 0.5$, $w_e = 0.5$, tuned via hyperparameter sweep.

#### 2.1.5 Structural Element Completion

For the element completion task (Phase 4.9), we introduce a fourth head that proposes missing elements from structural context. A random subset of GT elements is dropped during training (drop ratio $\rho \in [0.2, 0.8]$). Each surviving constraint that references a dropped element becomes a "violated" signal. The GNN must learn to detect these structural holes and predict:

$$\hat{\mathbf{b}}_k, \hat{t}_k = \text{MLP}_{\text{proposal}}(\mathbf{h}_{c_j}^{(1)})$$

where $\mathbf{h}_{c_j}^{(1)}$ is the aggregated constraint embedding from Hop 1. The proposal loss combines IoU-based bbox regression and cross-entropy type prediction:

$$\mathcal{L}_{\text{prop}} = \frac{1}{K} \sum_{k=1}^K \left[ \mathcal{L}_{\text{IoU}}(\hat{\mathbf{b}}_k, \mathbf{b}_k^*) + \alpha \cdot \text{CE}(\hat{t}_k, t_k^*) \right]$$

The training is self-supervised: no human annotation is needed beyond the existing GT layout, since the target is derived by masking.

### 2.2 Pipeline Overview

```
Screenshot → Lightweight VLM → Noisy JSON → [Bipartite Graph] → GraphSAGE → Δ𝐱 → Corrected JSON
```

The pipeline takes a noisy VLM output JSON, constructs a bipartite constraint graph from the detected elements, applies GraphSAGE message passing over this structure, and predicts per-element coordinate corrections.

**Figure 1: System architecture pipeline.**
![Figure 1: System architecture pipeline](figures/fig%201.png)

### 2.3 Bipartite Graph Construction

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

**Figure 2: Bipartite message passing flow.**
![Figure 2: Bipartite message passing flow](figures/fig%202.png)

### 2.4 GraphSAGE Encoder and Prediction Heads

We use a two-layer heterogeneous GraphSAGE encoder that performs bipartite message passing:

1. **Hop 1** (Element → Constraint): Each constraint node aggregates features from all elements linked to it via mean pooling.
2. **Hop 2** (Constraint → Element): Each element node aggregates updated constraint features back.

After encoding, three independent MLP heads operate on the refined embeddings:

- **Coordinate Refinement Head**: Predicts $\Delta\mathbf{x}_i = (\Delta x, \Delta y, \Delta w, \Delta h)$ per element.
- **Violation Detection Head**: Binary classifier predicting if a constraint is violated.
- **Existence Head**: Binary classifier predicting whether a detected element is real or hallucinated.

The loss function combines coordinate MSE, violation BCE, and existence BCE: $\mathcal{L} = w_c\mathcal{L}_{\text{coord}} + w_v\mathcal{L}_{\text{vio}} + w_e\mathcal{L}_{\text{exist}}$.

---

## 4. Experimental Results

We evaluate on the RICO dataset (500 screenshots, ~12K GUI elements) and ScreenSpot (~5K screenshots across mobile/web/desktop). All experiments use AdamW optimizer with cosine annealing, 128-d hidden dimension, and 2-layer bipartite GraphSAGE.

### 4.1 Constraint-Aware Confidence Scoring

A GNN-trained confidence head predicts each VLM detection's reliability based on spatial context. The model learns to identify false positives from structural inconsistencies alone.

| Metric | Value |
|--------|:-----:|
| AUROC | **0.989** |
| Accuracy | **93.2%** |
| Precision | 99.1% |
| Recall | 90.7% |

The near-perfect AUROC demonstrates that spatial context alone is sufficient to distinguish real GUI elements from random imposters.

### 4.2 Structural Element Completion

The GNN detects "holes" in the constraint graph — missing elements that leave incomplete spatial relationships — and proposes their positions and types. Training is self-supervised: randomly drop 60–80% of GT elements, then train GNN to predict the missing ones.

| Drop Ratio | GNN IoU | NN Baseline IoU | GNN > NN? |
|:----------:|:-------:|:---------------:|:---------:|
| 0.6 | **0.123** | 0.088 | ✅ +40% |
| 0.8 | **0.097** | 0.062 | ✅ +56% |

The GNN significantly outperforms a nearest-neighbor baseline when substantial structure is missing, confirming it learns genuine structural priors rather than simple interpolation.

### 4.3 Real VLM End-to-End Pipeline

We deploy the trained model behind Qwen3-VL Flash on 200 real screenshots and measure detection quality before/after GNN correction:

| Metric | Before (VLM only) | After (VLM + GNN) | Δ |
|--------|:-----------------:|:-----------------:|:-:|
| Recall (pooled) | 0.235 | **0.282** | **+4.7pp** |
| F1 (pooled) | 0.291 | **0.320** | **+2.9pp** |
| Precision (pooled) | 0.382 | 0.369 | −1.4pp |

The GNN recovers 226 missed elements via constraint-based proposals (+226 TP, −226 FN), with a modest precision cost of 1.4pp. Fine-tuning on real VLM data further improves F1 by +2.1pp.

**Figure 3: Experimental results.**
![Figure 3: Experimental results](figures/fig%203.png)

### 4.4 Ablation Studies

Constraint type ablation reveals which spatial priors contribute most:

| Constraint Set | Violation Acc | Drop |
|:--------------:|:-------------:|:----:|
| Full (10 types) | 0.908 | — |
| No CONTAINMENT | 0.889 | **−1.9pp** |
| No ALIGNMENT | 0.903 | −0.5pp |
| No SPACING | 0.906 | −0.2pp |
| CONTAINMENT only | 0.904 | — |

CONTAINMENT is the most critical constraint type — removing it causes the largest accuracy drop (−1.9pp). This aligns with the intuition that parent-child containment relationships provide the strongest structural signal.

### 4.5 Visual Feature Fusion

We evaluated augmenting the structural features (5-d bbox coordinates) with visual features from a pre-trained ViT-Tiny encoder. Simple concatenation yields small improvements in violation accuracy (+0.9pp). DINOv2 (86M params, 768-dim features) showed no clear advantage over ViT-Tiny (5.7M), suggesting visual features provide diminishing returns beyond structural context.

---

## 5. Current Status and Next Steps

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
