# Algorithm: Heterogeneous Bipartite GNN for GUI Spatial Error Correction

> **Mathematical basis and core logic.**
> Maps directly to `src/bipartite_gnn_gui/`.

---

## 1. Problem Formulation

A lightweight VLM (e.g. Qwen3.5-2B) takes a screenshot and outputs a list of
predicted GUI elements with bounding boxes. These predictions are **noisy**: the
VLM's spatial understanding is approximate, with typical errors of 5–25 px in
position and 3–15% in size, even when semantic labeling is correct.

We treat this as a **structured refinement problem**. The input is a set of N
noisy element predictions. The output is a set of per-element correction deltas
$\Delta\mathbf{x}_i = (\Delta x, \Delta y, \Delta w, \Delta h)_i$ that, when
applied, produce a corrected layout consistent with GUI spatial design
principles.

The key insight: GUI layouts are **not random**. They obey predictable spatial
rules — alignment, equal spacing, containment, uniform sizing. These rules are
the priors that allow us to correct VLM noise that would otherwise be
unrecoverable from a single element in isolation.

---

## 2. Why a Bipartite Heterogeneous Graph?

### 2.1 Representational Argument

A GUI layout has two kinds of entities with fundamentally different semantics:

| Entity type | What it represents | Feature space |
|---|---|---|
| **Element node** | A concrete UI widget | Spatial position (cx, cy, w, h), type one-hot, confidence |
| **Constraint node** | A spatial relationship | Constraint type one-hot, tolerance, geometric params |

A homogeneous graph (e.g. all-element with pairwise edges) conflates these.
Element–element edges encode pairwise relations implicitly. Constraint nodes
make relations **explicit, typed, and learnable** — each constraint is a
first-class object with its own embedding that can be updated through message
passing.

### 2.2 Bipartite Structure: Why Two Sets?

The graph is bipartite by construction:

$$
G = (V_e \cup V_c,\; E),\qquad E \subseteq V_e \times V_c
$$

Edges only go between elements and constraints, **never** element–element or
constraint–constraint. This enforces an inductive bias:

1. **Elements communicate only through shared constraints.** Two buttons don't
   directly message each other — they message through the "ALIGN_LEFT"
   constraint they both participate in. This forces the model to justify any
   coordinate change in terms of a named spatial rule.

2. **Constraints aggregate local evidence.** A constraint node sees all the
   elements that allegedly satisfy it. If three buttons are supposedly
   left-aligned but one is 10 px off, the constraint embedding encodes that
   discrepancy.

3. **Two-hop message passing is sufficient.** One hop from element → constraint
   (constraint aggregates), one hop from constraint → element (element
   updates). Every element sees every other element that shares a constraint
   with it, exactly two hops away.

### 2.3 Heterogeneous vs. Homogeneous

A homogeneous graph assigns every node the same feature space and the same
update function. This fails for our setting because an element node
$\mathbf{h}_e \in \mathbb{R}^{D_e}$ and a constraint node $\mathbf{h}_c \in
\mathbb{R}^{D_c}$ have different dimensionalities and semantics. A heterogeneous
GNN applies **type-specific linear transformations** before message passing,
ensuring each node type is projected into a compatible space without losing its
type-specific structure.

Concretely, the encoder applies separate MLPs to element and constraint features
**before** any message passing:

$$
\mathbf{h}_e^{(0)} = \text{MLP}_e(\mathbf{x}_e),\qquad
\mathbf{h}_c^{(0)} = \text{MLP}_c(\mathbf{x}_c)
$$

---

## 3. Node and Edge Feature Spaces

### 3.1 Element Node Features

Each element $e_i$ carries a raw feature vector:

$$
\mathbf{x}_{e_i} = [x_1, y_1, x_2, y_2, p_i] \in \mathbb{R}^5
$$

where $(x_1, y_1, x_2, y_2)$ is the predicted bounding box (normalized to
$[0, 1]^4$) and $p_i \in [0, 1]$ is the VLM's detection confidence.

> **Planned:** The feature dimension will grow to ~23-d with one-hot type
> encoding (18+ element types: button, text, image, input, icon, container,
> list, etc.) and explicit spatial features (cx, cy, w, h) instead of raw
> xyxy corners.

### 3.2 Constraint Node Features

Each constraint $c_j$ carries a parameter vector. The current stub
implementation uses a 1-d tolerance value. The planned fixed dimension is:

$$
\mathbf{x}_{c_j} = [\underbrace{t_1, \ldots, t_{10}}_{\text{type one-hot}},\;
                     \varepsilon_j,\; w_j] \in \mathbb{R}^{12}
$$

where $\varepsilon_j$ is the tolerance and $w_j$ is a weight (1.0 for
GT-derived constraints in training, lower for heuristic constraints at
inference).

### 3.3 Edge Features (Planned)

Edges carry 4-d geometric features describing the spatial relationship between
the connected element and the constraint:

$$
\mathbf{e}_{ij} = [d_{ij},\; \Delta x_{ij},\; \Delta y_{ij},\;
                   \text{IoU}_{ij}] \in \mathbb{R}^4
$$

These are computed from the element's predicted bbox and the constraint's
parameters. Currently, `edge_attr` is **not set** — the stub uses topology
alone.

---

## 4. Message Passing Architecture

### 4.1 The Two-Hop Bipartite Flow

```
           hop 1                  hop 2
    ┌───────────────────┐   ┌───────────────────┐
    │ Element → Constraint │   │ Constraint → Element │
    │ (aggregate evidence) │   │ (update positions)   │
    └───────────────────┘   └───────────────────┘

    e₁ ──┐                        e₁' ←──┐
    e₂ ──┤ → c₁ (align_left)  →   e₂' ←──┤ → c₁
    e₃ ──┘                        e₃' ←──┘
```

**Hop 1 — Constraint aggregation:** Each constraint $c_j$ gathers features from
all elements linked to it:

$$
\mathbf{h}_{c_j}^{(k+1)} = \sigma\!\left(
    \mathbf{W}_c^{(k)} \cdot \text{MEAN}\!\left(
        \{\mathbf{h}_{e_i}^{(k)} : (e_i, c_j) \in E\}
    \right) + \mathbf{b}_c^{(k)}
\right)
$$

**Hop 2 — Element refinement:** Each element $e_i$ gathers features from all
constraints it participates in:

$$
\mathbf{h}_{e_i}^{(k+1)} = \sigma\!\left(
    \mathbf{W}_e^{(k)} \cdot \text{MEAN}\!\left(
        \{\mathbf{h}_{c_j}^{(k+1)} : (e_i, c_j) \in E\}
    \right) + \mathbf{b}_e^{(k)}
\right)
$$

In PyG, this is implemented as `SAGEConv` layers wrapped with `to_hetero`,
which automatically dispatches the correct message function to each edge type.

### 4.2 Current Stub Implementation

The current encoder (`BipartiteGraphSAGE`) is a feed-forward stand-in:
independent MLPs for element and constraint nodes with **no message passing**.
The full `SAGEConv + to_hetero` implementation is the next development
milestone.

### 4.3 Number of Layers

Two bipartite message-passing rounds (`n_layers = 2`) means each element sees
constraints that are 2 hops away. Since the graph is strictly bipartite, 2 hops
covers the entire receptive field — every element can see every other element
that shares a constraint. Additional layers would allow higher-order effects
(constraints influencing other constraints through shared elements), but this
is a secondary effect unlikely to help for the core spatial correction task.

---

## 5. Formal Constraint Definitions

Each constraint type defines a mathematical predicate over element bounding
boxes. The constraint is **satisfied** when the predicate value is below a
tolerance $\varepsilon$.

### 5.1 Alignment Constraints

For two elements with boxes $(x_1, y_1, x_2, y_2)$ and
$(x_1', y_1', x_2', y_2')$:

| Constraint | Predicate | Interpretation |
|---|---|---|
| ALIGN_LEFT | $|x_1 - x_1'|$ | Left edges aligned |
| ALIGN_RIGHT | $|x_2 - x_2'|$ | Right edges aligned |
| ALIGN_TOP | $|y_1 - y_1'|$ | Top edges aligned |
| ALIGN_BOTTOM | $|y_2 - y_2'|$ | Bottom edges aligned |
| CENTER_X | $|(x_1 + x_2)/2 - (x_1' + x_2')/2|$ | Horizontal centers aligned |
| CENTER_Y | $|(y_1 + y_2)/2 - (y_1' + y_2')/2|$ | Vertical centers aligned |

### 5.2 Size Constraints

| Constraint | Predicate | Interpretation |
|---|---|---|
| SAME_SIZE | $\max\left(\frac{|w - w'|}{w'},\; \frac{|h - h'|}{h'}\right)$ | Width and height equal within relative tolerance |

### 5.3 Spatial Configuration Constraints

| Constraint | Predicate | Interpretation |
|---|---|---|
| SPACING | $\vert\text{gap}_{i,i+1} - \text{gap}_{i+1,i+2}\vert$ | Consistent gap between consecutive elements |
| CONTAINMENT | $x_1' \leq x_1 \;\wedge\; y_1' \leq y_1 \;\wedge\; x_2 \leq x_2' \;\wedge\; y_2 \leq y_2'$ | Element $i$ is fully inside element $j$ |
| GRID | Row/column membership from clustering element centers | Elements form a regular 2D array |

### 5.4 Constraint Extraction: Train vs. Inference

| Aspect | Training | Inference |
|---|---|---|
| Element source | Ground-truth bboxes | VLM predicted bboxes |
| Tolerance $\varepsilon$ | 0.02 (tight, from clean GT) | 0.05 (loose, accomodates VLM noise) |
| Constraint filter | Keep all | Drop constraints with $w_j < 0.3$ |
| Constraint weight $w_j$ | 1.0 (known-correct) | Heuristic confidence score |

In training, constraints are extracted from ground-truth annotations — they are
the "correct" spatial rules that the model should learn to enforce. At
inference, constraints are proposed heuristically from VLM predictions and may
be erroneous; low-weight constraints are dropped to prevent propagating bad
structural information.

> **Note:** The current stub does not implement this train/inference split.
> `extract_all_constraints` operates identically in all modes and returns a
> single ALIGN_LEFT constraint on the first two elements when N ≥ 2.

---

## 6. Prediction Heads

Three independent MLP heads operate on the refined element/constraint
embeddings produced by the encoder:

### 6.1 Coordinate Refinement Head

Maps each element embedding to a 4-d delta:

$$
\Delta\mathbf{x}_i = \text{MLP}_{\text{coord}}(\mathbf{h}_{e_i}^{(L)})
    \in \mathbb{R}^4
$$

The corrected bounding box is:

$$
\hat{\mathbf{x}}_i = \mathbf{x}_i + \Delta\mathbf{x}_i
$$

No activation on the output — deltas can be positive or negative. At inference,
deltas are optionally clamped to $[-0.5, 0.5]$ to prevent blowup on extremely
noisy inputs.

### 6.2 Violation Prediction Head

Maps each constraint embedding to a scalar probability:

$$
v_j = \sigma(\text{MLP}_{\text{vio}}(\mathbf{h}_{c_j}^{(L)}))
    \in [0, 1]
$$

$v_j \approx 1$ means the constraint is likely violated (the bounding boxes
that triggered this constraint don't actually satisfy it). This auxiliary
signal helps the model learn which constraints are informative vs.
coincidental.

### 6.3 Existence Prediction Head

Maps each element embedding to a scalar probability:

$$
p_i = \sigma(\text{MLP}_{\text{exist}}(\mathbf{h}_{e_i}^{(L)}))
    \in [0, 1]
$$

$p_i \approx 0$ means the element is likely a hallucination (VLM predicted
something that doesn't exist). This head enables the model to suppress false
detections as an alternative to correcting their coordinates.

### 6.4 Architecture

Each head is a 2-layer MLP:

$$
\text{MLP}(\mathbf{h}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2
$$

The coordinate head outputs 4 dimensions, violation and existence heads output
1 (followed by sigmoid).

---

## 7. Loss Function

The total loss is a weighted sum of three component losses:

$$
\mathcal{L} = w_c \cdot \mathcal{L}_{\text{coord}}
             + w_v \cdot \mathcal{L}_{\text{vio}}
             + w_e \cdot \mathcal{L}_{\text{exist}}
$$

### 7.1 Coordinate Loss

Mean squared error between predicted and ground-truth deltas:

$$
\mathcal{L}_{\text{coord}} = \frac{1}{N} \sum_{i=1}^{N}
    \|\Delta\mathbf{x}_i - \Delta\mathbf{x}_i^{\text{gt}}\|_2^2
$$

The ground-truth delta is the difference between the VLM's predicted bbox and
the annotated bbox: $\Delta\mathbf{x}_i^{\text{gt}} = \mathbf{x}_i^{\text{gt}}
- \mathbf{x}_i^{\text{pred}}$.

**Why MSE rather than Smooth L1 or IoU loss?** For GUI coordinate correction,
MSE penalizes large errors quadratically, which is appropriate — a 20 px error
is more than twice as bad as a 10 px error in terms of perceived layout
quality. Smooth L1 would underweight large corrections.

### 7.2 Violation Loss

Binary cross-entropy between predicted violation scores and ground-truth labels:

$$
\mathcal{L}_{\text{vio}} = -\frac{1}{M} \sum_{j=1}^{M}
    \left[ y_j \log v_j + (1 - y_j) \log(1 - v_j) \right]
$$

where $y_j = \mathbf{1}[\text{constraint } c_j \text{ is actually violated}]$.
In training, violation labels are derived by comparing GT-derived constraints
against the VLM's noisy predictions.

### 7.3 Existence Loss

Binary cross-entropy between predicted existence probabilities and ground truth:

$$
\mathcal{L}_{\text{exist}} = -\frac{1}{N} \sum_{i=1}^{N}
    \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$

where $y_i = 1$ if element $i$ is a real GUI element (matched to GT), and
$y_i = 0$ if it is a hallucination (FP).

### 7.4 Loss Weighting

Default: $w_c = w_v = w_e = 1.0$. The coordinate loss typically dominates in
magnitude (4D MSE vs. scalar BCE). In practice, the weights should be tuned so
that each loss component contributes roughly equally to the total gradient at
the start of training. A heuristic: scale $w_v$ and $w_e$ by
$\approx\text{hidden\_dim}/4$ to compensate for the dimensionality difference.

---

## 8. Training Dynamics

### 8.1 Optimization

- **Optimizer:** AdamW with weight decay $10^{-5}$ (applied to non-bias params)
- **Learning rate:** $10^{-3}$ peak, linear warmup over 1000 steps, cosine
  annealing to $10^{-6}$
- **Gradient clipping:** max L2 norm = 1.0
- **Mixed precision:** FP16 (AMP) when CUDA available
- **Batch size:** 32 HeteroData graphs per batch

### 8.2 Early Stopping

Training stops when validation loss fails to improve for 20 consecutive epochs.
The checkpoint with the best validation loss is retained.

### 8.3 Expected Convergence Behavior

The coordinate loss should decrease monotonically as the model learns the
average spatial bias of each constraint type. The violation and existence
losses may plateau early — they are auxiliary signals that help regularize the
encoder but have limited learnable signal beyond the first ~20 epochs.

A converged model should produce:

- **Position error reduction** of 30–50% relative to the raw VLM output
- **Element recall improvement** of 10–20% at IoU > 0.5
- **Alignment error reduction** of 40–60% (the largest relative gain, since
  alignment is what the graph structure most directly encodes)

---

## 9. Inference Pipeline

```
VLM JSON → parse → ElementNodes → extract constraints → HeteroData
    → encoder(h) → coord_head → Δx → x + Δx → clamp → corrected JSON
```

1. **Parse** VLM JSON into `VLMOutputElement` objects with normalized bboxes.
2. **Extract constraints** heuristically from predicted bboxes (loose tolerance
   $\varepsilon = 0.05$, drop low-confidence constraints).
3. **Build** `HeteroData` bipartite graph.
4. **Encode** through `BipartiteGraphSAGE` to get refined embeddings.
5. **Predict** coordinate deltas from element embeddings.
6. **Apply** deltas: $\hat{\mathbf{x}}_i = \mathbf{x}_i + \Delta\mathbf{x}_i$.
7. **Clamp** corrected coordinates to $[0, 1]$ (valid image space).
8. **Denormalize** to absolute pixel coordinates if needed.
9. **Output** corrected JSON with the same schema as the VLM input.

The violation and existence heads are **not used** during inference (they are
auxiliary training signals). However, future work could use them to flag
low-confidence elements for human review.

---

## 10. Why This Works: The Key Intuitions

### 10.1 Constraints as Inductive Bias

A standard MLP corrector sees $\mathbb{R}^{5N}$ and must learn a mapping to
$\mathbb{R}^{4N}$ without any structural prior. The bipartite GNN decomposes
this into local message-passing operations, each parameterized by constraint
type. The model doesn't need to learn that left-alignment is a thing — it's
given that structure explicitly.

### 10.2 Message Passing as Constraint Satisfaction

The two-hop flow (element → constraint → element) implements a differentiable
relaxation of constraint satisfaction. When three elements are connected to an
ALIGN_LEFT constraint:

1. The constraint node receives all three x-coordinates.
2. It can detect outliers (one x₁ is far from the others).
3. It sends back a gradient signal that pushes the outlier toward the consensus.

This is analogous to one round of belief propagation in a factor graph, where
constraint nodes are factors and element nodes are variables.

### 10.3 Multi-Task Learning as Regularization

The violation and existence heads are auxiliary tasks that force the encoder
to produce embeddings that are useful for multiple purposes. This prevents the
encoder from collapsing to a trivial solution (e.g. always predicting zero
deltas). The existence head in particular gives the model an "escape hatch" for
hallucinated elements — instead of trying to correct a non-existent button, it
can suppress it.

### 10.4 Scale Separation

The GNN operates on **normalized** coordinates in $[0, 1]^2$. This means the
same model works for screenshots of any resolution. The coordinate deltas are
also in normalized space, so $\Delta x = 0.01$ means 1% of the image width
regardless of whether the image is 720 px or 4K.

---

## 11. Current Status & Remaining Work

| Component | Status | Description |
|---|---|---|
| Graph schema | ✅ Implemented | `ElementNode`, `ConstraintNode`, `EdgeType`, 10 `ConstraintType` values |
| Graph builder | ✅ Implemented | `BipartiteGraphBuilder.build()` producing valid `HeteroData` |
| Constraint extraction | 🔶 Stub | Only ALIGN_LEFT stub; 9 extractors return `[]` |
| Encoder | 🔶 Stub | Feed-forward MLP stand-in; no message passing |
| Prediction heads | ✅ Implemented | 3 MLP heads with proper activations |
| Loss function | ✅ Implemented | 3-component weighted loss |
| Trainer | 🔶 Stub | `fit()` is a no-op |
| Inference | 🔶 Stub | `correct_layout()` calls `model(data)` |
| Full SAGEConv encoder | ❌ Planned | `to_hetero` + `SAGEConv` bipartite MP |
| Edge features | ❌ Planned | 4-d `edge_attr` on both edge stores |
| Full constraint extraction | ❌ Planned | All 10 types with train/inference modes |

---

## References

- Hamilton, W. et al. "Inductive Representation Learning on Large Graphs."
  NeurIPS 2017. (*GraphSAGE*)
- Fey, M. & Lenssen, J.E. "Fast Graph Representation Learning with PyTorch
  Geometric." ICLR 2019 Workshop. (*PyG HeteroData*)
- The 10 spatial constraint types are derived from GUI layout conventions
  (material design, iOS HIG, web CSS box model) rather than from a specific
  paper.
