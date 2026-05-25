# 图模式设计 (Graph Schema Design)

> **Phase 2.4–2.5 — High-Level Design for bipartite-gnn-gui**
>
> This document defines the HeteroData schema, constraint extraction algorithms,
> graph builder API, visualization hooks, and optional augmentation transforms.
> It is the design blueprint for `src/bipartite_gnn_gui/graph/` and derives
> tangible feature dimensions for every node type, edge type, and constraint.

---

## 1. HeteroData 表结构

The bipartite graph has two node stores (`element`, `constraint`) and two
edge stores (`satisfies`, `satisfied_by`).  Every node and edge is represented
as a **fixed-length float32 feature vector** so the GNN encoder receives dense,
well-conditioned numerical input.

### 1.1 ElementNode  (`"element"`)

Each element node encodes *what* the element is (type one-hot), *where* it is
(spatial), and *how trustworthy* the VLM prediction is (confidence).

| Feature group | Dims | Description |
|---|---|---|
| Type one-hot | 18 | One-hot encoding over the 18 element type classes (see §1.1.1). |
| Spatial | 4 | Normalised centre + size: `cx, cy, w, h` (in `[0, 1]`). |
| Confidence | 1 | Scalar confidence score from the VLM output, in `[0, 1]`. |
| **Total** | **23** | Concatenated vector `[type_onehot(18), cx, cy, w, h, confidence]`. |

**Node type key:** `"element"`

> **Note:** The existing `ElementNode` dataclass (`graph/schema.py`) is a
> *logical* representation.  The `HeteroGraphBuilder` is responsible for
> converting logical nodes into the fixed-length tensors above.

#### 1.1.1 Element Type Enumeration (18 classes)

The 18-element vocabulary is shared between VLM parsing and GT loading.  It is
defined as a Python `IntEnum` (or `StrEnum`) that all downstream code imports
from `bipartite_gnn_gui.graph.schema`.

```
 0 — unknown       (fallback / unclassified)
 1 — button
 2 — text
 3 — image
 4 — input         (text input, textarea)
 5 — icon
 6 — container     (generic grouping element)
 7 — list          (ordered / unordered list root)
 8 — checkbox
 9 — radio
10 — dropdown      (select, combobox)
11 — slider
12 — tab           (tab bar item)
13 — link          (hyperlink / anchor)
14 — header
15 — footer
16 — nav           (navigation bar)
17 — card          (semantic card / tile)
```

**Rationale for 18 types:**
- Covers the union of labels found in GUI-360° and ScreenSpot annotations.
- Sufficient granularity for constraint extraction (e.g., `container` is a
  common parent) without exploding the one-hot dimension.
- `unknown` (0) acts as the safe fallback for VLM outputs whose label does not
  map to any known class.

**Mapping specification:**

```python
class ElementType(IntEnum):
    UNKNOWN     = 0
    BUTTON      = 1
    TEXT        = 2
    IMAGE       = 3
    INPUT       = 4
    ICON        = 5
    CONTAINER   = 6
    LIST        = 7
    CHECKBOX    = 8
    RADIO       = 9
    DROPDOWN    = 10
    SLIDER      = 11
    TAB         = 12
    LINK        = 13
    HEADER      = 14
    FOOTER      = 15
    NAV         = 16
    CARD        = 17

def element_type_to_onehot(t: ElementType) -> torch.Tensor:
    """Return a (18,) float32 one-hot vector."""
    vec = torch.zeros(18, dtype=torch.float32)
    vec[int(t)] = 1.0
    return vec
```

#### 1.1.2 Spatial Feature Extraction

Given a bbox in `xywh` normalised coordinates `[x, y, w, h]` (all ∈ [0, 1]):

```
cx = x + w / 2
cy = y + h / 2
```

The spatial vector is `[cx, cy, w, h]`.  Normalisation guards ensure every
coordinate is clamped to `[0, 1]`; if the raw bbox is already normalised by
the data layer (`CoordinateNormalizer`), this step is the identity.

> `x, y` refer to the *top-left* corner.  This matches the conventions in both
> VLM output and GT data (see Phase 1 analysis).

---

### 1.2 ConstraintType 枚举 (10 types)

| # | Enum value | Semantic | Typical params |
|---|---|---|---|
| 1 | `ALIGN_LEFT` | Two or more elements share the same left edge (x). | `tolerance` |
| 2 | `ALIGN_RIGHT` | Elements share the same right edge (x + w). | `tolerance` |
| 3 | `ALIGN_TOP` | Elements share the same top edge (y). | `tolerance` |
| 4 | `ALIGN_BOTTOM` | Elements share the same bottom edge (y + h). | `tolerance` |
| 5 | `CENTER_X` | Vertical centre-lines (cx) are aligned. | `tolerance` |
| 6 | `CENTER_Y` | Horizontal centre-lines (cy) are aligned. | `tolerance` |
| 7 | `SAME_SIZE` | Elements have similar width *and* height. | `tolerance` |
| 8 | `SPACING` | Adjacent elements are equally spaced (gap consistency). | `tolerance`, `expected_gap` |
| 9 | `CONTAINMENT` | One element's bbox fully encloses another. | `margin` |
| 10 | `GRID` | Elements form a regular row/column arrangement. | `tolerance`, `rows`, `cols` |

```python
class ConstraintType(str, Enum):
    ALIGN_LEFT   = "align_left"
    ALIGN_RIGHT  = "align_right"
    ALIGN_TOP    = "align_top"
    ALIGN_BOTTOM = "align_bottom"
    CENTER_X     = "center_x"
    CENTER_Y     = "center_y"
    SAME_SIZE    = "same_size"
    SPACING      = "spacing"
    CONTAINMENT  = "containment"
    GRID         = "grid"
```

> The existing `ConstraintType(str, Enum)` in `graph/schema.py` is already
> correct and will not be changed; this section formalises its semantics.

---

### 1.3 ConstraintNode  (`"constraint"`)

Each constraint node stores the constraint type and two scalar parameters.

| Feature group | Dims | Description |
|---|---|---|
| Type one-hot | 10 | One-hot encoding over the 10 `ConstraintType` values. |
| Params | 2 | Normalised scalar parameters: `tolerance` and `weight` (in `[0, 1]`). |
| **Total** | **12** | Concatenated vector `[type_onehot(10), tolerance, weight]`. |

**Node type key:** `"constraint"`

**Parameter semantics:**

- **`tolerance`** — The threshold used when the constraint was detected (e.g.,
  `0.02` in training mode, `0.05` in inference mode).  Normalised to `[0, 1]` by
  dividing by a configurable `max_tolerance` (default `0.1`).
- **`weight`** — A scalar importance weight assigned to the constraint.  In training
  mode all GT-derived constraints have weight `1.0`.  In inference mode,
  low-confidence constraints may be assigned a weight < 1.0.  This value feeds into
  the `alignment_consistency_loss` as a per-constraint loss multiplier.

**Why two params?**  Twelve is a small, dense feature vector.  `tolerance`
captures detection quality and `weight` captures reliability.  Together they
let the GNN learn to pay more attention to high-confidence constraints.

---

### 1.4 Edge Features  (`E`)

Every edge (both directions) carries a 4-dimensional feature vector computed
from the **element** side of the relationship.

| Feature | Dims | Description |
|---|---|---|
| Spatial distance | 1 | Euclidean distance between element and constraint-node "anchor" (computed as mean centre of all elements involved in the constraint — if only one element, this is `‖(cx, cy) − (c̄x, c̄y)‖₂`). |
| Relative position dx | 1 | Horizontal offset from the constraint anchor: `cx − c̄x`. |
| Relative position dy | 1 | Vertical offset from the constraint anchor: `cy − c̄y`. |
| IoU | 1 | IoU between this element's bbox and the union-rectangle of all elements in the constraint group. |
| **Total** | **4** | Concatenated vector. |

**Edge type keys (PyG convention):**

| Edge store key | Direction |
|---|---|
| `("element", "satisfies", "constraint")` | element → constraint |
| `("constraint", "satisfied_by", "element")` | constraint → element (reverse) |

Both edge stores carry the same 4-dimensional edge features.

> **Implementation note:** `torch_geometric` automatically creates a reverse
> edge type when `to_hetero` wraps the convolution.  If edges are created
> manually (as in the current `BipartiteGraphBuilder`), the reverse edges
> must be created explicitly via `torch.flip(edge_index, dims=[0])`.

---

## 2. 约束提取算法 (Constraint Extraction Algorithms)

The extraction functions take a list of element bboxes and return a list of
`ConstraintNode` objects.  Training mode uses ground-truth bboxes (clean);
inference mode uses VLM-predicted bboxes (noisy and may contain false positives
or missed elements).

### 2.1 Training Mode (GT-based)

Ground-truth annotations are assumed accurate.  Use **narrow tolerance**
`eps = 0.02` (2 % of normalised coordinate space).  Every detected constraint is
assigned `weight = 1.0`.

#### 2.1.1 Alignment Constraints

```
extract_alignment_constraints(elements: list[dict], eps: float = 0.02)
    → list[ConstraintNode]
```

**Algorithm:**

1. For each of the 6 alignment axes (`left`, `right`, `top`, `bottom`,
   `center_x`, `center_y`), group elements whose axis coordinate differs by
   ≤ `eps`.
2. A group must contain at least **2** elements to produce a constraint.
3. For each group, emit one `ConstraintNode` of the corresponding
   `ConstraintType` with `source_indices` = all element indices in the group
   and `target_indices` = same (alignment is symmetric).

**Complexity:** O(n²) per axis — acceptable for typical GUI screens (< 200
elements).

#### 2.1.2 Containment Constraints

```
extract_containment_constraints(elements: list[dict], margin: float = 0.01)
    → list[ConstraintNode]
```

**Algorithm:**

1. For every ordered pair (A, B) where A ≠ B:
   - A contains B iff:
     ```
     A.x ≤ B.x + margin
     A.y ≤ B.y + margin
     (A.x + A.w) ≥ (B.x + B.w) − margin
     (A.y + A.h) ≥ (B.y + B.h) − margin
     ```
2. Emit one `ConstraintNode` of type `CONTAINMENT` with
   `source_indices = [idx(A)]` (parent), `target_indices = [idx(B)]` (child),
   `params = {"tolerance": margin, "weight": 1.0}`.

**Complexity:** O(n²).  A small `margin` avoids false containments due to
bbox noise.

#### 2.1.3 Spacing Constraints

```
extract_spacing_constraints(elements: list[dict], eps: float = 0.02)
    → list[ConstraintNode]
```

**Algorithm:**

1. **Horizontal spacing:** Sort elements by x.  For every consecutive triple
   (A, B, C), compute the two gaps:
   ```
   gapAB = B.x − (A.x + A.w)
   gapBC = C.x − (B.x + B.w)
   ```
   If `|gapAB − gapBC| ≤ eps` and both gaps ≥ 0, emit a `SPACING` constraint
   with `source_indices = [idx(A), idx(B), idx(C)]`.

2. **Vertical spacing:** Repeat with elements sorted by y, using
   `gapAB = B.y − (A.y + A.h)`, etc.

3. `params = {"tolerance": eps, "weight": 1.0}`.

**Complexity:** O(n log n) from sorting.

#### 2.1.4 Grid Constraints

```
extract_grid_constraints(elements: list[dict], eps: float = 0.02)
    → list[ConstraintNode]
```

**Algorithm:**

1. Cluster element centre-y coordinates using 1D agglomerative clustering with
   distance ≤ `eps`.  Each cluster is a candidate **row**.
2. Row is valid if it contains ≥ 2 elements whose centre-x coordinates are
   monotonically increasing.
3. For each valid row, check that elements share similar height
   (`|h_i − h_j| ≤ eps`) and are approximately equally spaced in x.
4. For each valid column (symmetric procedure on centre-x), check shared
   width and equal y-spacing.
5. Emit one `GRID` constraint per valid row/column group with
   `params = {"tolerance": eps, "weight": 1.0}` and `source_indices` listing all
   elements in the group.

**Complexity:** O(n log n + n·k) where k is the number of clusters.

---

### 2.2 Inference Mode (Heuristic)

VLM predictions are noisier:
- Coordinates may be offset by several percent.
- False positive elements may exist.
- Elements may be missing (false negatives).
- Confidence scores may be unreliable.

**Differences from training mode:**

| Aspect | Training (GT) | Inference (Heuristic) |
|---|---|---|
| Tolerance `eps` | `0.02` (2 %) | `0.05` (5 %) |
| Confidence filtering | None | Drop elements with `confidence < 0.3` before extraction |
| Constraint weight | Always `1.0` | Derived from element confidences: `weight = mean(confidence of constrained elements)` |
| Spacing sign check | Required (gap ≥ 0) | Relaxed: gap ≥ −0.01 (tolerate slight overlap) |
| Grid element count | ≥ 2 | ≥ 3 (more conservative) |
| Max constraints | Unlimited | Cap at `5 × n_elements` to prevent explosion |

**Unified entry point:**

```python
def extract_constraints(
    elements: list[dict],
    mode: str = "train",          # "train" | "infer"
    eps: float | None = None,     # auto-set from mode if None
    min_confidence: float | None = None,
) -> list[ConstraintNode]:
    """Extract all constraints using the specified mode.

    Args:
        elements: List of element dicts with keys: bbox [cx,cy,w,h], label, confidence.
        mode: "train" (GT-based, eps=0.02) or "infer" (heuristic, eps=0.05).
        eps: Override the default tolerance.
        min_confidence: Override the confidence threshold (infer mode only).

    Returns:
        Flat list of ConstraintNode objects.
    """
    ...
```

**Internal delegation:**

```
extract_constraints(elements, mode)
  ├── filter low-confidence elements   (infer only)
  ├── extract_alignment_constraints(elements, eps)
  ├── extract_containment_constraints(elements, margin)
  ├── extract_spacing_constraints(elements, eps)
  ├── extract_grid_constraints(elements, eps)
  └── cap total count                  (infer only)
```

---

## 3. HeteroGraphBuilder

The graph builder converts a list of logical elements and constraints into a
`torch_geometric.data.HeteroData` object with tensors that are ready for
the GNN encoder.

### 3.1 Class Interface

```python
class HeteroGraphBuilder:
    """Build a HeteroData graph from element and constraint descriptions.

    This class is responsible for:
      - Converting element attributes into fixed-length feature vectors.
      - Converting constraint attributes into fixed-length feature vectors.
      - Computing per-edge features (spatial distance, relative position, IoU).
      - Creating forward and reverse edge index tensors.
    """

    def __init__(self, config: Config):
        """Initialise builder with validated configuration.

        Args:
            config: Global Config object.  Key fields used:
              - config.model.hidden_dim (for sanity checks only)
              - The builder itself determines feature dimensions.
        """
        self.config = config
        self._element_encoder = None  # placeholder
        self._constraint_encoder = None  # placeholder

    def build(
        self,
        vlm_output: VLMOutput,
        gt: GroundTruth | None = None,
        mode: str = "train",
    ) -> HeteroData:
        """Full graph construction pipeline.

        Args:
            vlm_output: Parsed VLM predictions (bboxes, labels, confidences).
            gt: Optional ground-truth annotations.  When provided (training),
                constraints are extracted from GT bboxes.  When None (inference),
                constraints are extracted from VLM bboxes.
            mode: "train" or "infer".

        Returns:
            HeteroData with keys:
              - data["element"].x         : (N_elem, 23)  float32
              - data["constraint"].x      : (N_con, 12)   float32
              - data["element", "satisfies", "constraint"].edge_index : (2, E) int64
              - data["element", "satisfies", "constraint"].edge_attr  : (E, 4) float32
              - data["constraint", "satisfied_by", "element"].edge_index : (2, E) int64
              - data["constraint", "satisfied_by", "element"].edge_attr  : (E, 4) float32
        """
        ...

    # ── Internal helpers ────────────────────────────────────────────────

    def _build_element_nodes(
        self,
        elements: list[dict],
    ) -> torch.Tensor:
        """Convert element dicts to (N_elem, 23) feature tensor.

        Each element dict must have: bbox [cx,cy,w,h], label (str), confidence (float).
        """
        ...

    def _build_constraint_nodes(
        self,
        constraints: list[ConstraintNode],
    ) -> torch.Tensor:
        """Convert ConstraintNode list to (N_con, 12) feature tensor.

        Extracts type one-hot from ConstraintType enum, normalises params.
        """
        ...

    def _build_edges(
        self,
        elements: list[dict],
        constraints: list[ConstraintNode],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build edge index and edge attributes.

        Returns:
            (edge_index, edge_attr) where:
              - edge_index: (2, total_edges) int64 — element→constraint pairs.
              - edge_attr:  (total_edges, 4) float32 — per-edge features.
        Reverse direction is created by the caller (build method).
        """
        ...
```

### 3.2 Build Pipeline (Data-flow Diagram)

```
VLMOutput + (optional) GroundTruth
                │
                ├── elements (list of dicts with bbox, label, confidence)
                │
                ├── constraints ← extract_constraints(
                │       elements_used = gt.elements if mode=="train" else vlm.elements,
                │       mode = mode,
                │   )
                │
                ▼
        _build_element_nodes(elements)        → (N_elem, 23)
        _build_constraint_nodes(constraints)  → (N_con,  12)
        _build_edges(elements, constraints)   → (2, E), (E, 4)
                │
                ▼
        HeteroData {
            "element":              { "x": (N_elem, 23) },
            "constraint":           { "x": (N_con, 12)  },
            ("element", "satisfies", "constraint"): {
                "edge_index": (2, E),
                "edge_attr":  (E, 4),
            },
            ("constraint", "satisfied_by", "element"): {
                "edge_index": (2, E),     # flipped
                "edge_attr":  (E, 4),     # same features
            },
        }
```

### 3.3 Edge Construction Detail

For each constraint node, edges connect it to **all elements listed in its
`source_indices` and `target_indices`**.  Because a constraint is always
satisfied by (or applied to) a group of elements, a single constraint node
has degree = `|source_indices| + |target_indices|`.

The edge feature for edge **e_k** (element i → constraint j) is:

```python
def _compute_edge_features(elem_bbox: list[float], group_bboxes: list[list[float]]) -> list[float]:
    """Compute 4D edge features from an element bbox and its constraint group.

    Args:
        elem_bbox: [cx, cy, w, h] of the element.
        group_bboxes: List of bboxes (all elements in the constraint group).

    Returns:
        [spatial_distance, dx, dy, iou]
    """
    group_centers = [(b[0], b[1]) for b in group_bboxes]
    anchor_cx = sum(c[0] for c in group_centers) / len(group_centers)
    anchor_cy = sum(c[1] for c in group_centers) / len(group_centers)

    spatial_distance = math.hypot(elem_bbox[0] - anchor_cx, elem_bbox[1] - anchor_cy)
    dx = elem_bbox[0] - anchor_cx
    dy = elem_bbox[1] - anchor_cy

    # IoU with the union rectangle of all group bboxes
    iou = _compute_iou_with_union(elem_bbox, group_bboxes)

    return [spatial_distance, dx, dy, iou]
```

---

## 4. 可视化 (Visualization)

Visualisation functions overlay the bipartite graph on the original screenshot
for qualitative analysis and debugging.

### 4.1 Function Signatures

```python
def plot_graph_on_screenshot(
    img: PIL.Image.Image | np.ndarray,
    hetero_data: HeteroData,
    element_types: list[str],
    constraint_types: list[str],
    title: str | None = None,
    figsize: tuple[int, int] = (16, 10),
    save_path: str | Path | None = None,
) -> matplotlib.figure.Figure:
    """Overlay bipartite graph edges and nodes on the screenshot.

    Elements are drawn as colored bounding boxes (color = element type).
    Constraints are drawn as lines/arrows connecting element groups.
    Constraint type is indicated by line style and color.

    Args:
        img: Screenshot image (PIL Image or numpy array, RGB).
        hetero_data: The HeteroData graph to visualize.
        element_types: List of element type strings (length = N_elem), one per node.
        constraint_types: List of constraint type strings (length = N_con).
        title: Optional figure title.
        figsize: Figure size in inches.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object.
    """
    ...

def color_by_element_type(element_type: str) -> str:
    """Return a matplotlib-compatible color for an element type.

    Pre-defined palette of 18 colors, one per ElementType.
    Returns '#888888' (gray) for unknown.
    """
    ...

def color_by_constraint_type(constraint_type: str) -> tuple[str, str]:
    """Return (color, linestyle) for a constraint type.

    Alignment → solid line, same color per axis.
    Containment → dashed line, blue.
    Spacing → dotted line, green.
    Grid → dash-dot line, orange.
    """
    ...

def export_graph(
    hetero_data: HeteroData,
    path: str | Path,
    format: str = "pt",
) -> None:
    """Export a HeteroData graph to disk.

    Args:
        hetero_data: The graph to export.
        path: Output file path.
        format: "pt" (torch.save), "pkl" (pickle), or "json" (serialise tensors as lists).
    """
    ...
```

### 4.2 Color Palettes

#### Element Type Colors

| Element type | Hex color | Rendered |
|---|---|---|
| unknown | `#CCCCCC` | gray |
| button | `#4C72B0` | blue |
| text | `#55A868` | green |
| image | `#C44E52` | red |
| input | `#8172B2` | purple |
| icon | `#937860` | brown |
| container | `#DA8BC3` | pink |
| list | `#8C8C8C` | dark gray |
| checkbox | `#CCB974` | gold |
| radio | `#64B5CD` | cyan |
| dropdown | `#4C72B0` | blue (same as button) |
| slider | `#F17CB0` | coral |
| tab | `#B2912F` | olive |
| link | `#1A85FF` | bright blue |
| header | `#D55E00` | orange |
| footer | `#D55E00` | orange (same as header) |
| nav | `#0072B2` | dark blue |
| card | `#009E73` | teal |

#### Constraint Colors & Styles

| Constraint type | Color | Linestyle |
|---|---|---|
| ALIGN_LEFT / ALIGN_RIGHT | `#E74C3C` | `—` (solid) |
| ALIGN_TOP / ALIGN_BOTTOM | `#3498DB` | `—` (solid) |
| CENTER_X / CENTER_Y | `#2ECC71` | `—` (solid) |
| SAME_SIZE | `#9B59B6` | `--` (dashed) |
| SPACING | `#1ABC9C` | `:` (dotted) |
| CONTAINMENT | `#E67E22` | `-.` (dash-dot) |
| GRID | `#F39C12` | `-.` (dash-dot) |

### 4.3 Visual Legend

The figure includes a legend showing:
- **Element types** (box colors) — all 18 types that appear in the graph.
- **Constraint types** (line styles) — all constraint types present.

If the legend exceeds 10 entries, it is split across two columns.

---

## 5. 数据增强 (Augmentation, Optional for Phase 4)

Graph-level augmentation transforms improve robustness and generalisation by
injecting controlled noise into the training graphs.  They are applied during
training only, not during inference or evaluation.

### 5.1 NodeDropout

```python
class NodeDropout:
    """Randomly drop element nodes with probability p.

    This simulates VLM false negatives — the constraint extractor must learn
    to infer missing elements from surviving ones.

    Args:
        p: Dropout probability (default 0.1).
    """

    def __init__(self, p: float = 0.1):
        assert 0.0 <= p < 1.0
        self.p = p

    def __call__(self, data: HeteroData) -> HeteroData:
        """Apply node dropout to element nodes.

        Drops element nodes and removes all edges connected to those nodes.
        Constraint nodes whose degree drops to 0 after element removal are
        also removed.

        Returns:
            A new HeteroData (the input is not mutated).
        """
        ...
```

**Edge cases:**
- If dropping elements would remove *all* elements (p high, few elements),
  keep at least 1 element (clamp: do not drop the last element).
- If a constraint node has degree 0 after element dropping, remove the
  constraint node as well.

### 5.2 CoordinateJitter

```python
class CoordinateJitter:
    """Add Gaussian noise to element spatial coordinates.

    Simulates VLM coordinate prediction noise to improve the model's robustness
    to imprecise bounding boxes.

    Args:
        sigma: Standard deviation of Gaussian noise in normalised space (default 0.01).
        clamp: Whether to clamp coordinates to [0, 1] after jittering.
    """

    def __init__(self, sigma: float = 0.01, clamp: bool = True):
        self.sigma = sigma
        self.clamp = clamp

    def __call__(self, data: HeteroData) -> HeteroData:
        """Add noise to the spatial features of element nodes.

        Only the 4 spatial coordinates (indices 18..21 in the feature vector)
        are perturbed.  The one-hot type and confidence scalars are unchanged.

        Returns:
            A new HeteroData (input not mutated).
        """
        ...
```

**Coordinates affected:**
The 4 spatial features at positions `[18, 19, 20, 21]` of each element's
feature vector are perturbed: `new_val = old_val + N(0, sigma²)`.

### 5.3 ConstraintPerturbation

```python
class ConstraintPerturbation:
    """Randomly flip a small fraction of constraint satisfaction edges.

    Simulates the GNN encountering incorrectly inferred constraints, forcing
    the violation head to learn to detect implausible constraints.

    Args:
        p: Probability of flipping each edge (default 0.05).
    """

    def __init__(self, p: float = 0.05):
        assert 0.0 <= p <= 1.0
        self.p = p

    def __call__(self, data: HeteroData) -> HeteroData:
        """Randomly remove a fraction p of edges.

        An edge is "flipped" by removing it from both the forward and reverse
        edge stores.  Constraint nodes that lose all edges are removed.

        Returns:
            A new HeteroData (input not mutated).
        """
        ...
```

### 5.4 GraphAugmentationPipeline

```python
class GraphAugmentationPipeline:
    """Apply a sequence of augmentation transforms.

    Args:
        transforms: Ordered list of augmentation callables.
        p: Probability of applying the entire pipeline (default 0.5).
           If not applied, the input is returned unchanged.
    """

    def __init__(
        self,
        transforms: list[Callable[[HeteroData], HeteroData]] | None = None,
        p: float = 0.5,
    ):
        self.transforms = transforms or [
            NodeDropout(p=0.1),
            CoordinateJitter(sigma=0.01),
            ConstraintPerturbation(p=0.05),
        ]
        self.p = p

    def __call__(self, data: HeteroData) -> HeteroData:
        """Apply transforms sequentially, each with its own probability.

        Returns:
            Augmented HeteroData (or unchanged input if pipeline not applied).
        """
        ...
```

---

## 6. Summary: Feature Dimensions

| Node / Edge | PyG key | Tensor shape | Contents |
|---|---|---|---|
| Element node | `data["element"].x` | `(N_elem, 23)` | `[type_onehot(18), cx, cy, w, h, confidence]` |
| Constraint node | `data["constraint"].x` | `(N_con, 12)` | `[type_onehot(10), tolerance, weight]` |
| Forward edge index | `data["element", "satisfies", "constraint"].edge_index` | `(2, E)` | `[row=elem_idx, col=con_idx]` |
| Forward edge attr | `data["element", "satisfies", "constraint"].edge_attr` | `(E, 4)` | `[dist, dx, dy, iou]` |
| Reverse edge index | `data["constraint", "satisfied_by", "element"].edge_index` | `(2, E)` | `[row=con_idx, col=elem_idx]` |
| Reverse edge attr | `data["constraint", "satisfied_by", "element"].edge_attr` | `(E, 4)` | Same as forward (identical features) |

**Encoding invariant:** All features are normalised to `[0, 1]` or centred
around zero.  The GNN encoder receives well-conditioned inputs without
requiring batch normalisation on the raw node features.

---

## 7. Design Decisions & Trade-offs

### 7.1 Why 23-dimensional element features?

- **18 type one-hot:** Covers the 18-element vocabulary defined in Phase 1.
  One-hot is preferred over learnable embeddings for this dimension because:
  (a) the vocabulary is fixed and small; (b) the GNN encoder already learns
  a dense representation in `hidden_dim` (256) space via SAGEConv.
- **4 spatial dims:** `(cx, cy, w, h)` encodes position and size in the
  minimal number of parameters.  Using the full `xyxy` (4 values) would be
  redundant since `w, h` can be derived.
- **1 confidence:** Carries the VLM's own uncertainty signal into the graph.

### 7.2 Why 12-dimensional constraint features?

- **10 type one-hot:** Direct encoding of the constraint type; the GNN learns
  type-specific message-passing behaviour.
- **2 params (`tolerance`, `weight`):** Provide the GNN with information
  about constraint *quality* without exploding the feature dimension.  Other
  constraint-specific parameters (e.g., `rows`, `cols` for GRID) are too
  sparse to include as fixed features and are better left for the GNN to
  implicitly discover from structure.

### 7.3 Why separate train/infer constraint modes?

Training with GT bboxes teaches the model the *ideal* constraint graph.
Inference with VLM bboxes simulates the *real-world* scenario where bboxes
are noisy.  A single model trained on clean graphs must still generalise
to noisy graphs — the wider tolerance and lower weight in inference mode
provide this robustness.

### 7.4 Why even-degree constraints?

Every constraint connects to all elements in its group (not just pairs).
This preserves the *n-ary* nature of alignment ("these 5 buttons are
left-aligned") while avoiding the combinatorial explosion of pairwise edges.
One constraint node with degree 5 is more expressive and compute-efficient
than 10 pairwise constraint nodes.

---

## 8. File Mapping

| Design section | Implementation file |
|---|---|
| §1.1 ElementNode, ElementType | `graph/schema.py` |
| §1.2 ConstraintType enum | `graph/schema.py` (already exists) |
| §1.3 ConstraintNode | `graph/schema.py` (already exists) |
| §1.4 Edge features | `graph/builder.py` (computed in `_build_edges`) |
| §2.1 Training-mode constraints | `graph/constraints.py` |
| §2.2 Inference-mode constraints | `graph/constraints.py` |
| §3 HeteroGraphBuilder | `graph/builder.py` |
| §4 Visualization | `graph/visualize.py` |
| §5 Augmentation | `graph/augment.py` (already exists) |
