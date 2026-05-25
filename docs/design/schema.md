# 图模式设计 (Graph Schema Design)

> Phase 2.4-2.5 — Heterogeneous Bipartite Graph Schema & Constraint Extraction Strategy
> Version: 2.0 | 2026-05-25

---

## 1. Schema Dataclasses

The graph module defines three schema objects in `src/bipartite_gnn_gui/graph/schema.py`.

### 1.1 ConstraintType

```python
class ConstraintType(str, Enum):
    """Supported spatial constraints (string-valued enum)."""

    ALIGN_LEFT     = "align_left"
    ALIGN_RIGHT    = "align_right"
    ALIGN_TOP      = "align_top"
    ALIGN_BOTTOM   = "align_bottom"
    CENTER_X       = "center_x"
    CENTER_Y       = "center_y"
    SAME_SIZE      = "same_size"
    SPACING        = "spacing"
    CONTAINMENT    = "containment"
    GRID           = "grid"
```

10 constraint types, each a string enum. The string values are used directly for
serialisation/logging and are the canonical identifiers throughout the codebase.

### 1.2 EdgeType

```python
class EdgeType(str, Enum):
    """Edge categories for the bipartite graph."""

    ELEMENT_TO_CONSTRAINT = "element_to_constraint"
    CONSTRAINT_TO_ELEMENT = "constraint_to_element"
```

Two edge categories, one for each direction of the bipartite graph. These enum
values are not directly used in the `HeteroData` key tuples (which use the shorter
"to" convention — see §2.3), but serve as semantic labels.

### 1.3 ElementNode

```python
@dataclass
class ElementNode:
    """Node describing a GUI element."""

    bbox: list[float]               # Bounding box (normalised xyxy or xywh, see note)
    label: str = "unknown"          # Element type label string
    confidence: float = 1.0         # Detection confidence ∈ [0, 1]
    element_id: str | None = None   # Optional unique identifier
    features: dict[str, float] = field(default_factory=dict)  # Extra key-value features
```

> **Note:** The `bbox` field stores 4 floats. The coordinate convention
> (xyxy vs xywh, normalised vs absolute) is determined by the upstream data
> pipeline. The builder (§2) uses `bbox` as-is without re-interpreting its
> semantics.

### 1.4 ConstraintNode

```python
@dataclass
class ConstraintNode:
    """Node describing a spatial constraint."""

    constraint_type: ConstraintType          # One of the 10 constraint types
    source_indices: list[int] = field(default_factory=list)   # Indices of source elements
    target_indices: list[int] = field(default_factory=list)   # Indices of target elements
    params: dict[str, float] = field(default_factory=dict)    # Extra parameters (e.g. tolerance)
```

- `source_indices` / `target_indices` are 0-based indices into the element list
  passed to the builder. Both lists are in scope for edge creation.
- `params` carries constraint-specific metadata (e.g. `{"tolerance": 0.02}` for
  alignment constraints).

---

## 2. Graph Construction

The builder lives in `src/bipartite_gnn_gui/graph/builder.py`.

### 2.1 BipartiteGraphBuilder

```python
class BipartiteGraphBuilder:
    """Build a bipartite graph from elements and constraints."""

    def build(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
    ) -> HeteroData:
        """Create a graph object with node and edge stores.

        Args:
            elements:   List of ElementNode objects (N_elem).
            constraints: List of ConstraintNode objects (N_con).

        Returns:
            HeteroData with the keys described in §2.2–2.4.
        """
```

> **Notes:**
> - There is no `mode` parameter — the builder is stateless and does not
>   distinguish train vs. inference. Any such distinction belongs to the caller.
> - The builder does **not** extract constraints internally. Constraint extraction
>   is a separate step (§3) whose output feeds into `build()`.
> - There are **no** private `_build_*` helper methods; all logic lives in the
>   single `build()` method.

### 2.2 Element Node Store (`data["element"]`)

```python
data["element"].x  # shape (N_elem, 5), dtype float32
```

Each row is `[x1, y1, x2, y2, confidence]` — the raw bbox (4 floats) concatenated
with the confidence score.

| Column | Index | Source Field        | Description                     |
|--------|-------|---------------------|---------------------------------|
| x1     | 0     | `element.bbox[0]`   | First bbox coordinate           |
| y1     | 1     | `element.bbox[1]`   | Second bbox coordinate          |
| x2     | 2     | `element.bbox[2]`   | Third bbox coordinate           |
| y2     | 3     | `element.bbox[3]`   | Fourth bbox coordinate          |
| conf   | 4     | `element.confidence`| Detection confidence            |

> **When `elements` is empty**, the store contains `torch.zeros((0, 5))`.

> **Planned enhancement:** Future versions will replace the raw bbox with explicit
> spatial features (cx, cy, w, h) and a type one-hot vector, raising the feature
> dimension to ≥ 23. See §7 for the design roadmap.

### 2.3 Constraint Node Store (`data["constraint"]`)

```python
data["constraint"].x  # shape (N_con, D), dtype float32
```

Each row is `list(constraint.params.values())` — a flat vector of the constraint's
parameter values. If `params` is empty, the row is `[0.0]`.

The feature dimension `D` is variable across constraints and depends on the params
dict size. In the current stub implementation (see §3), alignment constraints
emit `params={"tolerance": 0.02}`, so `D = 1`.

> **When `constraints` is empty**, the store contains `torch.zeros((0, 1))`.

> **Planned enhancement:** A fixed-dimension constraint feature vector with
> one-hot type encoding (10-d) + tolerance + weight = 12-d. See §7.

### 2.4 Edge Stores

Two directed edge types connect the bipartite nodes:

```python
# Forward: element → constraint
data["element", "to", "constraint"].edge_index   # shape (2, E), dtype long
# Reverse: constraint → element  (flipped forward index)
data["constraint", "to", "element"].edge_index   # shape (2, E), dtype long
```

**Edge construction logic:**

For each constraint at index `c`:
- For each element index in `constraint.source_indices ∪ constraint.target_indices`:
  - Add edge `(element_index, c)` to the forward store.

The reverse index is a simple `torch.flip` of the forward index along dim 0.

> **When either elements or constraints is empty**, the edge index is
> `torch.zeros((2, 0))` for both directions.

> **There is no `edge_attr`** set on either edge store. Edge features (spatial
> distance, dx, dy, IoU — previously spec'd as 4-d) are not computed by the
> current builder. They remain a planned enhancement (§7).

### 2.5 Fallback HeteroData

When PyTorch Geometric is not installed, the module provides a minimal fallback:

```python
class HeteroData(dict):
    """Minimal fallback when PyG is unavailable."""
    def __getattr__(self, item): ...
    def __setattr__(self, key, value): ...
```

This enables constructing graph-like objects in lightweight environments without
importing PyG. All PyG-specific features (message passing, to_hetero, etc.) are
unavailable in fallback mode.

---

## 3. Constraint Extraction

Implemented in `src/bipartite_gnn_gui/graph/constraints.py`.

### 3.1 Public Interface

```python
def extract_all_constraints(
    elements: Sequence[ElementNode],
) -> list[ConstraintNode]:
    """Extract all heuristic constraints.

    Returns a list of ConstraintNode objects discovered from the element list.
    Dispatches to per-type sub-extractors.
    """
```

| Sub-extractor                      | Signature                                                          | Status        |
|------------------------------------|--------------------------------------------------------------------|---------------|
| `extract_alignment_constraints`    | `(elements, tolerance=0.02) -> list[ConstraintNode]`               | **Stub** — returns one ALIGN_LEFT on elements [0,1] when N≥2 |
| `extract_containment_constraints`  | `(elements) -> list[ConstraintNode]`                               | **Stub** — always returns `[]` |
| `extract_spacing_constraints`      | `(elements, tolerance=0.02) -> list[ConstraintNode]`               | **Stub** — always returns `[]` |
| `extract_grid_constraints`         | `(elements) -> list[ConstraintNode]`                               | **Stub** — always returns `[]` |

> **Note:** The current implementation is a **minimal stub**. Only
> `extract_alignment_constraints` produces output (a single ALIGN_LEFT on the
> first two elements). All other extractors return empty lists. There is no
> `extract_same_size_constraints` function in the actual code (SAME_SIZE is
> defined in the enum but has no extractor yet).

### 3.2 Constraint Semantics (Design Intent)

| Type         | Semantic                                                       | Typical Scenario          |
|--------------|----------------------------------------------------------------|---------------------------|
| ALIGN_LEFT   | \|x1_i - x1_j\| < tolerance                                    | Button groups, list items |
| ALIGN_RIGHT  | \|x2_i - x2_j\| < tolerance                                    | Right-aligned panels      |
| ALIGN_TOP    | \|y1_i - y1_j\| < tolerance                                    | Same-row elements, navbars|
| ALIGN_BOTTOM | \|y2_i - y2_j\| < tolerance                                    | Bottom navs, footers      |
| CENTER_X     | \|cx_i - cx_j\| < tolerance                                    | Center-aligned modals     |
| CENTER_Y     | \|cy_i - cy_j\| < tolerance                                    | Same-line elements        |
| SAME_SIZE    | max(\|w_i-w_j\|/w_j, \|h_i-h_j\|/h_j) < tolerance              | Uniform buttons/icons     |
| SPACING      | \|gap_{i,i+1} - gap_{i+1,i+2}\| < tolerance                    | Equidistant lists/grids   |
| CONTAINMENT  | One bbox fully encloses another (with margin)                   | Container-child hierarchy |
| GRID         | Row + column clustering detects 2D arrangement                  | Tables, icon grids        |

### 3.3 Train vs. Inference

The current stub code makes **no distinction** between train and inference modes.
The design intent (to be implemented) is:

| Dimension         | Train                          | Inference                    |
|-------------------|--------------------------------|------------------------------|
| Element source    | GT bboxes                      | VLM predicted bboxes         |
| Tolerance (eps)   | 0.02 (tight)                   | 0.05 (loose)                 |
| Constraint filter | Keep all                       | Drop low-confidence (weight < 0.3) |

---

## 4. Visualisation

Implemented in `src/bipartite_gnn_gui/graph/visualize.py`.

```python
def plot_bipartite_graph(
    elements: Sequence[ElementNode],
    constraints: Sequence[ConstraintNode],
    ax: Any | None = None,
) -> Any:
    """Placeholder visualisation showing element and constraint counts.

    Args:
        elements:   ElementNode list.
        constraints: ConstraintNode list.
        ax:         Optional matplotlib Axes. Created if None.

    Returns:
        The matplotlib Axes, or None if matplotlib is not available.
    """
```

Current behaviour: renders a text-only matplotlib figure showing element and
constraint counts. Does **not** overlay bboxes on a screenshot, does **not**
render edges between nodes, and has no color-by-type support.

> **Planned:** The full-featured `plot_graph_on_screenshot`, `color_by_element_type`,
> `color_by_constraint_type`, and `export_graph` functions described in the original
> design are deferred to a later iteration.

---

## 5. Graph Augmentation

Implemented in `src/bipartite_gnn_gui/graph/augment.py`.

```python
@dataclass
class GraphAugmenter:
    """Apply light-weight stochastic augmentations."""

    node_dropout_rate: float = 0.0   # Fraction of elements to drop
    jitter_std: float = 0.0           # Std of Gaussian noise on bbox coords

    def augment(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
    ) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Return a copy of the input graph components.

        Currently a no-op pass-through. The dropout_rate and jitter_std
        parameters are accepted but not yet applied.
        """
```

> **Current state:** The `augment()` method is a **pass-through stub** — it
> returns `list(elements), list(constraints)` without applying any stochastic
> transformations. The parameters are stored for future use.

> **Design intent** (not yet implemented):
> - **NodeDropout**: randomly drop elements with probability `node_dropout_rate` to
>   simulate VLM missed detections.
> - **CoordinateJitter**: add Gaussian noise (std=`jitter_std`) to bbox coordinates
>   to simulate VLM localisation error.
> - **ConstraintPerturbation**: randomly flip constraint edge states.

---

## 6. Full HeteroData Key Reference

```python
data = HeteroData()

# ── Node stores ──────────────────────────────────────
data["element"].x       # torch.float32  (N_elem, 5)
                        # Columns: [bbox[0], bbox[1], bbox[2], bbox[3], confidence]

data["constraint"].x    # torch.float32  (N_con, D)  where D = len(params) or 1
                        # Current stub: D = 1 (tolerance value)

# ── Edge stores ──────────────────────────────────────
data["element", "to", "constraint"].edge_index     # torch.long  (2, E)
data["constraint", "to", "element"].edge_index     # torch.long  (2, E)  (flipped)
# No edge_attr on either store.
```

---

## 7. Design Roadmap (Planned Enhancements)

The current implementation is a functional stub. The following enhancements are
part of the design intent and will be implemented in later phases:

| # | Enhancement                                                    | Priority |
|---|----------------------------------------------------------------|----------|
| 1 | Full constraint extraction algorithms for all 10 types (§3.2)  | High     |
| 2 | One-hot type encoding for ElementNode (18+ types) → 23-d       | Medium   |
| 3 | Fixed-dim ConstraintNode features (one-hot type + tolerance + weight) → 12-d | Medium |
| 4 | Edge features: spatial_distance, dx, dy, IoU → 4-d `edge_attr` | Medium   |
| 5 | Train vs. inference mode separation with different tolerances   | Medium   |
| 6 | Full visualisation: `plot_graph_on_screenshot` with bbox overlay + edge rendering | Low |
| 7 | Working augmentation pipeline (dropout, jitter, perturbation)   | Low      |
| 8 | `export_graph` for JSON serialisation of graph structure         | Low      |
| 9 | Edge type semantics (using `EdgeType` enum values in HeteroData keys) | Low |

---

## 修订历史

| 版本 | 日期       | 变更                                                                                       |
|------|------------|--------------------------------------------------------------------------------------------|
| 2.0  | 2026-05-25 | 完全重写以与实际代码对齐：ConstraintType 改为 `str, Enum`；Builder 改为 `BipartiteGraphBuilder`；特征维度更新为 5/1；边键改为 `"to"`；标注当前 stub 状态；新增设计路线图 §7。 |
| 1.0  | 2026-05-25 | 初始版本（仅设计意图）                                                                     |
