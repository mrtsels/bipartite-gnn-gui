# 详细设计 (Detailed Design)

> **Phase 3.1–3.2 — Data Layer & Graph Construction Layer Class Design**
>
> Version: 1.0 | 2026-05-25
>
> This document specifies the class interfaces, algorithm logic, and data flow for the
> data preprocessing pipeline (§1) and graph construction layer (§2). All signatures
> match the **actual code** in `src/bipartite_gnn_gui/`. Stub implementations are
> clearly labelled; planned enhancements are noted separately.
>
> 本文档详细定义了数据预处理管线 (§1) 和图构建层 (§2) 的类接口、算法逻辑和数据流。
> 所有签名均与 `src/bipartite_gnn_gui/` 中的**实际代码**一致。Stub 实现已明确标注；
> 计划中的增强功能另行说明。

---

## 1. 数据层类设计 (Data Layer Class Design)

> **Phase 3.1** — Coordinate normalization, feature extraction, dataset wrapping,
> and batching strategies.
>
> Source files:
> - `src/bipartite_gnn_gui/data/vlm_output.py`
> - `src/bipartite_gnn_gui/data/ground_truth.py`
> - `src/bipartite_gnn_gui/data/preprocess.py`
> - `src/bipartite_gnn_gui/data/dataset.py`
> - `src/bipartite_gnn_gui/utils/bbox.py`

### 1.1 Data Loading: VLM Output

**File:** `src/bipartite_gnn_gui/data/vlm_output.py`
**Status:** ✅ Fully implemented

#### Data Classes

```python
@dataclass
class VLMOutputElement:
    """Single predicted GUI element."""

    bbox: list[float]           # 4-value bounding box (format depends on source)
    label: str = "unknown"      # Element type label string
    confidence: float = 1.0     # Detection confidence ∈ [0, 1]
    text: str | None = None     # OCR text or element text content
    element_id: str | None = None  # Optional unique identifier


@dataclass
class VLMOutput:
    """Container for parsed VLM predictions."""

    elements: list[VLMOutputElement] = field(default_factory=list)
    source: str | None = None          # File path or model name
    image_size: tuple[int, int] | None = None  # (width, height) in pixels

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
```

**Design notes:**

| Aspect | Actual Behaviour |
|--------|-----------------|
| Bbox format | Preserved as-is from the source (no coordinate convention enforced at parse time). The downstream `BipartiteGraphBuilder` also uses bbox values as-is. |
| Label mapping | No taxonomy normalisation at parse time. The raw label string is stored directly. |
| Missing fields | `bbox` defaults to `[0.0, 0.0, 0.0, 0.0]` via `payload.get("bbox", payload.get("box", ...))`. `confidence` defaults to `1.0`. |

#### Loading Functions

```python
def load_vlm_output(source: str | Path | Mapping[str, Any]) -> VLMOutput:
    """Load a VLM output from a path or mapping.

    Accepts:
      - A file path (str/Path) → reads JSON from disk.
      - A dict/Mapping → parses directly (no file I/O).

    The JSON payload is expected to have an "elements" or "predictions" key.
    Each element dict may use keys: bbox/box, label/type, confidence, text, id.
    """

class VLMOutputLoader:
    """Simple callable loader wrapper."""
    def __call__(self, source: str | Path | Mapping[str, Any]) -> VLMOutput:
        return load_vlm_output(source)
```

> **Planned enhancement (design intent):** The requirements doc (`vlm_format.md`) specifies
> model-aware parsers (`parse_qwen_output`, `parse_minimax_output`), a shared 20-type
> taxonomy, and xyxy normalised coordinate convention. The current implementation is a
> simpler, format-agnostic loader that defers these concerns. Phase 4.2.1 will reconcile
> the loader with the full requirements spec.

---

### 1.2 Data Loading: Ground Truth

**File:** `src/bipartite_gnn_gui/data/ground_truth.py`
**Status:** ✅ Fully implemented

#### Data Classes

```python
@dataclass
class GTElement:
    """Single annotated GUI element."""

    bbox: list[float]           # 4-value bounding box (format depends on source)
    label: str = "unknown"      # Element type label
    element_id: str | None = None  # Optional unique identifier


@dataclass
class GroundTruth:
    """Container for annotations."""

    elements: list[GTElement] = field(default_factory=list)
    source: str | None = None          # File path
    image_size: tuple[int, int] | None = None  # (width, height) in pixels
```

**Design decision — `GTElement` is lighter than `VLMOutputElement`:** The ground-truth
struct omits `confidence`, `text`, and `attributes` fields because GT annotations do not
carry detection uncertainty. This asymmetry is intentional and simplifies the matching
code.

#### Loading & Matching Functions

```python
def load_ground_truth(
    source: str | Path | Mapping[str, Any],
) -> GroundTruth:
    """Load annotations from a JSON file or mapping.

    Accepts "elements" or "annotations" as the elements key.
    Supports bbox/box, label/type, id as per-element keys.
    """

def match_elements(
    predicted: Sequence[Mapping[str, Any] | GTElement],
    ground_truth: Sequence[Mapping[str, Any] | GTElement],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Greedily match predicted elements to ground truth by IoU.

    Returns:
        list of (pred_index, gt_index, iou_score) triples.
        Only pairs with IoU ≥ iou_threshold are returned.

    Algorithm:
        For each predicted element (in order):
          1. Compute IoU against all unmatched GT elements.
          2. Select the GT element with maximum IoU.
          3. If max IoU ≥ threshold, record the match and mark GT as matched.
          4. Otherwise, the prediction remains unmatched.

    This is a greedy heuristic, NOT the Hungarian algorithm specified
    in the requirements doc. See §1.2.1 for the planned upgrade path.
    """
```

**Matching algorithm detail:**

```
Input:  predictions P[0..M-1], ground-truth G[0..N-1], threshold τ
Output: matches — list of (pred_idx, gt_idx, iou_score)

gt_remaining ← {0, 1, ..., N-1}
matches ← []

for pred_idx = 0 to M-1:
    best_match ← None
    best_score ← 0.0
    for gt_idx in gt_remaining:
        score ← IoU(P[pred_idx].bbox, G[gt_idx].bbox)
        if score > best_score:
            best_score ← score
            best_match ← gt_idx
    if best_match is not None and best_score ≥ τ:
        matches.append((pred_idx, best_match, best_score))
        gt_remaining.remove(best_match)

return matches
```

**Complexity:** O(M × N) per image. This is acceptable for typical GUI screens
(M, N ≤ 100), but a Hungarian-based implementation (via `scipy.optimize.linear_sum_assignment`)
would be more principled for production use.

> **⚠️ Planned upgrade (design intent):** The requirements doc (`gt_format.md` §6) specifies
> IoU-based Hungarian matching with type-conditioned filtering and explicit FP/FN output.
> The current `match_elements` uses a simple greedy assignment. Phase 4.2.2 will replace
> this with `scipy.optimize.linear_sum_assignment`.

---

### 1.3 Coordinate Normalization

**File:** `src/bipartite_gnn_gui/data/preprocess.py`
**Status:** ✅ Function implemented (no class yet)

#### Current Implementation

```python
def normalize_coordinates(
    box: Sequence[float],   # [x, y, w, h] — absolute pixel values
    width: float,           # image width in pixels
    height: float,          # image height in pixels
) -> list[float]:
    """Normalize absolute coordinates to the [0, 1] range.

    Each component is divided independently:
      x_norm = x / width
      y_norm = y / height
      w_norm = w / width
      h_norm = h / height

    Returns: [x_norm, y_norm, w_norm, h_norm]
    """
```

**Coordinate convention — xywh:**
The function treats the input as `(x, y, w, h)` where `(x, y)` is the **top-left corner**
and `(w, h)` is the **width and height**. Each value is normalised independently by its
corresponding image dimension.

**What the function does NOT do (current limitations):**
- Does not convert between bbox formats (xywh ↔ xyxy ↔ cxcywh).
- Does not collect dataset-wide statistics (mean, std, min, max).
- Does not handle the inverse transform (normalised → pixel) — use
  `bbox_to_tensor` + `apply_delta` from `utils/bbox.py` instead.

**Usage example:**

```python
# Input:  bbox at (100, 200) with size (300, 150) on a 1920×1080 image
result = normalize_coordinates([100, 200, 300, 150], 1920, 1080)
# result = [0.0521, 0.1852, 0.1562, 0.1389]
```

> **⚠️ Planned upgrade — `CoordinateNormalizer` class (design intent):**
>
> The requirements doc (Phase 3.1.1) calls for a `CoordinateNormalizer` class with
> `fit`/`transform` methods that capture dataset-wide statistics. The planned interface:
>
> ```python
> class CoordinateNormalizer:
>     """Stateful normalizer with fit/transform pattern.
>
>     Design intent (not yet implemented):
>       - fit(elements): collect global min/max/mean/std of all coordinates
>         across the training set.
>       - transform(bbox): apply the fitted statistics.
>       - inverse_transform(norm_bbox): map back to original scale.
>
>     This enables:
>       - Z-score normalization (μ=0, σ=1) for better training stability.
>       - Consistent normalization across train/val/test splits.
>       - Inverse transform for applying refined deltas back to pixel space.
>     """
> ```
>
> The current `normalize_coordinates` function handles the simple 0–1 normalisation
> case; the class-based approach adds statistical normalisation for improved model
> convergence.

---

### 1.4 Feature Extraction

**File:** `src/bipartite_gnn_gui/data/preprocess.py`
**Status:** ✅ Function implemented

#### Current Implementation

```python
def extract_element_features(
    element: dict[str, object],
) -> Tensor:
    """Convert a GUI element payload into a small feature tensor.

    Extracts:
      - bbox[:4] — the first 4 values of the "bbox" list (default [0,0,0,0]).
      - confidence — from "confidence" key (default 1.0).

    Returns:
        torch.float32 tensor of shape (5,).
        Values: [bbox[0], bbox[1], bbox[2], bbox[3], confidence]
    """
```

**Feature vector composition:**

| Index | Feature | Source Key | Default |
|-------|---------|-----------|---------|
| 0 | bbox[0] | `element["bbox"][0]` | `0.0` |
| 1 | bbox[1] | `element["bbox"][1]` | `0.0` |
| 2 | bbox[2] | `element["bbox"][2]` | `0.0` |
| 3 | bbox[3] | `element["bbox"][3]` | `0.0` |
| 4 | confidence | `element["confidence"]` | `1.0` |

> **Note:** This function produces the same 5-d feature vector that
> `BipartiteGraphBuilder.build()` constructs internally from `ElementNode` objects
> (it concatenates `element.bbox + [element.confidence]`). The function is a
> convenience helper for code paths that work with raw dicts rather than
> `ElementNode` instances.

**Current limitations (features NOT extracted):**
- No **type embedding** (label → one-hot) is produced.
- No **spatial feature decomposition** (cx, cy, w, h from xyxy) is performed.
- No **relative positioning** features (element-to-element).
- No **text features** (text content is not embedded).

> **⚠️ Planned enhancement (design intent):**
>
> ```python
> # Future feature vector design per the schema doc (§7):
> #   spatial:  [cx, cy, w, h]           — 4 values (centre + size, normalised)
> #   type:     one-hot over 20 labels    — 20 values
> #   confidence: scalar                  — 1 value
> #   Total: 25-d per element node
>
> def extract_spatial_features(bbox_xyxy: Tensor) -> Tensor:
>     """Convert xyxy bbox → (cx, cy, w, h) spatial features.
>
>     cx = (x1 + x2) / 2
>     cy = (y1 + y2) / 2
>     w  = x2 - x1
>     h  = y2 - y1
>     Returns: Tensor of shape (4,).
>     """
>
> def extract_type_embedding(
>     label: str,
>     taxonomy: list[str],
> ) -> Tensor:
>     """Map element type label to one-hot vector.
>
>     Case-insensitive match against the 20-type taxonomy.
>     Unrecognised labels → index 0 ("unknown" slot).
>     Returns: Tensor of shape (len(taxonomy),) with one 1.0.
>     """
> ```

---

### 1.5 BBox Utilities

**File:** `src/bipartite_gnn_gui/utils/bbox.py`
**Status:** ✅ Fully implemented

```python
def bbox_to_tensor(
    bbox: Sequence[float],
    device: torch.device | None = None,
) -> Tensor:
    """Convert a 4-value bbox sequence to a float32 tensor."""

def tensor_to_bbox(tensor: Tensor) -> Tuple[float, float, float, float]:
    """Convert a bbox tensor to a Python tuple."""

def xywh_to_xyxy(box: Tensor) -> Tensor:
    """Convert [x, y, w, h] → [x1, y1, x2, y2].

    x2 = x + w,  y2 = y + h
    """

def xyxy_to_xywh(box: Tensor) -> Tensor:
    """Convert [x1, y1, x2, y2] → [x, y, w, h].

    w = x2 - x1,  h = y2 - y1
    """

def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """Compute pairwise IoU between two box tensors.

    Supports:
      - Broadcasting: (N, 4) vs (M, 4) → (N, M) IoU matrix.
      - Auto-detection: xywh inputs are detected and converted to xyxy.
        Detection heuristic: if x2 < x1 for any box, treat as xywh.

    Returns:
        Tensor of IoU values in [0, 1]. Division-by-zero is handled
        by returning 0 where union ≤ 0.
    """

def apply_delta(box: Tensor, delta: Tensor) -> Tensor:
    """Apply a refinement delta to an [x, y, w, h] box.

    Returns: box + delta (element-wise addition).
    This is used by the model's CoordinateRefinementHead to adjust
    predicted bbox coordinates.
    """
```

**Coordinate format conventions:**

| Function | Input | Output |
|----------|-------|--------|
| `xywh_to_xyxy` | `[x, y, w, h]` | `[x, y, x+w, y+h]` |
| `xyxy_to_xywh` | `[x1, y1, x2, y2]` | `[x1, y1, x2-x1, y2-y1]` |
| `compute_iou` | Auto-detect xywh or xyxy | N/A |
| `apply_delta` | xywh only | xywh |

> **Design decision — auto-detection in `compute_iou`:** Instead of requiring the caller
> to specify the bbox format, `compute_iou` checks whether `x2 < x1` for any box across
> the batch. If so, it interprets the input as xywh and converts to xyxy before computing
> IoU. This is a pragmatic choice that avoids format bugs but should not be relied upon
> for performance-critical paths.

---

### 1.6 GUIDataset & DataModule

**File:** `src/bipartite_gnn_gui/data/dataset.py`
**Status:** ✅ Fully implemented (stub — no graph construction in dataset)

#### GUIDataset

```python
class GUIDataset(Dataset):
    """Simple dataset pairing VLM output with ground truth.

    Stores samples as a plain list of dicts. No graph construction
    is performed in the dataset; graph building happens in the
    training loop or collate function (planned).

    This design keeps the dataset agnostic to PyTorch Geometric,
    allowing it to be used in environments where PyG is not installed.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Initialize with a sequence of sample dicts.

        Each sample dict is expected to contain:
          - "vlm": VLMOutput (or dict of element data)
          - "gt": GroundTruth (or dict of annotation data)
          - Optional metadata keys (image_path, source, etc.)

        If samples is None, the dataset starts empty (useful for
        deferred loading patterns where samples are added later).
        """
        self.samples = list(samples or [])

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return the sample dict at the given index.

        This is a pure passthrough — no transformation or graph
        construction is performed. The caller (training loop) is
        responsible for building the HeteroData graph from the
        returned dict.

        Returns:
            The sample dict as-is.
        """
        return self.samples[index]
```

**Why a passthrough dataset?**
The current design intentionally delays graph construction to the training loop for two
reasons:

1. **Flexibility:** The training loop can choose whether to build graphs eagerly or
   lazily, and can apply different augmentations or constraint extraction strategies
   depending on mode (train vs. eval).
2. **PyG independence:** The dataset module remains importable without `torch_geometric`,
   enabling lightweight data inspection and CLI tools.

The cost is that each sample is a raw dict rather than a `HeteroData` object, shifting
complexity to the caller.

#### Collation & DataLoader

```python
def collate_variable_elements(
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep variable-size samples as a list.

    Because each GUI screenshot has a different number of elements
    (anywhere from 1 to 100+), standard tensor stacking along a batch
    dimension is not possible. This collate function simply returns
    the list as-is.

    Returns:
        The same list — a pure identity collation.
    """
    return batch


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader for variable-size GUI samples.

    Uses collate_variable_elements to preserve per-sample structure.
    Typical usage with batch_size=1 (since samples cannot be stacked).

    Args:
        dataset: Any PyTorch Dataset (typically GUIDataset).
        batch_size: Number of samples per batch (default 1).
        shuffle: Whether to shuffle between epochs.

    Returns:
        A torch.utils.data.DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_variable_elements,
    )
```

**Batching strategy — list-of-dicts:**

```
DataLoader with batch_size=1
  └─ __iter__ yields:
       [{"vlm": VLMOutput(...), "gt": GroundTruth(...)}]
       # list of 1 dict

DataLoader with batch_size=4
  └─ __iter__ yields:
       [sample_0, sample_1, sample_2, sample_3]
       # list of 4 dicts — each dict has different element counts
```

The training loop iterates over the list item-by-item and builds a separate
`HeteroData` graph for each. This is viable when the number of elements per
sample is moderate (≤ 100, as is typical for GUI screens).

> **⚠️ Planned upgrade — batched graph construction (design intent):**
>
> PyTorch Geometric supports mini-batching of `HeteroData` objects via
> `torch_geometric.loader.DataLoader`, which concatenates adjacency matrices
> diagonally to form a single large disconnected graph. A future iteration will:
>
> 1. Move graph construction into the dataset's `__getitem__`:
>    `GUIDataset.__getitem__(idx) -> HeteroData`
> 2. Replace `collate_variable_elements` with PyG's built-in collation
>    (`torch_geometric.loader.dataloader.Collater`).
> 3. Enable efficient mini-batch training with `batch_size > 1`.
>
> This upgrade depends on PyG being available and the `BipartiteGraphBuilder`
> being fast enough for on-the-fly construction during data loading.

#### GUIDataModule

```python
@dataclass
class GUIDataModule:
    """Lightweight container for train/val/test loaders.

    Holds three GUIDataset instances and provides factory methods
    for creating DataLoaders with the appropriate shuffle settings.
    Analogous to PyTorch Lightning's LightningDataModule but without
    the Lightning dependency.
    """

    train_dataset: GUIDataset | None = None
    val_dataset: GUIDataset | None = None
    test_dataset: GUIDataset | None = None
    batch_size: int = 1

    def train_dataloader(self) -> DataLoader | None:
        """Returns DataLoader with shuffle=True, or None if no train set."""
        return (
            None if self.train_dataset is None
            else create_dataloader(
                self.train_dataset, batch_size=self.batch_size, shuffle=True
            )
        )

    def val_dataloader(self) -> DataLoader | None:
        """Returns DataLoader with shuffle=False, or None if no val set."""
        return (
            None if self.val_dataset is None
            else create_dataloader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False
            )
        )

    def test_dataloader(self) -> DataLoader | None:
        """Returns DataLoader with shuffle=False, or None if no test set."""
        return (
            None if self.test_dataset is None
            else create_dataloader(
                self.test_dataset, batch_size=self.batch_size, shuffle=False
            )
        )
```

---

### 1.7 Data Layer Data Flow Summary

```
                       ┌──────────────────────┐
                       │   VLM JSON / dict     │
                       │   or GT JSON / dict   │
                       └──────────┬───────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   load_vlm_output()       │
                    │   load_ground_truth()     │
                    │   → VLMOutput / GT        │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   normalize_coordinates() │  ← optional (if pixel coords)
                    │   extract_element_        │  ← optional (raw dict path)
                    │     features()            │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   GUIDataset(samples)     │
                    │   → stores as list[dict]  │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   create_dataloader()     │
                    │   + collate_variable_     │
                    │     elements()            │
                    │   → list[dict] per batch  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  Training / Eval Loop    │
                    │  (graph construction     │
                    │   happens here)          │
                    └──────────────────────────┘
```

---

## 2. 图构建层类设计 (Graph Construction Layer Class Design)

> **Phase 3.2** — Schema definitions, constraint extraction, graph building,
> visualization, and augmentation.
>
> Source files:
> - `src/bipartite_gnn_gui/graph/schema.py`
> - `src/bipartite_gnn_gui/graph/constraints.py`
> - `src/bipartite_gnn_gui/graph/builder.py`
> - `src/bipartite_gnn_gui/graph/visualize.py`
> - `src/bipartite_gnn_gui/graph/augment.py`

### 2.1 Graph Schema

**File:** `src/bipartite_gnn_gui/graph/schema.py`
**Status:** ✅ Fully implemented

#### ConstraintType Enum

```python
class ConstraintType(str, Enum):
    """Supported spatial constraints (string-valued enum).

    String values serve as canonical identifiers for serialisation,
    logging, and human-readable output across the codebase.
    """

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

# 10 constraint types total, covering:
#   Alignment (6): ALIGN_LEFT, ALIGN_RIGHT, ALIGN_TOP, ALIGN_BOTTOM,
#                   CENTER_X, CENTER_Y
#   Size (1):      SAME_SIZE
#   Layout (2):    SPACING, GRID
#   Hierarchy (1): CONTAINMENT
```

#### EdgeType Enum

```python
class EdgeType(str, Enum):
    """Edge categories for the bipartite graph.

    These serve as semantic labels documenting the two directed
    edge types. The actual HeteroData edge keys use the shorter
    "to" convention (e.g., ("element", "to", "constraint")),
    NOT the enum values directly.
    """

    ELEMENT_TO_CONSTRAINT = "element_to_constraint"
    CONSTRAINT_TO_ELEMENT = "constraint_to_element"
```

#### ElementNode

```python
@dataclass
class ElementNode:
    """Node describing a GUI element in the bipartite graph.

    This is the unified graph-level representation. It is similar
    to VLMOutputElement but lives in the graph module to avoid
    circular imports between data and graph layers.
    """

    bbox: list[float]                    # 4-value bbox (format determined by upstream)
    label: str = "unknown"               # Element type label string
    confidence: float = 1.0              # Detection confidence ∈ [0, 1]
    element_id: str | None = None        # Optional unique identifier
    features: dict[str, float] = field(default_factory=dict)  # Extra key-value features
```

#### ConstraintNode

```python
@dataclass
class ConstraintNode:
    """Node describing a spatial constraint in the bipartite graph.

    Constraints are the second node type in the bipartite graph,
    connecting related elements through directed edges. Each
    constraint links a set of source and target element indices.
    """

    constraint_type: ConstraintType                      # One of the 10 enum values
    source_indices: list[int] = field(default_factory=list)  # Indices of source elements
    target_indices: list[int] = field(default_factory=list)  # Indices of target elements
    params: dict[str, float] = field(default_factory=dict)   # Constraint-specific parameters
```

**Semantic distinction — `source_indices` vs `target_indices`:**
For most constraint types (alignment, same-size, spacing, grid), both lists contain
the same indices — the constraint binds a set of elements together symmetrically.
For directional constraints like CONTAINMENT, `source_indices` refers to the
container element and `target_indices` refers to the contained elements. The builder
treats both lists identically for edge creation (union of both lists).

---

### 2.2 Constraint Extraction

**File:** `src/bipartite_gnn_gui/graph/constraints.py`
**Status:** ⚠️ Stub — only alignment produces output

#### Public Entry Point

```python
def extract_all_constraints(
    elements: Sequence[ElementNode],
) -> list[ConstraintNode]:
    """Extract all heuristic constraints from a list of GUI elements.

    Dispatches to four sub-extractors and concatenates their outputs.
    The current implementation produces at most one constraint
    (an ALIGN_LEFT on elements [0, 1] when N ≥ 2).

    Args:
        elements: Ordered list of ElementNode objects (may be empty).

    Returns:
        Flat list of ConstraintNode objects (may be empty).

    Implementation:
        constraints = []
        constraints.extend(extract_alignment_constraints(elements))
        constraints.extend(extract_containment_constraints(elements))
        constraints.extend(extract_spacing_constraints(elements))
        constraints.extend(extract_grid_constraints(elements))
        return constraints
    """
```

#### Sub-Extractors — Current Implementation

| Function | Signature | Status | Current Output |
|----------|-----------|--------|----------------|
| `extract_alignment_constraints` | `(elements, tolerance=0.02) -> list[ConstraintNode]` | **Stub** | Single ALIGN_LEFT on elements [0,1] if N≥2; otherwise `[]` |
| `extract_containment_constraints` | `(elements) -> list[ConstraintNode]` | **Stub** | Always `[]` |
| `extract_spacing_constraints` | `(elements, tolerance=0.02) -> list[ConstraintNode]` | **Stub** | Always `[]` |
| `extract_grid_constraints` | `(elements) -> list[ConstraintNode]` | **Stub** | Always `[]` |

**Current alignment stub detail:**

```python
def extract_alignment_constraints(
    elements: Sequence[ElementNode],
    tolerance: float = 0.02,
) -> list[ConstraintNode]:
    """Extract a small set of alignment constraints.

    Current behaviour (stub):
      - If len(elements) < 2 → return [].
      - Otherwise → return ONE ConstraintNode:
          type: ALIGN_LEFT
          source_indices: [0, 1]
          target_indices: [0, 1]
          params: {"tolerance": 0.02}

    This is a placeholder. The full implementation will perform O(N²)
    pairwise edge comparisons for all 6 alignment types.
    """
    if len(elements) < 2:
        return []
    return [
        ConstraintNode(
            constraint_type=ConstraintType.ALIGN_LEFT,
            source_indices=[0, 1],
            target_indices=[0, 1],
            params={"tolerance": tolerance},
        )
    ]
```

#### Planned Full Algorithms (Design Intent)

Each sub-extractor will implement element-pair comparison with tolerance thresholds.
The key design consideration is the O(N²) pairwise comparison cost for alignment,
containment, and same-size checks (tractable for N ≤ 100).

**Alignment & Same-Size Extraction (planned):**

```
Input: elements[0..N-1], tolerance ε
Output: list[ConstraintNode]

for i in 0..N-1:
    for j in i+1..N-1:
        xi1, yi1, xi2, yi2 = elements[i].bbox
        xj1, yj1, xj2, yj2 = elements[j].bbox

        wi = xi2 - xi1;  hi = yi2 - yi1
        wj = xj2 - xj1;  hj = yj2 - yj1
        cxi = (xi1 + xi2) / 2;  cyi = (yi1 + yi2) / 2
        cxj = (xj1 + xj2) / 2;  cyj = (yj1 + yj2) / 2

        if |xi1 - xj1| < ε → ALIGN_LEFT(i, j)
        if |xi2 - xj2| < ε → ALIGN_RIGHT(i, j)
        if |yi1 - yj1| < ε → ALIGN_TOP(i, j)
        if |yi2 - yj2| < ε → ALIGN_BOTTOM(i, j)
        if |cxi - cxj| < ε → CENTER_X(i, j)
        if |cyi - cyj| < ε → CENTER_Y(i, j)
        if max(|wi-wj|/max(wj,ε), |hi-hj|/max(hj,ε)) < ε → SAME_SIZE(i, j)
```

**Containment Extraction (planned):**

```
Input: elements[0..N-1]
Output: list[ConstraintNode]

for i in 0..N-1:
    for j in 0..N-1 (j ≠ i):
        if xi1 ≤ xj1 and yi1 ≤ yj1 and xi2 ≥ xj2 and yi2 ≥ yj2:
            → CONTAINMENT(container=i, contained=j, source=[i], target=[j])
```

**Spacing Extraction (planned):**

```
Input: elements[0..N-1], tolerance ε
Output: list[ConstraintNode]

For horizontal spacing:
    Sort elements by x1 (left edge).
    For each consecutive triple (a, b, c):
        gap_ab = b.x1 - a.x2  (horizontal gap between a and b)
        gap_bc = c.x1 - b.x2  (horizontal gap between b and c)
        if gap_ab > 0 and gap_bc > 0:
            if |gap_ab - gap_bc| / max(gap_ab, ε) < ε:
                → SPACING(a, b, c, axis="h")

For vertical spacing:
    Sort elements by y1 (top edge).
    Apply the same logic with vertical gaps.
```

**Grid Extraction (planned):**

```
Input: elements[0..N-1], tolerance ε
Output: list[ConstraintNode]

1. Cluster elements by cy (row detection):
   Sort by cy. Group elements whose cy values differ by < ε.

2. Within each row, cluster by cx (column detection):
   Sort by cx. Group elements whose cx values differ by < ε.

3. If a row has ≥ 3 columns OR a column has ≥ 3 rows:
   → GRID(row_indices=list of row element indices)
```

> **Note:** No `extract_same_size_constraints` function exists as a standalone
> extractor. SAME_SIZE checking is part of the `extract_alignment_constraints`
> function (as shown in the pairwise algorithm above), since it shares the same
> O(N²) pairwise comparison pattern.

#### Train vs. Inference Mode

The current code makes **no distinction** between training and inference modes.
The design intent (from the schema doc) is:

| Dimension | Train Mode | Inference Mode |
|-----------|-----------|----------------|
| Element source | GT bboxes (accurate) | VLM predicted bboxes (noisy) |
| Tolerance (ε) | 0.02 (tight) | 0.05 (loose) |
| Constraint filter | Keep all | Drop low-confidence (weight < 0.3) |

This distinction will be implemented by adding a `mode` parameter or by providing
separate extraction functions for train and inference contexts.

---

### 2.3 BipartiteGraphBuilder

**File:** `src/bipartite_gnn_gui/graph/builder.py`
**Status:** ✅ Fully implemented

#### Class Interface

```python
class BipartiteGraphBuilder:
    """Build a bipartite graph from elements and constraints.

    The builder is stateless — it has no constructor parameters,
    no mode flags, and no internal state. Each call to build() is
    fully independent and returns a new HeteroData object.

    Key design decisions:
      - The builder does NOT extract constraints internally.
        Constraint extraction is a separate step whose output
        feeds into build().
      - There are no private _build_* helper methods. All logic
        lives in the single build() method for clarity.
      - Edge features (edge_attr) are NOT computed. The current
        graph stores only edge indices.
    """

    def build(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
    ) -> HeteroData:
        """Create a graph object with node and edge stores.

        Args:
            elements:   List of ElementNode objects (length N_elem ≥ 0).
            constraints: List of ConstraintNode objects (length N_con ≥ 0).

        Returns:
            HeteroData with the stores documented in §2.3.1–2.3.3.
        """
```

#### 2.3.1 Element Node Store

```python
# data["element"].x
# Shape: (N_elem, 5), dtype: torch.float32
# Each row: [bbox[0], bbox[1], bbox[2], bbox[3], confidence]

element_features = [
    element.bbox + [element.confidence]
    for element in elements
]
data["element"].x = torch.tensor(
    element_features, dtype=torch.float32
) if element_features else torch.zeros((0, 5), dtype=torch.float32)
```

| Column | Index | Source | Description |
|--------|-------|--------|-------------|
| bbox[0] | 0 | `element.bbox[0]` | First bbox coordinate |
| bbox[1] | 1 | `element.bbox[1]` | Second bbox coordinate |
| bbox[2] | 2 | `element.bbox[2]` | Third bbox coordinate |
| bbox[3] | 3 | `element.bbox[3]` | Fourth bbox coordinate |
| confidence | 4 | `element.confidence` | Detection confidence |

When `elements` is empty: `torch.zeros((0, 5), dtype=torch.float32)`.

#### 2.3.2 Constraint Node Store

```python
# data["constraint"].x
# Shape: (N_con, D), dtype: torch.float32
# D = len(params) if params is non-empty, else 1
# Each row: list(constraint.params.values()) or [0.0]

constraint_features = [
    list(constraint.params.values()) or [0.0]
    for constraint in constraints
]
data["constraint"].x = torch.tensor(
    constraint_features, dtype=torch.float32
) if constraint_features else torch.zeros((0, 1), dtype=torch.float32)
```

**Variable feature dimension:** Since `params` is a free-form dict, the feature
dimension `D` varies by constraint type. In the current stub, alignment constraints
emit `params={"tolerance": 0.02}` so D = 1. Future implementations with richer
params (tolerance + weight + axis) would produce D ≥ 2. This variability means the
downstream GNN model must handle variable input dimensions or a fixed-dim encoding
must be introduced.

When `constraints` is empty: `torch.zeros((0, 1), dtype=torch.float32)`.

#### 2.3.3 Edge Stores

Two directed edge types connect the bipartite nodes:

```python
# Forward: element → constraint
# Shape: (2, E), dtype: torch.long
data["element", "to", "constraint"].edge_index

# Reverse: constraint → element (flipped forward)
# Shape: (2, E), dtype: torch.long
data["constraint", "to", "element"].edge_index
```

**Edge construction algorithm:**

```
source_index = []
target_index = []

for constraint_idx, constraint in enumerate(constraints):
    union_indices = constraint.source_indices + constraint.target_indices
    for element_idx in set(union_indices):  # deduplicate
        source_index.append(element_idx)    # element → constraint
        target_index.append(constraint_idx)

edge_index = torch.tensor(
    [source_index, target_index], dtype=torch.long
) if source_index else torch.zeros((2, 0), dtype=torch.long)

data["element", "to", "constraint"].edge_index = edge_index
data["constraint", "to", "element"].edge_index = torch.flip(
    edge_index, dims=[0]
) if edge_index.numel() else torch.zeros((2, 0), dtype=torch.long)
```

**Edge count `E`:** Equals the sum over all constraints of
`|source_indices ∪ target_indices|`. No edge features (`edge_attr`) are
computed — the stores contain only `edge_index`.

#### 2.3.4 Tensor Shape Examples

**Example 1: 3 elements, 1 ALIGN_LEFT constraint linking elements [0, 1]**

```
N_elem = 3, N_con = 1

data["element"].x = tensor([
    [0.10, 0.20, 0.30, 0.35, 0.95],   # element 0
    [0.10, 0.40, 0.30, 0.55, 0.88],   # element 1 (aligned left with 0)
    [0.50, 0.10, 0.70, 0.20, 0.92],   # element 2 (no constraint)
])  # shape (3, 5)

data["constraint"].x = tensor([[0.02]])  # shape (1, 1) — tolerance

data["element", "to", "constraint"].edge_index = tensor([
    [0, 1],   # source: element indices
    [0, 0],   # target: constraint index
])  # shape (2, 2) — 2 edges

data["constraint", "to", "element"].edge_index = tensor([
    [0, 0],   # source: constraint index (flipped)
    [0, 1],   # target: element indices (flipped)
])  # shape (2, 2)
```

**Example 2: Empty case — no elements or constraints**

```
N_elem = 0, N_con = 0

data["element"].x    = tensor([])  shape (0, 5)
data["constraint"].x = tensor([])  shape (0, 1)
data["element", "to", "constraint"].edge_index  = tensor([])  shape (2, 0)
data["constraint", "to", "element"].edge_index  = tensor([])  shape (2, 0)
```

**Example 3: 4 elements, 2 constraints (ALIGN_LEFT on [0,1,2], CONTAINMENT on [3,0])**

```
N_elem = 4, N_con = 2

Constraint 0: ALIGN_LEFT, source=[0,1,2], target=[0,1,2]
  → edges: (0→0), (1→0), (2→0)

Constraint 1: CONTAINMENT, source=[3], target=[0]
  → edges: (3→1), (0→1)

data["element", "to", "constraint"].edge_index = tensor([
    [0, 1, 2, 3, 0],
    [0, 0, 0, 1, 1],
])  # shape (2, 5)
```

#### 2.3.5 Fallback HeteroData

When PyTorch Geometric is not installed, a minimal dictionary-based fallback is
provided inline in `builder.py`:

```python
class HeteroData(dict):
    """Minimal fallback when PyG is unavailable.

    Supports dict-style key access and attribute-style access for
    compatibility. Does NOT support PyG-specific features:
      - No message passing (to_hetero, SAGEConv, etc.)
      - No batching (Batch.from_data_list)
      - No edge_index validation or utility methods
    """

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value
```

This fallback enables the builder to be imported and used in lightweight
environments (e.g., data preprocessing scripts, CI linting) without installing
PyTorch Geometric. All graph construction logic works identically; only the
container type changes.

#### 2.3.6 Complete HeteroData Key Reference

```
data = HeteroData()

# ── Node stores ──────────────────────────────────────
data["element"].x       # (N_elem, 5)   float32
                        # [bbox[0], bbox[1], bbox[2], bbox[3], confidence]

data["constraint"].x    # (N_con, D)   float32
                        # D = len(params) or 1
                        # list(params.values()) or [0.0]

# ── Edge stores ──────────────────────────────────────
data["element", "to", "constraint"].edge_index     # (2, E)  long
data["constraint", "to", "element"].edge_index     # (2, E)  long  (flipped)

# No edge_attr on either store.
```

---

### 2.4 Graph Visualization

**File:** `src/bipartite_gnn_gui/graph/visualize.py`
**Status:** ⚠️ Stub — text-only placeholder

#### Current Implementation

```python
def plot_bipartite_graph(
    elements: Sequence[ElementNode],
    constraints: Sequence[ConstraintNode],
    ax: Any | None = None,
) -> Any:
    """Plot a simple placeholder visualization.

    Args:
        elements:   List of ElementNode objects.
        constraints: List of ConstraintNode objects.
        ax:         Optional matplotlib Axes. Created if None.

    Returns:
        The matplotlib Axes object, or None if matplotlib is not available.

    Current behaviour:
        - Creates a matplotlib figure (6×4 inches) if no ax is provided.
        - Sets the title to "Elements: N | Constraints: M".
        - Hides the axes (ax.axis("off")).
        - Does NOT render:
            - Bounding box overlays
            - Constraint-to-element edges
            - Screenshot backgrounds
            - Color-coded element/constraint types
    """
```

**What the stub renders:**

```
┌─────────────────────────────────────┐
│  Elements: 14 | Constraints: 3      │
│                                     │
│          (empty figure)             │
│                                     │
└─────────────────────────────────────┘
```

#### Planned Visualization Suite (Design Intent)

The original design document specifies four visualization functions. None are yet
implemented. The planned interfaces and behaviors:

| Function | Planned Signature | Description |
|----------|-------------------|-------------|
| `plot_graph_on_screenshot` | `(image, elements, constraints, ax) -> Axes` | Overlay bbox rectangles and constraint edges on the original screenshot. Element nodes rendered as colored rectangles; constraint nodes as labeled diamond markers; edges as lines connecting related elements. |
| `color_by_element_type` | `(label: str) -> str` | Map each of the 20 element types to a distinct color from a colormap (e.g., matplotlib's tab20). Returns hex color string. |
| `color_by_constraint_type` | `(ctype: ConstraintType) -> str` | Map each of the 10 constraint types to a distinct color. Returns hex color string. |
| `export_graph` | `(data: HeteroData, path: str \| Path) -> None` | Serialize the graph structure (nodes, edges, types) to JSON for external visualization tools. |

---

### 2.5 Graph Augmentation

**File:** `src/bipartite_gnn_gui/graph/augment.py`
**Status:** ⚠️ Stub — pass-through (no-op)

#### Current Implementation

```python
@dataclass
class GraphAugmenter:
    """Apply light-weight stochastic augmentations to graph components.

    Attributes:
        node_dropout_rate (float): Fraction of elements to randomly
            drop. Range [0, 1]. Default 0.0 (no dropout).
            NOT YET APPLIED.
        jitter_std (float): Standard deviation of Gaussian noise
            added to bbox coordinates. Default 0.0 (no jitter).
            NOT YET APPLIED.
    """

    node_dropout_rate: float = 0.0
    jitter_std: float = 0.0

    def augment(
        self,
        elements: Sequence[ElementNode],
        constraints: Sequence[ConstraintNode],
    ) -> tuple[list[ElementNode], list[ConstraintNode]]:
        """Return a copy of the input graph components.

        Current behaviour (stub):
            Returns list(elements), list(constraints) — a shallow copy
            with no transformations applied. The dropout_rate and
            jitter_std parameters are stored but have no effect.

        Args:
            elements:   Input element nodes (N_elem items).
            constraints: Input constraint nodes (N_con items).

        Returns:
            A (elements, constraints) tuple — currently identical to
            the input (shallow copy only).
        """
        return list(elements), list(constraints)
```

#### Planned Augmentation Pipeline (Design Intent)

The augmenter will apply three stochastic transformations in sequence when the
respective parameters are non-zero. The augmentation is applied to the training
data to simulate VLM-like errors; the original (unaugmented) data serves as the
ground truth for loss computation.

**1. NodeDropout** — Simulate VLM missed detections:

```
For each element e at index i:
    if random() < node_dropout_rate:
        Remove element i from the elements list.
        Re-index remaining elements.
        Remove any constraint c where i ∈ c.source_indices ∪ c.target_indices.
        Re-index remaining constraint source/target indices.

Returns: (reduced_elements, filtered_constraints)
```

**2. CoordinateJitter** — Simulate VLM localization error:

```
For each surviving element e:
    for j in 0..3:
        e.bbox[j] += Normal(μ=0, σ=jitter_std)
    Clamp e.bbox values to [0, 1] range.
```

**3. ConstraintPerturbation** — Simulate constraint extraction errors:

```
Parameter: perturbation_rate (float, 0.0 to 1.0)

For each constraint c:
    if random() < perturbation_rate:
        With 50% probability:
            Randomly reassign c.constraint_type to another ConstraintType.
        With 50% probability:
            Remove c entirely.
```

The augmentation is only applied during training, not validation or testing.
The `GraphAugmenter` is designed to be called between constraint extraction and
graph building:

```
elements → extract_all_constraints() → constraints
    → augmenter.augment(elements, constraints) → (aug_elements, aug_constraints)
    → builder.build(aug_elements, aug_constraints) → HeteroData (training input)
```

---

### 2.6 Graph Layer Data Flow Summary

```
┌────────────────────┐    ┌─────────────────────┐
│  VLMOutputElement[]│    │  GTElement[]        │
│  (or ElementNode[])│    │  (Ground Truth)     │
└─────────┬──────────┘    └──────────┬──────────┘
          │                          │
          └──────────┬───────────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │ ElementNode[]       │  ← Converted to unified graph schema
          │ (N_elem items)      │
          └─────────┬───────────┘
                    │
          ┌─────────┴───────────┐
          │                     │
          ▼                     ▼
┌─────────────────────┐  ┌─────────────────────┐
│ extract_all_        │  │ GraphAugmenter      │
│ constraints()       │  │ .augment()          │
│                     │  │                     │
│ → extract_alignment │  │ Currently pass-     │
│ → extract_containment│ │ through. Planned:   │
│ → extract_spacing   │  │ - NodeDropout       │
│ → extract_grid      │  │ - CoordinateJitter  │
│                     │  │ - ConstraintPerturb │
│ → ConstraintNode[]  │  │                     │
│   (N_con items)     │  │ → (Elem[], Con[])   │
└─────────┬───────────┘  └──────────┬──────────┘
          │                          │
          └──────────────┬───────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ BipartiteGraph-     │
              │ Builder.build()     │
              │                     │
              │ → HeteroData        │
              │                     │
              │  data["element"].x  │
              │    (N_elem, 5)      │
              │  data["constraint"].x│
              │    (N_con, D)       │
              │  data["e","to","c"] │
              │    .edge_index (2,E)│
              │  data["c","to","e"] │
              │    .edge_index (2,E)│
              └─────────┬───────────┘
                        │
                        ▼
              ┌─────────────────────┐
              │ plot_bipartite_     │
              │ graph()  (stub)     │
              │ → matplotlib Axes   │
              │   text-only title   │
              └─────────────────────┘
```

---

## 3. Implementation Status Summary

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `VLMOutputElement` / `VLMOutput` | `data/vlm_output.py` | ✅ Implemented | Format-agnostic loader |
| `load_vlm_output` / `VLMOutputLoader` | `data/vlm_output.py` | ✅ Implemented | Accepts path or dict |
| `GTElement` / `GroundTruth` | `data/ground_truth.py` | ✅ Implemented | Lighter than VLM equivalent |
| `load_ground_truth` | `data/ground_truth.py` | ✅ Implemented | Accepts path or dict |
| `match_elements` | `data/ground_truth.py` | ✅ Implemented | Greedy (planned: Hungarian) |
| `normalize_coordinates` | `data/preprocess.py` | ✅ Implemented | Standalone function, xywh → [0,1] |
| `extract_element_features` | `data/preprocess.py` | ✅ Implemented | 5-d tensor (bbox[:4] + conf) |
| `GUIDataset` | `data/dataset.py` | ✅ Implemented | Passthrough — no graph in `__getitem__` |
| `collate_variable_elements` | `data/dataset.py` | ✅ Implemented | Identity collation |
| `create_dataloader` | `data/dataset.py` | ✅ Implemented | Uses identity collate_fn |
| `GUIDataModule` | `data/dataset.py` | ✅ Implemented | Train/val/test loader factory |
| `bbox_to_tensor` / `tensor_to_bbox` | `utils/bbox.py` | ✅ Implemented | |
| `xywh_to_xyxy` / `xyxy_to_xywh` | `utils/bbox.py` | ✅ Implemented | |
| `compute_iou` | `utils/bbox.py` | ✅ Implemented | Auto-detects xywh vs xyxy |
| `apply_delta` | `utils/bbox.py` | ✅ Implemented | xywh delta application |
| `ConstraintType` (10 values) | `graph/schema.py` | ✅ Implemented | `str, Enum` |
| `EdgeType` (2 values) | `graph/schema.py` | ✅ Implemented | `str, Enum` |
| `ElementNode` (5 fields) | `graph/schema.py` | ✅ Implemented | |
| `ConstraintNode` (4 fields) | `graph/schema.py` | ✅ Implemented | |
| `extract_alignment_constraints` | `graph/constraints.py` | ⚠️ Stub | Single ALIGN_LEFT if N≥2 |
| `extract_containment_constraints` | `graph/constraints.py` | ⚠️ Stub | Returns `[]` |
| `extract_spacing_constraints` | `graph/constraints.py` | ⚠️ Stub | Returns `[]` |
| `extract_grid_constraints` | `graph/constraints.py` | ⚠️ Stub | Returns `[]` |
| `extract_all_constraints` | `graph/constraints.py` | ⚠️ Stub | Dispatches to above |
| `BipartiteGraphBuilder.build()` | `graph/builder.py` | ✅ Implemented | Element (N,5) + Constraint (N,D) |
| `HeteroData` fallback | `graph/builder.py` | ✅ Implemented | Dict-based when PyG absent |
| `plot_bipartite_graph` | `graph/visualize.py` | ⚠️ Stub | Text-only title |
| `GraphAugmenter.augment()` | `graph/augment.py` | ⚠️ Stub | Pass-through, params unused |

**Legend:** ✅ = Implemented and functional | ⚠️ Stub = Minimal placeholder, returns empty/identity

---

## 4. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Stateless `BipartiteGraphBuilder`** | No constructor, no mode flags, no internal state. Each `build()` call is self-contained. This simplifies testing (no setup/teardown) and makes the builder trivially parallelizable. |
| **Separate constraint extraction from graph building** | The builder accepts pre-extracted `ConstraintNode` lists. This allows the extraction strategy to vary (train vs. inference, different tolerance values) without changing the builder. |
| **Passthrough `GUIDataset` (no graph in `__getitem__`)** | Keeps dataset module free of PyG dependency. Graph construction happens in the training loop, giving the caller full control over augmentation and constraint strategies. |
| **Identity `collate_variable_elements`** | GUI screens have variable element counts (1–100+), making tensor stacking impossible. The identity collation returns a plain list; the training loop handles each sample individually. |
| **Greedy matching (`match_elements`)** | A pragmatic choice for early development. The Hungarian algorithm (via `scipy`) is planned for Phase 4.2.2. |
| **Raw bbox in element features (no type embedding)** | The 5-d feature vector ([bbox[0..3], confidence]) is the minimal viable representation. Type one-hot embedding and spatial feature decomposition are planned enhancements. |
| **Variable constraint feature dimension** | Constraint `params` is a free-form dict, so the feature dimension varies per constraint type. A fixed-dim encoding (one-hot type + tolerance + weight → 12-d) is planned. |
| **No edge features (`edge_attr`)** | The current graph stores only edge indices. Edge features (spatial distance, dx, dy, IoU) are a planned enhancement for richer message passing. |

---

## 3. 模型层类设计 (Model Layer Class Design)

> **Phase 3.3** — Heterogeneous GNN encoder, prediction heads, full model assembly,
> and loss functions.
>
> Source files:
> - `src/bipartite_gnn_gui/model/encoder.py`
> - `src/bipartite_gnn_gui/model/heads.py`
> - `src/bipartite_gnn_gui/model/losses.py`
> - `src/bipartite_gnn_gui/model/model.py`

### 3.1 BipartiteGraphSAGE Encoder

**File:** `src/bipartite_gnn_gui/model/encoder.py`
**Status:** ⚠️ Stub — uses MLP stand-in (planned: SAGEConv)

#### Current Implementation

```python
class BipartiteGraphSAGE(nn.Module):
    """Small feed-forward stand-in for the planned GraphSAGE encoder."""

    def __init__(
        self,
        input_dim: int = 5,           # Feature dimension (matches data["element"].x)
        hidden_dim: int = 128,        # Hidden/output dimension for all layers
        output_dim: int | None = None, # Output dimension (default: hidden_dim)
        num_layers: int = 2,          # Number of sequential layers
    ) -> None:
        """Build two independent feed-forward encoders.

        Creates element_encoder and constraint_encoder as separate
        nn.Sequential stacks. Each stack is: Linear → ReLU → ... → Linear.
        The final layer has no activation.
        """

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        """Encode element and constraint node features.

        Input:
          data: HeteroData (or dict) with:
            data["element"].x     → (N_elem, input_dim)  float32
            data["constraint"].x  → (N_con, input_dim)   float32

        Output:
          {
            "element":    Tensor of shape (N_elem, output_dim),
            "constraint": Tensor of shape (N_con, output_dim),
          }

        Notes:
          - If a node type is missing (e.g., no constraint nodes),
            the corresponding key is omitted from the output dict.
          - Accesses node features via data.x_dict (PyG API) or
            data[node_type].x (fallback for dict-based HeteroData).
        """
```

**Forward pass tensor shapes:**

```
Input:
  data["element"].x      → (N_elem, 5)    [bbox[0..3], confidence]
  data["constraint"].x   → (N_con, D)     [params values, variable D]

Encode (two independent Seqential stacks):
  element_encoder(N_elem, 5)    → (N_elem, 128)
  constraint_encoder(N_con, D)  → (N_con, 128)

Output:
  {"element": (N_elem, 128), "constraint": (N_con, 128)}
```

**Encoder architecture detail:**

```
element_encoder = Sequential(
    Linear(5, 128),       ← layer 0
    ReLU(),               ← activation (not applied after final layer)
    Linear(128, 128),     ← layer 1 (final, no activation)
)

constraint_encoder = Sequential(
    Linear(D, 128),       ← layer 0
    ReLU(),
    Linear(128, 128),     ← layer 1 (final)
)
```

The two encoders are **independent** — no message passing or cross-attention
between element and constraint nodes occurs. The encoder treats each node in
isolation, which means the current implementation does **not** leverage the
bipartite graph structure (edge_index is never read).

#### Planned Upgrade — True GraphSAGE with to_hetero()

The design intent (from Phase 3.3.1) specifies a heterogeneous GraphSAGE encoder
that performs actual message passing:

```
Planned architecture:

  HeteroLinear(element_dim, hidden_dim)   ← initial projection
  HeteroLinear(constraint_dim, hidden_dim)

  SAGEConv × 2 layers, wrapped with to_hetero():
    First layer:  element → aggregate neighbor constraints → element + ReLU
                  constraint → aggregate neighbor elements → constraint + ReLU
    Second layer: same as above

  Information flow:
    element nodes ←→ constraint nodes (bidirectional message passing)

  Output:  {"element": (N_elem, hidden_dim),
            "constraint": (N_con, hidden_dim)}
```

The planned encoder will also include `reset_parameters()` for weight
initialisation — a method absent from the current stub.

**Key gaps between current stub and planned implementation:**

| Feature | Current Stub | Planned (Phase 4.4.1) |
|---------|-------------|----------------------|
| Message passing | None (isolated MLPs) | Two-layer SAGEConv + `to_hetero()` |
| Edge use | `edge_index` ignored | Aggregates from neighbour nodes |
| `reset_parameters()` | Not implemented | Xavier/Glorot initialisation |
| Activation | ReLU (except final layer) | ReLU + Dropout after each conv |
| `torch_geometric` dependency | Not required | Required (`SAGEConv`, `HeteroLinear`, `to_hetero`) |

> **⚠️ Stub note:** The current `BipartiteGraphSAGE` processes element and
> constraint features through **independent MLP stacks** without graph
> convolution. It serves as a functional placeholder enabling end-to-end
> pipeline testing with simpler architectures. Phase 4.4.1 will replace the
> MLP stacks with true `SAGEConv` layers wrapped by `to_hetero()`.

---

### 3.2 Prediction Heads

**File:** `src/bipartite_gnn_gui/model/heads.py`
**Status:** ✅ Fully implemented

#### Shared Base Class

```python
class _MLPHead(nn.Module):
    """Two-layer MLP with ReLU activation."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim),   # Hidden layer (same dim as input)
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),  # Output layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
```

**Architecture:** `Linear(in, in) → ReLU → Linear(in, out)`

#### 3.2.1 CoordinateRefinementHead

```python
class CoordinateRefinementHead(_MLPHead):
    """Predict per-element coordinate refinement deltas.

    Takes encoded element features and produces a 4-d delta vector
    for each element node.

    Signature:
        __init__(self, input_dim: int = 128)
        forward(self, x: Tensor) -> Tensor
    """

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__(input_dim=input_dim, output_dim=4)
        # MLP: Linear(128, 128) → ReLU → Linear(128, 4)
```

**Tensor shapes:**
```
Input:  x            (N_elem, 128)  ← encoded element features from encoder
       network:
         Linear(128, 128) → (N_elem, 128)
         ReLU()
         Linear(128, 4)   → (N_elem, 4)
Output: (N_elem, 4)  [Δcx, Δcy, Δw, Δh] — raw deltas (no activation)
```

**Interpretation:**
- Output is applied to element bboxes via `bbox + delta` (see `apply_delta` in `utils/bbox.py`).
- The coordinate convention is xywh (centre x, centre y, width, height).
- No sigmoid or tanh activation — raw deltas can be positive or negative.

#### 3.2.2 ViolationPredictionHead

```python
class ViolationPredictionHead(_MLPHead):
    """Predict per-constraint violation scores.

    Takes encoded constraint features and produces a violation score
    for each constraint node.

    Signature:
        __init__(self, input_dim: int = 128)
        forward(self, x: Tensor) -> Tensor
    """

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__(input_dim=input_dim, output_dim=1)
        # MLP: Linear(128, 128) → ReLU → Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))
```

**Tensor shapes:**
```
Input:  x             (N_con, 128)  ← encoded constraint features from encoder
       super().forward(x) → (N_con, 1)    raw logit
       sigmoid()          → (N_con, 1)    ∈ [0, 1]
Output: (N_con, 1)  violation confidence score
```

**Interpretation:** Score ∈ [0, 1] indicating how likely the constraint is
violated in the current VLM prediction. High score → constraint is probably
broken and needs correction.

#### 3.2.3 ExistencePredictionHead

```python
class ExistencePredictionHead(_MLPHead):
    """Predict per-element existence probabilities.

    Takes encoded element features and produces an existence probability
    for each element node.

    Signature:
        __init__(self, input_dim: int = 128)
        forward(self, x: Tensor) -> Tensor
    """

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__(input_dim=input_dim, output_dim=1)
        # MLP: Linear(128, 128) → ReLU → Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(super().forward(x))
```

**Tensor shapes:**
```
Input:  x             (N_elem, 128)  ← encoded element features from encoder
       super().forward(x) → (N_elem, 1)    raw logit
       sigmoid()          → (N_elem, 1)    ∈ [0, 1]
Output: (N_elem, 1)  existence probability
```

**Interpretation:** Score ∈ [0, 1] indicating how likely the element is a
genuine GUI component (not a false positive). Used to suppress spurious
VLM detections during inference.

#### Head Summary

| Head | Input Shape | Hidden | Output Shape | Activation | Use |
|------|------------|--------|-------------|------------|-----|
| `CoordinateRefinementHead` | `(N_elem, 128)` | `Linear(128,128) → ReLU → Linear(128,4)` | `(N_elem, 4)` | None (raw) | Refine element positions |
| `ViolationPredictionHead` | `(N_con, 128)` | `Linear(128,128) → ReLU → Linear(128,1)` | `(N_con, 1)` | Sigmoid | Detect broken constraints |
| `ExistencePredictionHead` | `(N_elem, 128)` | `Linear(128,128) → ReLU → Linear(128,1)` | `(N_elem, 1)` | Sigmoid | Suppress false positives |

---

### 3.3 BipartiteGNNCorrector — Full Model

**File:** `src/bipartite_gnn_gui/model/model.py`
**Status:** ✅ Fully implemented

```python
class BipartiteGNNCorrector(nn.Module):
    """End-to-end GUI layout correction model.

    Assembles:
      - BipartiteGraphSAGE encoder (stub → two MLP stacks)
      - CoordinateRefinementHead  (MLP → 4-d deltas)
      - ViolationPredictionHead   (MLP → 1-d score, sigmoid)
      - ExistencePredictionHead   (MLP → 1-d prob, sigmoid)

    The model does NOT own a loss function; that is external
    (BipartiteGNNLoss in losses.py).
    """

    def __init__(
        self,
        input_dim: int = 5,       # Element node feature dimension
        hidden_dim: int = 128,    # Hidden dimension (shared across encoder and heads)
    ) -> None:
        self.encoder = BipartiteGraphSAGE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        self.coordinate_head = CoordinateRefinementHead(input_dim=hidden_dim)
        self.violation_head = ViolationPredictionHead(input_dim=hidden_dim)
        self.existence_head = ExistencePredictionHead(input_dim=hidden_dim)

    def forward(self, data: Any) -> dict[str, torch.Tensor]:
        """Run complete inference.

        Args:
            data: HeteroData (or dict) with element.x and constraint.x.

        Returns:
            dict with keys:
              - "coord":     Tensor shape (N_elem, 4) — raw coordinate deltas
              - "violation": Tensor shape (N_con, 1) — violation scores [0,1]
              - "existence": Tensor shape (N_elem, 1) — existence probs [0,1]

            If no element nodes are present, "coord" and "existence"
            keys are omitted. If no constraint nodes are present,
            "violation" key is omitted.
        """
```

**Forward pass data flow:**

```
HeteroData input:
  data["element"].x      (N_elem, 5)
  data["constraint"].x   (N_con, D)
           │
           ▼
  ┌─────────────────────────────┐
  │  BipartiteGraphSAGE.forward │
  │  (two independent MLP stacks)│
  │                             │
  │  encoded["element"]         │
  │    = element_encoder(x)     │  (N_elem, 128)
  │  encoded["constraint"]       │
  │    = constraint_encoder(x)  │  (N_con, 128)
  └──────────┬──────────────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
  ┌──────────────┐  ┌─────────────────┐
  │ coord_head   │  │ violation_head  │
  │   (N,128)→4  │  │   (M,128)→1     │
  └──────┬───────┘  └────────┬────────┘
         │                   │
         ▼                   ▼
  (N_elem, 4)           (N_con, 1)
  "coord"               "violation"
                             │
     ┌───────────────────────┘
     │
     ▼
  ┌──────────────────┐
  │ existence_head   │
  │   (N,128)→1      │
  └────────┬─────────┘
           │
           ▼
      (N_elem, 1)
      "existence"
```

**Design decision — heads dispatch by node type:**
- `coordinate_head` and `existence_head` consume element node encodings only.
- `violation_head` consumes constraint node encodings only.
- This is enforced by the `forward` method checking `"element" in encoded` and
  `"constraint" in encoded`.

**Design decision — no `compute_loss` on model:**
The `BipartiteGNNCorrector` is a pure `nn.Module` that only defines `forward`.
Loss computation is external (`BipartiteGNNLoss`), unlike some frameworks where
`compute_loss` is a model method. This separation keeps the model focused on
inference and allows flexible loss configurations without inheriting from the model.

---

### 3.4 BipartiteGNNLoss — Weighted Multi-Task Loss

**File:** `src/bipartite_gnn_gui/model/losses.py`
**Status:** ✅ Fully implemented

#### Component Loss Functions

```python
def compute_coord_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error on coordinate refinement deltas.

    Args:
        prediction: (N_elem, 4) — predicted deltas [Δcx, Δcy, Δw, Δh].
        target:     (N_elem, 4) — ground-truth deltas.

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(prediction, target)


def compute_violation_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary Cross Entropy on violation scores.

    Args:
        prediction: (N_con, 1) — predicted violation scores ∈ [0, 1].
        target:     (N_con, 1) — binary labels (0 = valid, 1 = violated).

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy(prediction, target)


def compute_existence_loss(prediction: Tensor, target: Tensor) -> Tensor:
    """Binary Cross Entropy on existence probabilities.

    Args:
        prediction: (N_elem, 1) — predicted existence probs ∈ [0, 1].
        target:     (N_elem, 1) — binary labels (0 = spurious, 1 = real).

    Returns:
        Scalar BCE loss.
    """
    return F.binary_cross_entropy(prediction, target)
```

> **⚠️ Planned upgrade — SmoothL1 for coord loss:**
> The task specification calls for `SmoothL1` (Huber) loss on coordinate
> refinement, which is more robust to outlier deltas than MSE. The current
> implementation uses MSE. Phase 4.4.3 may switch to `F.smooth_l1_loss`.

#### Weighted Combination

```python
@dataclass
class BipartiteGNNLoss:
    """Weighted multi-task loss for GUI layout correction.

    L_total = w_coord · L_coord + w_violation · L_violation + w_existence · L_existence
    """

    coord_weight: float = 1.0
    violation_weight: float = 1.0
    existence_weight: float = 1.0

    def __call__(
        self,
        prediction: dict[str, Tensor],
        target: dict[str, Tensor],
    ) -> Tensor:
        """Compute weighted sum of component losses.

        Args:
            prediction: Dict from BipartiteGNNCorrector.forward():
                {"coord": (N_elem, 4), "violation": (N_con, 1), "existence": (N_elem, 1)}
            target: Ground-truth dict with matching keys:
                {"coord": (N_elem, 4), "violation": (N_con, 1), "existence": (N_elem, 1)}

        Returns:
            Scalar tensor — total weighted loss.

        Notes:
            - Each loss component is skipped (contributes 0) if its key
              is missing from either prediction or target dict.
            - This enables training on datasets where some labels are
              unavailable (e.g., no violation labels available → skip
              violation loss).
        """
```

**Loss computation logic:**

```
total = 0.0

if "coord" in prediction and "coord" in target:
    total += coord_weight * MSE(pred["coord"], target["coord"])

if "violation" in prediction and "violation" in target:
    total += violation_weight * BCE(pred["violation"], target["violation"])

if "existence" in prediction and "existence" in target:
    total += existence_weight * BCE(pred["existence"], target["existence"])

return total
```

**Typical weight configurations:**

| Use Case | `coord_weight` | `violation_weight` | `existence_weight` | Rationale |
|----------|---------------|-------------------|-------------------|-----------|
| Balanced | 1.0 | 1.0 | 1.0 | All tasks equally important |
| Coord-focused | 10.0 | 1.0 | 1.0 | Prioritise spatial accuracy |
| FP suppression | 1.0 | 1.0 | 5.0 | Prioritise removing spurious detections |

---

### 3.5 Model Layer Data Flow Summary

```
graph layer                               model layer
───────────                               ───────────

HeteroData                                BipartiteGNNCorrector.forward(data)
  │                                                 │
  ├─ element.x (N_elem, 5) ──────────►  encoder.element_encoder  ──► (N_elem, 128)
  │                                                 │
  │                                   ┌─────────────┼─────────────┐
  │                                   │             │             │
  │                                   ▼             │             ▼
  │                          coord_head (→4)        │    existence_head (→1)
  │                          (N_elem, 4)            │    (N_elem, 1)
  │                                                 │
  ├─ constraint.x (N_con, D) ────────►  encoder.constraint_encoder ──► (N_con, 128)
  │                                                 │
  │                                                 ▼
  │                                        violation_head (→1)
  │                                        (N_con, 1)
  │
  ▼
  ┌────────────────────────────────────────────────────────────┐
  │  BipartiteGNNLoss(prediction, target)                      │
  │                                                            │
  │  L_total = 1.0·MSE(coord) + 1.0·BCE(violation)            │
  │          + 1.0·BCE(existence)                              │
  │                                                            │
  │  → scalar tensor                                           │
  └────────────────────────────────────────────────────────────┘
```

---

## 4. 训练与推理规划 (Training & Inference Plan)

> **Phase 3.4** — Trainer lifecycle, optimizer/scheduler strategy, early stopping,
> checkpointing, and inference pipeline design.
>
> Source files:
> - `src/bipartite_gnn_gui/model/trainer.py`
> - `src/bipartite_gnn_gui/model/inference.py`

### 4.1 Trainer

**File:** `src/bipartite_gnn_gui/model/trainer.py`
**Status:** ⚠️ Stub — minimal placeholder (no-op fit)

#### Current Implementation

```python
@dataclass
class Trainer:
    """Minimal trainer placeholder."""

    model: Any            # The model to train (e.g., BipartiteGNNCorrector)
    loss_fn: Any | None = None  # Loss function (e.g., BipartiteGNNLoss)

    def fit(self, *_: Any, **__: Any) -> None:
        """No-op fit method for now."""
        return None
```

The current `Trainer` is a **bare-bones dataclass** with no training logic.
It stores a model reference and a loss function but performs no optimisation.

#### Planned Full Implementation (Design Intent)

The Phase 4.4.5 implementation will expand the `Trainer` to include the full
training lifecycle:

**Planned class structure:**

```python
class Trainer:
    """Full training orchestrator.

    Planned constructor args:
        model: BipartiteGNNCorrector
        loss_fn: BipartiteGNNLoss
        optimizer: torch.optim.AdamW       (or auto-constructed from config)
        scheduler: CosineAnnealingLR + LinearWarmup
        device: torch.device
        config: TrainingConfig             (from config system)
        metrics_logger: MetricsLogger       (for experiment tracking)
        early_stopping_patience: int = 10
    """
```

**Planned lifecycle:**

```
Trainer.__init__()
  │
  ├─ .fit(train_loader, val_loader)
  │    │
  │    ├─ for epoch in 1..max_epochs:
  │    │    │
  │    │    ├─ train_epoch(train_loader)
  │    │    │    ├─ model.train()
  │    │    │    ├─ for batch in train_loader:
  │    │    │    │    ├─ data → device
  │    │    │    │    ├─ with autocast (if AMP):
  │    │    │    │    │    prediction = model(data)
  │    │    │    │    │    loss = loss_fn(prediction, target)
  │    │    │    │    ├─ scaler.scale(loss).backward()
  │    │    │    │    ├─ grad_clip (if enabled)
  │    │    │    │    ├─ scaler.step(optimizer)
  │    │    │    │    ├─ scaler.update()
  │    │    │    │    └─ scheduler.step()
  │    │    │    └─ return avg_train_loss
  │    │    │
  │    │    ├─ validate(val_loader)
  │    │    │    ├─ model.eval()
  │    │    │    ├─ with torch.no_grad():
  │    │    │    │    for batch in val_loader:
  │    │    │    │        prediction = model(data)
  │    │    │    │        loss = loss_fn(prediction, target)
  │    │    │    └─ return avg_val_loss
  │    │    │
  │    │    ├─ metrics_logger.log_metrics(...)
  │    │    │
  │    │    ├─ if val_loss is best:
  │    │    │    └─ checkpoint.save(model, optimizer, epoch, val_loss)
  │    │    │
  │    │    └─ if early_stopping triggered:
  │    │         └─ break
  │    │
  │    └─ return best_val_loss
```

**Planned optimizer & scheduler details:**

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Optimizer | **AdamW** | Decoupled weight decay (better than Adam + L2). Weight decay applied only to non-bias parameters. |
| Scheduler | **Cosine annealing with linear warmup** | Warmup prevents early instability; cosine decay smoothly reduces LR to near-zero by epoch end. |
| Warmup | Linear from 0 → `lr` over `warmup_steps` batches | Prevents large gradients in first few steps. |
| Grad clip | Max L2 norm = `grad_clip` (default 1.0) | Prevents exploding gradients in deep GNNs. |

**Planned AMP (Automatic Mixed Precision):**

Uses `torch.cuda.amp.autocast` + `GradScaler`:

```python
scaler = torch.cuda.amp.GradScaler(enabled=config.amp)

# In training step:
with torch.cuda.amp.autocast(enabled=config.amp):
    prediction = model(data)
    loss = loss_fn(prediction, target)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)          # for grad clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
scaler.step(optimizer)
scaler.update()
```

Falls back gracefully to FP32 on CPU or when `amp=False`.

**Planned early stopping:**

```
patience: int = 10           # Number of epochs without improvement
min_delta: float = 1e-4      # Minimum change to count as improvement

best_val_loss = +inf
patience_counter = 0

After each validation:
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint(...)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            stop training, restore best checkpoint
```

**Planned checkpoint format:**

```python
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "val_loss": val_loss,
    "config": config.to_dict(),
}
torch.save(checkpoint, f"{checkpoint_dir}/model_epoch_{epoch:03d}.pt")
# Also save "best_model.pt" for the best checkpoint.
```

> **⚠️ Stub note:** The current `Trainer` is a no-op dataclass. All training
> logic — `train_epoch`, `validate`, checkpointing, optimizer/scheduler setup,
> AMP, early stopping, metrics logging — will be implemented in Phase 4.4.5.

#### Key Gaps Between Current Stub and Planned Implementation

| Feature | Current | Planned (Phase 4.4.5) |
|---------|---------|----------------------|
| `fit()` logic | No-op | Full epoch loop |
| `train_epoch()` | N/A | Forward → loss → backward → step |
| `validate()` | N/A | `torch.no_grad()` eval loop |
| Optimizer | N/A | AdamW |
| Scheduler | N/A | Cosine annealing + linear warmup |
| Early stopping | N/A | Patience-based with best model restore |
| Checkpoint | N/A | Save/load model + optimizer + scheduler |
| AMP | N/A | `torch.cuda.amp.autocast` + `GradScaler` |
| Metrics logging | N/A | Per-epoch to `MetricsLogger` |
| Device strategy | N/A | `model.to(device)` + data transfer |

---

### 4.2 InferencePipeline

**File:** `src/bipartite_gnn_gui/model/inference.py`
**Status:** ⚠️ Stub — minimal forward pass

#### Current Implementation

```python
def correct_layout(model: Any, data: Any) -> Any:
    """Run inference and return the model outputs.

    Args:
        model: A BipartiteGNNCorrector (or compatible).
        data:  A HeteroData graph (or dict).

    Returns:
        The model's forward output dict:
        {"coord": ..., "violation": ..., "existence": ...}
    """
    return model(data)
```

The current implementation is a **trivial wrapper** around `model.forward()`.
It does not parse VLM JSON, build graphs, apply deltas to bboxes, clamp coordinates,
or produce corrected JSON output.

#### Planned Full Implementation (Design Intent)

The Phase 4.4.6 implementation will expand this into a full `InferencePipeline`
class supporting the complete correction workflow.

**Planned class structure:**

```python
class InferencePipeline:
    """End-to-end inference pipeline for GUI layout correction.

    Planned constructor args:
        model: BipartiteGNNCorrector
        graph_builder: BipartiteGraphBuilder
        device: torch.device
        amp: bool = False             # Enable mixed precision for inference
    """

    def correct_single(
        self,
        vlm_output: dict | VLMOutput,
        image_size: tuple[int, int] | None = None,
    ) -> dict:
        """Correct a single VLM output.

        Full pipeline:
          1. Parse VLM JSON → VLMOutput (if raw dict given).
          2. Extract ElementNode list from VLM output elements.
          3. Build HeteroData graph:
             elements → extract_all_constraints() → builder.build()
          4. Run model inference (model.forward).
          5. Apply coordinate deltas to original bboxes.
          6. Clamp corrected bboxes to valid range [0, 1].
          7. Filter low-confidence elements by existence score.
          8. Serialise to corrected JSON dict.

        Args:
            vlm_output: Raw VLM JSON dict or pre-parsed VLMOutput.
            image_size: Optional (width, height) for pixel → normalised conversion.

        Returns:
            Corrected JSON dict with refined elements.
        """
```

**Planned inference data flow:**

```
VLM JSON (raw dict)
  │
  ▼
┌─────────────────────────────┐
│ Step 1: Parse               │
│   load_vlm_output(json)     │
│   → VLMOutput               │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 2: Build Graph         │
│   ElementNode[] from        │
│   VLMOutput.elements        │
│                             │
│   extract_all_constraints() │
│   → ConstraintNode[]        │
│                             │
│   builder.build(elems,      │
│     constraints)            │
│   → HeteroData              │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 3: Model Inference     │
│   model.eval()              │
│   with torch.no_grad():     │
│     output = model(data)    │
│                             │
│   output ≈ {                │
│     "coord": (N, 4),        │
│     "violation": (M, 1),    │
│     "existence": (N, 1),    │
│   }                         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 4: Apply Deltas        │
│   For each element i:       │
│     corrected_bbox[i] =     │
│       original_bbox[i] +    │
│       output["coord"][i]    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 5: Clamp & Filter      │
│   Clamp bbox to [0, 1]      │
│                             │
│   If existence[i] < 0.5:    │
│     Drop element i          │
│   (suppress spurious        │
│    VLM predictions)         │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Step 6: Serialise           │
│   Corrected JSON dict       │
│   with refined elements     │
└─────────────────────────────┘
```

**Planned internal pipeline steps detail:**

```
correct_single(vlm_output, image_size=None) → corrected_json

Internal pipeline:
  1. _parse_input(raw) → VLMOutput
     - If already VLMOutput, use as-is.
     - If dict/str, call load_vlm_output().

  2. _build_graph(vlm_output) → HeteroData
     - Convert each VLMOutputElement to ElementNode.
     - Call extract_all_constraints(elements).
     - Call builder.build(elements, constraints).
     - Move HeteroData to self.device.

  3. _model_forward(data) → output_dict
     - model.eval()
     - with torch.no_grad():
         with autocast(enabled=self.amp):
           return model(data)

  4. _apply_deltas(elements, output_dict) → corrected_elements
     - For each element i:
         corrected_bbox = elements[i].bbox + output_dict["coord"][i]
     - If image_size provided, apply inverse normalisation.

  5. _clamp_and_filter(corrected_elements, output_dict) → final_elements
     - Clamp all bbox values to [0.0, 1.0] (or [0, W]/[0, H] if pixel).
     - Drop elements where output_dict["existence"][i] < 0.5.
     - Flag degenerate bboxes (x2 ≤ x1 or y2 ≤ y1).

  6. _serialise(final_elements) → dict
     - Convert to JSON-compatible format.
     - Include metadata (model version, timestamp, image_size).
```

**Planned device strategy:**

```python
def _to_device(self, data: HeteroData) -> HeteroData:
    """Move HeteroData tensors to the target device."""
    for store in data.node_stores + data.edge_stores:
        for key, value in store.items():
            if isinstance(value, torch.Tensor):
                store[key] = value.to(self.device)
    return data
```

**Planned batch inference:**

```python
def correct_batch(
    self,
    vlm_outputs: list[dict | VLMOutput],
) -> list[dict]:
    """Correct multiple VLM outputs in batch.

    Builds a mini-batch HeteroData via PyG's batch concatenation.

    Args:
        vlm_outputs: List of VLM output dicts.

    Returns:
        List of corrected JSON dicts, one per input.
    """
```

> **⚠️ Stub note:** The current `correct_layout` is a one-liner that wraps
> `model(data)`. The full `InferencePipeline` with VLM parsing, graph
> construction, delta application, clamping, and filtering will be
> implemented in Phase 4.4.6.

#### Key Gaps Between Current Stub and Planned Implementation

| Feature | Current | Planned (Phase 4.4.6) |
|---------|---------|----------------------|
| VLM JSON parsing | Not done | `load_vlm_output()` inside pipeline |
| Graph construction | Not done | Builder + constraint extraction |
| Delta application | Not done | `apply_delta()` + `bbox_to_tensor` |
| Bbox clamping | Not done | Clamp to [0, 1] or image bounds |
| Existence filtering | Not done | Drop elements with score < 0.5 |
| Device strategy | Not done | Manual tensor transfer |
| AMP support | Not done | `torch.cuda.amp.autocast` |
| Batch inference | Not done | PyG mini-batch via `Collater` |
| `correct_single` | Not implemented | Full parse → graph → model → apply → serialise |
| `correct_batch` | Not implemented | Batch graph construction + model forward |
| Serialisation | Not done | Corrected JSON dict output |

---

## Implementation Status Summary

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `BipartiteGraphSAGE` | `model/encoder.py` | ⚠️ Stub | Two independent MLP stacks (no SAGEConv) |
| `CoordinateRefinementHead` | `model/heads.py` | ✅ Implemented | MLP → 4-d deltas |
| `ViolationPredictionHead` | `model/heads.py` | ✅ Implemented | MLP → 1-d sigmoid score |
| `ExistencePredictionHead` | `model/heads.py` | ✅ Implemented | MLP → 1-d sigmoid prob |
| `BipartiteGNNCorrector` | `model/model.py` | ✅ Implemented | Encoder + 3 heads assembled |
| `BipartiteGNNLoss` | `model/losses.py` | ✅ Implemented | Weighted MSE + BCE + BCE |
| `compute_coord_loss` | `model/losses.py` | ✅ Implemented | MSE (planned: SmoothL1) |
| `compute_violation_loss` | `model/losses.py` | ✅ Implemented | BCE |
| `compute_existence_loss` | `model/losses.py` | ✅ Implemented | BCE |
| `Trainer` | `model/trainer.py` | ⚠️ Stub | No-op dataclass, no training logic |
| `correct_layout` | `model/inference.py` | ⚠️ Stub | One-liner model(data) wrapper |
| `InferencePipeline` | `model/inference.py` | ⚠️ Not implemented | Planned for Phase 4.4.6 |

**Legend:** ✅ = Implemented and functional | ⚠️ Stub = Minimal placeholder | ⚠️ Not implemented = Planned but not started

---

## Key Design Decisions (Model Layer)

| Decision | Rationale |
|----------|-----------|
| **Independent MLP stacks in encoder (stub)** | Enables end-to-end pipeline testing before GNN convolution is implemented. Two separate `nn.Sequential` stacks process element and constraint nodes independently. |
| **No message passing in current encoder** | The `BipartiteGraphSAGE` encoder reads `x_dict` but never accesses `edge_index`. This is a deliberate simplification for the stub phase. |
| **Heads dispatched by node type** | `coord_head` and `existence_head` operate on element encodings; `violation_head` on constraint encodings. This semantic assignment reflects the node type's role in the bipartite graph. |
| **Loss function external to model** | `BipartiteGNNLoss` is a standalone callable, not a method on `BipartiteGNNCorrector`. This decouples the model from loss configuration and allows flexible weighting without model subclassing. |
| **Dict-based loss input** | Uses `prediction: dict[str, Tensor]` and `target: dict[str, Tensor]` instead of positional tuple arguments. This allows graceful skipping of missing loss components (e.g., when no violation labels are available). |
| **MSE for coordinate loss (not SmoothL1)** | MSE is simpler and the default in the current stub. SmoothL1 (planned) is more robust to outliers but requires additional tuning of the beta parameter. |
| **No `reset_parameters()` on encoder** | The current MLP-based encoder relies on PyTorch's default Linear initialisation. A `reset_parameters()` method will be added when SAGEConv is introduced (Phase 4.4.1). |

## 修订历史 (Revision History)

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-05-25 | 初始版本：数据层与图构建层的详细类设计，与 `src/` 实际代码对齐，标注 stub 状态。 |
| 1.1 | 2026-05-25 | 新增 Phase 3.3–3.4：模型层 (encoder/heads/model/losses) 类设计、训练器与推理管线规划，与 `src/` 实际代码对齐，标注 stub 状态。 |
| 1.2 | 2026-05-25 | 新增 Phase 3.5：评估层 (metrics/evaluator/baselines/qualitative) 类设计，与 `src/` 实际代码对齐，标注 stub 状态。 |

---

## 5. 评估层设计 (Evaluation Layer Design)

> **Phase 3.5** — Metric definitions, evaluator orchestration, baseline comparison,
> and qualitative analysis.
>
> Source files:
> - `src/bipartite_gnn_gui/eval/metrics.py`
> - `src/bipartite_gnn_gui/eval/evaluator.py`
> - `src/bipartite_gnn_gui/eval/baselines.py`
> - `src/bipartite_gnn_gui/eval/qualitative.py`

### 5.1 Metrics (`metrics.py`)

**File:** `src/bipartite_gnn_gui/eval/metrics.py`
**Status:** ✅ Implemented (core metrics; `AlignmentError` is stub)

#### 5.1.1 Metric Class Design

All metrics are implemented as `@dataclass` callable objects rather than bare
functions. Each metric class has a `__call__(self, prediction: Tensor, target: Tensor) -> Tensor`
signature that accepts torch tensors and returns a scalar tensor.

```python
from dataclasses import dataclass
from torch import Tensor

@dataclass
class PositionError:
    """Euclidean position error."""

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.norm(prediction[..., :2] - target[..., :2], dim=-1).mean()
```

**Design decision — dataclass callables instead of functions:**
Using `@dataclass` callables allows metrics to carry configurable parameters
(e.g., `iou_threshold`) as instance attributes without requiring closures
or partial application. This makes the metric objects self-documenting and
easily serialisable.

#### 5.1.2 Individual Metrics

```python
@dataclass
class PositionError:
    """Euclidean distance between predicted and target element centers.

    Formula:
        (1/N) · Σᵢ ‖pred[...,:2]ᵢ − target[...,:2]ᵢ‖₂

    Input shape:
        prediction: (N, 4) or (..., 4) — [cx, cy, w, h] format
        target:     (N, 4) or (..., 4)

    Output:  Scalar tensor — mean L₂ distance on center coordinates.
    """

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.norm(prediction[..., :2] - target[..., :2], dim=-1).mean()


@dataclass
class SizeError:
    """Euclidean distance between predicted and target element dimensions.

    Formula:
        (1/N) · Σᵢ ‖pred[...,2:4]ᵢ − target[...,2:4]ᵢ‖₂

    Input shape:
        prediction: (N, 4) or (..., 4) — [cx, cy, w, h] format
        target:     (N, 4) or (..., 4)

    Output:  Scalar tensor — mean L₂ distance on size dimensions.
    """

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.norm(prediction[..., 2:4] - target[..., 2:4], dim=-1).mean()


@dataclass
class AlignmentError:
    """⚠️ Stub — mean absolute error between prediction and target tensors.

    The requirements doc specifies a per-alignment-group max deviation
    aggregation. The current implementation computes a naive MAE across
    all coordinate dimensions, ignoring alignment group structure entirely.

    Planned (design intent):
        - Group elements by alignment type (left/right/top/bottom/center).
        - For each group, compute max deviation from the group's axis.
        - Aggregate as mean of per-group max deviations.

    Input shape:
        prediction: (N, 4) or any shape
        target:     matching shape

    Output:  Scalar tensor — mean absolute difference.
    """

    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return torch.abs(prediction - target).mean()


@dataclass
class ElementRecall:
    """Fraction of target (ground-truth) elements matched to a prediction.

    Computes pairwise IoU between prediction and target boxes, then
    checks whether each target box has at least one prediction with
    IoU ≥ threshold. This is a **max-pooled** per-target check —
    a target is considered matched if ANY prediction overlaps it
    sufficiently.

    Formula:
        recall = |{t ∈ targets : max_p IoU(p, t) ≥ τ}| / |targets|

    Args:
        iou_threshold: float = 0.5 — minimum IoU to count as a match.
    """

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
            device = prediction_boxes.device if prediction_boxes.numel() else target_boxes.device
            return torch.tensor(0.0, device=device)
        iou = compute_iou(prediction_boxes, target_boxes)
        return (iou.max(dim=-1).values >= self.iou_threshold).float().mean()


@dataclass
class ElementPrecision:
    """Fraction of predicted elements matched to a target.

    Computes pairwise IoU and checks whether each predicted box has
    at least one target with IoU ≥ threshold. This is a **max-pooled**
    per-prediction check — a prediction is considered matched if ANY
    target box overlaps it sufficiently.

    Formula:
        precision = |{p ∈ predictions : max_t IoU(p, t) ≥ τ}| / |predictions|

    Args:
        iou_threshold: float = 0.5 — minimum IoU to count as a match.
    """

    iou_threshold: float = 0.5

    def __call__(self, prediction_boxes: Tensor, target_boxes: Tensor) -> Tensor:
        if prediction_boxes.numel() == 0 or target_boxes.numel() == 0:
            device = prediction_boxes.device if prediction_boxes.numel() else target_boxes.device
            return torch.tensor(0.0, device=device)
        iou = compute_iou(prediction_boxes, target_boxes)
        return (iou.max(dim=-2).values >= self.iou_threshold).float().mean()
```

**Key difference — max-pooled matching vs. Hungarian assignment:**

The current `ElementRecall` and `ElementPrecision` use independent per-box
max-pooling rather than one-to-one bipartite matching. This means:

| Approach | Recall (max over dim -1) | Precision (max over dim -2) |
|----------|-------------------------|----------------------------|
| **Current** | Each GT box independently checked against ALL predictions | Each pred box independently checked against ALL GTs |
| **Hungarian (requirements)** | One prediction can match at most one GT (one-to-one assignment) | Same |

The current approach is simpler and gradient-friendly (useful for loss functions),
but overestimates both recall and precision when multiple predictions cluster around
the same GT element. For pure evaluation purposes, Hungarian matching (one-to-one)
is more faithful — see the planned upgrade note below.

> **⚠️ Planned upgrade — Hungarian matching:** The requirements doc (`metrics.md` §4–5)
> specifies greedy bipartite matching with one-to-one assignment. The current max-pooled
> approach will be supplemented with a Hungarian-based matching layer (via
> `scipy.optimize.linear_sum_assignment`) for evaluation-mode metrics. The max-pooled
> versions will remain available for training-time loss computation where gradient flow
> is required.

#### 5.1.3 Metric Aggregation

```python
def compute_all_metrics(prediction_boxes: Tensor, target_boxes: Tensor) -> dict[str, float]:
    """Compute the standard metrics bundle.

    Instantiates each metric class with defaults and calls it on the inputs.
    This is the de-facto ALL_METRICS registry — a single function that computes
    all five metrics and returns them as a plain dict.

    Args:
        prediction_boxes: (N_pred, 4) tensor of predicted bboxes [cx, cy, w, h].
        target_boxes:     (N_gt, 4) tensor of ground-truth bboxes [cx, cy, w, h].

    Returns:
        dict with keys: "recall", "precision", "position_error",
        "size_error", "alignment_error". All values are Python floats.
    """
    return {
        "recall":          float(ElementRecall()(prediction_boxes, target_boxes).item()),
        "precision":       float(ElementPrecision()(prediction_boxes, target_boxes).item()),
        "position_error":  float(PositionError()(prediction_boxes, target_boxes).item()),
        "size_error":      float(SizeError()(prediction_boxes, target_boxes).item()),
        "alignment_error": float(AlignmentError()(prediction_boxes, target_boxes).item()),
    }
```

**Design decision — `compute_all_metrics()` as registry:**

The requirements doc specifies a module-level `ALL_METRICS: dict[str, Callable]`
dictionary. The actual code uses `compute_all_metrics()` instead, which
internally instantiates and calls all five metric classes. This is functionally
equivalent but trades configurability (no runtime metric selection) for
simplicity (one function, guaranteed ordering, consistent defaults).

#### 5.1.4 IoU Utility

```python
def compute_iou(box1: Tensor, box2: Tensor) -> Tensor:
    """Compute IoU between two bbox tensors.

    Thin wrapper around bipartite_gnn_gui.utils.bbox.compute_iou.
    Supports broadcasting: (N, 4) vs (M, 4) → (N, M) IoU matrix.
    Auto-detects xywh vs xyxy format.
    """
```

#### Metric Summary

| Metric | Class | Input Dims | Output | Status |
|--------|-------|-----------|--------|--------|
| `PositionError` | `@dataclass` callable | `(N,4)` vs `(N,4)` | Scalar float (L₂) | ✅ |
| `SizeError` | `@dataclass` callable | `(N,4)` vs `(N,4)` | Scalar float (L₂) | ✅ |
| `AlignmentError` | `@dataclass` callable | any vs any | Scalar float (MAE) | ⚠️ Stub |
| `ElementRecall` | `@dataclass` callable | `(N,4)` vs `(M,4)` | Scalar float ∈ [0,1] | ✅ |
| `ElementPrecision` | `@dataclass` callable | `(N,4)` vs `(M,4)` | Scalar float ∈ [0,1] | ✅ |
| `compute_all_metrics` | Function | `(N,4)` vs `(M,4)` | `dict[str, float]` | ✅ |

**Edge case handling (implemented):**
- **Empty tensors (no predictions or no targets):** `ElementRecall` and `ElementPrecision`
  return `0.0` when either input has zero elements. This avoids division-by-zero in
  the mean computation.
- **Device consistency:** When one input is empty, the output tensor is placed on the
  device of the non-empty input.

---

### 5.2 Evaluator (`evaluator.py`)

**File:** `src/bipartite_gnn_gui/eval/evaluator.py`
**Status:** ✅ Implemented (minimal wrapper; no per-category breakdown or bootstrap)

#### 5.2.1 Current Implementation

```python
@dataclass
class EvaluationResult:
    """Simple wrapper for evaluation output."""

    metrics: dict[str, float]


class Evaluator:
    """Evaluate predictions against targets."""

    def evaluate(self, prediction_boxes, target_boxes) -> EvaluationResult:
        """Run all metrics and return a wrapped result.

        Args:
            prediction_boxes: Tensor of predicted bboxes.
            target_boxes:     Tensor of ground-truth bboxes.

        Returns:
            EvaluationResult wrapping the output of compute_all_metrics().
        """
        return EvaluationResult(metrics=compute_all_metrics(prediction_boxes, target_boxes))
```

The `Evaluator` is a thin orchestration class. Its sole method `evaluate()` delegates
entirely to `compute_all_metrics()` and wraps the result dict in an `EvaluationResult`
dataclass for future extensibility.

#### 5.2.2 Current Limitations

| Feature | Status | Notes |
|---------|--------|-------|
| `evaluate(preds, gt) -> dict of all metrics` | ✅ | Via delegation to `compute_all_metrics()` |
| `per_category_breakdown` | ❌ Not implemented | Element-type-level metric breakdown not computed |
| Bootstrap / statistical significance | ❌ Not implemented | No resampling, no confidence intervals |
| Multi-sample aggregation | ❌ Not implemented | `evaluate()` handles one sample pair at a time; caller must aggregate across dataset |

> **⚠️ Planned upgrade — full evaluator (design intent):**
>
> The requirements doc (`metrics.md` §7) and TASK.md (Phase 3.5.1) specify:
>
> ```python
> class Evaluator:
>     """Full evaluation orchestrator (planned)."""
>
>     def evaluate(
>         self,
>         preds: list[VLMOutput],
>         gt: list[GroundTruth],
>     ) -> dict[str, float]:
>         """Compute all metrics across a full dataset."""
>
>     def per_category_breakdown(
>         self,
>         preds: list[VLMOutput],
>         gt: list[GroundTruth],
>     ) -> dict[str, dict[str, float]]:
>         """Metrics broken down by element type (button, text, image, ...)."""
>
>     def bootstrap(
>         self,
>         preds: list[VLMOutput],
>         gt: list[GroundTruth],
>         n_iter: int = 1000,
>     ) -> dict[str, dict[str, float]]:
>         """Bootstrap resampling for confidence intervals.
>
>         Returns: {metric_name: {"mean": ..., "std": ..., "ci_lower": ..., "ci_upper": ...}}
>         """
> ```
>
> The current `Evaluator` provides only the single-pair `evaluate()` method.
> Multi-sample aggregation, per-category breakdown, and bootstrap confidence
> intervals will be added in Phase 4.5.2.

---

### 5.3 Baselines (`baselines.py`)

**File:** `src/bipartite_gnn_gui/eval/baselines.py`
**Status:** ⚠️ Stub — all three baselines are identity (pass-through)

#### 5.3.1 Current Implementation

```python
from dataclasses import dataclass
from typing import Any


@dataclass
class BaselineNoCorrection:
    """Return inputs unchanged — VLM output as-is."""

    def __call__(self, data: Any) -> Any:
        return data


@dataclass
class BaselineRuleBased(BaselineNoCorrection):
    """⚠️ Stub — placeholder rule-based baseline.

    Currently inherits from BaselineNoCorrection and performs no
    correction. The planned implementation will apply heuristic
    fixes such as Non-Maximum Suppression (NMS) and alignment
    snapping using bbox-based rules.
    """


@dataclass
class BaselineMLPOnly(BaselineNoCorrection):
    """⚠️ Stub — placeholder MLP-only baseline.

    Currently inherits from BaselineNoCorrection and performs no
    correction. The planned implementation will use a plain MLP
    (no graph structure) to refine element positions, serving as
    a control to isolate the contribution of the GNN component.
    """
```

#### 5.3.2 Baseline Comparison Matrix

| Baseline | Class | Current Behavior | Planned Behavior |
|----------|-------|-----------------|-----------------|
| **VLM Raw Output** | `BaselineNoCorrection` | Identity pass-through | N/A (final) |
| **Rule-Based Correction** | `BaselineRuleBased` | Identity pass-through (same as above) | NMS + heuristic snapping |
| **MLP-Only** | `BaselineMLPOnly` | Identity pass-through (same as above) | MLP refinement without graph |

**Design decision — inheritance from `BaselineNoCorrection`:**
`BaselineRuleBased` and `BaselineMLPOnly` inherit from `BaselineNoCorrection`
rather than from a shared abstract base. This is a pragmatic choice for the
stub phase — all three currently share identical behavior (identity), and
inheritance avoids code duplication. When the planned implementations diverge,
each subclass will override `__call__` with its own logic.

**Naming discrepancy with requirements:**

| TASK.md / Requirements Name | Actual Code Name |
|----------------------------|------------------|
| `VLMOutputBaseline` | `BaselineNoCorrection` |
| `RuleBasedCorrection` | `BaselineRuleBased` |
| `MLPOnlyBaseline` | `BaselineMLPOnly` |

The code uses a `Baseline` prefix convention; the requirements use `Correction`
and `Baseline` suffixes. This is a cosmetic difference — the classes are
functionally equivalent in intent (the stubs are all identity).

> **⚠️ Stub note:** All three baselines currently perform zero correction.
> They exist as placeholders to establish the interface and enable pipeline
> integration testing (Phase 5.5). Full implementations — rule-based NMS
> correction and MLP-only refinement — will be added in Phase 4.5.3.

---

### 5.4 Qualitative Analysis (`qualitative.py`)

**File:** `src/bipartite_gnn_gui/eval/qualitative.py`
**Status:** ⚠️ Stub — all three functions are no-op (return `None`)

#### 5.4.1 Current Implementation

```python
from typing import Any


def side_by_side_plot(*_: Any, **__: Any) -> Any:
    """⚠️ Stub — placeholder for side-by-side comparison plot.

    Planned: Render two screenshots side-by-side with bounding box
    overlays showing VLM predictions (before) vs GNN-corrected (after).
    """
    return None


def plot_error_heatmap(*_: Any, **__: Any) -> Any:
    """⚠️ Stub — placeholder for spatial error heatmap.

    Planned: Render a 2D heatmap overlaid on the screenshot showing
    per-pixel localization error magnitude, highlighting regions where
    the model struggles most.
    """
    return None


def plot_category_breakdown(*_: Any, **__: Any) -> Any:
    """⚠️ Stub — placeholder for per-category performance chart.

    Planned: Render a grouped bar chart showing metrics (recall, precision,
    position error) broken down by element type (button, text, image, ...).
    """
    return None
```

#### 5.4.2 Planned Qualitative Analysis Suite (Design Intent)

The requirements doc and TASK.md (Phase 3.5.3) specify a richer set of qualitative
analysis functions. The current stubs map to the planned functions as follows:

| Requirements / TASK.md | Actual Code | Status |
|------------------------|-------------|--------|
| `side_by_side_comparison` | `side_by_side_plot` | ⚠️ Stub (returns `None`) |
| `case_study_report` | Not implemented | ❌ Not started |
| `failure_analysis` | `plot_error_heatmap` (partial) | ⚠️ Stub (returns `None`) |
| (additional) | `plot_category_breakdown` | ⚠️ Stub (returns `None`) |

**Planned function interfaces (design intent):**

```
side_by_side_comparison(
    image: PIL.Image | str,          # Original screenshot
    vlm_elements: list[ElementNode],  # VLM predicted elements
    corrected_elements: list[ElementNode],  # GNN-corrected elements
    gt_elements: list[ElementNode],   # Ground truth (optional)
    save_path: str | None = None,
) -> matplotlib.Figure

case_study_report(
    sample_paths: list[str],          # List of screenshot paths
    model: BipartiteGNNCorrector,
    output_dir: str,
) -> str                              # Path to generated HTML report

failure_analysis(
    evaluator: Evaluator,
    preds: list[VLMOutput],
    gt: list[GroundTruth],
    top_k: int = 20,                   # Number of worst cases to show
) -> list[dict]                        # Ranked failure cases with metadata
```

> **⚠️ Stub note:** All three qualitative analysis functions are no-ops.
> The function names in the actual code (`side_by_side_plot`, `plot_error_heatmap`,
> `plot_category_breakdown`) differ from the requirements names but cover the
> same conceptual space. Full implementations using matplotlib for plotting
> will be added in Phase 4.5.4.

---

### 5.5 Evaluation Layer Data Flow Summary

```
┌──────────────────────────────┐
│  Model output / Baseline     │
│  (BipartiteGNNCorrector      │
│   or Baseline*.__call__())   │
│                              │
│  → predicted boxes:          │
│    Tensor (N_pred, 4)        │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐     ┌──────────────────────────────┐
│  Ground Truth                │     │  compute_all_metrics()        │
│  → target boxes:             │────▶│                              │
│    Tensor (N_gt, 4)          │     │  ElementRecall()              │
│                              │     │    → per-GT max IoU ≥ τ       │
└──────────────────────────────┘     │  ElementPrecision()           │
                                     │    → per-pred max IoU ≥ τ     │
                                     │  PositionError()              │
                                     │    → L₂ norm of center diffs  │
                                     │  SizeError()                  │
                                     │    → L₂ norm of size diffs    │
                                     │  AlignmentError()             │
                                     │    → MAE (stub)               │
                                     │                              │
                                     │  → dict[str, float]           │
                                     └──────────────┬───────────────┘
                                                    │
                                                    ▼
                                     ┌──────────────────────────────┐
                                     │  Evaluator.evaluate()        │
                                     │  → EvaluationResult(metrics) │
                                     └──────────────────────────────┘
```

**Companion analysis (pluggable but currently stubs):**

```
                                     ┌──────────────────────────────┐
                                     │  Qualitative Analysis        │
                                     │                              │
                                     │  side_by_side_plot()   ⚠️    │
                                     │  plot_error_heatmap()  ⚠️    │
                                     │  plot_category_breakdown() ⚠️│
                                     │                              │
                                     │  All return None currently   │
                                     └──────────────────────────────┘
```

**Baseline comparison flow:**

```
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ BaselineNoCorrection    │   │ BaselineRuleBased       │   │ BaselineMLPOnly         │
│   (VLM raw output)      │   │   (identity stub)        │   │   (identity stub)        │
└───────────┬─────────────┘   └───────────┬─────────────┘   └───────────┬─────────────┘
            │                             │                             │
            └─────────────────┬───────────┴─────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Evaluator.evaluate │
                    │  → dict[str, float] │
                    │    per baseline     │
                    └─────────────────────┘
```

---

### 5.6 Implementation Status Summary (Evaluation Layer)

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `PositionError` | `eval/metrics.py` | ✅ Implemented | Dataclass callable, L₂ on centers |
| `SizeError` | `eval/metrics.py` | ✅ Implemented | Dataclass callable, L₂ on dimensions |
| `AlignmentError` | `eval/metrics.py` | ⚠️ Stub | Naive MAE (not group-based) |
| `ElementRecall` | `eval/metrics.py` | ✅ Implemented | Dataclass callable, max-pooled matching |
| `ElementPrecision` | `eval/metrics.py` | ✅ Implemented | Dataclass callable, max-pooled matching |
| `compute_iou` | `eval/metrics.py` | ✅ Implemented | Wraps `utils/bbox.compute_iou` |
| `compute_all_metrics` | `eval/metrics.py` | ✅ Implemented | De-facto registry, returns `dict[str, float]` |
| `EvaluationResult` | `eval/evaluator.py` | ✅ Implemented | Dataclass wrapper for metrics dict |
| `Evaluator.evaluate()` | `eval/evaluator.py` | ✅ Implemented | Thin wrapper around `compute_all_metrics` |
| `Evaluator.per_category_breakdown` | `eval/evaluator.py` | ❌ Not implemented | Planned for Phase 4.5.2 |
| `Evaluator.bootstrap()` | `eval/evaluator.py` | ❌ Not implemented | Planned for Phase 4.5.2 |
| `BaselineNoCorrection` | `eval/baselines.py` | ⚠️ Stub | Identity pass-through |
| `BaselineRuleBased` | `eval/baselines.py` | ⚠️ Stub | Inherits identity from above |
| `BaselineMLPOnly` | `eval/baselines.py` | ⚠️ Stub | Inherits identity from above |
| `side_by_side_plot` | `eval/qualitative.py` | ⚠️ Stub | Returns `None` |
| `plot_error_heatmap` | `eval/qualitative.py` | ⚠️ Stub | Returns `None` |
| `plot_category_breakdown` | `eval/qualitative.py` | ⚠️ Stub | Returns `None` |

**Legend:** ✅ = Implemented and functional | ⚠️ Stub = Minimal placeholder | ❌ = Not implemented

---

### 5.7 Key Design Decisions (Evaluation Layer)

| Decision | Rationale |
|----------|-----------|
| **Dataclass callables for metrics** | Allows per-instance configuration (e.g., `iou_threshold`) without closures. Makes metrics self-documenting and trivially serialisable. |
| **Max-pooled matching (not Hungarian)** | Simplifies implementation and preserves gradient flow for training-time use. Hungarian matching is planned for pure evaluation mode where gradient flow is not needed. |
| **`compute_all_metrics()` as registry** | Replaces the requirements-specified `ALL_METRICS` dict with a single function. Simpler to maintain and guarantees consistent metric ordering. Trade-off: no runtime metric selection. |
| **`EvaluationResult` dataclass** | Wraps the metrics dict in a typed container for future extensibility (e.g., adding metadata, timestamps, per-sample breakdowns without changing the return type). |
| **Baseline inheritance from `BaselineNoCorrection`** | Pragmatic for the stub phase — all three share identity behavior. Subclasses will diverge when implementations are added in Phase 4.5.3. |
| **No `case_study_report` or `failure_analysis` yet** | The actual code provides plotting stubs (`side_by_side_plot`, `plot_error_heatmap`, `plot_category_breakdown`) but none of the higher-level report-generation functions from the requirements. These are planned for Phase 4.5.4. |
