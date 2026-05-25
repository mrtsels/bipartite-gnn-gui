# 评估指标体系 (Metrics Definition)

## 1. PositionError

**Definition**: Mean Euclidean distance between predicted and ground-truth element centers,
computed across all matched element pairs.

**Formula**:

```
PositionError = (1 / N) · Σᵢ ||(cx̂ᵢ, cŷᵢ) - (cxᵢ, cyᵢ)||₂
```

Where:
- `(cx̂ᵢ, cŷᵢ)` = predicted center coordinates of element i (normalized to [0, 1])
- `(cxᵢ, cyᵢ)` = ground-truth center coordinates of element i
- `N` = number of matched element pairs

**Edge Cases**:
- No matched elements (N = 0) → returns `NaN`
- Single-element mismatch (some VLM or GT elements unmatched) → only matched pairs counted

**Units**: Normalized coordinates (fraction of screenshot dimension). Multiply by image
width/height for pixel values.

---

## 2. SizeError

**Definition**: Mean Euclidean distance between predicted and ground-truth element dimensions,
computed across all matched element pairs.

**Formula**:

```
SizeError = (1 / N) · Σᵢ ||(ŵᵢ, ĥᵢ) - (wᵢ, hᵢ)||₂
```

Where:
- `(ŵᵢ, ĥᵢ)` = predicted width and height of element i (normalized to [0, 1])
- `(wᵢ, hᵢ)` = ground-truth width and height of element i
- `N` = number of matched element pairs

**Edge Cases**:
- Zero-size elements (w = 0 or h = 0 in GT) → element excluded from computation with a
  warning logged; this prevents degenerate L₂ distances.
- No valid elements after filtering → returns `NaN`

**Units**: Normalized coordinates (fraction of screenshot dimension).

---

## 3. AlignmentError

**Definition**: Deviation score quantifying how well predicted elements within each alignment
group conform to the group's alignment axis (horizontal or vertical).

**Computation**:

1.  For each alignment group in the ground truth, identify its primary axis:
    - **Horizontal group**: elements should share the same `cy` (top-aligned, center-aligned,
      or bottom-aligned).
    - **Vertical group**: elements should share the same `cx` (left-aligned, center-aligned,
      or right-aligned).

2.  For each group, compute the deviation of each member element from the group's alignment
    axis:

    ```
    deviation(e, axis) =
        |e.cy - axis.cy|  for horizontal groups
        |e.cx - axis.cx|  for vertical groups
    ```

    where `axis.cy` and `axis.cx` are determined by the GT group's alignment reference
    point (mean of GT member centers on the relevant axis).

3.  Aggregate across all groups:

    ```
    AlignmentError = (1 / |G|) · Σg∈G max_e∈elements(g) deviation(e, axis_g)
    ```

    Using per-group **max** deviation penalizes groups where any single element breaks
    alignment, which is the perceptually salient failure mode.

**Edge Cases**:
- Empty groups (no elements) → excluded
- No alignment groups present → returns `0.0` (no alignment constraints to violate)
- Single-element groups → deviation is trivially 0.0

---

## 4. ElementRecall

**Definition**: Fraction of ground-truth elements that have a matching predicted element
(intersection over union ≥ threshold).

**Formula**:

```
ElementRecall = TP / (TP + FN)
```

Where:
- **True Positive (TP)**: A predicted element matched to a ground-truth element with
  IoU ≥ 0.5.
- **False Negative (FN)**: A ground-truth element with no matching prediction.

**Matching Strategy**: Greedy bipartite matching. Elements are matched in descending IoU order;
once a pair is matched, both elements are removed from consideration (one-to-one matching).

```
matches = []
remaining_preds = set(range(len(preds)))
remaining_gts = set(range(len(gts)))
for (p_idx, g_idx, iou) in sorted(all_pairs, key=iou, desc):
    if p_idx in remaining_preds and g_idx in remaining_gts and iou >= 0.5:
        matches.append((p_idx, g_idx))
        remaining_preds.remove(p_idx)
        remaining_gts.remove(g_idx)
TP = len(matches)
FN = len(remaining_gts)
```

**Edge Cases**:
- Zero GT elements → returns `1.0` (trivially, all GT elements were found)
- IoU threshold is configurable via the config system (default 0.5)

---

## 5. ElementPrecision

**Definition**: Fraction of predicted elements that match a ground-truth element.

**Formula**:

```
ElementPrecision = TP / (TP + FP)
```

Where:
- **True Positive (TP)**: As defined in ElementRecall.
- **False Positive (FP)**: A predicted element with no matching ground-truth element.

**Edge Cases**:
- Zero predicted elements → returns `1.0` (trivially, no false predictions)
- All predictions unmatched → returns `0.0`

---

## 6. ALL_METRICS Registry

A module-level dictionary mapping metric names to their callable implementations.

```python
from typing import Callable, Dict
from data.types import VLMOutput, GroundTruth

ALL_METRICS: Dict[str, Callable[[VLMOutput, GroundTruth], float]] = {
    "position_error":    compute_position_error,
    "size_error":        compute_size_error,
    "alignment_error":   compute_alignment_error,
    "element_recall":    compute_element_recall,
    "element_precision": compute_element_precision,
}
```

**Contract**:
- Each callable has signature `fn(preds: VLMOutput, gt: GroundTruth) -> float`.
- All return `float` values (including `NaN` where appropriate).
- Functions are stateless and deterministic — same inputs always produce same outputs.

**Usage**:
```python
from eval.metrics import ALL_METRICS

results = {}
for name, fn in ALL_METRICS.items():
    results[name] = fn(predictions, ground_truth)
```

---

## 7. Statistical Significance

To assess result reliability, bootstrap resampling with 1,000 iterations is performed for
every metric.

**Algorithm**:

```
def bootstrap_metric(preds, gt, metric_fn, n_iter=1000):
    n = len(preds.elements)  # sample-level bootstrap
    estimates = []
    for _ in range(n_iter):
        indices = np.random.choice(n, size=n, replace=True)
        sample_preds = subsample(preds, indices)
        sample_gt = subsample(gt, indices)
        estimates.append(metric_fn(sample_preds, sample_gt))
    return {
        "mean": np.mean(estimates),
        "std": np.std(estimates),
        "ci_lower": np.percentile(estimates, 2.5),
        "ci_upper": np.percentile(estimates, 97.5),
    }
```

**Output Format**:

```
Metric             Mean     Std     CI (95%)
--------------------------------------------------
position_error     0.0234   0.0012  [0.0210, 0.0258]
size_error         0.0189   0.0009  [0.0171, 0.0207]
alignment_error    0.0056   0.0004  [0.0048, 0.0064]
element_recall     0.8723   0.0154  [0.8419, 0.9021]
element_precision  0.9135   0.0128  [0.8880, 0.9383]
```

**Edge Cases**:
- Small test sets (n < 30) → bootstrap still valid but CIs will be wider; log a warning.
- `NaN` metric values → those iterations excluded from CI computation.
