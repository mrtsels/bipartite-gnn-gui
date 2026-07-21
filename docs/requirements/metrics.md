# Evaluation Metrics

This document defines the metrics for evaluating model performance.

## 1. Position Error

**Formula:**

```
PositionError = (1 / N) times Sum of ||(cx_pred, cy_pred) - (cx_gt, cy_gt)||2
```

Where:
- `(cx_pred, cy_pred)` = predicted center coordinates
- `(cx_gt, cy_gt)` = ground-truth center coordinates
- `N` = number of matched elements

**Interpretation:** Lower is better. Measures how far the predicted center deviates from the true center.

## 2. Size Error

**Formula:**

```
SizeError = (1 / N) times Sum of ||(w_pred, h_pred) - (w_gt, h_gt)||2
```

Where:
- `(w_pred, h_pred)` = predicted width and height
- `(w_gt, h_gt)` = ground-truth width and height

## 3. Alignment Error

**Edge Cases:**
- Empty groups (no elements) do not count.
- If no alignment groups exist, the metric returns `0.0`.

## 4. Element Recall and Precision

### Matching Algorithm

The bipartite matching uses the Hungarian algorithm.
Elements match by IoU in descending order.
Once the algorithm matches a pair, it removes both elements from further consideration.

### Recall

```
Recall = TP / (TP + FN)
```

Where:
- TP = true positives (predicted elements that match a GT element)
- FN = false negatives (GT elements with no match)

### Precision

```
Precision = TP / (TP + FP)
```

Where:
- FP = false positives (predicted elements with no match)

## 5. Confidence Intervals

For metrics that need confidence intervals, the tool performs bootstrap resampling with 1,000 iterations.

If a metric has no variance (trivially, all GT elements matched), the CI is still valid but wider.
