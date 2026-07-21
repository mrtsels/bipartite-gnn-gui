# Graph Schema Design

## Overview

This document defines the heterogeneous bipartite graph schema for the GUI structure correction task.

## Node Types

The graph has two node types:
- **Element nodes**: represent detected GUI elements.
- **Constraint nodes**: represent spatial constraints between elements.

### Element Node Features

| Feature | Type | Description |
|---------|------|-------------|
| bbox | float32[5] | (cx, cy, w, h) normalized coordinates |
| type | int64 | One-hot encoded element type index |

### Constraint Node Features

| Feature | Type | Description |
|---------|------|-------------|
| constraint_type | float32[10] | One-hot encoding of the constraint type (align, space, contain, same-size, grid) |
| tolerance | float32 | Detection threshold for this constraint |

## Edge Types

The graph has two edge types for message passing:
- **element_to_constraint**: `(element, constraint)` — connects each element to its incident constraints.
- **constraint_to_element**: `(constraint, element)` — reverse direction for the second message-passing hop.

### Edge Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| weight | float32 | Constraint confidence or strength |

## HeteroData Structure

```python
data = HeteroData()

# Node stores
data["element"].x = torch.randn(N_elements, 5)       # bbox features
data["constraint"].x = torch.randn(N_constraints, 11) # type + tolerance features

# Edge stores
data["element", "to", "constraint"].edge_index = ...  # adjacency
data["element", "to", "constraint"].weight = ...      # edge weights
data["constraint", "to", "element"].edge_index = ...  # reversed
data["constraint", "to", "element"].weight = ...      # reversed weights
```

## Message Passing Flow

The encoder performs two alternating hops:

1. **Hop 1 (element to constraint)**: Each constraint node aggregates features from its incident elements.
2. **Hop 2 (constraint to element)**: Each element node aggregates the updated constraint representations.

## Augmentation

During training, the augmenter applies:

| Augmentation | Description | Parameters |
|-------------|-------------|------------|
| Bbox jitter | Adds Gaussian noise to element bbox coordinates | `jitter_std` |
| Drop constraint | Randomly removes a fraction of constraints | `drop_ratio` |

## Visualization

The `plot_graph` function renders the graph structure overlaid on the screenshot:

- Element nodes: red rectangles with type label
- Constraint nodes: blue circles with type abbreviation
- Edges: gray lines with transparency proportional to edge weight
