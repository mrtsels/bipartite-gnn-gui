# 图模式设计 (Graph Schema Design)

> Phase 2.4-2.5 — Heterogeneous Bipartite Graph Schema & Constraint Extraction Strategy
> Version: 1.0 | 2026-05-25

---

## 1. HeteroData 表结构

本项目使用 PyTorch Geometric 的 `HeteroData` 来表示异构图 $G = (V_e \cup V_c, E)$。

### 1.1 Element Node ($V_e$)

**Node type key:** `"element"`

| 字段 | 维度 | 说明 |
|------|------|------|
| `type_onehot` | 18 | 元素类型 one-hot 编码（见 §4 分类体系） |
| `cx` | 1 | 归一化中心 x 坐标 [0,1] |
| `cy` | 1 | 归一化中心 y 坐标 [0,1] |
| `w` | 1 | 归一化宽度 [0,1] |
| `h` | 1 | 归一化高度 [0,1] |
| `confidence` | 1 | VLM 置信度得分 [0,1] |

**总特征维度:** $D_e = 18 + 4 + 1 = 23$

### 1.2 Constraint Node ($V_c$)

**Node type key:** `"constraint"`

| 字段 | 维度 | 说明 |
|------|------|------|
| `type_onehot` | 10 | 约束类型 one-hot 编码（见 §2） |
| `tolerance` | 1 | 匹配容忍度（训练: 0.02, 推理: 0.05） |
| `weight` | 1 | 约束重要性权重（默认: 1.0） |

**总特征维度:** $D_c = 10 + 1 + 1 = 12$

### 1.3 Edge Features ($E$)

**Edge types:**
- `("element", "satisfies", "constraint")` — 正向边
- `("constraint", "satisfied_by", "element")` — 反向边（消息传递用）

| 字段 | 维度 | 说明 |
|------|------|------|
| `spatial_distance` | 1 | 元素到约束的归一化空间距离 |
| `dx` | 1 | 元素中心与约束参考点的 x 偏移 |
| `dy` | 1 | 元素中心与约束参考点的 y 偏移 |
| `iou` | 1 | 元素与约束相关元素的 IoU（若适用） |

**总特征维度:** $D_e = 4$

### 1.4 PyG HeteroData 键结构

```python
data = HeteroData()

# Element nodes: (N_elem, 23)
data["element"].x = torch.empty(N_elem, 23)

# Constraint nodes: (N_con, 12)
data["constraint"].x = torch.empty(N_con, 12)

# Edge index: (2, num_edges)
data["element", "satisfies", "constraint"].edge_index = torch.empty(2, num_edges, dtype=torch.long)
data["constraint", "satisfied_by", "element"].edge_index = torch.empty(2, num_edges, dtype=torch.long)

# Edge features: (num_edges, 4)
data["element", "satisfies", "constraint"].edge_attr = torch.empty(num_edges, 4)
data["constraint", "satisfied_by", "element"].edge_attr = torch.empty(num_edges, 4)
```

---

## 2. ConstraintType 枚举

```python
from enum import IntEnum

class ConstraintType(IntEnum):
    ALIGN_LEFT     = 0  # 元素共享左边缘
    ALIGN_RIGHT    = 1  # 元素共享右边缘
    ALIGN_TOP      = 2  # 元素共享上边缘
    ALIGN_BOTTOM   = 3  # 元素共享下边缘
    CENTER_X       = 4  # 垂直中心线对齐
    CENTER_Y       = 5  # 水平中心线对齐
    SAME_SIZE      = 6  # 相似的宽度和高度
    SPACING        = 7  # 相邻元素间距一致
    CONTAINMENT    = 8  # 一个元素包含另一个
    GRID           = 9  # 行/列排列
```

### 约束语义与适用场景

| 类型 | 语义 | 适用场景 |
|------|------|----------|
| ALIGN_LEFT | $|x_1^i - x_1^j| < \epsilon$ | 按钮组、列表项 |
| ALIGN_RIGHT | $|x_2^i - x_2^j| < \epsilon$ | 右对齐面板、输入框 |
| ALIGN_TOP | $|y_1^i - y_1^j| < \epsilon$ | 同一行元素、导航栏 |
| ALIGN_BOTTOM | $|y_2^i - y_2^j| < \epsilon$ | 底部导航、页脚 |
| CENTER_X | $|cx^i - cx^j| < \epsilon$ | 居中对齐的弹窗、卡片 |
| CENTER_Y | $|cy^i - cy^j| < \epsilon$ | 同一行元素 |
| SAME_SIZE | $\max(\frac{|w^i-w^j|}{w^j}, \frac{|h^i-h^j|}{h^j}) < \epsilon$ | 等大小按钮、图标 |
| SPACING | $|\text{gap}^{i,i+1} - \text{gap}^{i+1,i+2}| < \epsilon$ | 等间距列表、网格 |
| CONTAINMENT | $x_1^j \ge x_1^i, x_2^j \le x_2^i, y_1^j \ge y_1^i, y_2^j \le y_2^i$ | 容器内元素 |
| GRID | 行/列聚类 + 对齐检测 | 表格、图标网格 |

> **注意:** 训练模式用 GT bbox 提取约束（精确），推理模式用 VLM 预测（有噪声，需更大 $\epsilon$）。

---

## 3. 约束提取算法

### 3.1 对齐约束 (Alignment)

```
输入: elements (list of bboxes), eps (float)
输出: list of ConstraintNode

对于每对 (i, j):
  1. 若 |x1_i - x1_j| < eps → ALIGN_LEFT
  2. 若 |x2_i - x2_j| < eps → ALIGN_RIGHT
  3. 若 |y1_i - y1_j| < eps → ALIGN_TOP
  4. 若 |y2_i - y2_j| < eps → ALIGN_BOTTOM
  5. 若 |cx_i - cx_j| < eps → CENTER_X
  6. 若 |cy_i - cy_j| < eps → CENTER_Y
```

### 3.2 包含约束 (Containment)

```
输入: elements (list of bboxes)
输出: list of ConstraintNode

对于每对 (i, j) 且 i ≠ j:
  若 x1_j ≥ x1_i - margin 且 x2_j ≤ x2_i + margin
    且 y1_j ≥ y1_i - margin 且 y2_j ≤ y2_i + margin:
    → CONTAINMENT (i 包含 j)
```

### 3.3 等大小约束 (Same Size)

```
输入: elements (list of bboxes), eps (float)
输出: list of ConstraintNode

对于每对 (i, j):
  size_diff = max(|w_i - w_j|/w_j, |h_i - h_j|/h_j)
  若 size_diff < eps → SAME_SIZE
```

### 3.4 间距约束 (Spacing)

```
输入: elements (list of bboxes), eps (float)
输出: list of ConstraintNode

1. 按 cx 对元素排序
2. 计算相邻间隙: gap_i = x1_{i+1} - x2_i
3. 若 |gap_i - gap_{i+1}| < eps → SPACING
```

### 3.5 网格约束 (Grid)

```
输入: elements (list of bboxes)
输出: list of ConstraintNode

1. 用 cy 进行一维聚类 (DBSCAN, eps=0.05) → 行
2. 用 cx 进行一维聚类 (DBSCAN, eps=0.05) → 列
3. 每行内检测对齐 + 等间距
4. 每列内检测对齐 + 等间距
5. 若行数 ≥ 2 且列数 ≥ 2 → GRID
```

### 3.6 主入口

```python
def extract_constraints(
    elements: list[VLMOutputElement],
    mode: Literal["train", "infer"] = "train",
) -> list[ConstraintNode]:
    """
    提取所有约束。

    Args:
        elements: VLM 解析后的元素列表
        mode: 'train' 用 GT bbox (eps=0.02), 'infer' 用 VLM bbox (eps=0.05)

    Returns:
        约束节点列表
    """
    eps = 0.02 if mode == "train" else 0.05
    constraints = []
    constraints.extend(extract_alignment_constraints(elements, eps))
    constraints.extend(extract_containment_constraints(elements))
    constraints.extend(extract_same_size_constraints(elements, eps))
    constraints.extend(extract_spacing_constraints(elements, eps))
    constraints.extend(extract_grid_constraints(elements))
    return constraints
```

---

## 4. HeteroGraphBuilder

```python
class HeteroGraphBuilder:
    """
    从 VLM 输出构建 HeteroData 异构图。

    Usage:
        builder = HeteroGraphBuilder(config)
        data = builder.build(vlm_output)
    """

    def __init__(self, config: ModelConfig):
        self.eps_train = 0.02
        self.eps_infer = 0.05

    def build(
        self,
        elements: list[VLMOutputElement],
        constraints: Optional[list[ConstraintNode]] = None,
        mode: str = "train",
    ) -> HeteroData:
        """
        构建异构图。

        Returns:
            HeteroData with keys:
            - "element".x: (N_elem, 23)
            - "constraint".x: (N_con, 12)
            - ("element","satisfies","constraint").edge_index: (2, E)
            - ("element","satisfies","constraint").edge_attr: (E, 4)
            - ("constraint","satisfied_by","element").edge_index: (2, E)
            - ("constraint","satisfied_by","element").edge_attr: (E, 4)
        """
        ...

    def _build_element_nodes(self, elements: list) -> Tensor:
        """构建元素节点特征矩阵 (N_elem, 23)。"""
        ...

    def _build_constraint_nodes(self, constraints: list) -> Tensor:
        """构建约束节点特征矩阵 (N_con, 12)。"""
        ...

    def _build_edges(
        self, elements: list, constraints: list
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        构建边索引和边特征。
        正向 + 反向边。
        """
        ...
```

---

## 5. 可视化

```python
def plot_graph_on_screenshot(
    img: np.ndarray | Image.Image,
    hetero_data: HeteroData,
    element_types: list[str],
    constraint_types: list[str],
    show_labels: bool = True,
) -> plt.Figure:
    """
    在截图上覆盖图结构可视化。
    - 元素节点: 矩形框 + 类型标签 (颜色编码)
    - 约束节点: 小圆点 (不同颜色)
    - 边: 连线 (颜色按约束类型)
    """
    ...

def color_by_element_type() -> dict:
    """每种元素类型的颜色映射表。"""
    ...

def color_by_constraint_type() -> dict:
    """每种约束类型的颜色映射表。"""
    ...

def export_graph(hetero_data: HeteroData, path: str):
    """将图结构导出为 JSON 格式，供外部工具查看。"""
    ...
```

---

## 6. 训练模式 vs 推理模式

| 维度 | 训练 (train) | 推理 (infer) |
|------|-------------|-------------|
| 元素来源 | GT element bbox | VLM 预测 bbox |
| 约束提取 | 精确 (eps=0.02) | 宽松 (eps=0.05) |
| 约束过滤 | 所有约束保留 | 低置信度约束丢弃 (weight < 0.3) |
| 图增强 | 应用 augmentation | 不增强 |

---

## 7. 图增强 (Augmentation)

```python
class NodeDropout:
    """随机丢弃部分元素节点 (p=0.1)，模拟 VLM 漏检。"""
    ...

class CoordinateJitter:
    """给坐标加高斯噪声 (sigma=0.01)，模拟 VLM 定位误差。"""
    ...

class ConstraintPerturbation:
    """随机翻转约束边状态 (p=0.05)，模拟约束提取误差。"""
    ...

class GraphAugmentationPipeline:
    """组合多个增强变换。"""
    ...
```

---

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2026-05-25 | 初始版本 |
