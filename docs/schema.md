# 图 Schema 设计

## 概述

本文档定义 GUI 结构修正任务的异构二分图 schema。

## 节点类型

图包含两种节点类型:
- **元素节点**:表示检测到的 GUI 元素。
- **约束节点**:表示元素之间的空间约束。

### 元素节点特征

| 特征 | 类型 | 描述 |
|---------|------|-------------|
| bbox | float32[5] | (cx, cy, w, h) 归一化坐标 |
| type | int64 | 元素类型索引 |

### 约束节点特征

| 特征 | 类型 | 描述 |
|---------|------|-------------|
| constraint_type | float32[10] | 约束类型 one-hot 编码(align, space, contain, same-size, grid) |
| tolerance | float32 | 该约束的检测阈值 |

## 边类型

图包含两种用于消息传递的边类型:
- **element_to_constraint**: `(element, constraint)` — 将每个元素连接到其关联约束。
- **constraint_to_element**: `(constraint, element)` — 反向,用于第二跳消息传递。

### 边属性

| 属性 | 类型 | 描述 |
|-----------|------|-------------|
| weight | float32 | 约束置信度或强度 |

## HeteroData 结构

```python
data = HeteroData()

# 节点存储
data["element"].x = torch.randn(N_elements, 5)       # bbox 特征
data["constraint"].x = torch.randn(N_constraints, 11) # 类型 + 容差特征

# 边存储
data["element", "to", "constraint"].edge_index = ...  # 邻接
data["element", "to", "constraint"].weight = ...      # 边权重
data["constraint", "to", "element"].edge_index = ...  # 反向
data["constraint", "to", "element"].weight = ...      # 反向权重
```

## 消息传递流程

编码器执行两跳交替:

1. **跳 1(元素 → 约束)**: 每个约束节点聚合其关联元素的特征。
2. **跳 2(约束 → 元素)**: 每个元素节点聚合更新后的约束表示。

## 数据增强

训练期间,增强器应用:

| 增强 | 描述 | 参数 |
|-------------|-------------|------------|
| Bbox jitter | 向元素 bbox 坐标添加高斯噪声 | `jitter_std` |
| Drop constraint | 随机移除一部分约束 | `drop_ratio` |

## 可视化

`plot_graph` 函数将图结构叠加渲染在截图上:

- 元素节点: 红色矩形,带类型标签
- 约束节点: 蓝色圆点,带类型缩写
- 边: 灰色线段,透明度与边权重成正比
