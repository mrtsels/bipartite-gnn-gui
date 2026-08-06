# 算法:用于 GUI 空间错误修正的异构二分图 GNN

> **数学基础与核心逻辑。**
> 与 `src/bipartite_gnn_gui/` 直接对应。

---

## 1. 问题定义

轻量级 VLM(例如 Qwen3-VL Flash)输入截图,输出带边界框的 GUI 元素列表。这些预测是**有噪声的**:VLM 的空间理解是近似的,即使语义标注正确,位置也常存在 5–25 px 的误差、尺寸存在 3–15% 的误差。

我们将此视为**结构化精修问题**。输入是 N 个含噪声的元素预测;输出是每个元素的修正量 $\Delta\mathbf{x}_i = (\Delta x, \Delta y, \Delta w, \Delta h)_i$,施加后得到与 GUI 空间设计原则一致的修正布局。

关键洞察:GUI 布局**并非随机**。它们遵循可预测的空间规则——对齐、等间距、包含、统一尺寸。这些规则是先验,使我们能够修正单看单个元素无法恢复的 VLM 噪声。

---

## 2. 为什么用异构二分图?

### 2.1 表示论依据

GUI 布局包含两类语义完全不同的实体:

| 实体类型 | 表示什么 | 特征空间 |
|---|---|---|
| **元素节点** | 具体的 UI 控件 | 空间位置 (cx, cy, w, h)、类型 one-hot、置信度 |
| **约束节点** | 一种空间关系 | 约束类型 one-hot、容差、几何参数 |

同构图(例如全元素 + 两两连边)会混淆这两者。元素–元素边隐式地编码两两关系;而约束节点使关系**显式、带类型、可学习**——每个约束都是一等对象,拥有自己的嵌入,可通过消息传递更新。

### 2.2 二分结构:为什么要两个集合?

图按构造即为二分:

$$
G = (V_e \cup V_c,\; E),\qquad E \subseteq V_e \times V_c
$$

边只存在于元素与约束之间,**绝无**元素–元素或约束–约束边。这带来一种归纳偏置:

1. **元素只能通过共享约束通信。** 两个按钮不会直接互传消息——它们通过共同参与的 "ALIGN_LEFT" 约束传递。这迫使模型以具名空间规则的形式为任何坐标变更提供依据。

2. **约束聚合局部证据。** 一个约束节点看到所有声称满足它的元素。若三个按钮声称左对齐但其中一个偏了 10 px,该差异就被编码进约束嵌入。

3. **两跳消息传递足够。** 一跳元素 → 约束(约束聚合),一跳约束 → 元素(元素更新)。每个元素都能看到与之共享约束的其他元素,恰好相距两跳。

### 2.3 异构 vs 同构

同构图给所有节点分配相同的特征空间和更新函数。这不适用于我们的场景,因为元素节点 $\mathbf{h}_e \in \mathbb{R}^{D_e}$ 与约束节点 $\mathbf{h}_c \in \mathbb{R}^{D_c}$ 维度不同、语义不同。异构 GNN 在消息传递前施加**类型特定的线性变换**,确保每种节点类型被投影到兼容空间,同时保留其类型特定结构。

具体地,编码器在**任何消息传递之前**对元素与约束特征分别应用独立的 MLP:

$$
\mathbf{h}_e^{(0)} = \text{MLP}_e(\mathbf{x}_e),\qquad
\mathbf{h}_c^{(0)} = \text{MLP}_c(\mathbf{x}_c)
$$

---

## 3. 节点与边特征空间

### 3.1 元素节点特征

每个元素 $e_i$ 携带原始特征向量:

$$
\mathbf{x}_{e_i} = [x_1, y_1, x_2, y_2, a_i] \in \mathbb{R}^5
$$

其中 $(x_1, y_1, x_2, y_2)$ 是预测边界框(归一化到 $[0, 1]^4$),$a_i$ 是归一化面积 $(x_2 - x_1)(y_2 - y_1)$。可选地拼接一个冻结的视觉特征 $v_i \in \mathbb{R}^d$(192 维 ViT-Tiny 或 768 维 DINOv2)。

### 3.2 约束节点特征

每个约束 $c_j$ 携带参数向量:

$$
\mathbf{x}_{c_j} = [\underbrace{t_1, \ldots, t_{10}}_{\text{类型 one-hot}},\;
             \text{spatial statistics}] 
$$

其中类型 one-hot 编码 10 种约束类型,空间统计量包括参与元素的平均两两距离、包含重叠率、对齐残差等。

### 3.3 边特征

边携带几何特征,描述连接元素与约束之间的空间关系:

$$
\mathbf{e}_{ij} = [d_{ij},\; \Delta x_{ij},\; \Delta y_{ij},\;
                   \text{IoU}_{ij}] \in \mathbb{R}^4
$$

这些由元素预测框与约束参数计算得到。

---

## 4. 消息传递架构

### 4.1 两跳二分图消息流

```
          跳 1                   跳 2
    ┌───────────────────┐   ┌───────────────────┐
    │ 元素 → 约束          │   │ 约束 → 元素          │
    │ (聚合证据)          │   │ (更新位置)          │
    └───────────────────┘   └───────────────────┘

    e₁ ──┐                        e₁' ←──┐
    e₂ ──┤ → c₁ (align_left)  →   e₂' ←──┤ → c₁
    e₃ ──┘                        e₃' ←──┘
```

**跳 1 — 约束聚合:** 每个约束 $c_j$ 从所有与之相连的元素收集特征:

$$
\mathbf{h}_{c_j}^{(k+1)} = \sigma\!\left(
    \mathbf{W}_c^{(k)} \cdot \text{MEAN}\!\left(
        \{\mathbf{h}_{e_i}^{(k)} : (e_i, c_j) \in E\}
    \right) + \mathbf{b}_c^{(k)}
\right)
$$

**跳 2 — 元素精修:** 每个元素 $e_i$ 从它参与的所有约束收集特征:

$$
\mathbf{h}_{e_i}^{(k+1)} = \sigma\!\left(
    \mathbf{W}_e^{(k)} \cdot \text{MEAN}\!\left(
        \{\mathbf{h}_{c_j}^{(k+1)} : (e_i, c_j) \in E\}
    \right) + \mathbf{b}_e^{(k)}
\right)
$$

在 PyG 中,这通过 `SAGEConv` 层实现(`BipartiteGraphSAGE`,见 `src/bipartite_gnn_gui/model/encoder.py`):元素 → 约束与约束 → 元素各一组卷积,交替进行。

### 4.2 层数

两轮二分消息传递(`n_layers = 2`)意味着每个元素看到两跳之内的约束。由于图严格二分,2 跳即覆盖整个感受野——每个元素都能看到与之共享约束的所有其他元素。增加层数可带来高阶效应(约束通过共享元素影响其他约束),但对核心空间修正任务帮助有限,属于次要效应。

---

## 5. 约束的正式定义

每种约束类型定义元素边界框上的数学谓词。当谓词值低于容差 $\varepsilon$ 时,约束**成立**。

### 5.1 对齐约束

两个元素,框分别为 $(x_1, y_1, x_2, y_2)$ 与 $(x_1', y_1', x_2', y_2')$:

| 约束 | 谓词 | 含义 |
|---|---|---|
| ALIGN_LEFT | $|x_1 - x_1'|$ | 左边缘对齐 |
| ALIGN_RIGHT | $|x_2 - x_2'|$ | 右边缘对齐 |
| ALIGN_TOP | $|y_1 - y_1'|$ | 上边缘对齐 |
| ALIGN_BOTTOM | $|y_2 - y_2'|$ | 下边缘对齐 |
| CENTER_X | $|(x_1 + x_2)/2 - (x_1' + x_2')/2|$ | 水平中心对齐 |
| CENTER_Y | $|(y_1 + y_2)/2 - (y_1' + y_2')/2|$ | 垂直中心对齐 |

### 5.2 尺寸约束

| 约束 | 谓词 | 含义 |
|---|---|---|
| SAME_SIZE | $\max\left(\frac{|w - w'|}{w'},\; \frac{|h - h'|}{h'}\right)$ | 相对容差内宽度、高度相等 |

### 5.3 空间配置约束

| 约束 | 谓词 | 含义 |
|---|---|---|
| SPACING | $\vert\text{gap}_{i,i+1} - \text{gap}_{i+1,i+2}\vert$ | 相邻元素间距一致 |
| CONTAINMENT | $x_1' \leq x_1 \;\wedge\; y_1' \leq y_1 \;\wedge\; x_2 \leq x_2' \;\wedge\; y_2 \leq y_2'$ | 元素 $i$ 完全位于元素 $j$ 内 |
| GRID | 由元素中心聚类得到的行/列归属 | 元素构成规则的二维阵列 |

### 5.4 约束提取:训练 vs 推理

| 方面 | 训练 | 推理 |
|---|---|---|
| 元素来源 | 真实标注框 | VLM 预测框 |
| 容差 $\varepsilon$ | 0.02(紧,来自干净 GT) | 0.05(松,容纳 VLM 噪声) |
| 约束过滤 | 保留全部 | 丢弃 $w_j < 0.3$ 的约束 |
| 约束权重 $w_j$ | 1.0(已知正确) | 启发式置信度分数 |

训练时,系统从真实标注中提取约束——它们是模型学习去执行的"正确"空间规则。推理时,系统从 VLM 预测中启发式地提出约束,这些约束可能有误,因此丢弃低权重约束以防止传播错误的结构信息。

---

## 6. 预测头

多个 MLP 头作用于编码器产生的精修嵌入(实现见 `src/bipartite_gnn_gui/model/heads.py`):

### 6.1 坐标修正头

将每个元素嵌入映射到 4 维修正量:

$$
\Delta\mathbf{x}_i = \text{MLP}_{\text{coord}}(\mathbf{h}_{e_i}^{(L)})
    \in \mathbb{R}^4
$$

修正后的边界框为:

$$
\hat{\mathbf{x}}_i = \mathbf{x}_i + \Delta\mathbf{x}_i
$$

输出不加激活——修正量可正可负。推理时,修正量可选地限制在 $[-0.5, 0.5]$ 以防止在极端噪声输入上爆炸。

### 6.2 违反检测头

将每个约束嵌入映射到标量概率:

$$
v_j = \sigma(\text{MLP}_{\text{vio}}(\mathbf{h}_{c_j}^{(L)}))
    \in [0, 1]
$$

$v_j \approx 1$ 表示约束很可能被违反(触发该约束的边界框实际上并不满足它)。这个辅助信号帮助模型学会区分有信息量的约束与偶然的约束。

### 6.3 存在性打分头

将每个元素嵌入映射到标量概率:

$$
p_i = \sigma(\text{MLP}_{\text{exist}}(\mathbf{h}_{e_i}^{(L)}))
    \in [0, 1]
$$

$p_i \approx 0$ 表示元素很可能是幻觉(VLM 预测了不存在的东西)。该头使模型能够抑制误检,作为修正坐标之外的另一种手段。

### 6.4 元素补全头

检测约束图中的"空洞",提出缺失元素的框与类型:

$$
\hat{\mathbf{b}}_k, \hat{t}_k = \text{MLP}_{\text{proposal}}(\mathbf{h}_{c_j}^{(L)})
$$

训练时随机删除一部分真实元素;被删元素留下的悬空约束正是缺失元素在图中留下的特征。该头从聚合的约束嵌入中预测缺失元素的框和类型,是**自监督**的——无需额外人工标注。

### 6.5 头结构

每个 MLP 头均为两层:

$$
\text{MLP}(\mathbf{h}) = \mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2
$$

坐标头输出 4 维;违反与存在性头输出 1 维(后接 sigmoid);补全头输出框坐标与类型 logits。

---

## 7. 损失函数

总损失是四项损失的加权和:

$$
\mathcal{L} = w_c \cdot \mathcal{L}_{\text{coord}}
             + w_v \cdot \mathcal{L}_{\text{vio}}
             + w_e \cdot \mathcal{L}_{\text{exist}}
             + w_p \cdot \mathcal{L}_{\text{prop}}
$$

默认权重 $w_c = 1.0, w_v = 0.5, w_e = 0.5, w_p = 0.5$(见 `src/bipartite_gnn_gui/model/losses.py`)。

### 7.1 坐标损失

预测修正量与真实修正量之间的均方误差:

$$
\mathcal{L}_{\text{coord}} = \frac{1}{N} \sum_{i=1}^{N}
    \|\Delta\mathbf{x}_i - \Delta\mathbf{x}_i^{\text{gt}}\|_2^2
$$

真实修正量是 VLM 预测框与标注框之差:$\Delta\mathbf{x}_i^{\text{gt}} = \mathbf{x}_i^{\text{gt}} - \mathbf{x}_i^{\text{pred}}$。

**为什么用 MSE 而非 Smooth L1 或 IoU 损失?** 对于 GUI 坐标修正,MSE 对较大误差按二次方惩罚,这正合适——从感知布局质量看,20 px 误差远不止 10 px 误差的两倍。Smooth L1 会低估大修正。

### 7.2 违反损失

预测违反分数与真实标签之间的二元交叉熵:

$$
\mathcal{L}_{\text{vio}} = -\frac{1}{M} \sum_{j=1}^{M}
    \left[ y_j \log v_j + (1 - y_j) \log(1 - v_j) \right]
$$

其中 $y_j = \mathbf{1}[\text{约束 } c_j \text{ 确实被违反}]$。训练时,违反标签通过将 GT 派生约束与 VLM 的含噪预测对比得出。

### 7.3 存在性损失

预测存在概率与真实标签之间的二元交叉熵:

$$
\mathcal{L}_{\text{exist}} = -\frac{1}{N} \sum_{i=1}^{N}
    \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]
$$

其中 $y_i = 1$ 表示元素 $i$ 是真实 GUI 元素(与 GT 匹配),$y_i = 0$ 表示它是幻觉(FP)。

### 7.4 补全损失

被删除元素的提议框与真实框之间的 IoU 损失,加上类型预测的交叉熵:

$$
\mathcal{L}_{\text{prop}} = \frac{1}{K} \sum_{k=1}^{K}
    \left[ \mathcal{L}_{\text{IoU}}(\hat{\mathbf{b}}_k, \mathbf{b}_k^{*})
           + \alpha \cdot \text{CE}(\hat{t}_k, t_k^{*}) \right]
$$

其中 $K$ 是存在缺失目标的被违反约束数。

### 7.5 损失加权

坐标损失通常在量级上占主导(4 维 MSE vs 标量 BCE)。实践中权重应使各损失分量在训练开始时对总梯度贡献大致相等。

---

## 8. 训练动态

### 8.1 优化

- **优化器:** AdamW,权重衰减 $10^{-5}$(仅作用于非 bias 参数)
- **学习率:** $10^{-3}$ 峰值,1000 步线性预热,余弦退火至 $10^{-6}$
- **梯度裁剪:** 最大 L2 范数 = 1.0
- **混合精度:** CUDA 可用时使用 FP16 (AMP)
- **批大小:** 每批 32 个 HeteroData 图

### 8.2 早停

当验证损失连续 20 个 epoch 无改善时停止训练,保留验证损失最优的 checkpoint。

### 8.3 实际效果

实验表明(详见 [`report/`](../report/)):违反检测准确率 90–94%,高缺失率下补全 IoU 较最近邻基线提升 +39%/+56%,200 张真实 VLM 截图端到端 F1 +2.0 pp。模型收敛行为与训练配置见 `experiments/` 与 `configs/`。

---

## 9. 推理管线

```
VLM JSON → parse → ElementNodes → extract constraints → HeteroData
    → encoder(h) → 4 heads → Δx + proposals → clamp → corrected JSON
```

1. **解析** VLM JSON 为归一化框的 `VLMOutputElement` 对象。
2. **提取约束** 从预测框中启发式提取(宽松容差 $\varepsilon = 0.05$,丢弃低置信度约束)。
3. **构建** `HeteroData` 二分图。
4. **编码** 通过 `BipartiteGraphSAGE` 得到精修嵌入。
5. **预测** 从元素嵌入预测坐标修正量;从约束嵌入提出缺失元素。
6. **应用** 修正量:$\hat{\mathbf{x}}_i = \mathbf{x}_i + \Delta\mathbf{x}_i$。
7. **合并** 补全提案经非极大值抑制 (NMS) 后与检测结果合并。
8. **裁剪** 修正坐标到 $[0, 1]$(有效图像空间)。
9. **反归一化** 如需要转换回绝对像素坐标。
10. **输出** 与 VLM 输入同 schema 的修正后 JSON。

违反与存在性头在推理时也可用于标记低置信度元素(如演示中的可靠性打分),供人工复核。

---

## 10. 为什么有效:关键直觉

### 10.1 约束作为归纳偏置

标准 MLP 修正器看到 $\mathbb{R}^{5N}$,必须在没有任何结构先验的情况下学习到 $\mathbb{R}^{4N}$ 的映射。二分 GNN 将其分解为局部的、按约束类型参数化的消息传递操作。模型不需要自己发现"左对齐"这回事——该结构被显式给出。

### 10.2 消息传递作为约束满足

两跳流程(元素 → 约束 → 元素)实现了约束满足的可微松弛。当三个元素连接到同一个 ALIGN_LEFT 约束:

1. 约束节点收到全部三个 x 坐标。
2. 它能检测离群点(某个 x₁ 与其他明显不同)。
3. 它回传梯度信号,将离群点推向共识。

这类似于因子图中的一个信念传播轮次:约束节点是因子,元素节点是变量。

### 10.3 多任务学习作为正则化

违反与存在性头是辅助任务,迫使编码器产生多种用途的嵌入。这防止编码器塌缩到平凡解(例如总是预测零修正量)。存在性头尤其给模型一个处理幻觉元素的"逃生通道"——不必去修正一个不存在的按钮,而是可以抑制它。

### 10.4 尺度分离

GNN 在 $[0, 1]^2$ 的**归一化**坐标上运行。这意味着同一模型适用于任何分辨率的截图。坐标修正量同样在归一化空间,$\Delta x = 0.01$ 表示图像宽度的 1%,无论图像是 720 px 还是 4K。

---

## 11. 当前状态

| 组件 | 状态 | 说明 |
|---|---|---|
| 图 schema | ✅ 已实现 | `ElementNode`、`ConstraintNode`、`EdgeType`、10 种 `ConstraintType` |
| 图构建 | ✅ 已实现 | `BipartiteGraphBuilder.build()` 生成合法 `HeteroData` |
| 约束提取 | ✅ 已实现 | 全部 10 种类型,含训练/推理模式 |
| 编码器 | ✅ 已实现 | `BipartiteGraphSAGE` 两跳 SAGEConv 消息传递 |
| 预测头 | ✅ 已实现 | 坐标 / 违反 / 存在性 / 补全 |
| 损失函数 | ✅ 已实现 | 4 项加权损失 |
| 训练器 | ✅ 已实现 | AdamW + 余弦退火 + 早停 |
| 推理 | ✅ 已实现 | `InferencePipeline` 端到端修正 |
| 评估 | ✅ 已实现 | PositionError / AlignmentError / Recall / Precision / F1 / IoU |

---

## 参考文献

- Hamilton, W. et al. "Inductive Representation Learning on Large Graphs." NeurIPS 2017.(*GraphSAGE*)
- Fey, M. & Lenssen, J.E. "Fast Graph Representation Learning with PyTorch Geometric." ICLR 2019 Workshop.(*PyG HeteroData*)
- 10 种空间约束类型源自 GUI 布局惯例(material design、iOS HIG、web CSS 盒模型),而非特定论文。
