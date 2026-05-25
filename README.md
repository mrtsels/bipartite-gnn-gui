# Bipartite-GNN-GUI

**Heterogeneous Bipartite GNN for GUI Structure Error Correction**

*异构二分图神经网络用于GUI结构错误修正*

---

## Project Overview / 项目概述

**English**

Bipartite-GNN-GUI is a research project that addresses the problem of GUI element parsing errors produced by lightweight Vision-Language Models (VLMs). When lightweight VLMs (e.g., Qwen3.5-2B, MiniMax-VL-01) parse GUI screenshots into structured JSON, they commonly suffer from **element omission** (missing UI elements) and **misalignment** (incorrect bounding box coordinates). This project proposes a post-correction framework that:

1. Takes the initial (noisy) JSON output from a lightweight VLM.
2. Constructs a **heterogeneous bipartite graph** with two node types:
   - *Element nodes* – each representing a detected GUI element (button, text, image, etc.).
   - *Constraint nodes* – representing spatial & structural priors (alignment, containment, spacing, grid).
3. Applies **GraphSAGE message passing** across the bipartite structure to propagate constraint information to element nodes.
4. Predicts a **coordinate refinement vector** Δ𝐱ᵢ = (Δx, Δy, Δw, Δh) for every element, correcting the VLM's initial bounding box prediction.

**中文**

Bipartite-GNN-GUI 是一个研究项目，旨在解决轻量级视觉语言模型 (VLM) 在 GUI 元素解析中产生的错误。当轻量级 VLM（如 Qwen3.5-2B、MiniMax-VL-01）将 GUI 截图解析为结构化的 JSON 时，经常出现**元素遗漏**（缺少 UI 元素）和**位置偏移**（边界框坐标错误）。本项目提出了一个后修正框架：

1. 接收轻量级 VLM 输出的初始（含噪声）JSON 结果。
2. 构建**异构二分图**，包含两种节点类型：
   - *元素节点* – 每个检测到的 GUI 元素（按钮、文本、图片等）。
   - *约束节点* – 表示空间和结构先验知识（对齐、包含、间距、网格）。
3. 通过二分图结构进行 **GraphSAGE 消息传递**，将约束信息传播到元素节点。
4. 为每个元素预测**坐标修正向量** Δ𝐱ᵢ = (Δx, Δy, Δw, Δh)，从而修正 VLM 预测的初始边界框。

---

## Background / 背景

**English**

Lightweight VLMs (under 3B parameters) are attractive for on-device GUI understanding due to their low latency and memory footprint. However, our empirical analysis shows that:

- **Element omission**: Models frequently miss 10–30% of visible GUI elements, especially small icons, dividers, and nested containers.
- **Misalignment**: Even when elements are detected, bounding box coordinates can be off by 10–50+ pixels, leading to broken layout trees and incorrect downstream action prediction.
- **Structural inconsistencies**: Detected layouts often violate basic GUI design principles (misaligned groups, inconsistent spacing, overlapping elements).

Existing approaches rely on fine-tuning larger VLMs (7B+) or cascading object detectors, both of which are computationally expensive. Our method instead treats GUI correction as a **structured prediction on a bipartite graph**, leveraging spatial constraints without requiring additional detection models or VLM fine-tuning.

**中文**

轻量级 VLM（<3B 参数）因其低延迟和低内存占用而在设备端 GUI 理解中具有吸引力。然而，我们的实证分析表明：

- **元素遗漏**：模型经常遗漏 10–30% 的可见 GUI 元素，尤其是小图标、分割线和嵌套容器。
- **位置偏移**：即使检测到元素，边界框坐标也可能偏离 10–50+ 像素，导致布局树破损和下游动作预测错误。
- **结构不一致**：检测到的布局经常违反基本 GUI 设计原则（组对齐错误、间距不一致、元素重叠）。

现有方法依赖于微调更大的 VLM（7B+）或级联目标检测器，两者计算成本都很高。我们的方法将 GUI 修正视为**二分图上的结构化预测**，利用空间约束，无需额外的检测模型或 VLM 微调。

---

## Method / 方法

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│  Lightweight VLM │────▶│  Initial Noisy JSON   │────▶│  Bipartite Graph      │
│  (Qwen3.5-2B /   │     │  (elements w/ coords) │     │  (Element × Constraint)│
│   MiniMax-VL-01) │     └──────────────────────┘     └───────────┬───────────┘
└─────────────────┘                                               │
                                                                  ▼
┌──────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│  Corrected GUI JSON  │◀────│  Coordinate Refinement │◀────│  GraphSAGE            │
│  (refined bboxes)    │     │  Δ𝐱ᵢ = (Δx,Δy,Δw,Δh) │     │  Message Passing      │
└──────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

### Key Components / 关键组件

| Component | Description |
|-----------|-------------|
| **Lightweight VLM** | Produces initial JSON with predicted element types and bounding boxes (x, y, w, h). |
| **Bipartite Graph** | Heterogeneous graph `G = (V_e ∪ V_c, E)` where `V_e` = element nodes, `V_c` = constraint nodes. Edges encode spatial relationships. |
| **GraphSAGE** | Inductive message-passing layers that aggregate neighbor information to refine node representations. |
| **Refinement Head** | MLP that predicts Δ𝐱ᵢ from the final element node embeddings. |
| **Loss Function** | `ℒ = ℒ_coord + λ₁ℒ_violation + λ₂ℒ_alignment` — combines coordinate regression, structural violation penalty, and alignment consistency. |

---

## Installation / 安装

### Prerequisites / 前置要求

- Python 3.10+
- CUDA-capable GPU (recommended, but CPU inference is supported)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-org/bipartite-gnn-gui.git
cd bipartite-gnn-gui

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

---

## Datasets / 数据集

| Dataset | Description | GUI Elements | Screenshots |
|---------|-------------|-------------|-------------|
| **GUI-360°** | 360° comprehensive GUI understanding dataset with pixel-level annotations. | ~50K | ~3.5K |
| **ScreenSpot** | GUI grounding dataset with fine-grained element annotations across mobile, web, and desktop. | ~30K | ~5K |

---

## Metrics / 评估指标

| Metric | Formula | Description |
|--------|---------|-------------|
| **PositionError** | `‖(x̂, ŷ) − (x, y)‖₂` | Euclidean distance between predicted and ground-truth top-left corner. |
| **SizeError** | `‖(ŵ, ĥ) − (w, h)‖₂` | Euclidean distance between predicted and ground-truth width & height. |
| **AlignmentError** | `∑₍ᵢ,ⱼ₎∈A |dxᵢ − dxⱼ| + |dyᵢ − dyⱼ|` | Deviation from expected alignment groups. |

---

## Project Structure / 项目结构

```
bipartite-gnn-gui/
├── README.md
├── TASK.md
├── pyproject.toml
├── .gitignore
├── requirements.txt
├── src/
│   └── bipartite_gnn_gui/
│       ├── __init__.py
│       ├── data/          # Dataset loading & preprocessing
│       ├── graph/         # Heterogeneous bipartite graph construction
│       ├── model/         # GraphSAGE GNN model & refinement head
│       ├── eval/          # Evaluation metrics (PositionError, SizeError, AlignmentError)
│       └── utils/         # Utility functions
└── tests/
    └── __init__.py
```

---

## License / 许可证

MIT License

## Citation / 引用

```bibtex
@software{bipartite_gnn_gui,
  title = {Bipartite-GNN-GUI: Heterogeneous Bipartite GNN for GUI Structure Error Correction},
  year = {2026},
  url = {https://github.com/your-org/bipartite-gnn-gui}
}
```
