# Task List — Bipartite-GNN-GUI

> Phase-based development plan following the structured engineering methodology:
> 需求分析 → 概要设计 → 详细设计 → 开发 → 集成测试 → 性能测试 → 实施 → 方案
>
> **P1 ✅ → P2 ✅ → P3 ✅ → P4 ✅ → P5 ✅ → P6 ✅ → P7 🔶 → P8 🔶 → P9 ⬜ → P10 ⬜**

---

## Phase 1: 需求分析 (Requirements Analysis) ✅

**Goal:** Understand the problem domain, analyze data formats, define what success looks like.
**Key artifacts:** `docs/requirements/` — data format specs, use case diagram, metrics definition.

---

### 1.1 VLM 输出格式分析 (`docs/requirements/vlm_format.md`)

**Verify:** Document covers all fields from both Qwen3.5-2B and MiniMax-VL-01 output JSONs.
**Depends on:** nothing.

- [x] **1.1.1** 收集 Qwen3.5-2B JSON 输出样例，分析字段结构和坐标格式
- [x] **1.1.2** 收集 MiniMax-VL-01 JSON 输出样例，分析字段结构和坐标格式
- [x] **1.1.3** 定义 `VLMOutputElement` 和 `VLMOutput` 数据类结构（骨架，不编码）
- [x] **1.1.4** 确定 `parse_qwen_output` / `parse_minimax_output` 接口和错误处理策略
- [x] **1.1.5** 确定全局元素类型分类体系（共享于 VLM 和 GT 之间）

### 1.2 Ground Truth 格式分析 (`docs/requirements/gt_format.md`)

**Verify:** Document covers GUI-360° and ScreenSpot annotation structure.

- [x] **1.2.1** 分析 GUI-360° JSON 标注格式
- [x] **1.2.2** 分析 ScreenSpot JSON 标注格式
- [x] **1.2.3** 定义 `GTElement` 和 `GroundTruth` 数据类结构
- [x] **1.2.4** 确定 VLM 预测 ↔ Ground Truth 匹配策略（IoU 代价矩阵 + 匈牙利算法）
- [x] **1.2.5** 确定评估中的 FP/FN 定义

### 1.3 用例定义与核心功能规划 (`docs/requirements/use_case.md`)

- [x] **1.3.1** 创建 Mermaid 用例图：VLM JSON → Graph → GNN → Corrected JSON
- [x] **1.3.2** 规划系统模块划分和模块间接口契约

### 1.4 非功能性需求：评估指标体系 (`docs/requirements/metrics.md`)

- [x] **1.4.1** 定义 `PositionError`
- [x] **1.4.2** 定义 `SizeError`
- [x] **1.4.3** 定义 `AlignmentError`
- [x] **1.4.4** 定义 `ElementRecall`
- [x] **1.4.5** 定义 `ElementPrecision`
- [x] **1.4.6** 定义 `ALL_METRICS` 注册策略和统计显著性方法

---

## Phase 2: 概要设计 (High-Level Design) ✅

**Goal:** Define system architecture, data schema, and component interaction.
**Key artifacts:** `docs/design/high_level.md`

---

- [x] **2.1.1-2.1.4** 配置系统设计: DataConfig, ModelConfig, TrainingConfig, Config 复合结构
- [x] **2.2.1-2.2.4** 日志与实验跟踪: setup_logger, MetricsLogger 基类, Wandb/Tensorboard/Noop
- [x] **2.3.1-2.3.5** 依赖管理: scipy, pydantic, wandb/tensorboard extras
- [x] **2.4.1-2.4.4** 图模式设计: ElementNode, ConstraintType(10种), ConstraintNode, EdgeFeatures
- [x] **2.5.1-2.5.6** 约束提取策略: Alignment/Containment/Spacing/Grid, 训练 vs 推理模式

---

## Phase 3: 详细设计 (Detailed Design) ✅

**Goal:** Define class hierarchies, interfaces, algorithms, and deployment plan.
**Key artifacts:** `docs/design/detailed.md`

---

- [x] **3.1.1-3.1.4** 数据层: CoordinateNormalizer, FeatureExtractor, GUIDataset, collate_dataloader
- [x] **3.2.1-3.2.4** 图构建层: HeteroGraphBuilder, 可视化, 增强变换, HeteroData 键结构
- [x] **3.3.1-3.3.4** 模型层: HeteroGraphSAGE, 3 个预测 Head, BipartiteGNNCorrector, CombinedLoss
- [x] **3.4.1-3.4.4** 训练与推理: Trainer, AdamW+cosine, 早停, InferencePipeline
- [x] **3.5.1-3.5.4** 评估层: Evaluator, 基线接口, 定性分析, 报告生成

---

## Phase 4: 开发 (Development) ✅

**Goal:** Implement all modules following the designs from Phases 1–3.

---

### 4.1-4.5 核心模块

所有核心模块已实现并验证:

- [x] **4.1** 基础设施: BBox 工具, 配置系统, 日志系统, 依赖声明
- [x] **4.2** 数据层: VLM 解析, GT 加载, 预处理, Dataset/DataLoader
- [x] **4.3** 图构建: Schema, 约束提取, Builder, 可视化, 增强
- [x] **4.3a** 数据适配: ScreenSpot, RICO View Hierarchy 加载器
- [x] **4.4** 模型层: 编码器, 预测头, 损失函数, 完整模型, 训练器, 推理管线
- [x] **4.5** 评估层: 指标, 评估器, 基线 (NoOp/Identity/Jitter), 定性分析

### 4.6 实验阶段

- [x] **4.6.1** 训练管线标准化: GraphDataset, run_experiment.py, configs/experiment.yaml
- [x] **4.6.2** 超参实验对比: 6 配置 sweep, Best: hd128 big-noise (val_loss=0.0537)
- [x] **4.6.3** VLM 推理管线: Qwen3-VL Flash (2947 elem), Qwen+Plus (7312), LLaVA (61), Moondream (弱)
- [x] **4.6.4** 实验总结: 核心发现 — GNN 无法战胜精度过高的 VLM, 也无法补足检测过弱的 VLM

### 4.7 方向调整

- [x] 核心发现: GNN 在精度上无法超越 VLM → 转向两个新方向
- [x] 新方向文档: `docs/research/direction_confidence_completion.md`

### 4.8 方向 1 — 约束感知置信度打分 ✅

> **Script:** `scripts/train_confidence.py`
> **Idea:** GNN 预测每个 VLM 检测的可靠性分数, 过滤低置信度检测

**方法:** GT 元素 (正样本) + 随机 imposter 元素 (负样本) → 训练存在性头部

**实验结果 (500 RICO, 50% imposter ratio):**

| 指标 | 值 |
|------|-----|
| Accuracy | **93.2%** |
| Precision | 99.1% |
| Recall | 90.7% |
| AUROC | **0.989** |

| # | Task | Status |
|---|------|--------|
| 4.8.1 | `scripts/train_confidence.py` — 训练管线 | ✅ |
| 4.8.2 | Imposter 生成（随机 bbox + 随机类型） | ✅ |
| 4.8.3 | 评估: AUROC, Precision/Recall, Accuracy | ✅ |
| 4.8.4 | 500 张 RICO 验证实验 | ✅ |

### 4.9 方向 2 — 结构性元素补全 ✅

> **Docs:** `docs/research/direction_confidence_completion.md`
> **Idea:** GNN 检测约束图中的"空洞"，预测缺失元素的位置和类型

**核心实验结果 (2000 RICO, 60% drop):**

| 任务 | 指标 | 结果 |
|------|------|------|
| 违反检测 | Accuracy | **95%** |
| 元素提议 | MSE | **0.044** |
| 类型预测 | 8 类 logits | ✅ |
| 元素提议 vs NN 基线 | IoU 提升 | **+40%** (drop≥0.6) |

| # | Task | Status |
|---|------|--------|
| 4.9.1 | `data/masking.py` — 合成元素删除管线 | ✅ |
| 4.9.2 | `model/heads.py:ElementProposalHead` — 元素提议头 | ✅ |
| 4.9.3 | 违反检测验证 (自监督预训练) | ✅ |
| 4.9.4 | 联合训练违反 + 提议头 | ✅ |
| 4.9.5 | 系统评估: 4 drop ratios × 2 seeds, 基线对比 | ✅ |
| 4.9.6 | 类型预测: 提议头输出 8 类 logits | ✅ |

**完整评估 (4 个 drop ratio, 500 RICO, 双 seed 平均):**

| drop | GNN Acc | GNN MSE | GNN IoU | NN MSE | NN IoU | GNN > NN? |
|------|---------|---------|---------|--------|--------|-----------|
| 0.2 | 92.5% | 0.073 | 0.047 | **0.020** | **0.057** | ❌ |
| 0.4 | 90.8% | 0.051 | 0.079 | **0.032** | **0.110** | ❌ |
| **0.6** | **91.4%** | 0.049 | **0.123** | 0.044 | 0.088 | **✅ (+40% IoU)** |
| **0.8** | **90.8%** | **0.044** | **0.097** | 0.048 | 0.062 | **✅** |

### 4.10 真实 VLM 测试 ⚠️

> **Script:** `scripts/evaluate_vlm_completion.py`

Qwen3-VL Flash 预测 (200 images) 通过完成管线运行。
RICO GT 稀疏 (obfuscated class names, 非可见元素多) 导致仅 32/193 图产生有效图。
**基础设施就绪**, 需更好的 GT 数据 (ScreenSpot, 人工标注) 才能评估。

---

## Phase 5: 集成测试 (Integration Testing) ✅

**Goal:** Verify end-to-end pipelines work on synthetic and real data.
**Status: 942 tests pass**

| Sub-phase | Items | 测试文件 |
|-----------|-------|---------|
| **5A** 原始管线 | 数据流 → 图构建 → 模型前向 → 端到端 → 基线 | `test_integration_5a.py` |
| **5B** 完成管线 | 违反图 → 遮掩 → 提议头 → 联合训练冒烟 → 评估冒烟 → 基线正确性 | `test_integration_5b.py` |

- [x] **5A.1** 数据管线: 合成 JSON → parse → Dataset → DataLoader
- [x] **5A.2** 图构建: 合成 JSON → constraints → HeteroData → verify keys
- [x] **5A.3** 模型前向: 梯度回传, loss 标量, 训练不 crash
- [x] **5A.4** 端到端: VLM JSON → InferencePipeline → corrected JSON
- [x] **5A.5** 评估基线: baselines + Evaluator → 所有指标
- [x] **5B.1** 违反图构建: drop=0/0.5/1 边界验证
- [x] **5B.2** 遮掩管线: mask_ratio=0/0.6/1 验证
- [x] **5B.3** 提议头: 输出形状, 梯度, Sigmoid 范围
- [x] **5B.4** 联合训练冒烟: `train_violation.py --n 10 --epochs 2`
- [x] **5B.5** 评估冒烟: `evaluate_completion.py --n 10 --epochs 2`
- [x] **5B.6** 基线正确性: NN, Center 基线数值合理

---

## Phase 6: 性能测试 (Performance Testing) ✅

**Goal:** Establish performance baselines and ensure practical usability.
**Script:** `scripts/benchmark_performance.py` → `experiments/benchmarks/performance_results.json`
**Report:** `docs/research/phase6_benchmark_report.md`

| Benchmark | Metrics | Result |
|-----------|---------|--------|
| **6.1** 数据加载吞吐 | 200 RICO JSONs → graph build | 2.1ms/img = **467 img/s** |
| **6.2** 图构建扩展性 | 10 / 50 / 100 / 500 elem | 0.2ms → 255ms (O(N²)) |
| **6.3** 训练吞吐量 | 50 graphs × 3 epochs, hidden=64 | **357 steps/s** |
| **6.4** 推理延迟 | 100 graphs p50/p95/p99 | **0.53ms / 0.96ms / 1.11ms** |

**结论:** GNN 从不是瓶颈。推理 0.5ms p50, VLM 才是限速步骤 (~2s/图)。

---

## Phase 7: 实施 (Implementation — 实验运行)

**Goal:** Define and execute experiment methodology, ensure reproducibility.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 7.1 | `experiments/run.py` 统一入口 | ✅ Done | 4 子命令: train-violation, train-confidence, evaluate-completion, constraint-ablation |
| 7.2 | 约束类型消融 | ✅ Done | CONTAINMENT 最关键 (acc drop 90.8→88.9%); alignment 提供最优空间信号 |
| 7.3 | 图构建超参敏感性 | ✅ Done | Phase 4.6.2 sweep: hd 64/128/256 + lr 1e-3/5e-4 |
| 7.4 | VLM 噪声鲁棒性 | ✅ Done | Phase 4.6.3-4.6.4: 5 类 VLM 全覆盖 |
| 7.5 | 跨数据集泛化 | ✅ Done | RICO→ScreenSpot: 28.1% zero-shot → 72.1% after VLM pseudo-GT fine-tune (+44pp) |
| 7.6 | 可复现性 | ✅ Done | seed_everything + deterministic 已验证 |

---

## Phase 8: 方案 (Solution — 文档与资料更新)

**Goal:** Update product/technical documentation for usability and publication.

| # | Item | Status | Notes |
|---|------|--------|-------|
| 8.1 | README.md 更新 | ✅ Done | 安装/用法/实验结果完整 |
| 8.2 | configs/default.yaml 示例 | ❌ 低优先级 | 现有 configs/experiment.yaml 可用 |
| 8.3 | examples/ 目录 | ❌ 低优先级 | scripts/ 目录已有完整示例 |
| 8.4 | pyproject.toml 最终版 | ✅ Done | 依赖/entry points 已配置 |

---

## Phase 9: Web Demo (Web 演示) ⬜

**Goal:** Single-page web app: upload screenshot → VLM + GNN → side-by-side bbox overlay.

**Dependencies:** InferencePipeline + trained checkpoint.

| # | Item | Status |
|---|------|--------|
| 9.1 | FastAPI 后端: `/api/correct`, `/api/health`, VLM 适配器, CLI 参数 | ⬜ |
| 9.2 | 前端: 上传区 + Canvas bbox overlay + before/after 切换 + JSON 对比 | ⬜ |
| 9.3 | 测试与文档: e2e 测试, Web README | ⬜ |
| 9.4 | 部署: Dockerfile, docker-compose | ⬜ |

---

## Phase 10: HTML/CSS 代码生成 ⬜

**Goal:** Convert corrected element JSON into a standalone HTML file.

**Dependencies:** InferencePipeline, Phase 9 API.

| # | Item | Status |
|---|------|--------|
| 10.1 | `web/codegen/html_generator.py`: bbox → absolute CSS, label → HTML tag 映射 | ⬜ |
| 10.2 | `POST /api/generate-html` 端点 + ?download 支持 | ⬜ |
| 10.3 | 前端: HTML 预览区, 下载/复制按钮 | ⬜ |
| 10.4 | 单元测试: 空列表, 各类型, z-index 排序, 边界情况 | ⬜ |

---

## 方法论对照

| 阶段 | TASK 对应 | 产出 |
|------|-----------|------|
| 需求分析 | Phase 1 | `docs/requirements/` |
| 概要设计 | Phase 2 | `docs/design/high_level.md` |
| 详细设计 | Phase 3 | `docs/design/detailed.md` |
| 开发 | Phase 4 | `src/bipartite_gnn_gui/` |
| 集成测试 | Phase 5 | 942 tests pass |
| 性能测试 | Phase 6 | Benchmark report |
| 实施 | Phase 7 | Experiment scripts |
| 方案 | Phase 8 | README, docs, examples |
| Web 演示 | Phase 9 | `web/` (FastAPI + frontend) |
| 代码生成 | Phase 10 | `web/codegen/` (JSON → HTML/CSS) |

---

## 执行原则

1. **Phase 1-3 轻量、Phase 4 厚重**: 分析和设计产出 markdown 文档而非代码
2. **不回溯**: Phase 1 的分析假设在整个项目中保持不变
3. **Phase 5-6 可在 Phase 4 中间穿插**: 模块开发完即可运行集成测试
4. **Phase 7 依赖 Phase 4-6 全部完成**: 实验使用完整的系统
5. **PR 每 checkbox 一个**: 推分支 → PR → 合并 (Ship Incrementally)
6. **Phase 9 依赖 4.4.6 (InferencePipeline) + checkpoint**: mock 模式可并行开发
7. **Phase 10 可独立开发**: 纯函数, 无 ML 依赖

---

## Stretch Goals

| # | 描述 |
|---|------|
| S1 | Attention-based constraint importance weighting |
| S2 | Cross-attention between VLM features and graph features |
| S3 | Multi-scale graph: hierarchical container → child → leaf |
| S4 | Synthetic GUI layout generator for data augmentation |
| S5 | ONNX / TorchScript export for deployment |
