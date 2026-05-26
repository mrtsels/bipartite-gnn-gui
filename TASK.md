# Task List — Bipartite-GNN-GUI

> Phase-based development plan following the structured engineering methodology:
> 需求分析 → 概要设计 → 详细设计 → 开发 → 集成测试 → 性能测试 → 实施 → 方案
>
> **~110 subtasks across 8 phases.**

---

## Phase 1: 需求分析 (Requirements Analysis) ✅

**Goal:** Understand the problem domain, analyze data formats, define what success looks like.
**Key artifacts:** `docs/requirements/` — data format specs, use case diagram, metrics definition.

---

### 1.1 VLM 输出格式分析 (`docs/requirements/vlm_format.md`)

**Verify:** Document covers all fields from both Qwen3.5-2B and MiniMax-VL-01 output JSONs.
**Depends on:** nothing.

- [x] **1.1.1** 收集 Qwen3.5-2B JSON 输出样例，分析字段结构和坐标格式
  - 记录: bbox 格式 (xyxy/xywh/cxcywh), 坐标系原点, 绝对/相对值
  - 产出: `vlm_format.md` 中 Qwen 格式说明

- [x] **1.1.2** 收集 MiniMax-VL-01 JSON 输出样例，分析字段结构和坐标格式
  - 同上，记录差异
  - 产出: `vlm_format.md` 中 MiniMax 格式说明

- [x] **1.1.3** 定义 `VLMOutputElement` 和 `VLMOutput` 数据类结构（骨架，不编码）
  - 确定必填字段 vs 可选字段、缺失值默认值策略
  - 确定坐标归一化需求（absolute→relative, 原点转换）

- [x] **1.1.4** 确定 `parse_qwen_output` / `parse_minimax_output` 接口和错误处理策略
  - 接口签名、返回值类型、异常情况

- [x] **1.1.5** 确定全局元素类型分类体系（共享于 VLM 和 GT 之间）
  - 类型枚举: button, text, image, input, icon, container, list, etc.

### 1.2 Ground Truth 格式分析 (`docs/requirements/gt_format.md`)

**Verify:** Document covers GUI-360° and ScreenSpot annotation structure.
**Depends on:** nothing.

- [x] **1.2.1** 分析 GUI-360° JSON 标注格式，记录字段和结构
  - 产出: `gt_format.md` 中 GUI-360° 说明

- [x] **1.2.2** 分析 ScreenSpot JSON 标注格式，记录字段和结构
  - 产出: `gt_format.md` 中 ScreenSpot 说明

- [x] **1.2.3** 定义 `GTElement` 和 `GroundTruth` 数据类结构（骨架，不编码）
  - 确定跨数据集的统一表示

- [x] **1.2.4** 确定 VLM 预测 ↔ Ground Truth 匹配策略
  - IoU 代价矩阵 + 匈牙利算法，筛选阈值策略

- [x] **1.2.5** 确定评估中的 FP/FN 定义（unmatched VLM → FP, unmatched GT → FN）

### 1.3 用例定义与核心功能规划 (`docs/requirements/use_case.md`)

**Verify:** Use case diagram captures all primary and secondary flows.
**Depends on:** 1.1, 1.2.

- [x] **1.3.1** 创建 Mermaid 用例图：VLM JSON → Graph → GNN → Corrected JSON
  - 主流程：predict → parse → build graph → encode → refine → output
  - 支撑流程：train model, evaluate, visualize, configure

- [x] **1.3.2** 规划系统模块划分和模块间接口契约
  - data → graph → model → eval 的边界和交换数据类型

### 1.4 非功能性需求：评估指标体系 (`docs/requirements/metrics.md`)

**Verify:** Each metric has a clear definition and expected behavior on edge cases.
**Depends on:** nothing.

- [x] **1.4.1** 定义 `PositionError`: `‖(x̂,ŷ) − (x,y)‖₂` 平均欧氏距离
- [x] **1.4.2** 定义 `SizeError`: `‖(ŵ,ĥ) − (w,h)‖₂` 平均欧氏距离
- [x] **1.4.3** 定义 `AlignmentError`: 对齐组偏差度量
- [x] **1.4.4** 定义 `ElementRecall`: IoU > threshold 的 GT 元素占比
- [x] **1.4.5** 定义 `ElementPrecision`: 匹配到 GT 的预测元素占比
- [x] **1.4.6** 定义 `ALL_METRICS` 注册策略和统计显著性方法（bootstrap / Wilcoxon）

---

## Phase 2: 概要设计 (High-Level Design) ✅

**Goal:** Define system architecture, data schema, and component interaction.
**Key artifacts:** `docs/design/high_level.md` — architecture diagram, schema specs.

---

### 2.1 配置系统设计

**Verify:** Config schema covers all training/evaluation hyperparameters with defaults.
**Depends on:** nothing.

- [x] **2.1.1** 设计 `DataConfig`: raw_dir, processed_dir, dataset_names, val_split, test_split
- [x] **2.1.2** 设计 `ModelConfig`: hidden_dim, n_layers, dropout, encoder_type, head_dims
- [x] **2.1.3** 设计 `TrainingConfig`: lr, epochs, batch_size, seed, weight_decay, warmup_steps, grad_clip, amp
- [x] **2.1.4** 设计 `Config` 复合结构、YAML 文件布局、校验策略（pydantic schema）

### 2.2 日志与实验跟踪架构设计

**Verify:** Logger architecture supports console file and optional external tracking.
**Depends on:** nothing.

- [x] **2.2.1** 设计结构化日志格式和 `setup_logger` 接口
- [x] **2.2.2** 设计 `MetricsLogger` 抽象基类接口
- [x] **2.2.3** 设计 `WandbMetricsLogger` 和 `TensorboardMetricsLogger`（作为可选 extra）
- [x] **2.2.4** 设计 `NoopMetricsLogger` 降级策略和 optional import 处理

### 2.3 依赖管理策略 (`pyproject.toml`)

**Verify:** Dependency groups are cleanly separated into core, dev, test, wandb, tensorboard.
**Depends on:** nothing.

- [x] **2.3.1** 规划 `scipy` 声明（匈牙利匹配 + 统计检验）
- [x] **2.3.2** 规划 `pydantic` 声明（config 校验）
- [x] **2.3.3** 规划 `wandb` optional extra
- [x] **2.3.4** 规划 `tensorboard` optional extra
- [x] **2.3.5** 规划 `[dev]` 和 `[test]` extras 分组策略

### 2.4 图模式设计：HeteroData "表结构" (`docs/design/schema.md`)

**Verify:** Schema covers all node/edge types, feature dimensions, and their PyG HeteroData keys.
**Depends on:** 2.1.

- [x] **2.4.1** 设计 `ElementNode`: type one-hot, spatial features (cx, cy, w, h), confidence
  - 确定特征维度: D_elem = num_types + 4 + 1

- [x] **2.4.2** 设计 `ConstraintType` 枚举: ALIGN_LEFT/RIGHT/TOP/BOTTOM, CENTER_X/Y, SAME_SIZE, SPACING, CONTAINMENT, GRID (10种)
  - 每种约束的语义和适用场景

- [x] **2.4.3** 设计 `ConstraintNode`: type one-hot + params
  - 确定特征维度: D_con = 10 + param_dim

- [x] **2.4.4** 设计 `EdgeFeatures`: spatial_distance, relative_position (dx, dy), IoU
  - 确定特征维度: D_edge = 4

### 2.5 约束提取策略设计（系统功能规划）

**Verify:** Strategy document specifies training-time and inference-time extraction flows.
**Depends on:** 2.4.

- [x] **2.5.1** 设计 Alignment 约束提取算法（共享边缘检测、tolerance 参数）
- [x] **2.5.2** 设计 Containment 约束提取算法（bbox 包含关系检测）
- [x] **2.5.3** 设计 Spacing 约束提取算法（相邻元素间隙一致性检测）
- [x] **2.5.4** 设计 Grid 约束提取算法（行/列排列检测）
- [x] **2.5.5** 设计训练模式 (GT-based) vs 推理模式 (Heuristic) 的约束提取策略差异
- [x] **2.5.6** 设计约束特征到 `HeteroData` 的映射方案

---

## Phase 3: 详细设计 (Detailed Design) ✅

**Goal:** Define class hierarchies, interfaces, algorithms, and deployment plan.
**Key artifacts:** `docs/design/detailed.md` — class diagrams, algorithm pseudocode, deployment spec.

---

### 3.1 数据层类设计

**Verify:** All interfaces and data flows between classes are specified.
**Depends on:** 2.1, 2.2.

- [x] **3.1.1** 设计 `CoordinateNormalizer` 类接口: fit/transform 方法签名
- [x] **3.1.2** 设计 `FeatureExtractor` 函数接口: spatial_features, type_embedding, confidence 签名
- [x] **3.1.3** 设计 `GUIDataset`: `__init__`, `__len__`, `__getitem__`, yield 的 dict 键
- [x] **3.1.4** 设计 `collate_variable_elements` 和 `create_dataloader` 接口

### 3.2 图构建层类设计

**Verify:** Builder class interface covers all HeteroData construction steps.
**Depends on:** 2.4, 2.5.

- [x] **3.2.1** 设计 `HeteroGraphBuilder`: `__init__`, `_build_element_nodes`, `_build_constraint_nodes`, `build(elements, constraints) -> HeteroData`
- [x] **3.2.2** 设计可视化函数接口: `plot_graph_on_screenshot`, `color_by_*`, `export_graph`
- [x] **3.2.3** 设计增强变换接口: `NodeDropout`, `CoordinateJitter`, `ConstraintPerturbation`, `GraphAugmentationPipeline`
- [x] **3.2.4** 设计 `HeteroData` 完整键结构文档

### 3.3 模型层类设计

**Verify:** Model forward pass tensor shapes are specified end-to-end.
**Depends on:** 3.1, 3.2.

- [x] **3.3.1** 设计 `HeteroGraphSAGE` 类: `__init__`, `_build_convs`, `forward`, `reset_parameters`
  - 信息流: element → constraint → element
  - 输出形状: `{"element": (N_elem, out_dim), "constraint": (N_con, out_dim)}`

- [x] **3.3.2** 设计三个预测 Head 接口:
  - `CoordinateRefinementHead`: MLP → (Δcx, Δcy, Δw, Δh)
  - `ViolationPredictionHead`: MLP → violation_score (sigmoid)
  - `ExistencePredictionHead`: MLP → existence_prob (sigmoid)

- [x] **3.3.3** 设计 `BipartiteGNNCorrector`: encoder + 3 heads 组装、forward 输出元组
- [x] **3.3.4** 设计 `CombinedLoss`: ℒ = w_c·ℒ_coord + w_v·ℒ_vio + w_a·ℒ_align + w_e·ℒ_exist

### 3.4 训练与推理规划 (`docs/design/deployment.md`)

**Verify:** Training and inference lifecycle is fully specified.
**Depends on:** 3.3.

- [x] **3.4.1** 设计 `Trainer` 生命周期: __init__ → fit → train_epoch ↔ validate → checkpoint
- [x] **3.4.2** 设计优化器/调度策略: AdamW + cosine annealing with warmup
- [x] **3.4.3** 设计早停和 checkpoint 格式
- [x] **3.4.4** 设计 `InferencePipeline`: vlm_json → HeteroData → model → apply delta → corrected JSON
  - 设备策略、AMP、batch 推理

### 3.5 评估层设计

**Verify:** Evaluator interface covers all defined metrics with per-category breakdown.
**Depends on:** 3.3.

- [x] **3.5.1** 设计 `Evaluator`: metrics 注册、evaluate、per_category_breakdown
- [x] **3.5.2** 设计基线接口: VLMOutputBaseline, RuleBasedCorrection, MLPOnlyBaseline
- [x] **3.5.3** 设计定性分析函数接口: side_by_side, case_study, attention_pattern, failure_analysis
- [x] **3.5.4** 设计报告生成函数接口: latex_table, comparison_fig, export_json/csv, summary_report

---

## Phase 4: 开发 (Development) 🔄

**Goal:** Implement all modules following the designs from Phases 1–3.
**Each subtask = write code + unit tests + verify passes.**

---

### 4.1 基础设施模块 ✅

- [x] **4.1.1** 实现 BBox 工具 (`src/bipartite_gnn_gui/utils/bbox.py`): `compute_iou`, `bbox_transform`, `apply_delta`
  - PR: #2
  - 测试: `test_utils_bbox.py`

- [x] **4.1.2** 实现配置系统 (`src/bipartite_gnn_gui/utils/config.py`): DataConfig, ModelConfig, TrainingConfig, Config, load_config, save_config
  - 创建 `configs/default.yaml`
  - PR: #5
  - 测试: `test_utils_config.py`

- [x] **4.1.3** 实现日志系统 (`src/bipartite_gnn_gui/utils/logging.py`): setup_logger, MetricsLogger, NoopMetricsLogger, WandbMetricsLogger, TensorboardMetricsLogger
  - PR: #6
  - 测试: `test_utils_logging.py`

- [x] **4.1.4** 实现依赖声明: 更新 `pyproject.toml`（scipy, pydantic 核心依赖; wandb, tensorboard optional extras）
  - PR: #7
  - 测试: `test_setup.py`

### 4.2 数据层 🔄

- [x] **4.2.1** 实现 VLM 输出解析 (`src/bipartite_gnn_gui/data/vlm_output.py`):
  VLMOutputElement, VLMOutput, parse_qwen_output, parse_minimax_output, normalize_coordinates
  - PR: #8
  - 测试: `test_data_vlm.py`

- [x] **4.2.2** 实现 Ground Truth 加载 (`src/bipartite_gnn_gui/data/ground_truth.py`):
  GTElement, GroundTruth, load_gui360_annotation, load_screenspot_annotation, match_predictions_to_ground_truth
  - PR: #9
  - 测试: `test_data_ground_truth.py`

- [x] **4.2.3** 实现数据预处理 (`src/bipartite_gnn_gui/data/preprocess.py`):
  CoordinateNormalizer, extract_spatial_features, extract_type_embedding, extract_confidence_scores, train_val_test_split
  - PR: #10
  - 测试: `test_data_preprocess.py`

- [x] **4.2.4** 实现数据集 (`src/bipartite_gnn_gui/data/dataset.py`):
  GUIDataset, collate_variable_elements, create_dataloader
  - PR: #11
  - 测试: `test_data_dataset.py`

### 4.3 图构建层

- [x] **4.3.1** 实现图模式 (`src/bipartite_gnn_gui/graph/schema.py`):
  ElementNode, ConstraintType, ConstraintNode, EdgeFeatures (含 to_tensor 方法)
  - PR: #12
  - 测试: `test_graph_schema.py`

- [ ] **4.3.2** 实现约束提取 (`src/bipartite_gnn_gui/graph/constraints.py`):
  extract_alignment/containment/spacing/grid_constraints, extract_constraints_ground_truth, propose_constraints_heuristic
  - 测试: `test_graph_constraints.py`

- [ ] **4.3.3** 实现图构建器 (`src/bipartite_gnn_gui/graph/builder.py`):
  HeteroGraphBuilder (含 build 方法和内部 _build_* 辅助方法)
  - 测试: `test_graph_builder.py`

- [ ] **4.3.4** 实现图可视化 (`src/bipartite_gnn_gui/graph/visualize.py`):
  plot_graph_on_screenshot, color_by_element_type, color_by_constraint_type, export_graph
  - 测试: `test_graph_visualize.py`

- [ ] **4.3.5** 实现图增强 (`src/bipartite_gnn_gui/graph/augment.py`):
  NodeDropout, CoordinateJitter, ConstraintPerturbation, GraphAugmentationPipeline
  - 测试: `test_graph_augment.py`

### 4.4 模型层

- [ ] **4.4.1** 实现异构编码器 (`src/bipartite_gnn_gui/model/encoder.py`):
  HeteroGraphSAGE (两层 SAGEConv + to_hetero + ReLU + Dropout + reset_parameters)
  - 测试: `test_model_encoder.py`

- [ ] **4.4.2** 实现预测头 (`src/bipartite_gnn_gui/model/heads.py`):
  CoordinateRefinementHead, ViolationPredictionHead, ExistencePredictionHead
  - 测试: `test_model_heads.py`

- [ ] **4.4.3** 实现损失函数 (`src/bipartite_gnn_gui/model/losses.py`):
  coordinate_refinement_loss, violation_loss, alignment_consistency_loss, existence_loss, CombinedLoss
  - 测试: `test_model_losses.py`

- [ ] **4.4.4** 实现完整模型 (`src/bipartite_gnn_gui/model/model.py`):
  BipartiteGNNCorrector (encoder + 3 heads + forward + compute_loss + train_step/validation_step)
  - 测试: `test_model_model.py`

- [ ] **4.4.5** 实现训练器 (`src/bipartite_gnn_gui/model/trainer.py`):
  Trainer (fit/train_epoch/validate + AdamW + cosine warmup + early stopping + checkpoint + AMP)
  - 测试: `test_model_trainer.py`

- [ ] **4.4.6** 实现推理管线 (`src/bipartite_gnn_gui/model/inference.py`):
  InferencePipeline (correct_single/correct_batch + _vlm_json_to_hetero + _apply_delta + clamp)
  - 测试: `test_model_inference.py`

### 4.5 评估层

- [ ] **4.5.1** 实现评估指标 (`src/bipartite_gnn_gui/eval/metrics.py`):
  position_error, size_error, alignment_error, element_recall, element_precision, ALL_METRICS
  - 测试: `test_eval_metrics.py`

- [ ] **4.5.2** 实现评估器 (`src/bipartite_gnn_gui/eval/evaluator.py`):
  Evaluator (evaluate + per_category_breakdown + statistical_significance with scipy)
  - 测试: `test_eval_evaluator.py`

- [ ] **4.5.3** 实现基线模型 (`src/bipartite_gnn_gui/eval/baselines.py`):
  VLMOutputBaseline, RuleBasedCorrection (pure-PyTorch NMS), MLPOnlyBaseline
  - 测试: `test_eval_baselines.py`

- [ ] **4.5.4** 实现定性分析 (`src/bipartite_gnn_gui/eval/qualitative.py`):
  side_by_side_comparison, case_study_report, plot_attention_patterns, failure_analysis
  - 测试: `test_eval_qualitative.py`

- [ ] **4.5.5** 实现报告生成 (`experiments/report.py`):
  generate_latex_table, generate_comparison_fig, export_results_json/csv, generate_summary_report
  - 测试: `test_experiment_report.py`

---

## Phase 5: 集成测试 (Integration Testing)

**Goal:** Verify end-to-end pipelines work on synthetic and real data.

---

- [ ] **5.1** 数据管线集成测试: VLM JSON → parse → normalize → extract features → Dataset → DataLoader
  - 使用合成 JSON 模拟 VLM 输出，验证完整的 data flow 不报错

- [ ] **5.2** 图构建集成测试: VLM JSON → constraints → HeteroData → visualize → augment → verify keys
  - 验证所有 HeteroData 键存在、形状正确、反向边建立

- [ ] **5.3** 模型前向集成测试: 合成 HeteroData → encoder → heads → loss → backward
  - 验证梯度可以回传、loss 是标量、训练一步后 loss 下降

- [ ] **5.4** 端到端管线测试: VLM JSON → InferencePipeline → corrected JSON
  - 验证输出 JSON 结构、坐标在边界内

- [ ] **5.5** 评估基线集成测试: 所有 baselines + Evaluator 在合成数据上运行
  - 验证每个 baseline 返回正确格式、Evaluator 产出所有指标

- [ ] **5.6** 实验脚本冒烟测试: `experiments/run.py` 在合成数据上执行全部 4 个实验
  - 验证每个实验脚本不 crash、产出结果文件

---

## Phase 6: 性能测试 (Performance Testing)

**Goal:** Establish performance baselines and ensure practical usability.

---

- [ ] **6.1** 数据加载性能基准: 测量 Dataset + DataLoader 在批量数据上的吞吐量
  - 记录: samples/sec, 内存占用

- [ ] **6.2** 图构建性能基准: 测量 10/50/100/500 个元素时的图构建时间
  - 记录: 平均构建时间 vs 元素数量曲线

- [ ] **6.3** 模型训练吞吐量基准: 测量训练时 samples/sec (batch size 8/16/32/64)
  - 记录: GPU 利用率、显存占用、AMP 加速比

- [ ] **6.4** 推理延迟基准: 测量单样本/批量推理延迟 (CPU vs GPU)
  - 记录: p50/p95/p99 延迟, batch size 对延迟的影响

---

## Phase 7: 实施 (Implementation — 实验运行)

**Goal:** Define and execute experiment methodology, ensure reproducibility.

---

- [ ] **7.1** 创建 `experiments/run.py` 统一入口
  - argparse: `--config`, `--experiment`, `--overrides`
  - 加载配置 → 执行指定实验 → 记录/保存结果

- [ ] **7.2** 实验1: 约束类型消融 (`experiments/ablation_constraints.py`)
  - 逐个移除约束类型 (alignment/containment/spacing/grid), 测量性能变化
  - 输出: ablation_results.json

- [ ] **7.3** 实验2: 图构建超参敏感性 (`experiments/sensitivity_graph.py`)
  - 改变: 约束容忍度、节点特征维度、边特征组合
  - 输出: sensitivity_results.json

- [ ] **7.4** 实验3: VLM 噪声鲁棒性 (`experiments/robustness_noise.py`)
  - 人工增加坐标噪声、随机丢失元素 → 测量性能衰减曲线
  - 输出: robustness_results.json

- [ ] **7.5** 实验4: 跨数据集泛化 (`experiments/cross_dataset.py`)
  - train on GUI-360°, eval on ScreenSpot; 反之亦然
  - 输出: cross_dataset_results.json

- [ ] **7.6** 可复现性设置: seed_everything, deterministic algorithms, 超参数日志
  - 每次训练保存完整 config + git commit hash + 环境信息

---

## Phase 8: 方案 (Solution — 文档与资料更新)

**Goal:** Update product/technical documentation for usability and publication.

---

- [ ] **8.1** 更新 `README.md`: 安装指南、快速开始示例、命令行用法
- [ ] **8.2** 创建 `configs/default.yaml` 注释完善的示例配置（含所有参数说明）
- [ ] **8.3** 创建 `examples/` 目录: 训练、评估、推理的完整使用示例
- [ ] **8.4** 更新 `pyproject.toml` 最终版本: 确认所有依赖声明、entry points、metadata

---

## 方法论对照

| 方法论阶段 | TASK 对应 | 产出 |
|-----------|-----------|------|
| 需求分析 | Phase 1 | `docs/requirements/` (数据格式、用例、指标) |
| 概要设计 | Phase 2 | `docs/design/high_level.md` (架构、schema、策略) |
| 详细设计 | Phase 3 | `docs/design/detailed.md` (类图、算法、部署) |
| 开发 | Phase 4 | `src/bipartite_gnn_gui/` (全部代码实现) |
| 集成测试 | Phase 5 | 端到端管线冒烟测试 |
| 性能测试 | Phase 6 | 基准测试数据 |
| 实施 | Phase 7 | `experiments/` (实验脚本与结果) |
| 方案 | Phase 8 | README、文档、使用示例 |

---

## 执行原则

1. **Phase 1-3 轻量、Phase 4 厚重**: 分析和设计产出 markdown 文档而非代码，每个文档 1-2 页即可进入下一阶段
2. **不回溯**: Phase 1 完成的分析假设在整个项目中保持不变；设计变更通过 Phase 4 的代码 review 处理，不重写需求文档
3. **Phase 5-6 可在 Phase 4 中间穿插**: 当一个模块开发完毕，可以立即运行集成测试，不需要等全部模块完成
4. **Phase 7 依赖 Phase 4-6 全部完成**: 实验使用完整的系统运行真实数据
5. **每个 checkbox 一个 PR**: 完成 → 推分支 → 提 PR → 合并 (遵循 CLAUDE.md Ship Incrementally)

## Stretch Goals

| # | 描述 |
|---|------|
| **S1** | Attention-based constraint importance weighting (可学习边权重) |
| **S2** | Cross-attention between VLM features and graph features |
| **S3** | Multi-scale graph: hierarchical container → child → leaf element |
| **S4** | Synthetic GUI layout generator for data augmentation |
| **S5** | Real-time web demo of VLM → correction pipeline |
| **S6** | ONNX / TorchScript export for deployment |
