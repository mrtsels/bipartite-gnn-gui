# Task List — Bipartite-GNN-GUI

> Phase-based development plan for Heterogeneous Bipartite GNN for GUI Structure Error Correction.
>
> **105 subtasks across 4 phases, 26 test files.**
> Each bullet is a single PR: implement → test → push → ship.

---

## Phase 1: Data Pipeline & Infrastructure

**Goal:** Set up data loading, preprocessing, and project infrastructure.
**30 subtasks, 8 test files.**
**Dependency chain:** 1.5(Config) → 1.6(Logging) → 1.1(VLM) + 1.2(GT) + 1.7(Bbox) → 1.3(Preprocess) → 1.4(Dataset)

---

### 1.1 VLM 输出解析器 (`data/vlm_output.py`)

**Verify:** `import` succeeds, `parse_qwen_output(json_str)` returns `VLMOutput` with correct fields.
**Depends on:** nothing.

- [ ] **1.1.1** 定义 `VLMOutputElement` dataclass
  - 字段: `bbox: tuple[float, float, float, float]`, `type: str`, `confidence: float`, `text: str`, `attributes: dict`
  - `test_data_vlm.py: test_vlm_output_element_fields`

- [ ] **1.1.2** 定义 `VLMOutput` dataclass
  - 字段: `elements: list[VLMOutputElement]`, `image_size: tuple[int, int]`, `source: str` (qwen/minimax)
  - `test_data_vlm.py: test_vlm_output_fields`

- [ ] **1.1.3** 实现 `parse_qwen_output(json_str: str) -> VLMOutput`
  - 处理 Qwen3.5-2B JSON 输出格式
  - `test_data_vlm.py: test_parse_qwen_valid / test_parse_qwen_missing_fields`

- [ ] **1.1.4** 实现 `parse_minimax_output(json_str: str) -> VLMOutput`
  - 处理 MiniMax-VL-01 JSON 输出格式
  - `test_data_vlm.py: test_parse_minimax_valid / test_parse_minimax_missing_fields`

- [ ] **1.1.5** 实现 `normalize_coordinates(vlm_out: VLMOutput, target_fmt: str) -> VLMOutput`
  - 支持 absolute→relative, relative→absolute, 不同原点约定转换
  - `test_data_vlm.py: test_normalize_absolute_to_relative / test_normalize_relative_to_absolute`

---

### 1.2 真实标注加载器 (`data/ground_truth.py`)

**Verify:** `load_gui360_annotation(path)` loads 100% of fields; `match_predictions_to_ground_truth` returns correct pair counts on synthetic data.
**Depends on:** 1.7 (bbox IoU).

- [ ] **1.2.1** 定义 `GTElement` dataclass
  - 字段: `bbox: tuple[float, float, float, float]`, `type: str`, `category: str`, `attributes: dict`
  - `test_data_ground_truth.py: test_gt_element_fields`

- [ ] **1.2.2** 定义 `GroundTruth` dataclass
  - 字段: `elements: list[GTElement]`, `image_size: tuple[int, int]`, `dataset_source: str`
  - `test_data_ground_truth.py: test_ground_truth_fields`

- [ ] **1.2.3** 实现 `load_gui360_annotation(path: str) -> GroundTruth`
  - GUI-360° JSON 标注格式解析
  - `test_data_ground_truth.py: test_load_gui360`

- [ ] **1.2.4** 实现 `load_screenspot_annotation(path: str) -> GroundTruth`
  - ScreenSpot JSON 标注格式解析
  - `test_data_ground_truth.py: test_load_screenspot`

- [ ] **1.2.5** 实现 `match_predictions_to_ground_truth(vlm_out, gt, iou_threshold) -> tuple[MatchedPairs, list[VLMOutputElement], list[GTElement]]`
  - 使用 IoU 代价矩阵 + 匈牙利算法 (scipy.optimize.linear_sum_assignment)
  - 返回 `MatchedPairs` dataclass (vlm_element, gt_element, iou), 以及 unmatched VLM 元素 (FP), unmatched GT 元素 (FN)
  - `test_data_ground_truth.py: test_bipartite_matching_exact / test_bipartite_matching_unmatched`

---

### 1.3 数据预处理 (`data/preprocess.py`)

**Verify:** A 1920×1080 screenshot → all bboxes in [0,1]; train/val/test split preserves total count.
**Depends on:** 1.1, 1.2.

- [ ] **1.3.1** 实现 `CoordinateNormalizer`
  - `fit(bboxes: list[tuple])` + `transform(bboxes, image_size) -> Tensor`: 缩放到 [0,1]
  - `test_data_preprocess.py: test_coordinate_normalizer_fit_transform`

- [ ] **1.3.2** 实现 `extract_spatial_features(bboxes: Tensor, image_size: tuple) -> Tensor`
  - 输出: (N, 4) = [cx, cy, w, h] （相对值）
  - `test_data_preprocess.py: test_spatial_features_shape / test_spatial_features_values`

- [ ] **1.3.3** 实现 `extract_type_embedding(elements: list[VLMOutputElement], vocab: dict[str, int]) -> Tensor`
  - 支持 one-hot 和可学习 embedding 两种模式
  - `test_data_preprocess.py: test_type_one_hot / test_type_embedding`

- [ ] **1.3.4** 实现 `extract_confidence_scores(vlm_output: VLMOutput) -> Tensor`
  - 输出: (N, 1) 置信度张量
  - `test_data_preprocess.py: test_confidence_scores`

- [ ] **1.3.5** 实现 `train_val_test_split(samples: list, ratios: tuple[float,float,float], seed: int) -> tuple[list,list,list]`
  - 随机划分 + 可选分层按类别划分
  - `test_data_preprocess.py: test_train_val_test_split_counts / test_reproducible_with_seed`

---

### 1.4 数据集类 (`data/dataset.py`)

**Verify:** `GUIDataset` yields dict of tensors; `DataLoader` with custom collate works.
**Depends on:** 1.1, 1.2, 1.3.

- [ ] **1.4.1** 实现 `GUIDataset.__init__(samples: list)` + `__len__() -> int`
  - `test_data_dataset.py: test_dataset_len`

- [ ] **1.4.2** 实现 `GUIDataset.__getitem__(idx: int) -> dict[str, Tensor]`
  - 返回: `{"element_features": Tensor, "gt_boxes": Tensor, "constraints": list, "image_size": tuple}`
  - `test_data_dataset.py: test_dataset_getitem_keys / test_dataset_getitem_shapes`

- [ ] **1.4.3** 实现 `collate_variable_elements(batch: list[dict]) -> dict[str, Tensor | list]`
  - 处理变长元素集的 padding 或 batch 封装
  - `test_data_dataset.py: test_collate_variable_batch`

- [ ] **1.4.4** 实现 `create_dataloader(dataset, batch_size, shuffle, collate_fn) -> DataLoader`
  - `test_data_dataset.py: test_create_dataloader_batch_shape`

---

### 1.5 配置系统 (`utils/config.py`)

**Verify:** `load_config("example.yaml")` returns validated Config; invalid config raises.
**Depends on:** nothing.

- [ ] **1.5.1** 定义 `DataConfig` dataclass: `raw_dir, processed_dir, dataset_names, val_split, test_split`
  - `test_utils_config.py: test_data_config_defaults`

- [ ] **1.5.2** 定义 `ModelConfig` dataclass: `hidden_dim, n_layers, dropout, encoder_type, head_dims`
  - `test_utils_config.py: test_model_config_fields`

- [ ] **1.5.3** 定义 `TrainingConfig` dataclass: `lr, epochs, batch_size, seed, weight_decay, warmup_steps, grad_clip, amp`
  - `test_utils_config.py: test_training_config_fields`

- [ ] **1.5.4** 定义 `Config` 复合 dataclass + `load_config(yaml_path: str) -> Config` + `save_config(cfg: Config, path: str)`
  - 使用 `pydantic` 做类型校验
  - 创建 `configs/default.yaml` 示例配置文件
  - `test_utils_config.py: test_load_config_yaml / test_config_validation_error`

---

### 1.6 日志与实验跟踪 (`utils/logging.py`)

**Verify:** Logger writes to both console and file; MetricsLogger records and flushes.
**Depends on:** nothing.

- [ ] **1.6.1** 实现 `setup_logger(name: str, log_file: str | None, level: str) -> logging.Logger`
  - 控制台 + 可选文件输出，统一格式
  - `test_utils_logging.py: test_setup_logger_console / test_setup_logger_file`

- [ ] **1.6.2** 定义 `MetricsLogger` 抽象基类: `log_metric(name, value, step)`, `log_metrics(dict, step)`, `flush()`
  - `test_utils_logging.py: test_metrics_logger_interface`

- [ ] **1.6.3** 实现 `WandbMetricsLogger(MetricsLogger)`
  - WandB 初始化、指标记录、配置记录
  - `test_utils_logging.py: test_wandb_logger` (mock wandb)

- [ ] **1.6.4** 实现 `TensorboardMetricsLogger(MetricsLogger)`
  - `SummaryWriter` 封装
  - `test_utils_logging.py: test_tensorboard_logger` (mock writer)

---

### 1.7 BBox 工具函数 (`utils/bbox.py`)

**新增文件** — 基础 bbox 工具，被 1.2、1.3、2.2、4.1 依赖。

**Verify:** IoU is correct for 0, 0.5, 1.0 overlap; format conversion round-trips.
**Depends on:** nothing.

- [ ] **1.7.1** 实现 `compute_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor`
  - (N,4) × (M,4) → (N,M) 成对 IoU
  - `test_utils_bbox.py: test_iou_identical / test_iou_no_overlap / test_iou_partial`

- [ ] **1.7.2** 实现 `bbox_transform(boxes: Tensor, src_fmt: str, dst_fmt: str) -> Tensor`
  - 支持格式: `xyxy` (左上右下), `cxcywh` (中心宽高), `xywh` (左上宽高)
  - `test_utils_bbox.py: test_bbox_transform_roundtrip`

- [ ] **1.7.3** 实现 `apply_delta(boxes: Tensor, delta: Tensor) -> Tensor`
  - delta shape: (N,4) = (Δcx, Δcy, Δw, Δh) 作用于 cxcywh 格式
  - `test_utils_bbox.py: test_apply_delta_shape / test_apply_delta_values`

---

## Phase 2: Bipartite Graph Construction

**Goal:** Build heterogeneous bipartite graphs from VLM JSON output.
**22 subtasks, 6 test files.**
**Dependency chain:** 2.1(Schema) → 2.2(Constraints) + 2.1 → 2.3(Builder) → 2.4(Visualize), 2.5(Augment)

---

### 2.1 图模式定义 (`graph/schema.py`)

**Verify:** All enum values present; `to_tensor()` returns correct shapes.
**Depends on:** nothing.

- [ ] **2.1.1** 定义 `ElementNode` dataclass
  - 字段: `bbox: tuple`, `type: str`, `confidence: float`, `spatial_features: Tensor`
  - `to_tensor() -> Tensor`: 拼接所有特征为 (1, D) 张量
  - `test_graph_schema.py: test_element_node_to_tensor`

- [ ] **2.1.2** 定义 `ConstraintType` 枚举
  - 值: `ALIGN_LEFT, ALIGN_RIGHT, ALIGN_TOP, ALIGN_BOTTOM, CENTER_X, CENTER_Y, SAME_SIZE, SPACING, CONTAINMENT, GRID`
  - `test_graph_schema.py: test_constraint_type_values / test_constraint_type_count`

- [ ] **2.1.3** 定义 `ConstraintNode` dataclass
  - 字段: `type: ConstraintType`, `involved_elements: list[int]`, `params: dict`
  - `to_tensor(num_types: int) -> Tensor`: one-hot 编码的约束类型 (1, num_types) + 参数
  - `test_graph_schema.py: test_constraint_node_to_tensor_shape`

- [ ] **2.1.4** 定义 `EdgeFeatures` dataclass
  - 字段: `spatial_distance: float`, `relative_position: tuple[float,float]`, `iou: float`
  - `to_tensor() -> Tensor`: (1, 4) = [dist, dx, dy, iou]
  - `test_graph_schema.py: test_edge_features_to_tensor`

---

### 2.2 约束提取 (`graph/constraints.py`)

**Verify:** On a known 2-element layout, alignment constraints are correct; heuristic proposer returns at least 1 constraint.
**Depends on:** 2.1.

- [ ] **2.2.1** 实现 `extract_alignment_constraints(elements: list[ElementNode], tolerance: float) -> list[ConstraintNode]`
  - 检测共享左/右/上/下边缘 (within tolerance), 以及中心对齐
  - `test_graph_constraints.py: test_alignment_same_left / test_alignment_same_center`

- [ ] **2.2.2** 实现 `extract_containment_constraints(elements: list[ElementNode]) -> list[ConstraintNode]`
  - 父-子容器关系检测 (bbox 完全包含)
  - `test_graph_constraints.py: test_containment_nested / test_containment_overlap_not_contain`

- [ ] **2.2.3** 实现 `extract_spacing_constraints(elements: list[ElementNode], tolerance: float) -> list[ConstraintNode]`
  - 相邻元素间一致间隙检测
  - `test_graph_constraints.py: test_spacing_equal_gaps`

- [ ] **2.2.4** 实现 `extract_grid_constraints(elements: list[ElementNode], tolerance: float) -> list[ConstraintNode]`
  - 行/列排列模式检测
  - `test_graph_constraints.py: test_grid_row_layout / test_grid_column_layout`

- [ ] **2.2.5** 实现 `extract_constraints_ground_truth(elements: list[ElementNode]) -> list[ConstraintNode]`
  - 从 GT 布局提取所有类型约束 → 训练 label 来源
  - `test_graph_constraints.py: test_extract_gt_all_types`

- [ ] **2.2.6** 实现 `propose_constraints_heuristic(vlm_elements: list[ElementNode]) -> list[ConstraintNode]`
  - 从 VLM 预测启发式提出约束（放宽 tolerance, 补充缺失约束）
  - `test_graph_constraints.py: test_propose_heuristic_returns_constraints`

---

### 2.3 二分图构建器 (`graph/builder.py`)

**Verify:** `build()` produces `HeteroData` with correct node/edge types and tensor shapes.
**Depends on:** 2.1, 2.2.

- [ ] **2.3.1** 实现 `HeteroGraphBuilder.__init__()`
  - 存储 element 和 constraint 特征维度
  - `test_graph_builder.py: test_builder_init`

- [ ] **2.3.2** 实现 `_build_element_nodes(elements: list) -> Tensor`
  - (N_elem, D_elem) 元素特征张量，通过 `ElementNode.to_tensor()` 堆叠
  - `test_graph_builder.py: test_element_node_tensor_shape`

- [ ] **2.3.3** 实现 `_build_constraint_nodes(constraints: list) -> Tensor`
  - (N_con, D_con) 约束特征张量，通过 `ConstraintNode.to_tensor()` 堆叠
  - `test_graph_builder.py: test_constraint_node_tensor_shape`

- [ ] **2.3.4** 实现 `build(elements, constraints) -> HeteroData`
  - 创建 `(element, belongs_to, constraint)` 边
  - 添加反向边 `(constraint, affects, element)`
  - 边属性: `EdgeFeatures.to_tensor()`
  - 最终 `HeteroData` 包含: `element_node.x`, `constraint_node.x`, `(elem, belongs_to, con).edge_index`, `(elem, belongs_to, con).edge_attr`, `(con, affects, elem).edge_index`, `(con, affects, elem).edge_attr`
  - `test_graph_builder.py: test_build_hetero_data_keys / test_build_edge_shapes / test_build_reverse_edges`

---

### 2.4 图可视化 (`graph/visualize.py`)

**Verify:** Function runs without error on synthetic data; outputs exist.
**Depends on:** 2.3.

- [ ] **2.4.1** 实现 `plot_graph_on_screenshot(hetero_data, screenshot: np.ndarray, ax) -> ax`
  - 在 matplotlib Axes 上叠加截图 + 图
  - `test_graph_visualize.py: test_plot_runs_without_error`

- [ ] **2.4.2** 实现 `color_by_element_type(ax, hetero_data, type_colors: dict)` — color-code element nodes by type
  - `test_graph_visualize.py: test_color_by_type_no_error`

- [ ] **2.4.3** 实现 `color_by_constraint_type(ax, hetero_data, constraint_colors: dict)` — color-code constraint nodes by type
  - `test_graph_visualize.py: test_color_by_constraint_no_error`

- [ ] **2.4.4** 实现 `export_graph(fig, path: str, fmt: str = "png")`
  - 支持 PNG、SVG 格式导出
  - `test_graph_visualize.py: test_export_graph_file_exists`

---

### 2.5 图增强 (`graph/augment.py`)

**Verify:** Each transform changes data; `GraphAugmentationPipeline` composes all.
**Depends on:** 2.3.

- [ ] **2.5.1** 实现 `NodeDropout(p_drop: float)` — 随机丢弃元素节点 (模拟 VLM 遗漏)
  - `forward(hetero_data) -> HeteroData`
  - `test_graph_augment.py: test_node_dropout_removes_nodes`

- [ ] **2.5.2** 实现 `CoordinateJitter(std: float)` — 坐标加高斯噪声 (模拟 VLM 偏差)
  - `forward(hetero_data) -> HeteroData`
  - `test_graph_augment.py: test_coordinate_jitter_changes_values`

- [ ] **2.5.3** 实现 `ConstraintPerturbation(p_add: float, p_remove: float)` — 增删随机约束
  - `forward(hetero_data) -> HeteroData`
  - `test_graph_augment.py: test_constraint_perturbation_changes_count`

- [ ] **2.5.4** 实现 `GraphAugmentationPipeline(transforms: list)` — 增强组合链
  - `forward(hetero_data) -> HeteroData`: 依次应用每个 transform
  - `test_graph_augment.py: test_pipeline_composition`

---

## Phase 3: GNN Model — GraphSAGE

**Goal:** Implement GraphSAGE-based correction model with violation prediction and coordinate refinement.
**26 subtasks, 7 test files.**
**Dependency chain:** 3.1(Encoder) + 3.2(Heads) → 3.3(Model) + 3.4(Losses) → 3.5(Trainer) → 3.6(Inference)

---

### 3.1 异构编码器 (`model/encoder.py`)

**Verify:** Forward pass returns dict with correct key and tensor shape.
**Depends on:** nothing (pure torch/nn.Module).

- [ ] **3.1.1** 实现 `HeteroGraphSAGE.__init__(hidden_dim, out_dim, n_layers, dropout)`
  - 用 `nn.ModuleDict` 存储各类型 SAGEConv
  - `test_model_encoder.py: test_encoder_init`

- [ ] **3.1.2** 实现 `_build_convs()` — 使用 `torch_geometric.nn.to_hetero()` 或 `HeteroConv` 包装 SAGEConv
  - 两层结构: SAGEConv → ReLU → Dropout → SAGEConv
  - `test_model_encoder.py: test_encoder_conv_types`

- [ ] **3.1.3** 实现 `forward(x_dict, edge_index_dict) -> dict[str, Tensor]`
  - 信息流: element → constraint → element (二分图两次消息传递)
  - 输出 dict: `{"element": (N_elem, out_dim), "constraint": (N_con, out_dim)}`
  - `test_model_encoder.py: test_encoder_forward_shape / test_encoder_forward_element_key`

- [ ] **3.1.4** 实现 `reset_parameters()` — 递归重置所有可学习参数
  - `test_model_encoder.py: test_encoder_after_reset_different`

---

### 3.2 预测头 (`model/heads.py`)

**Verify:** Each head output shape matches expectation; output value ranges are correct.
**Depends on:** nothing (but 3.2 results feed into 3.3).

- [ ] **3.2.1** 实现 `CoordinateRefinementHead(in_dim, hidden_dim) -> nn.Module`
  - MLP: in_dim → hidden_dim → ReLU → Dropout → 4 (Δcx, Δcy, Δw, Δh)
  - 输出形状: (N, 4)
  - `test_model_heads.py: test_delta_head_output_shape / test_delta_head_forward`

- [ ] **3.2.2** 实现 `ViolationPredictionHead(in_dim) -> nn.Module`
  - MLP: in_dim → hidden_dim → ReLU → 1 → Sigmoid
  - 输出形状: (N, 1), 值域 [0, 1]
  - `test_model_heads.py: test_violation_head_output_range`

- [ ] **3.2.3** 实现 `ExistencePredictionHead(in_dim) -> nn.Module`
  - MLP: in_dim → hidden_dim → ReLU → 1 → Sigmoid
  - 输出形状: (N, 1), 值域 [0, 1]
  - `test_model_heads.py: test_existence_head_output_range`

---

### 3.3 完整模型 (`model/model.py`)

**Verify:** End-to-end forward returns 3 outputs with correct shapes; loss computation works.
**Depends on:** 3.1, 3.2, 3.4.

- [ ] **3.3.1** 定义 `BipartiteGNNCorrector.__init__(encoder, delta_head, vio_head, exist_head)`
  - 组装子模块
  - `test_model_model.py: test_corrector_init`

- [ ] **3.3.2** 实现 `forward(x_dict, edge_index_dict) -> tuple[Tensor, Tensor, Tensor]`
  - 返回: `(deltas: (N_elem, 4), violations: (N_con, 1), existence: (N_elem, 1))`
  - `test_model_model.py: test_corrector_forward_shapes`

- [ ] **3.3.3** 实现 `compute_loss(batch, deltas, violations, existence) -> dict[str, Tensor]`
  - 调用 `CombinedLoss` 计算各分量 + 组合损失
  - `test_model_model.py: test_compute_loss_keys`

- [ ] **3.3.4** 实现 `train_step(batch)` + `validation_step(batch)`
  - 封装 forward + loss 为单步调用
  - `test_model_model.py: test_train_step_output`

---

### 3.4 损失函数 (`model/losses.py`)

**Verify:** Each loss returns scalar Tensor; combined loss respects weight ratios.
**Depends on:** nothing.

- [ ] **3.4.1** 实现 `coordinate_refinement_loss(pred_delta, target_delta) -> Tensor`
  - SmoothL1Loss (nn.HuberLoss or F.smooth_l1_loss)
  - `test_model_losses.py: test_coord_loss_zero_when_perfect / test_coord_loss_nonzero_when_wrong`

- [ ] **3.4.2** 实现 `violation_loss(pred_violation, target_violation) -> Tensor`
  - BCELoss (F.binary_cross_entropy)
  - `test_model_losses.py: test_violation_loss_bce`

- [ ] **3.4.3** 实现 `alignment_consistency_loss(deltas: Tensor, alignment_groups: list[list[int]]) -> Tensor`
  - ℒ_align = 对每组内所有配对的 deltas 差的 L2 范数平方
  - `test_model_losses.py: test_alignment_loss_zero_when_consistent`

- [ ] **3.4.4** 实现 `existence_loss(pred_exist, target_exist) -> Tensor`
  - BCELoss
  - `test_model_losses.py: test_existence_loss_shape`

- [ ] **3.4.5** 实现 `CombinedLoss(weights: dict[str, float])`
  - `forward(loss_components: dict) -> Tensor`: ℒ = w_c * ℒ_coord + w_v * ℒ_vio + w_a * ℒ_align + w_e * ℒ_exist
  - 支持权重调度器（可选 warmup）
  - `test_model_losses.py: test_combined_loss_weighted`

---

### 3.5 训练器 (`model/trainer.py`)

**Verify:** Training loop decreases loss; checkpoint is loadable; early stopping triggers.
**Depends on:** 3.3, 3.4, 1.5, 1.6.

- [ ] **3.5.1** 实现 `Trainer.__init__(model, config, logger, metrics_logger)`
  - `test_model_trainer.py: test_trainer_init`

- [ ] **3.5.2** 实现 `train_epoch(dataloader) -> dict[str, float]`
  - 遍历 dataloader → train_step → 梯度累积 → 返回 epoch 平均指标
  - `test_model_trainer.py: test_train_epoch_runs`

- [ ] **3.5.3** 实现 `validate(dataloader) -> dict[str, float]`
  - 无梯度验证循环
  - `test_model_trainer.py: test_validate_runs`

- [ ] **3.5.4** 实现 `_configure_optimizer(lr, weight_decay)` + `_configure_scheduler(warmup_steps, total_steps)`
  - 优化器: AdamW; 调度: cosine annealing with linear warmup
  - `test_model_trainer.py: test_optimizer_config / test_scheduler_config`

- [ ] **3.5.5** 实现 `_early_stop(val_metric, patience)` + `_checkpoint(path, metric)` + `_load_checkpoint(path)`
  - 早停: 当 val_metric 连续 patience 个 epoch 未改善时停止
  - 检查点: 保存 model + optimizer + scheduler + epoch state dict
  - `test_model_trainer.py: test_early_stop_triggers / test_checkpoint_roundtrip`

- [ ] **3.5.6** 实现 `fit(train_loader, val_loader) -> Trainer` + `_amp_context()`
  - `torch.cuda.amp.autocast()` 混合精度训练
  - 调用 _early_stop 和 _checkpoint
  - `test_model_trainer.py: test_fit_completes`

---

### 3.6 推理管线 (`model/inference.py`)

**Verify:** `correct_single` returns dict with corrected bboxes; batch preserves order.
**Depends on:** 3.3, 2.3.

- [ ] **3.6.1** 实现 `InferencePipeline.__init__(model, graph_builder, device)`
  - 将模型移至 device
  - `test_model_inference.py: test_pipeline_init`

- [ ] **3.6.2** 实现 `correct_single(vlm_json: dict, screenshot: np.ndarray) -> dict`
  - VLM JSON → hetero_data (via graph_builder) → model forward → apply Δ → 返回修正后 JSON
  - `test_model_inference.py: test_correct_single_output_keys`

- [ ] **3.6.3** 实现 `correct_batch(inputs: list[tuple[dict, np.ndarray]]) -> list[dict]`
  - `test_model_inference.py: test_correct_batch_preserves_order`

- [ ] **3.6.4** 实现 `_vlm_json_to_hetero(vlm_json) -> HeteroData` + `_apply_delta(hetero_data, deltas) -> dict`
  - 内部管线步骤：JSON 解析 → 约束提出 → 图构建 → 修正应用 → JSON 输出
  - 确保输出的修正后 bboxes 不超出截图边界 (clamp to [0, image_size])
  - `test_model_inference.py: test_delta_clamp_boundary`

---

## Phase 4: Evaluation & Experiments

**Goal:** Evaluate the model on benchmark datasets and baselines.
**32 subtasks, 5 test files.**
**Dependency chain:** 4.1(Metrics) → 4.2(Evaluator) + 4.3(Baselines) + 4.5(Qualitative) → 4.4(Experiments) → 4.6(Report)

---

### 4.1 评估指标 (`eval/metrics.py`)

**Verify:** Metrics return expected values on hand-crafted test cases.
**Depends on:** nothing (pure tensor ops).

- [ ] **4.1.1** 实现 `position_error(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor`
  - ‖(x̂,ŷ) − (x,y)‖₂ 平均欧氏距离
  - `test_eval_metrics.py: test_position_error_zero_when_perfect / test_position_error_value`

- [ ] **4.1.2** 实现 `size_error(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor`
  - ‖(ŵ,ĥ) − (w,h)‖₂ 平均欧氏距离
  - `test_eval_metrics.py: test_size_error_value`

- [ ] **4.1.3** 实现 `alignment_error(pred_boxes: Tensor, gt_boxes: Tensor, groups: list[list[int]]) -> Tensor`
  - 对齐组偏差: 组内各元素的对齐差异度量
  - `test_eval_metrics.py: test_alignment_error_zero_when_perfect`

- [ ] **4.1.4** 实现 `element_recall(pred_boxes: Tensor, gt_boxes: Tensor, iou_thresh: float) -> float`
  - fraction of GT elements with matched prediction at IoU > threshold
  - `test_eval_metrics.py: test_recall_perfect / test_recall_half`

- [ ] **4.1.5** 实现 `element_precision(pred_boxes: Tensor, gt_boxes: Tensor, iou_thresh: float) -> float`
  - fraction of predictions matching a GT element
  - `test_eval_metrics.py: test_precision_perfect / test_precision_half`

- [ ] **4.1.6** 定义 `ALL_METRICS: dict[str, Callable]` 将所有指标注册为字典
  - `test_eval_metrics.py: test_all_metrics_have_expected_keys`

---

### 4.2 评估器 (`eval/evaluator.py`)

**Verify:** `evaluate()` returns dict with same keys as registered metrics; per-category breakdown sums to total.
**Depends on:** 4.1.

- [ ] **4.2.1** 实现 `Evaluator.__init__(metrics: dict[str, Callable])`
  - `test_eval_evaluator.py: test_evaluator_init`

- [ ] **4.2.2** 实现 `evaluate(predictions: list[dict], ground_truths: list[dict]) -> dict[str, float]`
  - 遍历所有指标 → 聚合均值
  - `test_eval_evaluator.py: test_evaluate_returns_dict`

- [ ] **4.2.3** 实现 `per_category_breakdown(predictions, ground_truths, categories: list[str]) -> dict[str, dict[str, float]]`
  - 按类型 (button/text/image/input) 单独计算各指标
  - `test_eval_evaluator.py: test_per_category_breakdown_keys`

- [ ] **4.2.4** 实现 `statistical_significance(baseline_scores, proposed_scores, n_bootstrap: int, metric: str) -> dict`
  - 配对 bootstrap 置信区间 + p-value (或 Wilcoxon signed-rank)
  - 返回: `{"p_value": float, "ci_95": tuple[float, float], "significant": bool}`
  - `test_eval_evaluator.py: test_significance_test_returns_keys`

---

### 4.3 基线模型 (`eval/baselines.py`)

**Verify:** Each baseline returns same format as model output (dict with "boxes", "types" keys).
**Depends on:** 1.1, 1.2.

- [ ] **4.3.1** 实现 `VLMOutputBaseline(vlm_json: dict) -> dict`
  - 基线1: 原样返回 VLM 输出 (不做修正)
  - `test_eval_baselines.py: test_vlm_baseline_identity`

- [ ] **4.3.2** 实现 `RuleBasedCorrection(vlm_json: dict) -> dict`
  - 基线2: NMS 去重 + snap-to-grid + 基于边距的调整
  - `test_eval_baselines.py: test_rule_based_returns_dict`

- [ ] **4.3.3** 实现 `MLPOnlyBaseline(input_dim, hidden_dim)`
  - 基线4: MLP 直接预测 Δ (无图结构)
  - `fit(train_data, val_data)` + `predict(vlm_json) -> dict`
  - `test_eval_baselines.py: test_mlp_baseline_forward_shape`

- [ ] **4.3.4** 创建 `experiments/baseline_finetune_vlm.py` 占位脚本
  - 基线3 占位: Fine-tune VLM (仅当计算资源允许)
  - 输出: "Not implemented — requires GPU cluster"

---

### 4.4 实验脚本 (`experiments/`)

**Verify:** Each script runs end-to-end on synthetic data.
**Depends on:** 3.5, 3.6, 4.2, 4.3.

- [ ] **4.4.1** 实验1: 约束类型消融 — `experiments/ablation_constraints.py`
  - 逐个移除约束类型 (alignment/containment/spacing/grid), 测量性能变化
  - 输出: ablation_results.json

- [ ] **4.4.2** 实验2: 图构建超参敏感性 — `experiments/sensitivity_graph.py`
  - 改变: 约束容忍度、节点特征维度、边特征组合
  - 输出: sensitivity_results.json

- [ ] **4.4.3** 实验3: VLM 噪声鲁棒性 — `experiments/robustness_noise.py`
  - 人工增加坐标噪声、随机丢失元素 → 测量性能衰减
  - 输出: robustness_results.json

- [ ] **4.4.4** 实验4: 跨数据集泛化 — `experiments/cross_dataset.py`
  - train on GUI-360°, eval on ScreenSpot; 反之亦然
  - 输出: cross_dataset_results.json

- [ ] **4.4.5** 统一入口 — `experiments/run.py`
  - argparse: `--config`, `--experiment`, `--overrides`
  - 加载配置 → 执行实验 → 记录/保存结果

---

### 4.5 定性分析 (`eval/qualitative.py`)

**Verify:** Functions create output files without error.
**Depends on:** 4.1, 2.4.

- [ ] **4.5.1** 实现 `side_by_side_comparison(gt, vlm, corrected, save_path)`
  - 三栏: Ground Truth | VLM Output | Corrected
  - `test_eval_qualitative.py: test_sbs_creates_file`

- [ ] **4.5.2** 实现 `case_study_report(samples, model, graph_builder, save_dir)`
  - 最佳改进案例 + 失败模式 + 边界案例
  - 输出每个案例的 HTML/markdown 描述 + 图

- [ ] **4.5.3** 实现 `plot_attention_patterns(model, hetero_data, save_path)`
  - 热图展示哪些约束节点对哪些元素节点的修正影响最大

- [ ] **4.5.4** 实现 `failure_analysis(incorrect_cases: list, save_path: str)` — 系统化分析失败原因
  - 分类统计: 位置错误 vs 类型错误 vs 缺失 vs 多余
  - 输出 JSON + 饼图/柱状图

---

### 4.6 报告生成 (`experiments/report.py`)

**Verify:** LaTeX table string renders; figure file exists.
**Depends on:** 4.4 results.

- [ ] **4.6.1** 实现 `generate_latex_table(results: dict, metrics: list[str], caption: str) -> str`
  - 输出: LaTeX table 源码 (可直接编译)
  - `test_experiment_report.py: test_latex_table_renders`

- [ ] **4.6.2** 实现 `generate_comparison_fig(all_results: dict, save_dir: str)`
  - 分组柱状图: 每组 = 指标, 每个柱子 = 方法 (VLM baseline / Rule / MLP / GNN)
  - `test_experiment_report.py: test_comparison_fig_creates_file`

- [ ] **4.6.3** 实现 `export_results_json(results: dict, path: str)` + `export_results_csv(results: dict, path: str)`
  - JSON 保留完整结构; CSV 为表格格式便于分析
  - `test_experiment_report.py: test_export_json_valid / test_export_csv_header`

- [ ] **4.6.4** 实现 `generate_summary_report(all_results: dict, template_path: str | None) -> str`
  - 自动生成摘要报告 (markdown): 关键发现、改进幅度、失败分析
  - `test_experiment_report.py: test_summary_report_contains_keys`

---

## Milestones

| Milestone | Subtasks | Description |
|-----------|----------|-------------|
| **M1** | 1.1 → 1.7 | Phase 1 complete: data loaders working, config/logging ready, smoke test passes |
| **M2** | 2.1 → 2.5 | Phase 2 complete: graph construction verified, visualization renders, augmentation works |
| **M3** | 3.1 → 3.6 | Phase 3 complete: model converges on validation set, inference pipeline produces output |
| **M4** | 4.1 → 4.6 | Phase 4 complete: all metrics, baselines, experiments, and report artifacts created |

## Stretch Goals (after M4)

| Task | Description |
|------|-------------|
| **S1** | Attention-based constraint importance weighting (learnable edge weights) |
| **S2** | Cross-attention between VLM features and graph features |
| **S3** | Multi-scale graph: hierarchical container → child → leaf element |
| **S4** | Synthetic GUI layout generator for data augmentation |
| **S5** | Real-time web demo of VLM -> correction pipeline |
| **S6** | ONNX / TorchScript export for deployment |
