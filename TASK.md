# 任务列表 — Bipartite-GNN-GUI

---

## Phase 1: 需求分析 ✅

**目标:** 分析问题域与数据格式，定义成功标准。
**产出:** `docs/requirements/` — 数据格式规范、用例图、评估指标。

### 1.1 VLM 输出格式分析 (`docs/requirements/vlm_format.md`)

| # | 条目 | 状态 |
|---|------|--------|
| 1.1.1 | 收集 Qwen3.5-2B JSON 样例，分析字段结构与坐标格式 | ✅ |
| 1.1.2 | 收集 MiniMax-VL-01 JSON 样例，分析字段结构与坐标格式 | ✅ |
| 1.1.3 | 定义 `VLMOutputElement` / `VLMOutput` 数据类结构 | ✅ |
| 1.1.4 | 确定 `parse_qwen_output` / `parse_minimax_output` 接口与错误处理策略 | ✅ |
| 1.1.5 | 确定全局元素类型分类体系（VLM 与 GT 共享） | ✅ |

### 1.2 Ground Truth 格式分析 (`docs/requirements/gt_format.md`)

| # | 条目 | 状态 |
|---|------|--------|
| 1.2.1 | 分析 GUI-360° JSON 标注格式 | ✅ |
| 1.2.2 | 分析 ScreenSpot JSON 标注格式 | ✅ |
| 1.2.3 | 定义 `GTElement` / `GroundTruth` 数据类结构 | ✅ |
| 1.2.4 | VLM 预测 ↔ GT 匹配策略（IoU 代价矩阵 + 匈牙利算法） | ✅ |
| 1.2.5 | 确定评估中的 FP/FN 定义 | ✅ |

### 1.3 用例与核心功能规划 (`docs/requirements/use_case.md`)

| # | 条目 | 状态 |
|---|------|--------|
| 1.3.1 | Mermaid 用例图：VLM JSON → Graph → GNN → Corrected JSON | ✅ |
| 1.3.2 | 系统模块划分与模块间接口契约 | ✅ |

### 1.4 评估指标体系 (`docs/requirements/metrics.md`)

| # | 条目 | 状态 |
|---|------|--------|
| 1.4.1 | `PositionError` | ✅ |
| 1.4.2 | `SizeError` | ✅ |
| 1.4.3 | `AlignmentError` | ✅ |
| 1.4.4 | `ElementRecall` | ✅ |
| 1.4.5 | `ElementPrecision` | ✅ |
| 1.4.6 | `ALL_METRICS` 注册策略与显著性检验 | ✅ |

---

## Phase 2: 概要设计 ✅

**目标:** 系统架构、数据 schema、组件交互设计。

| # | 条目 | 状态 |
|---|------|--------|
| 2.1 | 配置系统：DataConfig / ModelConfig / TrainingConfig / Config | ✅ |
| 2.2 | 日志与实验跟踪：setup_logger / MetricsLogger / Wandb / Tensorboard / Noop | ✅ |
| 2.3 | 依赖管理：scipy / pydantic / wandb / tensorboard extras | ✅ |
| 2.4 | 图模式：ElementNode / ConstraintType（10 种）/ ConstraintNode / EdgeFeatures | ✅ |
| 2.5 | 约束提取策略：Alignment / Containment / Spacing / Grid，训练 vs 推理模式 | ✅ |

---

## Phase 3: 详细设计 ✅

**目标:** 类层次、接口、算法与部署计划。

| # | 条目 | 状态 |
|---|------|--------|
| 3.1 | 数据层：CoordinateNormalizer / FeatureExtractor / GUIDataset / collate_dataloader | ✅ |
| 3.2 | 图构建层：HeteroGraphBuilder / 可视化 / 增强变换 / HeteroData 键结构 | ✅ |
| 3.3 | 模型层：HeteroGraphSAGE / 3 预测头 / BipartiteGNNCorrector / CombinedLoss | ✅ |
| 3.4 | 训练与推理：Trainer / AdamW+cosine / 早停 / InferencePipeline | ✅ |
| 3.5 | 评估层：Evaluator / 基线接口 / 定性分析 / 报告生成 | ✅ |

---

## Phase 4: 开发 ✅

**目标:** 按 Phase 1–3 设计实现全部模块。

### 4.1–4.5 核心模块

| # | 条目 | 状态 |
|---|------|--------|
| 4.1 | 基础设施：BBox 工具 / 配置 / 日志 / 依赖声明 | ✅ |
| 4.2 | 数据层：VLM 解析 / GT 加载 / 预处理 / Dataset/DataLoader | ✅ |
| 4.3 | 图构建：Schema / 约束提取 / Builder / 可视化 / 增强 | ✅ |
| 4.3a | 数据适配：ScreenSpot / RICO View Hierarchy 加载器 | ✅ |
| 4.4 | 模型层：编码器 / 预测头 / 损失 / 完整模型 / 训练器 / 推理管线 | ✅ |
| 4.5 | 评估层：指标 / 评估器 / 基线 (NoOp/Identity/Jitter) / 定性分析 | ✅ |

### 4.6 实验阶段

| # | 条目 | 状态 |
|---|------|--------|
| 4.6.1 | 训练管线标准化：GraphDataset / run_experiment.py / configs/experiment.yaml | ✅ |
| 4.6.2 | 超参 sweep：6 配置，Best: hd128 big-noise (val_loss=0.0537) | ✅ |
| 4.6.3 | VLM 推理管线：Qwen3-VL Flash (2947 elem) / Qwen+Plus (7312) / LLaVA (61) / Moondream (弱) | ✅ |
| 4.6.4 | 实验总结：GNN 无法战胜精度过高的 VLM，也无法补足检测过弱的 VLM | ✅ |

### 4.7 方向调整

| # | 条目 | 状态 |
|---|------|--------|
| 4.7.1 | 核心发现：GNN 精度上无法超越 VLM → 转向两个新方向 | ✅ |
| 4.7.2 | 新方向调研：置信度打分 + 元素补全（文档已归档） | ✅ |

### 4.8 方向 1 — 约束感知置信度打分 ✅

**思路:** GNN 预测每个 VLM 检测的可靠性分数，过滤低置信度检测。
**方法:** GT 元素（正样本）+ 随机 imposter 元素（负样本）→ 训练存在性头部。
**结果 (500 RICO, 50% imposter ratio):**

| 指标 | 值 |
|------|-----|
| Accuracy | **93.2%** |
| Precision | 99.1% |
| Recall | 90.7% |
| AUROC | **0.989** |

| # | 条目 | 状态 |
|---|------|--------|
| 4.8.1 | `scripts/train_confidence.py` 训练管线 | ✅ |
| 4.8.2 | Imposter 生成（随机 bbox + 随机类型） | ✅ |
| 4.8.3 | 评估：AUROC / Precision / Recall / Accuracy | ✅ |
| 4.8.4 | 500 张 RICO 验证实验 | ✅ |

### 4.9 方向 2 — 结构性元素补全 ✅

**思路:** GNN 检测约束图中的"空洞"，预测缺失元素的位置和类型。
**核心结果 (2000 RICO, 60% drop):** 违反检测 Acc **95%** · 提议 MSE **0.044** · 提议 IoU 较 NN 基线 **+40%** (drop≥0.6)

| # | 条目 | 状态 |
|---|------|--------|
| 4.9.1 | `data/masking.py` 合成元素删除管线 | ✅ |
| 4.9.2 | `model/heads.py:ElementProposalHead` 提议头 | ✅ |
| 4.9.3 | 违反检测验证（自监督预训练） | ✅ |
| 4.9.4 | 联合训练违反 + 提议头 | ✅ |
| 4.9.5 | 系统评估：4 drop ratios × 2 seeds，基线对比 | ✅ |
| 4.9.6 | 类型预测：提议头输出 8 类 logits | ✅ |

**完整评估 (4 drop ratios, 500 RICO, 双 seed 平均):**

| drop | GNN Acc | GNN MSE | GNN IoU | NN MSE | NN IoU | GNN > NN? |
|------|---------|---------|---------|--------|--------|-----------|
| 0.2 | 92.5% | 0.073 | 0.047 | **0.020** | **0.057** | ❌ |
| 0.4 | 90.8% | 0.051 | 0.079 | **0.032** | **0.110** | ❌ |
| **0.6** | **91.4%** | 0.049 | **0.123** | 0.044 | 0.088 | **✅ (+40% IoU)** |
| **0.8** | **90.8%** | **0.044** | **0.097** | 0.048 | 0.062 | **✅** |

### 4.10 真实 VLM 测试 ⚠️

**脚本:** `scripts/evaluate_vlm_completion.py`
Qwen3-VL Flash 预测 (200 images) 经完成管线运行；RICO GT 稀疏导致仅 32/193 图产生有效图。**基础设施就绪**，需更好 GT 数据（ScreenSpot / 人工标注）才能评估。

### 4.11 DINOv2 视觉特征升级 ✅

**目标:** 用 DINOv2-base 替换 vit_tiny，评估视觉特征升级的收益。

| # | 条目 | 状态 |
|---|------|--------|
| 4.11.1 | 下载/缓存 DINOv2-base (~346MB) | ✅ |
| 4.11.2 | 预计算 500 RICO 特征（`scripts/precompute_dinov2_features.py`） | ✅ |
| 4.11.3 | 训练 + 评估（简单拼接，对比 vit_tiny 基线） | ✅ |

**结果 (500 RICO, seed=42, hd=128, drop=0.4):**

| 指标 | +vit_tiny (192-d) | +DINOv2 (768-d) | Δ |
|--------|:-----------------:|:---------------:|:-:|
| Violation Acc | 0.846 | 0.854 | +0.008 |
| Proposal MSE | 0.081 | 0.085 | −0.005 |
| Type Acc | 0.405 | 0.403 | −0.002 |

**结论:** DINOv2 无明显优势：violation +0.8pp，proposal/type 持平；参数 +30% (245K→319K)，预计算慢 6 倍 (30s vs 173s)。**vit_tiny 仍是推荐选择**。

---

## Phase 5: 集成测试 ✅

**目标:** 合成与真实数据上的端到端验证。
**状态:** 942 tests pass

| 子阶段 | 覆盖 | 测试文件 |
|-----------|------|---------|
| **5.1** 原始管线 | 数据流 → 图构建 → 模型前向 → 端到端 → 基线 | `test_integration_5a.py` |
| **5.2** 完成管线 | 违反图 → 遮掩 → 提议头 → 联合训练冒烟 → 评估冒烟 → 基线正确性 | `test_integration_5b.py` |

| # | 条目 | 状态 |
|---|------|--------|
| 5.1.1 | 数据管线：合成 JSON → parse → Dataset → DataLoader | ✅ |
| 5.1.2 | 图构建：合成 JSON → constraints → HeteroData → verify keys | ✅ |
| 5.1.3 | 模型前向：梯度回传，loss 标量，训练不 crash | ✅ |
| 5.1.4 | 端到端：VLM JSON → InferencePipeline → corrected JSON | ✅ |
| 5.1.5 | 评估基线：baselines + Evaluator → 所有指标 | ✅ |
| 5.2.1 | 违反图构建：drop=0/0.5/1 边界验证 | ✅ |
| 5.2.2 | 遮掩管线：mask_ratio=0/0.6/1 验证 | ✅ |
| 5.2.3 | 提议头：输出形状、梯度、Sigmoid 范围 | ✅ |
| 5.2.4 | 联合训练冒烟：`train_violation.py --n 10 --epochs 2` | ✅ |
| 5.2.5 | 评估冒烟：`evaluate_completion.py --n 10 --epochs 2` | ✅ |
| 5.2.6 | 基线正确性：NN / Center 基线数值合理 | ✅ |

---

## Phase 6: 性能测试 ✅

**目标:** 建立性能基线，确保实际可用性。
**脚本:** `scripts/benchmark_performance.py`

| # | 基准 | 指标 | 结果 |
|---|-----------|---------|--------|
| 6.1 | 数据加载吞吐 | 200 RICO JSONs → graph build | 2.1ms/img = **467 img/s** |
| 6.2 | 图构建扩展性 | 10/50/100/500 elem | 0.2ms → 255ms (O(N²)) |
| 6.3 | 训练吞吐量 | 50 graphs × 3 epochs, hidden=64 | **357 steps/s** |
| 6.4 | 推理延迟 | 100 graphs p50/p95/p99 | **0.53 / 0.96 / 1.11ms** |

**结论:** GNN 不是瓶颈（推理 0.5ms p50）；VLM 才是限速步骤 (~2s/图)。

---

## Phase 7: 实施 (实验运行) ✅

**目标:** 定义并执行实验方法学，确保可复现性。

| # | 条目 | 状态 | 备注 |
|---|------|--------|-------|
| 7.1 | `experiments/run.py` 统一入口 | ✅ | 4 子命令：train-violation / train-confidence / evaluate-completion / constraint-ablation |
| 7.2 | 约束类型消融 | ✅ | CONTAINMENT 最关键 (acc 90.8→88.9%) |
| 7.3 | 图构建超参敏感性 | ✅ | sweep: hd 64/128/256 + lr 1e-3/5e-4 |
| 7.4 | VLM 噪声鲁棒性 | ✅ | 5 类 VLM 全覆盖 |
| 7.5 | 跨数据集泛化 | ✅ | RICO→ScreenSpot: 28.1% → 72.1% (+44pp, VLM pseudo-GT fine-tune) |
| 7.6 | 可复现性 | ✅ | seed_everything + deterministic 已验证 |

---

## Phase 9: 研究 — 受控实验 ✅

**目标:** 验证 Phase 7 结论，建立统计显著性，回应评审批评。

### 9.1 受控两模型对比

评审关键批评：两模型比较同时变了两个变量（约束类型 AND 头配置）。需全类型 × 3 头配置对照。

| # | 条目 | 状态 |
|---|------|--------|
| 9.1.1 | 全类型 × violation-only (no coord loss) | ✅ |
| 9.1.2 | 全类型 × proposal-only (no violation loss) | ✅ |
| 9.1.3 | 5 seed 评估 + 置信区间 | ✅ |

**结果 (5 个 seed):**

| 配置 | seed 42 | seed 73 | seed 99 | seed 123 | seed 256 | 均值 ± 标准差 |
|--------|:-------:|:-------:|:-------:|:--------:|:--------:|:----------:|
| Full × joint | 0.9062 | 0.8612 | 0.8568 | 0.8747 | 0.8803 | **0.8758 ± 0.0195** |
| Full × violation-only | 0.9263 | 0.9003 | 0.8799 | 0.8871 | 0.8974 | **0.8982 ± 0.0177** |
| Full × proposal-only | 0.4802 | 0.4521 | 0.5149 | 0.5927 | 0.4029 | **0.4886 ± 0.0712** |

**结论:** violation-only (0.898 ± 0.018) 明显优于 joint (0.876 ± 0.020)——多任务联合训练损害违反检测，证实评审怀疑。

### 9.2 Real VLM 端到端评估

评审关键批评：所有实验都用合成元素删除，唯一真实 VLM 测试 (4.9.7) acc 仅 27.6%、IoU 0.000。真实 VLM 错误模式与随机删除不同（type-dependent / 位置偏置 / 结构相关）。

| # | 条目 | 状态 |
|---|------|--------|
| 9.2.1 | RICO real VLM 端到端评估（4.9.7 复现 + 改进） | ✅ |
| 9.2.2 | ScreenSpot 人工 GT 接入（ThinkPad SMB，610 images） | ✅ |
| 9.2.3 | ScreenSpot 真实 VLM 端到端评估 | ✅ |

**RICO 结果 (196 images, Qwen3-VL Flash, center-distance Hungarian, threshold=0.1):**

| 指标 | 值 |
|--------|-------|
| VLM Precision / Recall / F1 | 0.382 / 0.235 / 0.291 |
| GNN Existence Acc / AUROC | 0.665 / 0.703 |
| VLM error rate / correction ceiling | 0.765 / ~0.508 (66.5% errors addressable) |

**ScreenSpot 对比 (600 images):**

| 指标 | RICO | ScreenSpot |
|--------|:----:|:----------:|
| VLM Precision / Recall / F1 | 0.382 / 0.235 / 0.291 | 0.028 / 0.383 / 0.052 |
| GNN Existence Acc / AUROC | 0.665 / 0.703 | 0.972 / 0.489 |
| GNN Pos / Neg mean (TP/FP) | 0.536 / 0.398 | 0.481 / 0.481 |

**结论:** RICO 上 VLM recall 极低 (0.235)，存在性头 (AUROC 0.703) 有区分度，可修正约 50% VLM 错误。ScreenSpot 上 VLM 大量 FP (Prec=0.028, 17K predictions vs 1.2K GT)，存在性头退化为全负预测 (Acc 0.972 无意义，AUROC 0.489 无区分度)——置信度模型不跨域迁移。

### 9.3 类型预测

评审发现训练目标不一致：多元素删除时 bbox 取平均但 type 取第一个。

| # | 条目 | 状态 |
|---|------|--------|
| 9.3.1 | 单元素删除实验（目标一致） | ✅ |
| 9.3.2 | 增加 type loss weight 验证可训练性 | ✅ |

**结果 (n=5000, 288 graphs):**

| 指标 | type_weight=0.5 | type_weight=2.0 |
|--------|:---------------:|:---------------:|
| Val Acc | 0.917 | 0.889 |
| Prop MSE | 0.087 | 0.087 |
| Type Acc | **0.618** | **0.618** |

**结论:** Type Acc 封顶 ~62%——约束上下文只含空间/结构信息，语义类型区分本质受限；加大 type loss 权重无效。

### 9.4 置信度模型部署

唯一强保留发现，可直接使用。

| # | 条目 | 状态 |
|---|------|--------|
| 9.4.1 | 真实数据重训模型替换 `checkpoints/confidence_scoring/` | ✅ |
| 9.4.2 | ScreenSpot 跨域验证 | ✅ |

**结果:** 真实数据 AUROC **0.780**（合成模型 0.989 高估真实性能）；ScreenSpot 跨域 AUROC **0.554**、Acc@0.5=0.040，TP/FP 置信度均 ≈0.90，无法区分跨域 FP 模式。

### 9.5 全管线对比 (修正前 vs 修正后)

**脚本:** `experiments/eval_real_vlm_pipeline.py`
**模型:** `violation_detection/best_model.pt` (hd=128)。joint 模型存在性头在真实数据上塌缩 (~0.48)，completion 模型违反头输出 ~0——仅专用违反检测模型产出有效提议。

| 指标 | 修正前 (仅 VLM) | 修正后 (VLM+GNN) | Δ |
|--------|:-----------------:|:---------------:|:-:|
| Precision (pooled) | 0.3821 | 0.3686 | **−0.0135** |
| Recall (pooled) | 0.2351 | 0.2823 | **+0.0472** |
| F1 (pooled) | 0.2911 | 0.3197 | **+0.0286** |
| TP / FP / FN | 1126 / 1821 / 3663 | 1352 / 2316 / 3437 | +226 / +495 / −226 |

**机制:** 2947 VLM elements → 721 proposals (NMS 后) → 3668 corrected (GT 4789)。

**结论:** 提议恢复 226 个遗漏元素 (recall +4.7pp)，但引入 495 个新 FP (precision −1.4pp)，净 F1 +2.9pp；存在性头在真实 VLM 上无过滤价值。
**⚠️ 更正:** 本表为早期记录，曾受 checkpoint 加载错误影响（strict=False 丢弃 89% 权重）；权威数字见 Phase 11 重要更正（正确加载后 F1 +2.0pp，recall +2.2pp，precision +1.1pp）。

### 9.6 真实数据微调

**脚本:** `experiments/finetune_real_vlm.py`
**思路:** 用真实 VLM 预测 + GT 匹配微调（不再用合成 drop）：matched = TP，unmatched = FP，violation/coord 目标全 0。
**数据:** 200 RICO VLM 预测，80/20 划分 → 160 训练 / 40 验证。
**训练:** 30 epochs, lr=1e-4, AdamW, 按验证 loss 保存最优。

| 指标 | 修正前 | 修正后 | Δ |
|--------|:------:|:-----:|:-:|
| Completion F1 (pooled) | 0.3748 | 0.3955 | **+0.0207** |
| Precision (pooled) | 0.3998 | 0.4165 | +0.0167 |
| Recall (pooled) | 0.3528 | 0.3765 | +0.0237 |

**机制:** TP 327→349 (+22) · FP 491→489 (−2) · FN 600→578 (−22) · proposals 214 (val)。

**结论:** 真实数据微调提升全部补全指标 (F1 +2.1pp)，TP +22、FP 基本持平；违反/存在性指标不变。增益小于原始管线——合成 drop 已捕获主要结构模式，微调边际收益递减。

### 9.7 交叉注意力融合 (编码器前置) ✅

**目标:** 用可学习跨注意力融合替换简单拼接（struct→Q, visual→KV → residual + LayerNorm → GNN encoder）。

| # | 条目 | 状态 |
|---|------|--------|
| 9.7.1 | `attention.py:CrossAttentionFusion` | ✅ struct (5-d) × visual (192-d) → 64-d |
| 9.7.2 | `attention.py:SplitAndFuse` | ✅ 自动拆分 197-d → 5+192，纯结构回退 |
| 9.7.3 | `model.py:fusion_dim` 参数 | ✅ `BipartiteGNNCorrector` 支持 |
| 9.7.4 | `experiments/train_with_visual.py` 对比实验 | ✅ Concat vs Cross-Attention |

**结果 (3 seeds, 500 RICO, hd=128, drop=0.4):**

| 指标 | 简单拼接 (PR#30) | 交叉注意力 | Δ |
|--------|:---------------------:|:---------------:|:-:|
| Violation Acc | 0.8465 ± 0.0011 | 0.8498 ± 0.0309 | +0.0034 |
| Proposal MSE | 0.0807 ± 0.0043 | 0.0623 ± 0.0043 | **−0.0183** |
| Type Acc | 0.4469 ± 0.0080 | 0.4403 ± 0.0163 | −0.0066 |

**结论:** Proposal MSE 三个 seed 一致改善 (−18~22%)，violation acc 高方差不稳定，type acc 持平；+24.6K 参数 (257K→282K) 性价比存疑——**简单拼接仍是推荐方案**，除非以提议质量为优先。

---

## Phase 10: 方案 (文档与资料更新) ✅

| # | 条目 | 状态 |
|---|------|--------|
| 10.1 | README.md 更新 | ✅ |
| 10.2 | pyproject.toml 最终版 | ✅ |

---

## Phase 11: 网页演示 ✅

**目标:** 单页 web app：上传截图 → VLM + GNN → 双栏 bbox overlay。
**实现:** 轻量单进程 FastAPI + 原生 JS SPA（无 Docker/MySQL）。

| # | 条目 | 状态 |
|---|------|--------|
| 11.1 | 开发文档（web_demo strategy + review） | ✅ |
| 11.2 | FastAPI 后端 (`api/main.py` + `api/pipeline.py`)，joint checkpoint 44/44 keys shape-filter 加载 | ✅ |
| 11.3 | 前端 (`web/index.html`)：双栏 Canvas + 案例导航 + 指标卡 + 上传模式 | ✅ |
| 11.4 | 预计算 hero cases（`scripts/prepare_demo_cases.py` → `demo_data/cases.json` 12 案例） | ✅ |
| 11.5 | 端到端验证：health/cases/case/screenshot/predict 全部 200，942 tests 通过 | ✅ |

**重要更正:** 原 `experiments/eval_real_vlm_pipeline.py` 用 hd=128 加载 hd=16 checkpoint（strict=False 丢弃 89% 权重）→ 文档中记录的 +2.9pp F1 为随机权重假象。正确加载 joint 模型后全量 200 图 F1 +1pp；精选案例 ΔF1 +0.15~0.26（10027: +0.156, 10043: +0.260）。

**范围说明:** HTML/CSS 代码生成（原 Phase 15: `web/codegen/html_generator.py` + `POST /api/generate-html` + 前端 HTML 预览区）未纳入 demo 范围，条目已关闭，不追踪。

---

## Phase 12: 终期报告 ✅

**目标:** UG 暑期实习终期报告并提交 SharePoint。
**格式:** 单栏 LaTeX 报告，面向有知识但不一定是领域专家的读者。

| # | 条目 | 状态 |
|---|------|--------|
| 12.1 | 结构规划：问题（为什么重要）→ 现有方案 → 方案 + 结果，非专家可读 | ✅ |
| 12.2 | 报告正文 (`report/report.tex`)：intro/背景、现有方案、方法、实验、结论 | ✅ |
| 12.3 | 图表复用 + 编译验证（pdflatex ×2 + bibtex，零 error） | ✅ |
| 12.4 | 与导师 (Prof Lau) 沟通确认要求，提交 SharePoint（用户执行） | ✅ |
| 12.5 | 终稿审查 + 状态更新 + commit/push | ✅ |

**结果:** `report/main.pdf` — 10 页单栏，零 error 零 overfull，13 条参考文献。
**内容:** 问题重要性 → 现有方案局限（微调/检测器级联/生成模型）→ 方法（图构建 + 消息传递 + 4 预测头 + 自监督补全）→ 实验（消融/训练目标/补全/真实 VLM 端到端）→ 讨论（局限 + Goldilocks 规律 + 未来工作）。
**导师确认:** 已与 Prof. Lau 沟通并确认报告要求，可提交 SharePoint。

---

## Phase 13: 海报 ✅

**目标:** 实习最终展示研究海报。
**格式:** A0 竖版 Beamer poster（gemini 主题 + CUHK 配色），TikZ/pgfplots 原生图表，latexmk + lualatex 编译。

| # | 条目 | 状态 |
|---|------|--------|
| 13.1 | 海报草稿（核心发现 + 流程图 + 实验结果） | ✅ |
| 13.2 | 版式与视觉设计 | ✅ |
| 13.3 | 打印版本 (PDF) | ✅ |

**结果:** `poster/poster.pdf` — 单张 A0 竖版，零 error 编译通过。
**内容（三栏）:** ① 动机与问题（VLM 遗漏 10–30% / 偏移 10–50+ px）+ 方法流程图；② 图构建（10 种约束、节点特征、两跳消息传递、自监督补全）+ RICO / ScreenSpot 截图对比；③ 结果（违反检测 90–94%、补全 IoU +39%/+56%、端到端恢复 **106** 元素 F1 +2.0pp、57K 参数 ~0.5ms/图）+ 结论与局限。
**视觉设计:** CUHK 配色与 Logo（5pt 白描边），图表全部 TikZ/pgfplots 原生绘制（与报告一致，无 PNG 插图），多轮迭代打磨（十余次提交）。

---

## Phase 14: 方案 (最终整理) ✅

| # | 条目 | 状态 |
|---|------|--------|
| 14.1 | README.md 重写（实际状态 + 结果 + demo + mermaid 流程图） | ✅ |
| 14.2 | pyproject.toml 最终版 | ✅ |
| 14.3 | LICENSE (MIT) + 仓库清理（内部文档删除、提交材料 gitignore）+ CLAUDE.md → AGENTS.md 符号链接 | ✅ |

---

## 方法论对照

| 阶段 | TASK 对应 | 产出 |
|------|-----------|------|
| 需求分析 | Phase 1 | `docs/requirements/` |
| 概要设计 | Phase 2 | 概要设计文档（已归档） |
| 详细设计 | Phase 3 | 详细设计文档（已归档） |
| 开发 | Phase 4 | `src/bipartite_gnn_gui/` |
| 集成测试 | Phase 5 | 942 个测试通过 |
| 性能测试 | Phase 6 | 性能基准报告 |
| 实施 | Phase 7 | 实验脚本 |

---

## 执行原则

1. **轻分析、重实现**: 分析与设计产出文档，开发产出代码
2. **不回溯**: 需求分析假设全程不变
3. **测试穿插**: 模块完成即可集成测试，无需等待全部开发完
4. **实验用完整系统**: 实施依赖全部开发与测试完成
5. **增量交付**: 每完成一项即提交 (PR/commit)
6. **无依赖模块并行**: 推理管线 (mock) 与视觉特征预计算可提前独立开发

---

## 延伸目标

| # | 描述 |
|---|------|
| S1 | 基于注意力的约束重要性加权 |
| S2 | VLM 特征与图特征之间的交叉注意力 |
| S3 | 多尺度图：容器层级 → 子元素 → 叶子元素 |
| S4 | 合成 GUI 布局生成器用于数据增强 |
| S5 | ONNX / TorchScript 导出部署 |
