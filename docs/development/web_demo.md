# Web Demo — 开发与使用文档

> Phase 11。实际实现为**轻量单进程 FastAPI + 原生 JS 前端**，无 Docker/MySQL。
> 设计决策与策略依据见 [`web_demo_strategy.md`](web_demo_strategy.md)，代码审查清单见 [`demo_review_plan.md`](demo_review_plan.md)。

---

## 1. 架构总览

```
┌──────────────┐     ┌──────────────────────────────┐
│  浏览器       │────▶│  FastAPI (api/main.py)       │
│  index.html  │ 8765│  ├─ GET  /                   │  → 前端 SPA
│  双栏 Canvas  │     │  ├─ GET  /api/cases          │  → 案例列表
│  bbox overlay│     │  ├─ GET  /api/case/{id}      │  → 单案例数据
│              │     │  ├─ GET  /api/screenshot/{id}│  → 截图文件
│              │     │  ├─ POST /api/predict        │  → VLM+GNN 推理
│              │     │  └─ GET  /api/health         │
│              │     │                              │
│              │     │  └── DemoPipeline (api/pipeline.py)
│              │     │        joint checkpoint (44/44 keys)
│              │     │        violation_threshold=0.60
│              │     └──────────────────────────────┘
```

### 1.1 与旧版（Docker 方案）的区别

| 维度 | 旧设计（已废弃） | 实际实现 |
|------|----------------|---------|
| 后端 | Docker + wheel + MySQL + nginx | 单进程 FastAPI，`python3 api/main.py` |
| 前端 | nginx 托管 + 热更新 | FastAPI 直接 serve `web/index.html` |
| 数据库 | MySQL 8.0 持久化推理历史 | 无数据库 |
| 推理 | 每次实时 VLM 调用 | **预计算 12 张 hero cases** + 可选实时上传 |
| 模型 | 未指定（旧代码加载错误 checkpoint） | **joint 模型**（shape-filter 44/44 keys） |

---

## 2. 目录结构

```
bipartite-gnn-gui/
├── api/
│   ├── main.py               # FastAPI app + 路由（8765 端口）
│   ├── pipeline.py           # DemoPipeline: checkpoint 加载 + VLM API + GNN 推理
│   └── requirements.txt      # fastapi/uvicorn/pillow 等
├── web/
│   └── index.html            # 单页 SPA（双栏 Canvas + 案例导航 + 上传）1151 行
├── scripts/
│   └── prepare_demo_cases.py # 预计算 hero cases（12 张已验证案例）
├── demo_data/                # 生成物（gitignore，运行 prepare 脚本生成）
│   ├── cases.json            # 12 案例完整数据（bbox + 指标）
│   ├── summary.json          # 聚合统计
│   └── screenshots/          # 案例截图
├── checkpoints/
│   └── violation_detection_joint/best_model.pt   # 唯一可信 checkpoint
└── docs/development/
    ├── web_demo.md           # ← 本文档
    ├── web_demo_strategy.md  # 策略与详细实现计划
    └── demo_review_plan.md   # 代码审查清单
```

---

## 3. 快速开始

### 3.1 安装依赖

```bash
pip install -e ".[demo]"   # 或: pip install -e . && pip install fastapi uvicorn python-multipart requests python-dotenv
```

### 3.2 预计算案例（首次运行必做）

```bash
python scripts/prepare_demo_cases.py
# 输出: demo_data/cases.json (12 案例) + demo_data/summary.json + demo_data/screenshots/
```

### 3.3 启动服务器

```bash
python api/main.py          # 默认 http://localhost:8765
# 或自定义端口
PORT=9000 python api/main.py
```

浏览器打开 `http://localhost:8765`。

### 3.4 验证

```bash
curl http://localhost:8765/api/health
# {"status":"ok","model":{"name":"BipartiteGNNCorrector","params":220439,"hidden_dim":128},"device":"cpu","violation_threshold":0.6}
```

---

## 4. 后端 API

### 4.1 `GET /` — 前端 SPA

返回 `web/index.html`（内联 CSS/JS，无构建步骤）。

### 4.2 `GET /api/cases` — 案例列表

```json
[
  {"id": "10027", "name": "Weather App", "metrics": {"before": {...}, "after": {...}}},
  ...
]
```

### 4.3 `GET /api/case/{case_id}` — 单案例完整数据

```json
{
  "id": "10027",
  "name": "Weather App",
  "screenshot": "10027.jpg",
  "img_w": 1080, "img_h": 1920,
  "vlm_elements": [{"bbox": [0.1, 0.2, 0.3, 0.4], "label": "text", "id": 0}, ...],
  "proposals": [{"bbox": [...], "violation_score": 0.72}, ...],
  "metrics": {
    "before": {"detections": 27, "tp": 22, "fp": 5, "fn": 50, "precision": 0.815, "recall": 0.306, "f1": 0.444},
    "after":  {"detections": 38, "tp": 33, "fp": 5, "fn": 39, "precision": 0.868, "recall": 0.458, "f1": 0.600}
  }
}
```

bbox 均为**归一化坐标** `[x1, y1, x2, y2] ∈ [0,1]`。

### 4.4 `GET /api/screenshot/{case_id}` — 截图文件

返回 JPEG 文件。

### 4.5 `POST /api/predict` — 上传模式（实时推理）

multipart 表单，字段 `file`（截图）。

- 需 `DASHSCOPE_API_KEY`（`.env` 或环境变量），否则返回 400：
  `{"error": "VLM API key not configured. Set DASHSCOPE_API_KEY in .env to enable upload mode."}`
- 成功返回：`{id, img_w, img_h, vlm_elements, proposals, metrics(null), vlm_time_ms, gnn_time_ms}`
- 实测延迟：VLM ~8-14s（API 调用），GNN ~6ms（本地推理）

### 4.6 `GET /api/health` — 健康检查

```json
{"status": "ok", "model": {"name": "BipartiteGNNCorrector", "params": 220439, "hidden_dim": 128}, "device": "cpu", "violation_threshold": 0.6}
```

---

## 5. 模型加载（重要）

### 5.1 唯一可信 checkpoint

| Checkpoint | 状态 | 说明 |
|-----------|:----:|------|
| `violation_detection_joint/best_model.pt` | ✅ **使用** | hd=128, 44/44 keys 全匹配 |
| `violation_detection/best_model.pt` | ❌ 废弃 | 实际 hd=16，旧评估脚本加载它产生随机权重假象 |
| `violation_detection_violation_only/` | ❌ 不可用 | proposal head 未训练，输出全无效 bbox |
| `violation_detection/visual_fusion_model.pt` | ❌ 不可用 | 需 197-d 视觉输入（ViT 特征） |

### 5.2 shape-filter 加载

`api/pipeline.py` 用 **shape-filter** 加载（而非 `strict=True`/`strict=False`）：

```python
filtered = {k: v for k, v in state.items()
            if k in model_state and v.shape == model_state[k].shape}
# 断言 matched >= 30/44 + critical layers 存在，否则 raise
model.load_state_dict(filtered, strict=False)
```

`_detect_hidden_dim()` 从 checkpoint 第一层 weight 推断 hidden_dim（兼容 hd=16/128）。

### 5.3 violation_threshold = 0.60

实测 sweep（200 RICO）：0.50 产生过多 FP（ΔTP+180/ΔFP+822），0.60-0.75 平衡较好。

### 5.4 VLM 坐标基准（重要 bug 修复）

**qwen3-vl-flash 返回的 bbox 坐标基于固定 1080×960 内部帧，与输入图像尺寸无关。**

`generate_vlm_predictions.py` 记录的是原图尺寸（1080×1920），若按此归一化 → y 坐标减半 → 所有框只覆盖上半屏。
实测验证：全部 200 个 VLM 预测文件坐标均在 1080×960 内；同一张 540×960 输入返回 x_max=974（超出 540，证明基准是 1080）。

**修复：** `api/pipeline.py`、`api/main.py`、`scripts/prepare_demo_cases.py`、`scripts/visualize_demo_cases.py`
统一按 `VLM_COORD_W=1080, VLM_COORD_H=960` 归一化，忽略 JSON 中的 image_width/image_height。

**影响：** 坐标修正后 hero case 筛选从 12 → 6（此前部分 TP 是错误坐标下的假匹配），红框从覆盖 48% → 96% 截图高度。

---

## 6. 前端要点

### 6.1 布局

```
┌──────────────────────────────────────────────────┐
│  Header: GUI Structural Correction Demo          │
├────────────┬─────────────────────────────────────┤
│ 侧边栏      │  主区                                │
│ 案例列表    │  ┌─────────────┬──────────────┐     │
│ ◀ 1/12 ▶   │  │ Canvas L    │ Canvas R     │     │
│ [Weather+0.156]│ (VLM 红框) │ (红+蓝虚线)   │     │
│ [Settings+0.260]└─────────────┴──────────────┘     │
│ ...        │  指标卡 (Before/After/Δ 表格)         │
│ [上传截图]  │  状态行 "GNN 找回 +11 个漏检元素"      │
├────────────┴─────────────────────────────────────┤
│  免责声明: 全量 200 图平均 F1 +1pp ...             │
└──────────────────────────────────────────────────┘
```

### 6.2 Canvas 渲染

- 双栏同帧绘制（`requestAnimationFrame`）
- 归一化 bbox → canvas 像素：保持宽高比缩放 + 居中
- Retina 屏：`ctx.scale(dpr, dpr)`
- 颜色：红 = VLM 检测，蓝虚线 = GNN 提议，绿 = GT 匹配
- 左图红 X 标记 GT 漏检中心

### 6.3 交互防护（已实现）

竞态防护（requestId）、fetch try/catch、resize debounce、键盘 ←→ 导航 preventDefault、
拖拽 preventDefault、指标 toFixed(3)、空案例提示、上传 400 错误条。

---

## 7. 开发工作流

```bash
# 改前端 → 直接刷新浏览器（无构建）
# 改后端 → 重启进程
kill %1; python api/main.py

# 重新预计算案例（换 checkpoint 或阈值后）
python scripts/prepare_demo_cases.py
```

---

## 8. 已知边界

1. **效果因图而异**：全量 200 图平均 F1 +1pp；demo 展示的 12 张是精选案例（ΔTP≥4 且 ΔFP≤5）
2. **上传模式需要 VLM API key**：无 key 时仅案例模式可用
3. **pillow-heif 缺失时**：HEIC/HEIF 上传不支持（JPEG/PNG 正常），日志提示不崩溃
4. **模型推理 CPU only**：~6ms/图，无 GPU 依赖
5. **端口冲突**：用 `PORT` 环境变量覆盖
6. **`.env` 必须放在项目根目录**（`api/main.py` 从上级目录加载）

---

## 9. 测试

```bash
pytest tests/ -q          # 942 passed（2026-07-31 验证）
```

端到端手动验证（2026-07-31 实测通过）：
- `/api/health` → 200, 220439 params, hd=128, threshold=0.6
- `/api/cases` → 12 案例
- `/api/case/10027` → F1 0.444→0.600, 27 elems, 11 proposals
- `/api/case/99999` → 404 + error
- `/api/screenshot/10027` → 200 JPEG 1080×1920
- `/api/predict`（真实 VLM key）→ 17-26 elems, 7-8 proposals, VLM 8-14s + GNN ~6ms
- 浏览器：案例切换（点击/键盘）、指标卡、双栏 Canvas 红/蓝/绿框渲染正确
