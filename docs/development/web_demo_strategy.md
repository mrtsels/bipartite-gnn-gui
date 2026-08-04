# Web Demo 策略 v3 — 基于实测验证

> 用真实 checkpoint 跑完 200 张 RICO + 人工检查可视化结果后的结论。
> 上一版的问题：没有实际运行管线，凭文档里的数字写方案，而文档数字本身有误。

---

## 0. 实测发现：文档里的结果不可信

重跑管线时发现关键 bug：`experiments/eval_real_vlm_pipeline.py` 用 `hidden_dim=128` 加载
`violation_detection/best_model.pt`（实际 hidden_dim=16），`strict=False` 静默丢弃 39/44 个
权重（89%）→ **之前记录的 +2.9pp F1 是用随机权重跑出来的假象。**

用 shape-filter 正确加载后重新评估：

| Checkpoint | 加载 | 有效 proposals | 真实 F1 Δ |
|-----------|------|--------------|----------|
| `violation_detection/best_model.pt` | 5/44 (11%) | — | ❌ 随机权重 |
| `violation_detection_violation_only/best_model.pt` | 44/44 | **0**（proposal head 未训练） | ❌ 无用 |
| `violation_detection/visual_fusion_model.pt` | 43/44 | 部分 | ⚠️ 需 197-d 视觉输入，纯结构输入下输入层随机 |
| **`violation_detection_joint/best_model.pt`** | **44/44** | **有效** | **✅ 唯一可信** |

**joint 模型全量 200 图真实结果（threshold sweep）：**

| 阈值 | ΔF1 | ΔTP | ΔFP | 每正确提议代价 |
|------|-----|-----|-----|--------------|
| 0.50 | +0.008 | +180 | +822 | 1 : 4.6 |
| 0.60 | +0.010 | +157 | +644 | 1 : 4.1 |
| **0.75** | **+0.010** | **+103** | **+336** | **1 : 3.3** |

**结论：全量平均 F1 只提升 ~1pp，且每个正确提议带来 3-4 个错误提议。** GNN 补全的收益在统计上是真实的，但整体幅度很小。

**但关键发现：精选案例差异巨大。** 单图 F1 提升可达 +0.15~+0.26，视觉上蓝框精确覆盖 VLM 漏检区：

| 图片 | ΔTP | ΔFP | F1 before→after | 视觉效果 |
|------|-----|-----|----------------|---------|
| **10027** (天气 App) | **+11** | **+0** | 0.444→0.600 | 蓝色框精确填补天气预报区、每日预报区所有漏检 |
| **10043** (设置页) | **+7** | +3 | 0.229→0.489 | VLM 只检出 4 个 TP → GNN 补到 11 个，全页覆盖 |
| 10068 | +7 | +3 | 0.304→0.500 | 明显 |
| 10005 | +4 | +3 | 0.500→0.576 | 较明显 |
| 10059 | +6 | +5 | 0.571→0.632 | 较明显 |
| 10064 | +0 | +5 | 0.158→0.140 | ❌ 变差 |

---

## 1. Demo 核心体验：补全缺失元素

**用户上传一张截图 → 看到 GNN 用蓝色框把 VLM 漏掉的真元素一个个找回来。**

这就是「有 GNN vs 没 GNN」的区别——不是精修坐标（那是假象），是**补全漏检**：

```
左边（VLM 检测）：  红色框稀疏，大片元素没有框，红 X 标记着漏检
右边（VLM+GNN）：   蓝色框精确落在漏检元素上，视觉上"图被补全了"
```

### 1.1 关键设计决策：预选 hero cases + 实时推理双模式

**为什么必须预选：** 全量平均 ΔF1=+1pp，意味着随机上传的图大概率看不出差别
（甚至变差，如 10064）。demo 必须展示**挑选过的、确实有效**的案例。

```
┌────────────────────────────────────────────┐
│ 模式选择                                     │
│                                            │
│ ① 精选案例 (推荐)  — 12 张预计算的 hero     │
│    每张都经过验证：ΔTP≥+4 且 ΔFP 相对小       │
│    秒开，无需 VLM API                        │
│                                            │
│ ② 上传自己的截图 — 实时推理                  │
│    需要 DASHSCOPE_API_KEY 调用 VLM           │
│    诚实提示："效果因图而异，设置页/列表页最佳"  │
└────────────────────────────────────────────┘
```

### 1.2 Hero case 筛选标准（已验证可执行）

从 200 张 RICO 里筛（用 joint 模型 @ threshold=0.6）：

1. ΔTP ≥ +4（GNN 至少找回 4 个真元素）
2. ΔFP ≤ +5（代价可接受，或 FP+0 更佳）
3. 图片内容常见（天气、设置、音乐、购物 App 优先，观众有代入感）
4. 人工检查可视化：蓝框必须"落在真元素上"而非乱放

现有 7 张验证通过（10027, 10043, 10033, 10059, 10068, 10005, + 需补 5 张）
→ 每张生成 before/after 叠加图 + 指标卡。

### 1.3 页面布局

```
┌────────────────────────────────────────────────────────┐
│  [GUI Structural Correction Demo]                      │
│  基于约束图的 GNN：把 VLM 漏检的元素找回来                       │
├────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │  案例导航          │  │  Canvas（前后对比）           │ │
│  │  ◀ 1/12 ▶         │  │                            │ │
│  │  [天气 App]        │  │  ┌──────────┬──────────┐   │ │
│  │  [设置页]          │  │  │  VLM     │ VLM+GNN  │   │ │
│  │  [音乐播放器]       │  │  │ (红色框)  │ (红+蓝框) │   │ │
│  │  ...              │  │  └──────────┴──────────┘   │ │
│  │                   │  │                            │ │
│  │  [上传截图]         │  │  图例：红=VLM 蓝=GNN提议      │ │
│  └──────────────────┘  └─────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 指标卡（随案例更新）                                 │  │
│  │  检测数 27→38  TP 22→33  FP 5→5                  │  │
│  │  F1 0.444 → 0.600  (+0.156)                      │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```

### 1.4 每张图的展示内容

- **左图**：VLM 原始检测（红色半透明框）+ 漏检标记（红 X 标注 GT 中心）
- **右图**：VLM 框 + GNN 提议（蓝色实线框）+ 正确匹配的 GT（绿色细框）
- **指标卡**：检测数 / TP / FP / F1 前后对比，Δ 用颜色标出
- **提示文案**（诚实）：「该案例 GNN 找回 X 个漏检元素；全量 200 图平均 F1 +1pp，
  效果因布局结构而异」

---

## 2. 不做什么（实测排除）

| 不做 | 为什么 |
|------|--------|
| ❌ 坐标精修展示 | 真实模型没有可展示的坐标修正效果（原 +2.9pp 是随机权重假象） |
| ❌ 置信度热力着色 | confidence_scoring checkpoint 在真实 VLM 上 TP/FP 分数无区分（0.938 vs 0.931），AUROC 0.780 记录对不上实际 checkpoint |
| ❌ violation-only 模型 | proposal head 未训练，输出 0 个有效提议 |
| ❌ visual_fusion 模型 | 需要每元素 197-d 视觉特征（ViT），纯结构输入下输入层是随机权重 |
| ❌ 类型预测展示 | Type Acc ~62% 且只对合成单删除有效 |
| ❌ ScreenSpot | 跨域失败（AUROC 0.49-0.55） |
| ❌ Docker/MySQL | 单进程 FastAPI 足够 |

---

## 3. 技术实现

### 3.1 后端（单进程 FastAPI）

```
api/demo_app.py
  ├── 启动时：加载 joint checkpoint（44/44 keys 已验证）
  ├── GET  /api/cases            → hero cases 列表
  ├── GET  /api/case/{id}        → 单案例 pre-computed 结果（含指标）
  ├── POST /api/predict          → 上传截图 + VLM API → GNN 推理 → 结果
  └── GET  /api/health
```

- 预计算 12 张 hero case 的 JSON（VLM 框 + 提议框 + 指标），前端直接渲染，**零推理延迟**
- 实时模式：`DASHSCOPE_API_KEY` 可选，未配置时仅显示精选案例模式
- checkpoint 加载用 shape-filter（已验证 joint 模型 44/44）

### 3.2 前端（原生 JS + Canvas，无构建）

- `web/index.html`：左侧案例导航，右侧双栏 Canvas
- 双栏画同一张截图：左=VLM 框（红），右=VLM 框+提议（红+蓝）+GT 匹配（绿）
- 指标卡随案例切换，Δ 值绿色/红色高亮
- 上传模式：拖拽 → POST → 渲染结果 + 提示效果因图而异

### 3.3 数据准备脚本

```
scripts/prepare_demo_cases.py
  1. joint 模型 @ threshold=0.6 跑 200 张 RICO（已验证 ~0.5s 全量）
  2. 按 ΔTP ≥ 4 且 ΔFP ≤ 5 筛选
  3. 输出 demo_data/cases.json（每案例含 boxes + metrics）
  4. 人工抽查可视化，保留视觉合格的
```

---

## 4. 开发路线（详细实现计划）

> 对照 demo_v3 策略，基于现有 `api/main.py` + `api/pipeline.py` + `web/index.html` 改造。

### 总体改造量

| 文件 | 状态 | 改造 |
|------|:----:|------|
| `api/pipeline.py` | 存在但用错 checkpoint | **改**：joint checkpoint + shape-filter + threshold |
| `api/main.py` | 存在但只有 upload API | **加**：hero cases APIs + serve screenshots |
| `web/index.html` | 存在但只是上传页 | **重写**：双栏 Canvas + 案例导航 + 指标卡 |
| `scripts/prepare_demo_cases.py` | 不存在 | **新建** |
| `pyproject.toml` | 缺少 demo deps | **加**：`[demo]` optional-dependencies |
| `api/requirements.txt` | 已完整 | 不加 |

---

### 任务 1：Fix `api/pipeline.py` — 正确 checkpoint + shape-filter 加载

**文件：** `api/pipeline.py`（约 611 行）

**现状问题：**
- 第 351 行：默认 load `violation_detection_violation_only/best_model.pt`（⚠️ proposal head 未训练）
- 第 368 行：`load_state_dict(ckpt, strict=True)` 对 hd=16 的 checkpoint 会崩溃
- 第 341 行：默认 `violation_threshold=0.3`（产生过多 FP）

**改为：**

```python
# 在 DemoPipeline.__init__ 中：
# 1. 默认 checkpoint → checkpoints/violation_detection_joint/best_model.pt
# 2. 加载时 shape-filter（仅匹配形状的 key 才加载）
# 3. violation_threshold 默认 0.60
```

**具体改动清单：**
- 第 345 行：`self._builder` 后加 `self.hidden_dim = self._detect_hidden_dim(checkpoint_path)`
- 新建 `_detect_hidden_dim()` 方法：读 checkpoint 第一层权重 shape 推断 hd
- 第 355-368 行：替换为 `_safe_load_state()` 方法（shape-filter + reshape 匹配）
- 第 341 行：`violation_threshold=0.60`
- 第 351 行：default path → `violation_detection_joint/best_model.pt`
- 第 466 行：violation score 判断保持不变（用 self.violation_threshold）

**验证：** `python -c "from api.pipeline import DemoPipeline; p = DemoPipeline(); print(p.health())"`
→ 应输出 model loaded, params≈220K, hd=128, 44 keys matched

---

### 任务 2：`scripts/prepare_demo_cases.py` — 预计算英雄案例

**目的：** 跑 joint 模型 @ threshold=0.6 在所有 200 张 RICO VLM 上，筛出 ΔTP≥4 的案例，
生成 `demo_data/cases.json` + 拷贝截图。

**输入：**
- `data/vlm_predictions/rico_qwen_flash/*.json`（200 个 VLM 预测）
- `data/rico_local/combined/{id}.json`（RICO GT）
- `data/rico_local/combined/{id}.jpg`（截图）

**输出：**
```
demo_data/
├── cases.json          # 案例列表 + 完整数据
├── summary.json        # 聚合指标
└── screenshots/
    ├── 10027.jpg
    ├── 10043.jpg
    └── ...
```

**cases.json 结构：**
```json
[
  {
    "id": "10027",
    "name": "Weather App",
    "screenshot": "10027.jpg",
    "img_w": 1080, "img_h": 1920,
    "vlm_elements": [
      {"bbox": [x1,y1,x2,y2], "label": "text", "id": 0},
      ...
    ],
    "proposals": [
      {"bbox": [x1,y1,x2,y2], "violation_score": 0.72},
      ...
    ],
    "metrics": {
      "before": {"detections": 27, "tp": 22, "fp": 5, "fn": 50, "precision": 0.815, "recall": 0.306, "f1": 0.444},
      "after":  {"detections": 38, "tp": 33, "fp": 5, "fn": 39, "precision": 0.868, "recall": 0.458, "f1": 0.600}
    }
  },
  ...
]
```

**筛选逻辑：**（复用 `scripts/recheck_eval_pipeline.py` 的核心逻辑）
1. 加载 joint model（shape-filter）
2. 遍历 200 张：VLM 预测 → constraints → GNN 推理 → proposals
3. Hungarian 匹配 before/after → 计算 ΔTP, ΔFP
4. 筛选：ΔTP ≥ 4 且 ΔFP ≤ 5
5. 按 ΔTP 降序排列，取前 12
6. 人工命名（"Weather App", "Settings Page", ...）

**验证：** `python scripts/prepare_demo_cases.py` → 输出 "12 cases saved to demo_data/cases.json"

---

### 任务 3：`api/main.py` — 增加 hero cases API

**新增端点：**

```python
@app.get("/api/cases")
async def list_cases() -> JSONResponse:
    """返回所有 pre-computed 案例的摘要列表。"""
    with open(DEMO_DATA / "cases.json") as f:
        cases = json.load(f)
    # 只返回摘要（不含完整 bbox 列表，减少 payload）
    return [{"id": c["id"], "name": c["name"], "metrics": c["metrics"]} for c in cases]

@app.get("/api/case/{case_id}")
async def get_case(case_id: str) -> JSONResponse:
    """返回单个案例的完整数据（含所有 bbox）。"""
    # 从 cases.json 读取对应案例

@app.get("/api/screenshot/{case_id}")
async def get_screenshot(case_id: str) -> FileResponse:
    """返回案例的截图文件。"""
    # 从 demo_data/screenshots/ 读取
```

**修改点：**
- 第 1-49 行：加导入 `json, FileResponse`
- 第 50 行：加 `DEMO_DATA = Path(__file__).parent.parent / "demo_data"`
- 第 80 行后：插入以上三个新路由
- 保留现有 `/api/predict`, `/api/gnn-only`, `/api/health`, `/` （任务 5 中优化）

**验证：** `curl http://localhost:8765/api/cases` → 返回 12 个案例摘要

---

### 任务 4：`web/index.html` — 重写前端为双栏 Canvas + 案例导航

**现状：** 369 行，基本的上传页

**目标：** 完全重写为双栏对比 + 案例导航的 SPA

**布局（CSS Grid）：**
```
┌──────────────────────────────────────────────────┐
│  .header: title + subtitle                        │
├────────────┬─────────────────────────────────────┤
│ .sidebar   │  .main                               │
│ 案例列表    │  .canvas-area                        │
│ ◀ ▶ 导航   │  ┌─────────────┬──────────────┐     │
│ [Weather]  │  │  Canvas L   │  Canvas R    │     │
│ [Settings] │  │  (VLM)      │  (VLM+GNN)   │     │
│ [Music]    │  └─────────────┴──────────────┘     │
│ ...        │  .metrics-bar                        │
│            │  Before: F1=0.444 → After: F1=0.600 │
│ ─────────  │  .legend                             │
│ [Upload]   │  ■ VLM  ■ GNN proposal  ■ GT match  │
├────────────┴─────────────────────────────────────┤
│  .footer: disclaimer                              │
└──────────────────────────────────────────────────┘
```

**技术选型：**
- 纯原生 JS，零依赖
- Canvas API 绘制 bbox overlay（不是 PIL 预渲染——让前端画，支持缩放和交互）
- Fetch API 加载案例 JSON
- 拖拽上传用原生 `dragenter`/`dragover`/`drop`

**Canvas 渲染逻辑：**
1. 加载截图 → `ctx.drawImage(img, 0, 0)`
2. 加载案例数据 → 遍历 vlm_elements → `ctx.strokeRect` 红框
3. 遍历 proposals → `ctx.strokeRect` 蓝框（虚线：`ctx.setLineDash`）
4. GT 匹配标记：`ctx.strokeRect` 绿细框
5. 指标卡动态更新

**状态管理（简单）：**
```javascript
const state = { cases: [], currentIdx: 0, mode: 'cases' }; // 'cases' | 'upload'
```

**关键函数：**
- `loadCases()` → fetch /api/cases → populate sidebar
- `renderCase(idx)` → fetch /api/case/{id} + screenshot → draw both canvases
- `nextCase()/prevCase()` → state.currentIdx ± 1 → renderCase
- `handleUpload(file)` → POST /api/predict → render from response

**验证：** 浏览器打开 `localhost:8765` → 看到 12 个案例列表 → 点击切换 → Canvas 渲染红/蓝框

---

### 任务 5：`api/main.py` — 优化 `/api/predict` 上传模式

**改动：**
1. 去掉 `vlm_model` Form 参数（固定用 qwen3-vl-flash）
2. `DASHSCOPE_API_KEY` 未配置时 → 返回 400 + 提示"仅支持预选案例模式，上传需配置 API key"
3. 返回格式对齐 case JSON（方便前端复用渲染）
4. 增加加载状态提示的响应字段

**返回格式：**
```json
{
  "id": "upload",
  "img_w": 1080, "img_h": 1920,
  "vlm_elements": [...],
  "proposals": [...],
  "metrics": {"before": {...}, "after": {...}},
  "image_b64": "data:image/jpeg;base64,...",
  "vlm_time_ms": 2340,
  "gnn_time_ms": 0.5
}
```

---

### 任务 6：`pyproject.toml` — 加 demo 依赖

在 `[project.optional-dependencies]` 中追加：

```toml
demo = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "python-multipart>=0.0.6",
    "pillow-heif>=0.21.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
]
```

（这样 `pip install -e ".[demo]"` 即可安装所有 demo 依赖）

---

### 任务 7：端到端测试

1. `pip install -e ".[demo]"` 安装依赖
2. `python scripts/prepare_demo_cases.py` 生成 cases.json
3. `python api/main.py` 启动服务器
4. 浏览器打开 `http://localhost:8765`
5. 验证：案例列表显示 12 项 → 点击切换 → Canvas 绘制正确
6. 验证：上传一张截图 → 返回结果 → Canvas 渲染
7. 验证：指标卡数字与 pre-computed 数据一致

---

### 工时估算

| 任务 | 工时 | 风险 |
|------|:----:|------|
| 1. Fix pipeline.py | 1h | checkpoint 加载兼容性 |
| 2. prepare_demo_cases.py | 1h | 复用已有 recheck 脚本，改动小 |
| 3. api/main.py heroes APIs | 0.5h | 简单 GET 端点 |
| 4. web/index.html 重写 | 3h | Canvas 双栏对齐 + 虚线提案框 |
| 5. /api/predict 优化 | 0.5h | 对齐 case JSON 格式 |
| 6. pyproject.toml | 5min | 一行 |
| 7. 端到端测试 | 0.5h | — |

**总计 ~6.5h**

---

## 5. 3 分钟演示脚本

```
0:00  "VLM 检测 GUI 元素有系统性的漏检——尤其小图标和列表项。"
0:15  "我们用约束图 + GNN 把漏检的元素找回来。"
0:30  (打开案例 1: 天气 App)
      "左边是 Qwen3-VL 的检测：27 个框，漏了 50 个元素（红 X）。"
0:50  "右边加了 GNN：蓝色框是模型根据布局结构提议的缺失元素——"
      "看，天气预报区、每日预报、温度数据全被补上了。"
1:10  "TP 从 22 到 33，F1 从 0.44 到 0.60，而且没有引入新误检。"
1:30  (切到案例 2: 设置页)
      "设置页更夸张：VLM 只找到 4 个真元素，GNN 补到 11 个。"
1:50  "老实说：不是每张图都这么理想。全量 200 张平均 F1 提升约 1pp，"
      "每个正确提议会带 3-4 个错误提议——结构化布局效果最好。"
2:30  "感兴趣可以上传自己的截图试试。"
2:50  "总结：GNN 作为 VLM 的后处理层，能基于空间约束补全漏检，"
      "不需要重训 VLM，模型只有 20 万参数、推理 0.5ms。"
3:00  "谢谢"
```

---

## 6. 一句话

**Demo = 上传截图 → 看到 GNN 用蓝色框把 VLM 漏检的真元素找回来。**
精选案例上视觉差异巨大（F1 +0.15~0.26），全量统计上只有 +1pp——所以必须
**预选案例展示 + 文案诚实说明**，而不是假装随机图片都有效。
