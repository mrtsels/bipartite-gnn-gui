# Phase 11.1–11.2: Web Demo 可执行开发报告

> 2026-07-20 · 可行性验证完毕 · 所有关键路径已通过

---

## 0. 可行性验证总结

| 验证项 | 状态 | 结论 |
|--------|:----:|------|
| PyTorch/PyG 可用 | ✅ | torch 2.10, pyg 2.7, MPS 可用 |
| 模型加载 (screenspot_finetuned) | ✅ | 220K params, element_dim=5, hidden_dim=128 |
| VLM → Graph → GNN 端到端 | ✅ | 3 elements → 7 constraints → 3 coord/3 existence/7 violation |
| GNN proposal 可用 | ✅ | 每个 constraint 都有 proposal bbox (xywh) + type logits |
| tqdm/dashscope/timm? | ⚠️ | timm 未安装（不需要），dashscope/tqdm 未安装（API 调用需 requests） |
| FastAPI 可用性 | ⚠️ | 未安装，需 `pip install fastapi uvicorn python-multipart` |

**模型选择决策：** 使用 `checkpoints/violation_detection/screenspot_finetuned.pt`（hidden_dim=128, element_dim=5）。不需要视觉特征 (vit_tiny/timm)，纯结构特征即可运行，部署最简。

**已知局限（来自 Phase 9）：**
- existence head 在非 RICO 数据上得分 ~0.45（低于 0.5 阈值）→ demo 中不硬过滤，改为显示置信度
- violation detection + proposal 在真实 VLM 数据上的准确率约 66.5%
- GNN 对 VLM 召回率提升约 +2.9pp (F1 0.291 → 0.320)
- **Demo 的核心展示价值**: 结构推理管线 (VLM→Graph→GNN) 而非绝对精度

---

## 1. 架构调整

原开发文档的三层 Docker 架构过于冗重。Demo 阶段采用极简架构：

```
浏览器 (index.html)  ←→  FastAPI (:8765)
                            │
                    ┌───────┼───────┐
                    │       │       │
                  VLM API   GNN    Canvas overlay
               (DashScope) (本地)  (PIL draw)
```

**简化的理由：**
- Nginx 反向代理不是 demo 刚需 — FastAPI 直接 serve 静态文件即可
- MySQL 推理历史不是 demo 刚需 — YAGNI，等 Phase 11.3 再加
- Docker 部署不是 demo 刚需 — 本地 `python api/main.py` 即可演示，Dockerfile 放 Phase 11.4
- Wheel 预编译不是 demo 刚需 — 源码直接 import

---

## 2. Phase 11.1 — FastAPI 后端

### 目录结构

```
bipartite-gnn-gui/
├── api/
│   ├── main.py          # FastAPI app + 2 endpoints
│   ├── pipeline.py      # DemoPipeline 封装: VLM call + GNN inference + overlay
│   └── requirements.txt # fastapi, uvicorn, python-multipart, pillow, requests
├── checkpoints/          # 已有
└── web/
    └── index.html        # Phase 11.2
```

### 2.1 `api/requirements.txt`

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pillow>=10.0.0
requests>=2.31.0
```

### 2.2 `api/pipeline.py` — DemoPipeline

```python
class DemoPipeline:
    """封装: VLM API 调用 + GNN 推理 + bbox overlay 渲染"""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        # 加载 BipartiteGNNCorrector(hidden_dim=128)
        # 创建 BipartiteGraphBuilder
        # 创建 PIL ImageDraw 工具

    def detect_elements(self, img_bytes: bytes, api_key: str, model: str) -> dict:
        """调用 DashScope Qwen3-VL API 检测 GUI 元素"""
        # 1. base64 编码图片
        # 2. POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
        # 3. 解析返回 JSON → 标准化 VLMOutput
        # 4. 返回 {"elements": [...], "vlm_time_ms": int}
        # 容错：API 错误 → 返回 {"error": str, "elements": []}

    def gnn_propose(self, vlm_elements: list) -> dict:
        """基于 VLM 检测结果运行 GNN 约束分析"""
        # 1. VLM elements → ElementNode list (normalize bbox)
        # 2. extract_all_constraints → ConstraintNode list
        # 3. build HeteroData graph
        # 4. model forward → violation + proposal + existence
        # 5. 筛选 violated constraints (score > 0.3)
        # 6. 从 proposal head 提取 bbox (xywh→xyxy)
        # 返回 {"constraints": [...], "proposals": [...], "existence_scores": [...], "gnn_time_ms": int}

    def render_overlay(self, img_bytes: bytes, vlm_elements: list, gnn_proposals: list) -> bytes:
        """在原始截图上绘制 bbox overlay"""
        # PIL Image.open(img_bytes)
        # draw VLM bboxes: 红色半透明矩形 + label
        # draw GNN proposals: 蓝色虚线矩形 + "proposed" label
        # 返回 PNG bytes
```

**关键参数：**
- 建议的 GNN violation 阈值：0.3（不要用 0.5 — GNN 在不熟悉的数据上分数偏低）
- VLM API: 复用 `scripts/generate_vlm_predictions.py` 的 `call_qwen_vl` 模式
- 默认 VLM 模型: `qwen3-vl-flash`（更快，~2s）或 `qwen3-vl-plus`（更准，~5s）

### 2.3 `api/main.py` — FastAPI 路由

```
POST /api/predict
  - 上传文件: screenshot (multipart)
  - Form 参数: api_key, vlm_model (默认 qwen3-vl-flash)
  - 流程: VLM detect → GNN propose → render overlay
  - 返回 JSON:
    {
      "vlm": {
        "elements": [{"bbox": [x1,y1,x2,y2], "label": "button", "confidence": 0.9}, ...],
        "count": 12,
        "time_ms": 2340
      },
      "gnn": {
        "proposals": [{"bbox": [x1,y1,x2,y2], "violation": 0.55, "constraint_type": "grid"}, ...],
        "constraints_total": 45,
        "violations_found": 3,
        "proposals": 3,
        "time_ms": 15
      },
      "overlay_b64": "data:image/png;base64,..."
    }

POST /api/gnn-only
  - 上传文件: screenshot + vlm_json (已有 VLM 预测的 JSON 字符串)
  - 只跑 GNN，不调 VLM API
  - 返回同上但无 vlm.time_ms

GET /api/health
  → {"status": "ok", "model": "screenspot_finetuned", "device": "cpu", "params": 220439}
```

### 2.4 验证步骤

```bash
# 1. 安装依赖
pip install fastapi uvicorn python-multipart pillow requests

# 2. 启动服务
cd api && python main.py
# 输出: Uvicorn running on http://0.0.0.0:8765

# 3. 测试 health
curl http://localhost:8765/api/health | python -m json.tool

# 4. 测试 gnn-only（无 VLM API key）
curl -X POST http://localhost:8765/api/gnn-only \
  -F "file=@../tests/fixtures/sample_rico_screenshot.jpg" \
  -F 'vlm_json={"elements":[{"bbox":[100,100,200,200],"label":"button"},{"bbox":[300,100,400,200],"label":"text"}]}' \
  | python -m json.tool

# 5. 测试 predict（需 DASHSCOPE_API_KEY 环境变量）
curl -X POST http://localhost:8765/api/predict \
  -F "file=@screenshot.png" \
  | python -m json.tool
```

---

## 3. Phase 11.2 — 前端 (index.html)

### 3.1 页面结构

```
┌─────────────────────────────────────────────────┐
│  [Upload Area]  拖拽图片到这里 或 点击选择      │
│  [配置区]  VLM: [Flash ▼]  API Key: [••••]     │
│  [Run] 按钮                                      │
├────────────────────┬────────────────────────────┤
│      原图 + VLM     │     原图 + VLM + GNN       │
│   (红色 bbox)      │  (红色 VLM + 蓝色 GNN)     │
│                    │                            │
│   ┌─ 截图 ────┐   │   ┌─ 截图 ────────────┐   │
│   │ □ button   │   │   │ □ button  ┊ ┊ ┊ ┊ │   │
│   │ □ text     │   │   │ □ text    ┊proposed│   │
│   └────────────┘   │   └───────────────────┘   │
├────────────────────┴────────────────────────────┤
│  VLM: 12 elements (2.3s)                        │
│  GNN: 45 constraints, 3 violations → 3 proposals│
│  Total: 2.4s                                    │
└─────────────────────────────────────────────────┘
```

### 3.2 技术选型

- **纯 HTML + CSS + JS** — 不引入任何框架
- **Canvas overlay**: 用 `<canvas>` 在原始图片上叠加 bbox
- **拖拽上传**: 原生 `dragenter/dragover/drop` 事件
- **API 调用**: 原生 `fetch()` + `FormData`
- **响应式**: Flexbox 两栏布局，小屏自动变单栏

### 3.3 Canvas 绘制细节

```javascript
// VLM bbox: 红色实线矩形 + 白色标签
ctx.strokeStyle = 'rgba(255, 50, 50, 0.8)';
ctx.lineWidth = 2;
ctx.strokeRect(x1, y1, w, h);
ctx.fillStyle = 'rgba(255, 50, 50, 0.25)';
ctx.fillRect(x1, y1, w, h);  // 半透明填充
// Label
ctx.fillStyle = 'white';
ctx.font = '11px monospace';
ctx.fillText(label + ' ' + confidence, x1 + 2, y1 - 4);

// GNN proposal: 蓝色虚线 + "⚡ proposed" 标签
ctx.setLineDash([6, 4]);
ctx.strokeStyle = 'rgba(50, 130, 255, 0.9)';
ctx.strokeRect(x1, y1, w, h);
ctx.setLineDash([]);
ctx.fillStyle = 'rgba(50, 130, 255, 0.15)';
ctx.fillRect(x1, y1, w, h);
```

### 3.4 状态管理

```javascript
const state = {
    image: null,           // File 对象
    imageUrl: null,        // blob URL for <img> display
    vlmModel: 'qwen3-vl-flash',
    apiKey: '',            // 从 localStorage 读取
    result: null,          // API 返回的完整结果
    loading: false,
    error: null,
};

// localStorage 持久化 apiKey
// 预测结果不持久化（每次上传重新跑）
```

### 3.5 验证步骤

```bash
# 1. 直接打开 index.html（file:// 协议）
open web/index.html

# 2. 用 Python HTTP server 测试（解决 CORS）
cd web && python -m http.server 8888

# 3. 集成测试：FastAPI serve 静态文件
# main.py 中加:
# app.mount("/", StaticFiles(directory="../web", html=True))
# 访问 http://localhost:8765
```

---

## 4. 关键风险与缓解

| 风险 | 概率 | 缓解 |
|------|:----:|------|
| VLM API 调用慢（2-5s），用户等待焦虑 | 高 | 前端显示进度条 + 预估时间；提供 `gnn-only` 模式直接粘贴 VLM JSON |
| GNN proposal 在非 RICO 截图上不准 | 高 | UI 上标注"实验性"，蓝色虚线+低透明度，不声称是"修正" |
| existence 分数全部 < 0.5 | 高 | 不硬过滤，改为在标签上显示置信度条 |
| 大图片 base64 编码慢 | 中 | 前端 resize 到 max 1920px 宽后再上传 |
| MPS/CUDA 不可用 | 低 | 模型 220K 参数，CPU 推理 ~1-5ms，完全可接受 |
| timm 未安装 | 无影响 | 选用 element_dim=5 的 checkpoint，不需要视觉特征 |

---

## 5. 执行计划（11 个步骤，预计 60-90 分钟）

### Step 1: 创建 api/ 目录和 requirements.txt
- `mkdir -p api web`
- 写 `api/requirements.txt`（fastapi + uvicorn + pillow + requests）
- 验证: `pip install -r api/requirements.txt`

### Step 2: 实现 `api/pipeline.py` — 模型加载
- 加载 checkpoint 并 BipartiteGNNCorrector 实例化
- `build_graph(elements)` 函数
- 验证: `python -c "from api.pipeline import DemoPipeline; p = DemoPipeline(); print(p.model)"`

### Step 3: 实现 `api/pipeline.py` — VLM API 调用
- 复用 `scripts/generate_vlm_predictions.py` 的 `call_qwen_vl`
- 容错处理（超时、API 错误、JSON 解析失败）
- 验证: 用真实 API key 测试（或 skip — 非阻塞）

### Step 4: 实现 `api/pipeline.py` — GNN 推理
- constraint extraction → graph build → model forward
- violation 过滤 + proposal 提取
- 验证: 用合成 VLM JSON 测试端到端

### Step 5: 实现 `api/pipeline.py` — Overlay 渲染
- PIL 画图: 原图 + 红色 bbox (VLM) + 蓝色虚线 bbox (GNN proposals)
- base64 编码输出
- 验证: 保存 overlay 到文件并用 Preview 打开查看

### Step 6: 实现 `api/main.py` — FastAPI routes
- `/api/health`, `/api/predict`, `/api/gnn-only`
- 错误处理 + CORS 中间件
- 验证: `curl` 三个 endpoint

### Step 7: 实现 `web/index.html` — 上传区域 + 配置
- 拖拽上传区 + 文件选择按钮
- VLM 模型选择 + API Key 输入
- 上传后显示缩略图
- 验证: 浏览器打开，拖入图片确认显示

### Step 8: 实现 `web/index.html` — Canvas overlay 渲染
- 两个 `<canvas>`：Before (只有 VLM) 和 After (VLM + GNN)
- bbox 绘制函数（红/蓝/标签）
- 验证: 注入 mock 数据，确认 Canvas 渲染正确

### Step 9: 实现 `web/index.html` — API 调用 + 结果展示
- `fetch()` POST `/api/predict` 或 `/api/gnn-only`
- 加载状态 + 错误状态
- 统计栏更新
- 验证: 启动 FastAPI，上传图片跑全流程

### Step 10: 实现 `web/index.html` — 响应式 + 细节打磨
- 小屏幕单栏布局
- Loading spinner
- API key localStorage 持久化
- 验证: Chrome DevTools 移动端模拟

### Step 11: 集成后运行全流程
- FastAPI serve 静态文件 (`app.mount`)
- 上传截图 → VLM → GNN → overlay 全流程
- 验证: 浏览器完整走一遍，截图确认

---

## 6. 不需要做的事（YAGNI）

- ❌ Docker / docker-compose — Phase 11.4 再说
- ❌ MySQL inference_history 表 — Phase 11.3 再说
- ❌ Nginx 反向代理 — FastAPI 直接 serve 静态文件
- ❌ Wheel 预编译 — 源码 import 即可
- ❌ 视觉特征 (timm/vit_tiny) — 结构模型 (element_dim=5) 足够
- ❌ Confidence scoring 模型 — 单模型 pipeline 更简单
- ❌ 多 VLM 模型支持 — 只支持 Qwen3-VL (DashScope API)
- ❌ WebSocket 实时流 — 同步请求足够
