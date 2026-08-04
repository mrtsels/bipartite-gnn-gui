# Demo 代码 Review — 潜在 Bug 与预防方案

> 按组件逐层分析：pipeline.py → prepare_demo_cases.py → main.py → index.html → 集成。
> 每项标注严重程度：🔴 阻断 / 🟡 数据错误 / 🟢 体验问题。

---

## 1. `api/pipeline.py` — 模型加载与推理

### 1.1 🔴 Shape-filter 静默丢弃权重

**场景：** `_safe_load_state()` 用 shape 过滤，如果写错了 filter 条件（比如用 `v.shape == ms[k].shape` 但 key 名字对应错了），会静默丢弃大量权重 → 模型输出随机值 → 所有 proposal 都无效。

**预防：**
```python
def _safe_load_state(self, state, model_state):
    matched = {k: v for k, v in state.items()
               if k in model_state and v.shape == model_state[k].shape}
    n_matched = len(matched)
    n_total = len(state)
    if n_matched < 30:  # 44 keys total, 30 is a safe floor
        raise RuntimeError(
            f"Checkpoint mismatch: only {n_matched}/{n_total} keys matched. "
            f"Expected hidden_dim={self.hidden_dim}, checkpoint may differ."
        )
    # Extra: verify critical layers
    critical = ['encoder.element_proj.weight', 'violation_head.network.3.weight',
                'proposal_head.network.3.weight']
    for ck in critical:
        if ck not in matched:
            raise RuntimeError(f"Critical layer missing: {ck}")
    model.load_state_dict(matched, strict=False)
    logger.info("Loaded %d/%d keys (critical layers OK)", n_matched, n_total)
```

### 1.2 🔴 hidden_dim 检测失败

**场景：** checkpoint 的 `element_proj.weight` shape 可能是 `[16, 5]` 或 `[128, 5]` 或 `[128, 197]`（visual fusion）。如果 `_detect_hidden_dim()` 读错了 → 模型创建错误的 hd → 加载失败。

**预防：**
```python
def _detect_hidden_dim(self, ckpt_path):
    state = torch.load(str(ckpt_path), map_location="cpu")
    for k in ['encoder.element_proj.weight', 'encoder.e_to_c_convs.0.lin_l.weight']:
        if k in state:
            return state[k].shape[0]
    raise RuntimeError(f"Cannot detect hidden_dim from checkpoint: {ckpt_path}")
```

### 1.3 🟡 Proposal bbox 有效性

**场景：** joint 模型的 proposal head 输出 sigmoid 后的 `[x1,y1,x2,y2]`，但 sigmoid 是独立的 → 经常 `x1 > x2` 或 `y1 > y2`。当前代码在 `gnn_analyse` 第 468-471 行 clamp 但不检查有效性 → 前端画框时 `width = x2-x1 < 0` → Canvas 不报错但框位置错乱。

**预防：** 在 proposals 构建时过滤：
```python
bbox_xyxy = [max(0.0, min(1.0, v)) for v in bbox_xyxy]
if bbox_xyxy[2] <= bbox_xyxy[0] or bbox_xyxy[3] <= bbox_xyxy[1]:
    continue  # skip invalid proposal
```

### 1.4 🟡 元素过滤导致索引偏移

**场景：** `_vlm_json_to_element_nodes` 过滤掉无效 bbox → 返回的元素数组比 VLM 原始输出少。但 `existence_scores` 是按过滤后的数组索引的（第 484 行循环 `existence.shape[0]`）——如果前端期望 `existence_scores[i]` 对应 VLM JSON 第 i 个元素，会错位。

**预防：** 返回时保留原始索引映射，或在前端不做 existence 到原始元素的关联。当前 design 里前端不需要 existence scores → 不在返回中暴露即可。

### 1.5 🟢 Constraint 数为 0 的边界情况

**场景：** 元素 < 3 个 → `extract_all_constraints` 返回空列表 → `gnn_analyse` 第 421-429 行 early return → proposals 为空。这是正确的行为，但前端可能不知道这是"正常"还是"出错"。

**预防：** 在返回中加一个字段 `"fallback": true`：
```python
if not element_nodes:
    return {..., "fallback": "no_elements"}
```
前端收到 `fallback` 时显示提示："元素太少，无法构建约束图"。

### 1.6 🟢 Image dimension 为 0

**场景：** PIL 读取损坏图片 → `img.size` 返回 (0, 0) → 归一化时除以 0 → NaN bbox → 后续全部异常。

**预防：**
```python
pil_img = Image.open(BytesIO(img_bytes))
img_w, img_h = pil_img.size
if img_w <= 0 or img_h <= 0:
    return JSONResponse({"error": "Invalid image dimensions"}, status_code=400)
```

---

## 2. `scripts/prepare_demo_cases.py` — 数据准备

### 2.1 🟡 VLM JSON 格式不一致

**场景：** 部分 VLM JSON 用 `bbox_xyxy`，部分用 `bbox`。`load_vlm_elements` 已处理（`item.get("bbox_xyxy") or item.get("bbox")`），但如果有第三名字段（比如旧版本叫 `box`）→ 静默跳过该元素 → 元素数 ≠ 实际 VLM 输出。

**预防：** 预处理时统计跳过率：
```python
skipped = 0
total_raw = len(raw_elements)
for item in raw_elements:
    bbox = item.get("bbox_xyxy") or item.get("bbox")
    if not bbox:
        skipped += 1
        continue
if skipped > total_raw * 0.5:
    logger.warning("Image %s: %d/%d elements skipped — check VLM JSON format", img_id, skipped, total_raw)
```

### 2.2 🟡 GT JSON 缺失

**场景：** 某些 RICO 截图有 VLM 预测但没有 GT JSON → 无法计算匹配 → 这些图不能进案例筛选。目前用 `load_gt_elements` 返回 None 跳过 → 但不会记录有多少图被跳。

**预防：** 脚本结束时打印统计：
```
200 VLM files, 200 screenshots, 198 GT found → 198 evaluated
```

### 2.3 🟡 两次运行 metric 不一致

**场景：** Hungarian 匹配用 `scipy.optimize.linear_sum_assignment`，但如果有多个同等 cost 的最优解，不同运行可能选不同的 → 单案例的 TP/FP 数字可能轻微浮动。12 张案例的筛选结果可能不稳定。

**预防：** 如果浮动了 1-2 个 TP，对筛选结果影响不大（筛选用 ΔTP≥4，浮动范围 ±1）。用 seed 固定依赖（但 Hungarian 不依赖 seed）。

### 2.4 🟢 截图文件损坏

**场景：** `data/rico_local/combined/{id}.jpg` 可能是 0 字节或损坏的。

**预防：** 用 PIL 尝试打开 + verify：
```python
from PIL import Image
try:
    img = Image.open(screenshot_path)
    img.verify()  # verify integrity
    img = Image.open(screenshot_path)  # re-open after verify
except Exception:
    logger.warning("Skipping %s: corrupted screenshot", img_id)
    continue
```

### 2.5 🟢 案例命名

**场景：** 自动命名（"Weather App" 等）靠人工判断。如果漏命名或命名错误 → 列表里显示 "10027" → 用户困惑。

**预防：** 脚本里加 `NAMES = {"10027": "Weather App", ...}` 映射表。默认 fallback 到 `"Case {id}"`，不裸显数字 ID。

---

## 3. `api/main.py` — API 层

### 3.1 🟡 cases.json 不存在

**场景：** 用户没跑 `prepare_demo_cases.py` 就启动服务器 → `/api/cases` 抛 FileNotFoundError → 前端 crash。

**预防：**
```python
CASES_PATH = DEMO_DATA / "cases.json"
if not CASES_PATH.exists():
    logger.warning("cases.json not found — hero cases unavailable. Run scripts/prepare_demo_cases.py first.")
    default_cases = []  # fallback to empty

@app.get("/api/cases")
async def list_cases():
    if not CASES_PATH.exists():
        return JSONResponse([], status_code=200)  # empty, not error
    ...
```

### 3.2 🟢 Case ID 不存在

**场景：** URL `/api/case/99999` 但 `99999` 不在 cases.json 里 → 404。

**预防：** 前端对 fetch 404 做 error handling（见前端部分）。后端返回明确的错误消息：
```python
case = next((c for c in cases if c["id"] == case_id), None)
if case is None:
    return JSONResponse({"error": f"Case {case_id} not found"}, status_code=404)
```

### 3.3 🟢 API key 未配置时上传

**场景：** 用户点上传，但没设 `DASHSCOPE_API_KEY` → VLM 调用失败 → 前端显示错误。

**预防：** 服务端提前检查并返回友好消息：
```python
if not os.environ.get("DASHSCOPE_API_KEY"):
    return JSONResponse({
        "error": "VLM API key not configured. "
                 "Set DASHSCOPE_API_KEY in .env to enable upload mode. "
                 "Hero cases mode is still available."
    }, status_code=400)
```
前端收到 400 后展示提示文案，不崩溃。

### 3.4 🟢 大图片上传 → 超时

**场景：** 用户上传 4K 截图 → base64 编码后变大 → VLM API 处理慢 → 超过 uvicorn 默认 60s timeout → 前端 receive 超时错误。

**预防：**
- 前端限制文件大小（max 10MB）
- VLM API 调用已经有 3 次 retry（60s timeout）
- 前端 loading 状态应持续显示（不因超时自动关闭）

---

## 4. `web/index.html` — 前端

### 4.1 🔴 Canvas bbox 缩放错误

**场景：** 原始截图 1080×1920，Canvas 显示区域 400×700。前端需要把归一化 bbox `[x1,y1,x2,y2]` 映射到 Canvas 像素坐标。如果用错宽高 → 所有框偏移/错位。

**预防：**
```javascript
function normToCanvas(bbox, canvasW, canvasH) {
    // bbox is [x1, y1, x2, y2] in [0,1]
    return {
        x: bbox[0] * canvasW,
        y: bbox[1] * canvasH,
        w: (bbox[2] - bbox[0]) * canvasW,
        h: (bbox[3] - bbox[1]) * canvasH
    };
}
```
关键：必须保证 `canvasW / canvasH = imgW / imgH`（保持宽高比）。用以下缩放策略：
```javascript
const scale = Math.min(canvasMaxW / imgW, canvasMaxH / imgH);
const drawW = imgW * scale;
const drawH = imgH * scale;
// Center in canvas
const offsetX = (canvasW - drawW) / 2;
const offsetY = (canvasH - drawH) / 2;
// Then bbox pixels = norm * drawW/H + offset
```

### 4.2 🔴 Retina Canvas 模糊

**场景：** MBP Retina 屏幕上，Canvas 默认分辨率 = CSS 像素，但实际渲染像素 = CSS × devicePixelRatio。不处理 → bbox 线条模糊。

**预防：**
```javascript
const dpr = window.devicePixelRatio || 1;
canvas.width = displayWidth * dpr;
canvas.height = displayHeight * dpr;
canvas.style.width = displayWidth + 'px';
canvas.style.height = displayHeight + 'px';
ctx.scale(dpr, dpr);
// All drawing coordinates now in CSS-pixel scale
```

### 4.3 🔴 窗口 resize 后 Canvas 不重绘

**场景：** 用户调整浏览器窗口 → Canvas CSS 尺寸变化 → 但 Canvas 内部像素尺寸不变 → 内容变形或空白。

**预防：**
```javascript
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        resizeCanvases();
        renderCurrentCase();
    }, 200);  // debounce 200ms
});
```

### 4.4 🟡 快速切换案例 → 竞态条件

**场景：** 用户快速点击 Next → 两个 fetch 同时发出 → 后到的结果覆盖先到的 → 显示错误的案例数据。

**预防：**
```javascript
let currentRequestId = 0;

async function renderCase(idx) {
    const reqId = ++currentRequestId;
    const data = await fetch(`/api/case/${cases[idx].id}`).then(r => r.json());
    if (reqId !== currentRequestId) return;  // stale request, discard
    // ... render
}
```

### 4.5 🟡 Fetch 失败 → 静默崩溃

**场景：** `/api/case/{id}` 返回 404 或网络错误 → `.then(r => r.json())` 抛异常 → 未被 catch → 页面空白。

**预防：**
```javascript
try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    // render
} catch (err) {
    showError(`Failed to load case: ${err.message}`);
}
```

### 4.6 🟡 拖拽上传不生效

**场景：** `dragenter`/`dragover` 事件需要 `preventDefault()` 才能变成 drop target。忘了 → 拖拽时浏览器打开图片。

**预防：**
```javascript
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
    });
});
```

### 4.7 🟡 Upload 模式：重复渲染 base64

**场景：** `/api/predict` 返回 `image_b64`（base64 JPEG），然后前端用 `new Image()` 加载 → `drawImage`。但上传页面已经有 `URL.createObjectURL(file)` 可以直接显示，不需要等 API 返回再渲染 base64。

**预防：** 分两步：
1. 文件选中时立即显示预览：`preview.src = URL.createObjectURL(file)`
2. API 返回后：在预览图上叠加 bbox
3. 不上传 base64 图片在 response 中（减少 2MB payload）

修改 API 返回：去掉 `image_b64` 字段。

### 4.8 🟢 指标数字格式化

**场景：** `precision: 0.8152631578947368` → 直接 `toString()` → 占满指标卡宽度。

**预防：**
```javascript
function fmt(n) { return n.toFixed(3); }
// 显示: "0.815 → 0.868"
```

### 4.9 🟢 键盘导航导致页面滚动

**场景：** 左右箭头键切换案例时，页面也滚动（arrow keys 的默认行为）。

**预防：**
```javascript
document.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); prevCase(); }
    if (e.key === 'ArrowRight') { e.preventDefault(); nextCase(); }
});
```

### 4.10 🟢 两个 Canvas 不同步

**场景：** 左 Canvas 和右 Canvas 同时渲染。如果一个比另一个慢（比如右 Canvas 多画蓝色虚线），会出现左右视觉不同步。

**预防：** 用一个 `requestAnimationFrame` 绘制两个 Canvas：
```javascript
requestAnimationFrame(() => {
    drawVLMCanvas(ctxLeft, data);
    drawGNNCanvas(ctxRight, data);
});
```

---

## 5. 集成问题

### 5.1 🟡 CORS 在生产环境太宽松

**场景：** `allow_origins=["*"]` 在生产环境不安全。

**预防：** 这个 demo 只在 localhost 运行，`*` 没问题。文档注明 "localhost only"。如果要部署到服务器，改为 `["http://localhost:8765"]`。

### 5.2 🟢 端口 8765 可能被占用

**场景：** 其他进程占用 8765 → uvicorn 启动失败 → "Address already in use"。

**预防：** `main.py` 底部加端口配置：
```python
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8765"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
```

### 5.3 🟢 首次加载 checkpoint 慢

**场景：** 请求 `/api/health` 时才 lazy-load checkpoint → 第一个请求等待 1-2s → 用户以为服务器挂了。

**预防：** 在 `__main__` 启动时预加载：
```python
if __name__ == "__main__":
    logger.info("Pre-loading GNN checkpoint...")
    get_pipeline()  # warm up
    uvicorn.run(app, ...)
```

---

## 6. Review Checklist（实施时逐条检查）

### Pipeline
- [ ] `_safe_load_state` 验证 matched ≥ 30 keys + critical layers 存在
- [ ] `_detect_hidden_dim` 覆盖 hd=16/64/128 三种
- [ ] proposals 列表过滤 x2≤x1 的无效 bbox
- [ ] 0 元素 / 0 约束 → 返回 fallback 字段
- [ ] 图片尺寸为 0 → 返回 400
- [ ] checkpoint 路径用 `Path(__file__)` 解析，不依赖 cwd

### 数据准备
- [ ] 统计 VLM JSON 元素跳过率，>50% 警告
- [ ] 统计 GT 缺失数
- [ ] 截图文件先用 PIL verify 检查完整性
- [ ] 案例命名 fallback "Case {id}"

### API
- [ ] cases.json 不存在 → 返回空列表，不报 500
- [ ] 不存在的 case_id → 返回 404 + 错误消息
- [ ] 无 API key → 返回 400 + 友好提示
- [ ] 健康检查端点不受 checkpoint 加载状态影响
- [ ] 启动时预加载 pipeline，不等到第一个请求

### 前端
- [ ] Canvas 保持宽高比缩放，bbox 映射公式验证
- [ ] Retina 屏 devicePixelRatio 处理
- [ ] window resize → debounce 重绘
- [ ] 案例切换竞态条件 → requestId 机制
- [ ] fetch 失败 → try/catch + 错误提示
- [ ] 拖拽上传 → preventDefault 所有 drag 事件
- [ ] 上传模式去掉 response 中的 image_b64 → 用本地预览
- [ ] 指标数字 → toFixed(3)
- [ ] 键盘导航 → preventDefault + 边界处理
- [ ] 左右 Canvas 同帧绘制
- [ ] 0 元素 fallback 显示提示文案

### 集成
- [ ] 端口可配（PORT 环境变量）
- [ ] 启动时预加载 checkpoint
- [ ] README 更新启动命令
