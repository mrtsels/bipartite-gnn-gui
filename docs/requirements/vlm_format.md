# VLM 输出格式分析 (VLM Output Format Analysis)

> **Phase 1.1 — Requirements Analysis**
>
> 本文档定义了 VLM 输出的标准数据结构、坐标约定、元素分类体系、各模型期望 JSON schema，
> 以及解析与错误处理策略。本文档**不**包含实现代码；它提供的是 Phase 4 在
> `src/bipartite_gnn_gui/data/vlm_output.py` 中需要落实的协议契约。
>
> This document defines the canonical data structures, coordinate conventions, element
> taxonomy, per-model expected JSON schemas, and parsing / error-handling strategy for
> VLM outputs.  It does **not** contain implementation code; it provides the contract
> that Phase 4 will implement in `src/bipartite_gnn_gui/data/vlm_output.py`.

---

## 1. 背景 (Background)

本项目使用轻量级视觉语言模型（VLM）将 GUI 截屏解析为结构化 JSON 描述。支持的模型包括
**Qwen3.5-2B** 和 **MiniMax-VL-01**。两个模型以不同的 JSON 格式输出检测结果，
本文档统一这些差异并定义系统内部使用的标准数据结构。

This project uses lightweight VLMs (Qwen3.5-2B, MiniMax-VL-01) to parse GUI screenshots
into structured JSON.  The models emit detection results in slightly different JSON
shapes; this document unifies those differences and defines the canonical in-memory data
structures used by downstream graph-construction and training pipelines.

### 设计目标 (Design Goals)

| # | Goal                                       | Rationale                                                       |
| -- | ------------------------------------------ | ---------------------------------------------------------------- |
| 1 | 模型无关的内部表示 (model-agnostic IR)     | 添加新 VLM 只需新增解析器，下游代码无需更改。                  |
| 2 | 归一化坐标 (normalised coordinates)        | 消除分辨率依赖，便于跨数据集泛化。                              |
| 3 | 统一的元素分类 (shared element taxonomy)   | GT 标注和 VLM 预测使用同一套类型标签，使得匹配/评估可计算。    |
| 4 | 宽容解析 (lenient parsing)                 | 非致命错误不阻塞流水线；缺失字段使用合理默认值。                |
| 5 | 可追溯性 (traceability)                    | 每次解析记录 model name、timestamp、原始图片尺寸及 parse errors。 |

---

## 2. VLMOutputElement 数据结构

单个 bounding-box 预测，由 VLM 输出的一个 element 条目转换而来。

A single bounding-box prediction converted from one element entry in the VLM output.

```python
@dataclass
class VLMOutputElement:
    element_id: int                               # 全局唯一元素编号 (0-based)
    bbox: tuple[float, float, float, float]       # 归一化 xyxy: (x1, y1, x2, y2) ∈ [0,1]
    element_type: str                             # 元素类型，取自 §4 分类体系的 canonical key
    text_content: Optional[str] = None            # 可见文本（OCR 结果），无文本时为 None
    confidence: float = 1.0                       # 模型置信度 ∈ [0, 1]，缺失时默认 1.0
    attributes: dict = field(default_factory=dict) # 额外元数据（role, disabled, alt_text, etc.）
    source: str = ""                              # 来源模型标识，如 "qwen3.5-2b"
```

### 字段说明 (Field Descriptions)

| Field           | Type      | Required | Description                                                             |
| --------------- | --------- | -------- | ----------------------------------------------------------------------- |
| `element_id`    | `int`     | yes      | Globally unique zero-based element index within the parent `VLMOutput`. |
| `bbox`          | `tuple[float×4]` | yes | `(x1, y1, x2, y2)` in normalised `[0, 1]` screen coordinates.  §3.     |
| `element_type`  | `str`     | yes      | Element type from the shared taxonomy (§4).  Case-normalised.           |
| `text_content`  | `Optional[str]` | no  | Visible text detected by OCR.  `None` when no text is present.          |
| `confidence`    | `float`   | yes      | Detection confidence in `[0, 1]`.  Default `1.0` if missing.            |
| `attributes`    | `dict`    | yes      | Free-form key-value metadata (e.g. `role`, `disabled`, `alt_text`).     |
| `source`        | `str`     | yes      | VLM model identifier (`"qwen3.5-2b"` or `"minimax-vl-01"`).            |

---

## 3. VLMOutput 数据结构

单张截屏对应的全部预测结果集合。

A collection of all predictions for one screenshot.

```python
@dataclass
class VLMOutput:
    elements: list[VLMOutputElement]              # 有序元素列表 (ordered list)
    image_width: int = 0                          # 原始图片宽度（像素），0 表示未知
    image_height: int = 0                         # 原始图片高度（像素），0 表示未知
    model_name: str = ""                          # VLM 模型标识
    timestamp: str = ""                           # ISO-8601 时间戳
    parse_errors: list[str] = field(default_factory=list)  # 非致命解析问题日志
```

### 字段说明 (Field Descriptions)

| Field          | Type                    | Required | Description                                               |
| -------------- | ----------------------- | -------- | --------------------------------------------------------- |
| `elements`     | `list[VLMOutputElement]` | yes      | Ordered list of predicted elements.                       |
| `image_width`  | `int`                   | no       | Original image pixel width.  `0` when unknown.            |
| `image_height` | `int`                   | no       | Original image pixel height.  `0` when unknown.           |
| `model_name`   | `str`                   | no       | Model identifier, e.g. `"qwen3.5-2b"`.                    |
| `timestamp`    | `str`                   | no       | ISO-8601 timestamp of the inference / parse run.          |
| `parse_errors` | `list[str]`             | no       | Non-fatal parse issues encountered (malformed entries etc. |
|                |                         |          | that were skipped or defaulted).                          |

---

## 4. 元素类型分类体系 (Element Type Taxonomy)

18 种元素类型，按交互特性分为四大类。所有 GT 标注和 VLM 预测共享这一分类体系，
大小写不敏感匹配（`"Button"`, `"BUTTON"`, `"button"` → `button`），
未识别类型映射到 `other`。

18 element types organised into four interaction categories.  Both ground-truth
annotations and VLM predictions use the same taxonomy.  Matching is case-insensitive;
unrecognised types are mapped to `other` with a logged warning.

### 4.1 Interactive — 可交互元素

用户可直接操作的控件。

Controls the user can directly interact with.

| # | Canonical Key | 中文      | Description                                   |
| -- | ------------- | --------- | --------------------------------------------- |
| 1 | `button`      | 按钮      | Clickable button (text or icon).              |
| 2 | `input`       | 输入框    | Text input field, search box, textarea.       |
| 3 | `checkbox`    | 复选框    | Checkbox control with binary state.           |
| 4 | `radio`       | 单选框    | Radio button (mutually exclusive group).      |
| 5 | `slider`      | 滑块      | Range slider with continuous or discrete values. |
| 6 | `switch`      | 开关      | Toggle switch with on/off state.              |

### 4.2 Display — 展示类元素

只读内容，用于信息展示。

Read-only content for information display.

| # | Canonical Key | 中文      | Description                                   |
| -- | ------------- | --------- | --------------------------------------------- |
| 7 | `text`        | 文本      | Text block, paragraph, or inline span.        |
| 8 | `image`       | 图片      | Image, picture, or bitmap graphic.            |
| 9 | `icon`        | 图标      | Icon (glyph or vector, *not* a bitmap image). |
|10 | `label`       | 标签      | Non-editable text label associated with a control. |

### 4.3 Layout — 布局类元素

页面结构和组织容器。

Structural elements that organise the page layout.

| # | Canonical Key | 中文      | Description                                   |
| -- | ------------- | --------- | --------------------------------------------- |
|11 | `container`   | 容器      | Generic grouping container (div, section).    |
|12 | `card`        | 卡片      | Card-style container with border/shadow.      |
|13 | `tab`         | 标签页    | Tab in a tab-bar or tab-navigation.           |
|14 | `menu`        | 菜单      | Dropdown menu, context menu, or nav list.     |
|15 | `divider`     | 分割线    | Horizontal or vertical separator line.        |
|16 | `list`        | 列表      | Ordered or unordered list of items.           |

### 4.4 Overlay — 浮层类元素

叠加在页面之上的临时元素。

Ephemeral elements rendered above the page content.

| # | Canonical Key | 中文      | Description                                   |
| -- | ------------- | --------- | --------------------------------------------- |
|17 | `modal`       | 弹窗      | Modal dialog or overlay with backdrop.        |
|18 | `toast`       | 提示      | Toast / snackbar notification, auto-dismiss.  |
|19 | `banner`      | 横幅      | Top or bottom banner / announcement bar.      |

### 4.5 Fallback

| # | Canonical Key | 中文      | Description                                   |
| -- | ------------- | --------- | --------------------------------------------- |
|20 | `other`       | 其他      | Fallback for unrecognised or ambiguous types. |

> **注意 (Note):** 虽然 `other` 是第 20 个类型，但交互分类中共 18 个有效类型加 1 个容错类型。
> `other` 仅用于容错，不在交互分析中使用。
>
> Although `other` is type #20, there are 18 semantic types plus 1 fallback.
> `other` exists solely for robustness and is excluded from interaction analysis.

---

## 5. 坐标约定 (Coordinate Convention)

### 5.1 规范表示 (Canonical Representation)

所有 bounding box 均以**归一化 xyxy** 格式存储：

All bounding boxes are stored internally as **normalised xyxy**:

```
(x1, y1) — 左上角 top-left corner,  ∈ [0, 1]
(x2, y2) — 右下角 bottom-right corner, ∈ [0, 1]
```

- **原点 origin:** 图片**左上角** `(0, 0)` — top-left of the image.
- **终点 extent:** 图片**右下角** `(1, 1)` — bottom-right of the image.
- `x2 > x1` 且 `y2 > y1`（非退化约束，non-degenerate constraint）。

### 5.2 选择归一化坐标的理由 (Rationale for Normalised Coordinates)

| 理由                              | 说明                                                      |
| --------------------------------- | --------------------------------------------------------- |
| 模型无关 (model-agnostic)         | 不同 VLM 可能处理不同分辨率的图片；归一化后统一表示。     |
| 分辨率独立 (resolution-independent) | 训练和推理可使用任意分辨率，无需更改坐标。               |
| 便于几何计算 (simpler geometry)   | IOU、中心距离等计算直接使用归一化值。                      |
| 跨数据集泛化 (cross-dataset)      | 不同数据集（Rico, WebUI 等）的图片尺寸不同，归一化后可直接拼接。 |

### 5.3 归一化转换规则 (Normalisation Rules)

| 源格式 (Source Format)             | 转换公式 (Conversion)                                               |
| ---------------------------------- | ------------------------------------------------------------------- |
| 绝对像素 (absolute pixels)         | `x_norm = x_px / image_width;  y_norm = y_px / image_height`       |
| xywh (中心无关)                    | `x2 = x + w;  y2 = y + h`，然后归一化                                |
| cxcywh (中心+宽高)                | `x1 = cx - w/2;  y1 = cy - h/2;  x2 = cx + w/2;  y2 = cy + h/2`  |
| 左下角原点 (bottom-left origin)    | `y_norm = 1 - y_orig`（先归一化再翻转）                              |
| 非 [0,1] 范围                      | 仿射变换 (affine transform) 映射到 [0, 1]                            |

### 5.4 裁剪与退化检测 (Clamping & Degeneracy)

- 归一化后所有值 clamp 到 `[0.0, 1.0]`。
- `x2 <= x1` 或 `y2 <= y1` 的 bbox 被标记为**退化 (degenerate)**，丢弃并在 `parse_errors` 中记录警告。
- 略超出 `[0, 1]` 范围的值（如 `1.02`、`-0.01`）先 clamp 再记录。

After normalisation, all values are clamped to `[0.0, 1.0]`.  A bbox with
`x2 <= x1` or `y2 <= y1` is flagged as degenerate, dropped, and logged as a
warning in `parse_errors`.  Values slightly outside `[0, 1]` (e.g. `1.02`,
`-0.01`) are clamped and logged.

---

## 6. Qwen3.5-2B 期望输出格式 (Expected Format)

### 6.1 System Prompt 示例 (Example System Prompt)

调用 Qwen3.5-2B 时使用的 system prompt 示意，引导模型输出结构化 JSON：

Example system prompt used when calling Qwen3.5-2B to guide structured JSON output:

```
You are a GUI element detector. Given a screenshot, detect all visible UI elements.

Output a JSON object with the following structure:
{
  "elements": [
    {
      "bbox_xyxy": [x1, y1, x2, y2],
      "label": "<element_type>",
      "text": "<visible_text_or_null>"
    }
  ]
}

Rules:
- All coordinates are NORMALIZED to [0, 1], where (0,0) is top-left.
- bbox_xyxy format: [x1, y1, x2, y2] (top-left, bottom-right).
- label must be one of: button, text, image, input, icon, container, list,
  label, checkbox, radio, slider, switch, modal, toast, banner, card, tab,
  menu, divider.
- text is null when no visible text exists on the element.
- Include ALL visible elements in a single JSON block.
```

### 6.2 预期 JSON 结构 (Expected JSON Structure)

```json
{
  "image_id": "screenshot_001.png",
  "elements": [
    {
      "bbox_xyxy": [0.12, 0.34, 0.28, 0.40],
      "label": "button",
      "text": "Submit",
      "confidence": 0.95
    },
    {
      "bbox_xyxy": [0.30, 0.10, 0.60, 0.16],
      "label": "text",
      "text": "Welcome to the dashboard"
    },
    {
      "bbox_xyxy": [0.05, 0.50, 0.40, 0.58],
      "label": "input",
      "text": "Enter your name"
    }
  ]
}
```

### 6.3 字段映射 (Field Mapping)

| Qwen 字段        | VLMOutputElement 字段 | 说明                                                |
| ---------------- | --------------------- | --------------------------------------------------- |
| `bbox_xyxy[i]`   | `bbox`                | 直接使用，Qwen 通常已输出归一化 xyxy。               |
| `label`          | `element_type`        | 大小写不敏感匹配 taxonomy。                           |
| `text`           | `text_content`        | `null` → `None`；空字符串 `""` → `None`。            |
| `confidence`     | `confidence`          | 缺失时默认 `1.0`。                                   |
| *(无)*            | `attributes`          | Qwen 不提供 attributes，默认为 `{}`。                |

### 6.4 转换注意事项 (Conversion Notes)

- Qwen 通常直接输出归一化 xyxy 坐标；若 JSON 中包含 `image_width`/`image_height` 字段，
  需验证一致性。
- 部分旧版 Qwen 返回的 bbox 字段名可能为 `bbox` 而非 `bbox_xyxy`，解析器应同时支持。
- `text` 字段可能为空字符串 `""` 而非 `null`，解析器应统一处理：空字符串 → `None`。
- 旧版 Qwen 可能返回像素级坐标；若 bbox 各值 > 1.0 且存在 `image_width`/`image_height`，
  则认为需要归一化。

- Qwen typically outputs normalised xyxy directly; if `image_width`/`image_height` fields
  are present in the JSON, verify consistency.
- Some older Qwen versions use `bbox` instead of `bbox_xyxy`; the parser should support both.
- An empty string `""` in the `text` field should be treated as `None`.
- If bbox values are all > 1.0 and `image_width`/`image_height` are present, pixel-level
  coordinates are assumed and normalisation is applied.

---

## 7. MiniMax-VL-01 期望输出格式 (Expected Format)

### 7.1 API 差异 (API Differences)

与 Qwen3.5-2B 相比，MiniMax-VL-01 的关键差异：

- **绝对像素坐标**：默认输出像素级 bbox，需要配合 `image_width`/`image_height` 归一化。
- **更丰富的元数据**：支持 `attributes` 自由字段，可携带 `role`、`disabled`、`alt_text` 等。
- **置信度始终存在**：`confidence` 字段总是 0-1 之间的浮点数。
- **字段命名不同**：类型字段为 `category` 而非 `label`，文本字段为 `text_content`。

Key differences compared to Qwen3.5-2B:

- **Absolute pixel coordinates**: bboxes are in pixels by default; normalisation requires
  `image_width`/`image_height`.
- **Richer metadata**: supports free-form `attributes` (role, disabled, alt_text, etc.).
- **Confidence always present**: the `confidence` field is always a float in `[0, 1]`.
- **Different field names**: element type uses `category` instead of `label`; OCR text uses
  `text_content`.

### 7.2 预期 JSON 结构 (Expected JSON Structure)

```json
{
  "image_id": "screen_42",
  "image_width": 1920,
  "image_height": 1080,
  "elements": [
    {
      "bbox": [100, 200, 400, 350],
      "category": "button",
      "confidence": 0.92,
      "text_content": "Click me",
      "attributes": {
        "role": "primary",
        "disabled": false
      }
    },
    {
      "bbox": [450, 50, 800, 120],
      "category": "text",
      "confidence": 0.98,
      "text_content": "User Profile Settings",
      "attributes": {}
    },
    {
      "bbox": [100, 400, 400, 440],
      "category": "input",
      "confidence": 0.87,
      "text_content": "username@example.com",
      "attributes": {
        "placeholder": "Enter email"
      }
    }
  ]
}
```

### 7.3 字段映射 (Field Mapping)

| MiniMax-VL-01 字段 | VLMOutputElement 字段 | 说明                                                  |
| ------------------- | --------------------- | ----------------------------------------------------- |
| `bbox[i]`           | `bbox`                | 像素级坐标 → 除以 `(image_width, image_height)` 归一化。 |
| `category`          | `element_type`        | 大小写不敏感匹配 taxonomy。                             |
| `confidence`        | `confidence`          | 直接使用，始终在 `[0, 1]` 范围内。                     |
| `text_content`      | `text_content`        | `null` → `None`。                                      |
| `attributes`        | `attributes`          | 直接传递，无额外属性时为 `{}`。                        |

### 7.4 转换注意事项 (Conversion Notes)

- 必须先验证 `image_width` 和 `image_height` 均为正整数再归一化。
- `category` 可能返回复合类型字符串如 `"button-primary"`；解析器应先尝试精确匹配，
  再降级为前缀匹配（提取 `-` 前的部分），仍未匹配则映射到 `other`。
- `attributes` 为 `null` 时转为空字典 `{}`。

- Validate that `image_width` and `image_height` are positive integers before normalisation.
- The `category` field may contain compound strings like `"button-primary"`; parsers
  should first attempt exact match, then fall back to prefix match (extracting the part
  before `-`), and finally map to `other`.
- `null` in `attributes` is converted to an empty dict `{}`.

---

## 8. 解析策略 (Parsing Strategy)

### 8.1 顶层接口 (Top-Level Interface)

以下函数将在 Phase 4 中实现：

The following functions will be implemented in Phase 4:

```python
def parse_qwen_output(raw: dict | str) -> VLMOutput:
    """Parse raw Qwen3.5-2B output (dict or JSON string) into VLMOutput."""

def parse_minimax_output(raw: dict | str) -> VLMOutput:
    """Parse raw MiniMax-VL-01 output (dict or JSON string) into VLMOutput."""

def parse_vlm_output(raw: dict | str, model: str) -> VLMOutput:
    """Factory dispatcher: selects parser based on model hint.

    Args:
        raw: Raw JSON dict or string.
        model: One of 'qwen3.5-2b' or 'minimax-vl-01'.
    Returns:
        Parsed and normalised VLMOutput.
    Raises:
        VLMOutputParseError: If the input is irrecoverably malformed.
    """
```

### 8.2 处理流水线 (Processing Pipeline)

```
原始 VLM 输出 (Raw JSON)
  │
  ├─ Step 1: 提取 (Extract)
  │   If input is string → try json.loads().
  │   If wrapped in markdown ```json ... ``` → strip wrapper.
  │   Fallback: regex extraction of JSON-like substring.
  │
  ├─ Step 2: 模型检测 (Model Detection)
  │   Use explicit `model` hint, or heuristics:
  │     - Has `image_width`/`image_height` at root + `category` in elements → MiniMax
  │     - Has `bbox_xyxy` / `label` fields → Qwen
  │
  ├─ Step 3: 提取图片尺寸 (Extract Image Dimensions)
  │   image_width, image_height (default 0).
  │
  ├─ Step 4: 逐元素处理 (Per-Element Processing)
  │   for each element in raw['elements']:
  │     ├─ extract bbox array (normalise to [0,1] xyxy per §5)
  │     ├─ extract & normalise element_type (case-insensitive map per §4)
  │     ├─ extract optional fields (text_content, confidence, attributes)
  │     ├─ validate: clamp coords, check non-degenerate
  │     ├─ assign sequential element_id
  │     └─ on failure → skip element, append to parse_errors, continue
  │
  └─ Step 5: 返回 (Return)
        VLMOutput(elements=[...], ...) with parse_errors log.
```

### 8.3 JSON 提取 (JSON Extraction from String)

```
Input string
  → Check if valid JSON directly → Yes → parse
  → Check for ```json ... ``` code block → Yes → extract inner content
  → Check for ``` (no lang) ... ``` code block → Yes → extract inner content
  → Fallback: regex r'\{[\s\S]*"elements"[\s\S]*\}' → extract longest match
  → All strategies fail → raise VLMOutputParseError
```

### 8.4 错误处理等级 (Error Handling Severity Levels)

| 严重度 (Severity) | 条件 (Condition)                                 | 处理 (Action)                                |
| ----------------- | ------------------------------------------------ | -------------------------------------------- |
| **Warning**       | 未知元素类型标签 (unknown element type label)     | 映射到 `other`，记录到 `parse_errors`。       |
| **Warning**       | 缺少可选字段 (missing optional field)              | 使用默认值，记录到 `parse_errors`。           |
| **Warning**       | 退化 bbox (`x2 <= x1` 或 `y2 <= y1`)             | 丢弃该元素，记录到 `parse_errors`。           |
| **Warning**       | 坐标略微超出 `[0,1]`（clamp 后可修复）            | Clamp 并记录到 `parse_errors`。               |
| **Error**         | 缺少必需字段 (`bbox`、`label`/`category`)          | 丢弃该元素，记录到 `parse_errors`。           |
| **Fatal**         | 整个 JSON 不可解析（无效 JSON、缺少 `elements` key） | 抛出 `VLMOutputParseError`。                 |

### 8.5 默认值 (Default Values)

| 缺失字段 (Missing Field)    | 默认值 (Default)  |
| --------------------------- | ----------------- |
| `confidence`                | `1.0`             |
| `text_content` / `text`     | `None`            |
| `attributes`                | `{}`              |
| `image_width`               | `0`               |
| `image_height`              | `0`               |
| `timestamp`                 | `""`              |
| `parse_errors`              | `[]`              |

### 8.6 验证规则 (Validation Rules)

一个解析后的 `VLMOutput` 被认为**有效 (valid)**，当：

1. `elements` 非空（至少包含一个元素）。
2. 所有 element 的 `element_type` 均来自 §4 分类体系或 `other`。
3. 所有 element 的 `bbox` 值 clamp 后在 `[-0.05, 1.05]` 范围内。
   （更严格的边界检查在 graph-construction 阶段执行。）
4. 所有 element 的 `element_id` 从 0 开始连续递增。
5. `confidence` 在 `[0, 1]` 范围内。

A parsed `VLMOutput` is considered **valid** when:

1. `elements` is non-empty (at least one element).
2. All `element_type` values belong to the §4 taxonomy or `"other"`.
3. All `bbox` values are within `[-0.05, 1.05]` after clamping.
   (Tighter bounds are enforced at graph-construction time.)
4. All `element_id` values are contiguous starting from 0.
5. Every `confidence` value is in `[0, 1]`.

---

## 9. 示例 (Examples)

### 9.1 Example 1: 简单登录表单 (Simple Login Form)

截屏内容：一个登录页面，包含标题文字、用户名框、密码框、登录按钮。

A simple login page with a title, username field, password field, and login button.

**原始 Qwen3.5-2B 输出 (Raw Qwen Output):**

```json
{
  "image_id": "login_screen.png",
  "elements": [
    {
      "bbox_xyxy": [0.25, 0.10, 0.75, 0.18],
      "label": "text",
      "text": "Welcome Back",
      "confidence": 0.99
    },
    {
      "bbox_xyxy": [0.20, 0.30, 0.80, 0.38],
      "label": "input",
      "text": "Enter username",
      "confidence": 0.95
    },
    {
      "bbox_xyxy": [0.20, 0.42, 0.80, 0.50],
      "label": "input",
      "text": "Enter password",
      "confidence": 0.93
    },
    {
      "bbox_xyxy": [0.35, 0.58, 0.65, 0.65],
      "label": "button",
      "text": "Sign In",
      "confidence": 0.97
    }
  ]
}
```

**解析后的 VLMOutput (Parsed VLMOutput):**

```
VLMOutput(
    model_name="qwen3.5-2b",
    image_width=0,
    image_height=0,
    timestamp="2026-05-25T10:30:00Z",
    elements=[
        VLMOutputElement(element_id=0, bbox=(0.25,0.10,0.75,0.18),
                         element_type="text", text_content="Welcome Back",
                         confidence=0.99, attributes={}, source="qwen3.5-2b"),
        VLMOutputElement(element_id=1, bbox=(0.20,0.30,0.80,0.38),
                         element_type="input", text_content="Enter username",
                         confidence=0.95, attributes={}, source="qwen3.5-2b"),
        VLMOutputElement(element_id=2, bbox=(0.20,0.42,0.80,0.50),
                         element_type="input", text_content="Enter password",
                         confidence=0.93, attributes={}, source="qwen3.5-2b"),
        VLMOutputElement(element_id=3, bbox=(0.35,0.58,0.65,0.65),
                         element_type="button", text_content="Sign In",
                         confidence=0.97, attributes={}, source="qwen3.5-2b"),
    ],
    parse_errors=[]
)
```

### 9.2 Example 2: 弹窗与表单 (Modal with Form)

一个带有模态弹窗的页面，弹窗内有关闭按钮和确认/取消按钮。

A page with a modal overlay containing a close icon, text, and confirm/cancel buttons.

**原始 MiniMax-VL-01 输出 (Raw MiniMax Output):**

```json
{
  "image_id": "modal_example.png",
  "image_width": 1440,
  "image_height": 900,
  "elements": [
    {
      "bbox": [576, 18, 864, 882],
      "category": "modal",
      "confidence": 0.96,
      "text_content": null,
      "attributes": {"backdrop": true}
    },
    {
      "bbox": [820, 36, 848, 64],
      "category": "icon",
      "confidence": 0.88,
      "text_content": "✕",
      "attributes": {"role": "close", "clickable": true}
    },
    {
      "bbox": [600, 80, 840, 120],
      "category": "text",
      "confidence": 0.99,
      "text_content": "Delete Confirmation",
      "attributes": {"font_weight": "bold"}
    },
    {
      "bbox": [600, 140, 840, 200],
      "category": "text",
      "confidence": 0.98,
      "text_content": "Are you sure you want to delete this item? This action cannot be undone.",
      "attributes": {}
    },
    {
      "bbox": [600, 230, 700, 265],
      "category": "button",
      "confidence": 0.94,
      "text_content": "Cancel",
      "attributes": {"role": "secondary"}
    },
    {
      "bbox": [710, 230, 840, 265],
      "category": "button",
      "confidence": 0.95,
      "text_content": "Delete",
      "attributes": {"role": "danger", "type": "primary"}
    }
  ]
}
```

**解析后的 VLMOutput (Parsed VLMOutput):**

```
VLMOutput(
    model_name="minimax-vl-01",
    image_width=1440,
    image_height=900,
    timestamp="2026-05-25T11:00:00Z",
    elements=[
        VLMOutputElement(element_id=0,
            bbox=(0.40, 0.02, 0.60, 0.98),  # pixel bbox / (1440, 900)
            element_type="modal", text_content=None,
            confidence=0.96,
            attributes={"backdrop": True},
            source="minimax-vl-01"),
        VLMOutputElement(element_id=1,
            bbox=(0.5694, 0.04, 0.5889, 0.0711),
            element_type="icon", text_content="✕",
            confidence=0.88,
            attributes={"role": "close", "clickable": True},
            source="minimax-vl-01"),
        VLMOutputElement(element_id=2,
            bbox=(0.4167, 0.0889, 0.5833, 0.1333),
            element_type="text", text_content="Delete Confirmation",
            confidence=0.99,
            attributes={"font_weight": "bold"},
            source="minimax-vl-01"),
        VLMOutputElement(element_id=3,
            bbox=(0.4167, 0.1556, 0.5833, 0.2222),
            element_type="text",
            text_content="Are you sure you want to delete this item? This action cannot be undone.",
            confidence=0.98, attributes={}, source="minimax-vl-01"),
        VLMOutputElement(element_id=4,
            bbox=(0.4167, 0.2556, 0.4861, 0.2944),
            element_type="button", text_content="Cancel",
            confidence=0.94,
            attributes={"role": "secondary"},
            source="minimax-vl-01"),
        VLMOutputElement(element_id=5,
            bbox=(0.4931, 0.2556, 0.5833, 0.2944),
            element_type="button", text_content="Delete",
            confidence=0.95,
            attributes={"role": "danger", "type": "primary"},
            source="minimax-vl-01"),
    ],
    parse_errors=[]
)
```

### 9.3 Example 3: 带解析错误的边缘情况 (Edge Case with Parse Errors)

一个包含格式错误元素的输入，展示宽容解析行为。

An input with malformed elements, demonstrating lenient parsing behaviour.

**原始 Qwen3.5-2B 输出 (Raw Qwen Output with issues):**

```json
{
  "image_id": "edge_case.png",
  "elements": [
    {
      "bbox_xyxy": [0.10, 0.20, 0.30, 0.35],
      "label": "button",
      "text": "OK"
    },
    {
      "bbox_xyxy": [0.40, 0.50, 0.30, 0.55],
      "label": "input",
      "text": "degenerate bbox"
    },
    {
      "bbox_xyxy": [0.50, 0.10, 0.70, 0.20],
      "label": "unknown_widget_type",
      "text": "unrecognised type"
    },
    {
      "label": "button",
      "text": "missing bbox"
    },
    {
      "bbox_xyxy": [0.80, 0.90, 0.95, -0.05],
      "label": "icon",
      "text": "icon"
    },
    {
      "bbox_xyxy": [0.75, 0.80, 0.90, 0.88],
      "label": "checkbox",
      "text": ""
    }
  ]
}
```

**解析日志 (Parse Errors Log):**

```
parse_errors=[
    "element[1]: degenerate bbox (x2=0.30 <= x1=0.40), skipped",
    "element[2]: unknown type 'unknown_widget_type', mapped to 'other'",
    "element[3]: missing required field 'bbox_xyxy', skipped",
    "element[4]: bbox y2=-0.05 out of range, clamped to 0.0",
    "element[4]: degenerate bbox after clamp (y2=0.0 <= y1=0.90), skipped",
    "element[5]: empty text_content, treated as None",
]
```

**解析后的有效元素 (Valid Elements After Parsing):**

```
elements:
  [0] button   bbox=(0.10,0.20,0.30,0.35)  text="OK"            confidence=1.0
  [1] other    bbox=(0.50,0.10,0.70,0.20)  text="unrecognised"   confidence=1.0
  [2] checkbox bbox=(0.75,0.80,0.90,0.88) text=None             confidence=1.0
```

---

## 10. 与其他模块的关系 (Relationship to Other Phases)

| 消费者 (Consumer)           | Phase    | 关系说明                                                    |
| --------------------------- | -------- | ----------------------------------------------------------- |
| `ground_truth.py`           | 4.2.2    | 使用相同的分类体系和坐标约定进行 GT-VLM 匹配。              |
| `preprocess.py`             | 4.2.3    | `CoordinateNormalizer` 接受 `VLMOutput.elements` 作为输入。 |
| `builder.py`                | 4.3.3    | `HeteroGraphBuilder` 消费 `VLMOutputElement` 构建异质图。   |
| `inference.py`              | 4.4.6    | `InferencePipeline` 从原始 VLM JSON 开始调用解析器。        |
| 数据加载器 (Data Loader)   | 4.1      | 训练/评估 pipeline 使用 `parse_vlm_output()` 加载 VLM 预测。 |

---

## 11. 修订历史 (Revision History)

| Date       | Version | Author   | Changes                                     |
| ---------- | ------- | -------- | ------------------------------------------- |
| 2026-05-25 | 1.0     | —        | 初始版本：完整的数据结构、分类体系、解析策略。 |
