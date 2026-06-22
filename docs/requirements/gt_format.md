# Ground Truth 格式分析 (Ground Truth Format Analysis)

> **Phase 1.2 — Requirements Analysis**
>
> 本文档为本项目使用的 GUI-360° 和 ScreenSpot 两个 benchmark 数据集定义标注结构，
> 以及如何将其统一为内部数据结构、VLM 预测与 GT 的匹配策略、FP/FN 语义。
> 本文档**不**包含实现代码；它提供的是 Phase 4 在
> `src/bipartite_gnn_gui/data/ground_truth.py` 中需要落实的协议契约。
>
> This document defines the annotation structures of the GUI-360° and ScreenSpot
> benchmark datasets used in this project, how they are unified into an internal data
> structure, the prediction-to-ground-truth matching strategy, and FP/FN semantics.
> It does **not** contain implementation code; it provides the contract that
> Phase 4 will implement in `src/bipartite_gnn_gui/data/ground_truth.py`.

---

## 1. 背景 (Background)

本项目使用两个公开 benchmark 数据集来评估 VLM 在 GUI 元素检测任务上的性能：
**GUI-360°** 和 **ScreenSpot**。两个数据集的标注格式有所不同，
本节记录它们的原始格式并定义统一的内部表示，使下游匹配和评估模块无需感知来源差异。

This project uses two public benchmark datasets to evaluate VLM performance on GUI
element detection: **GUI-360°** and **ScreenSpot**.  The two datasets use different
annotation formats.  This document records their original schemas and defines a unified
internal representation so that downstream matching and evaluation code is agnostic to
the data source.

### 设计目标 (Design Goals)

| # | Goal                                         | Rationale                                                          |
| -- | -------------------------------------------- | ------------------------------------------------------------------ |
| 1 | 数据集无关的统一表示 (dataset-agnostic IR)   | 加载器处理格式差异；下游代码只看到统一结构。                      |
| 2 | 归一化坐标 (normalised coordinates)          | 与 `VLMOutputElement` 使用相同的 `[0,1]` xyxy 约定 (§5 `vlm_format.md`)。 |
| 3 | 共享元素分类体系 (shared element taxonomy)   | GT 元素类型使用与 VLM 相同的 20 种 canonical key（§4 `vlm_format.md`）。 |
| 4 | 可追溯性 (traceability)                      | 每条标注记录来源数据集及原始元数据。                              |

---

## 2. GUI-360° 数据集 (GUI-360° Dataset)

### 2.1 数据集概述 (Overview)

GUI-360° 是一个大规模 GUI 截屏数据集，覆盖 Android、iOS 和 Web 平台，
专为 UI 元素检测和 grounding 任务设计。

| 指标 (Metric)       | 数值 (Value)     |
| ------------------- | ---------------- |
| 截屏数量 (screenshots) | ~3,500          |
| 标注元素总量 (total annotations) | ~50,000 |
| 平均每图元素数 (avg. elements/image) | ~14 |
| 平台 (platforms)     | Android, iOS, Web |
| 坐标格式 (coordinate format) | **归一化** xyxy `[0, 1]` |

### 2.2 原始标注格式 (Raw Annotation Format)

每条 GUI-360° 标注包含元素的归一化坐标、类型标签和文本内容。

Each GUI-360° annotation contains normalised coordinates, a type label, and text content.

```json
{
  "image_id": "android_calculator_01",
  "image_width": 1080,
  "image_height": 2340,
  "platform": "android",
  "annotations": [
    {
      "element_id": "calc_btn_7",
      "bbox": [0.10, 0.05, 0.30, 0.12],
      "type": "button",
      "text": "7",
      "attributes": {
        "clickable": true,
        "resource_id": "com.android.calculator2:id/digit_7"
      }
    },
    {
      "element_id": "calc_display",
      "bbox": [0.05, 0.02, 0.95, 0.08],
      "type": "text",
      "text": "0",
      "attributes": {
        "is_editable": false
      }
    }
  ]
}
```

| 字段 (Field)       | 类型 (Type)    | 说明 (Description)                                          |
| ------------------ | -------------- | ----------------------------------------------------------- |
| `image_id`         | `str`          | 截屏唯一标识符。                                           |
| `image_width`      | `int`          | 原始图片宽度（像素）。                                      |
| `image_height`     | `int`          | 原始图片高度（像素）。                                      |
| `platform`         | `str`          | 平台标识：`"android"`、`"ios"` 或 `"web"`。                |
| `annotations`      | `list[dict]`   | 元素标注列表。                                              |
| `annotations[].element_id` | `str`   | 元素唯一 ID。                                               |
| `annotations[].bbox`       | `[float×4]` | **归一化** xyxy：`[x1, y1, x2, y2]`，所有值 ∈ `[0, 1]`。  |
| `annotations[].type`       | `str`    | 元素类型标签（见 §4 `vlm_format.md` 分类体系）。            |
| `annotations[].text`       | `str`    | OCR 文本或元素文本内容。无文本时为空字符串 `""`。           |
| `annotations[].attributes` | `dict`   | 平台相关的额外元数据（`clickable`、`resource_id` 等）。     |

### 2.3 加载数据类 (Loading Data Class)

用于从磁盘反序列化 GUI-360° JSON 文件的中间结构：

Intermediate data class used to deserialize GUI-360° JSON files from disk:

```python
@dataclass
class GUI360Annotation:
    element_id: str
    bbox: tuple[float, float, float, float]    # 归一化 xyxy [0,1]
    type: str                                  # 原始类型标签
    text: str = ""                             # OCR 文本
    attributes: dict = field(default_factory=dict)

@dataclass
class GUI360Record:
    image_id: str
    image_width: int
    image_height: int
    platform: str                              # "android" / "ios" / "web"
    annotations: list[GUI360Annotation]
```

> **Note:** GUI-360° bboxes are already normalised; no coordinate conversion is needed
> at load time.  The `type` field is case-normalised and mapped to the shared taxonomy
> during unification (§4).

---

## 3. ScreenSpot 数据集 (ScreenSpot Dataset)

### 3.1 数据集概述 (Overview)

ScreenSpot 是一个面向 Visual Grounding 任务的 GUI 截屏 benchmark，
要求模型根据文本描述定位目标 UI 元素。与 GUI-360° 不同，ScreenSpot
更关注 grounding 精度而非全图检测，每张图的标注数量相对较少。

| 指标 (Metric)       | 数值 (Value)     |
| ------------------- | ---------------- |
| 截屏数量 (screenshots) | ~5,000          |
| 标注元素总量 (total annotations) | ~30,000 |
| 平均每图元素数 (avg. elements/image) | ~6   |
| 平台分组 (groups)    | Mobile, Desktop, Web |
| 坐标格式 (coordinate format) | **绝对像素** xyxy |

### 3.2 原始标注格式 (Raw Annotation Format)

ScreenSpot 使用绝对像素坐标，加载时需要归一化。标注体量较小但每条标注都
经过人工验证，grounding 精度高。

ScreenSpot uses absolute pixel coordinates that must be normalised at load time.
Annotations are sparser but each one is manually verified with high grounding accuracy.

```json
{
  "image_id": "screenspot_mobile_0001",
  "image_width": 1080,
  "image_height": 1920,
  "group": "mobile",
  "annotations": [
    {
      "element_id": "ss_001_0",
      "bbox": [120, 400, 380, 520],
      "type": "text",
      "text": "Settings",
      "attributes": {
        "instruction": "Find the Settings menu item"
      }
    },
    {
      "element_id": "ss_001_1",
      "bbox": [200, 600, 500, 660],
      "type": "button",
      "text": "Save",
      "attributes": {
        "instruction": "Click the save button"
      }
    }
  ]
}
```

| 字段 (Field)               | 类型 (Type)    | 说明 (Description)                                          |
| -------------------------- | -------------- | ----------------------------------------------------------- |
| `image_id`                 | `str`          | 截屏唯一标识符。                                           |
| `image_width`              | `int`          | 原始图片宽度（像素），归一化必须使用。                      |
| `image_height`             | `int`          | 原始图片高度（像素），归一化必须使用。                      |
| `group`                    | `str`          | 平台分组：`"mobile"`、`"desktop"` 或 `"web"`。             |
| `annotations`              | `list[dict]`   | 元素标注列表。                                              |
| `annotations[].element_id` | `str`          | 元素唯一 ID。                                               |
| `annotations[].bbox`       | `[float×4]`    | **绝对像素** xyxy：`[x1, y1, x2, y2]`，需归一化。          |
| `annotations[].type`       | `str`          | 元素类型标签（见 §4 `vlm_format.md` 分类体系）。            |
| `annotations[].text`       | `str`          | 目标元素文本描述。                                          |
| `annotations[].attributes` | `dict`         | 额外元数据（`instruction`、`application` 等）。            |

### 3.3 坐标归一化 (Coordinate Normalisation)

ScreenSpot 的 bbox 为绝对像素值，必须在加载时转换：

ScreenSpot bboxes are in absolute pixels and must be converted at load time:

```python
x1_norm = x1_px / image_width
y1_norm = y1_px / image_height
x2_norm = x2_px / image_width
y2_norm = y2_px / image_height
```

> **约束 (Constraint):** `image_width` 和 `image_height` 必须为正整数且大于 0，
> 否则归一化失败并抛出 `GroundTruthParseError`。

### 3.4 加载数据类 (Loading Data Class)

```python
@dataclass
class ScreenSpotAnnotation:
    element_id: str
    bbox: tuple[float, float, float, float]    # 绝对像素 xyxy
    type: str                                  # 原始类型标签
    text: str = ""                             # 文本描述
    attributes: dict = field(default_factory=dict)

@dataclass
class ScreenSpotRecord:
    image_id: str
    image_width: int
    image_height: int
    group: str                                 # "mobile" / "desktop" / "web"
    annotations: list[ScreenSpotAnnotation]
```

## 3.5 RICO 数据集 (RICO Dataset)

### 3.5.1 数据集概述 (Overview)

RICO 是当前最大的移动端 UI 数据集，包含从 9.3K Android 应用挖掘的 66K+ 唯一 UI 屏幕和 3M+ UI 元素。每个 UI 带有一张截图和完整的 Android View Hierarchy，从中可提取所有可见元素的 bbox、类型和文本。

| 指标 (Metric)       | 数值 (Value)     |
| ------------------- | ---------------- |
| 截屏数量 (screenshots) | ~66,000        |
| 标注元素总量 (total annotations) | ~3,000,000 |
| 平均每图元素数 (avg. elements/image) | ~45      |
| 平台 (platforms)     | Android          |
| 坐标格式 (coordinate format) | **绝对像素** bounds `[x1, y1, x2, y2]` (int list) |

### 3.5.2 原始格式 (Raw Format)

RICO 数据以扁平目录结构分发（所有 JSON 和图片在同一目录下）：

```
combined/
  10101.json       # View Hierarchy
  10101.jpg        # paired screenshot
  10010.json
  10010.jpg
  ...
```

每个 View Hierarchy JSON 文件的结构如下：

```json
{
  "activity_name": "com.example.MainActivity",
  "activity": {
    "root": {
      "bounds": [0, 0, 1440, 2560],
      "class": "android.widget.FrameLayout",
      "visibility": "visible",
      "visible-to-user": true,
      "children": [
        {
          "bounds": [0, 0, 1440, 2560],
          "class": "android.widget.LinearLayout",
          "visibility": "visible",
          "visible-to-user": true,
          "children": [
            {
              "bounds": [50, 100, 200, 300],
              "class": "android.widget.Button",
              "text": "Submit",
              "content-desc": [null],
              "clickable": true,
              "visibility": "visible",
              "visible-to-user": true
            }
          ]
        }
      ]
    }
  }
}
```

| 字段 (Field)          | 类型 (Type)      | 说明 (Description)                        |
| --------------------- | ---------------- | ----------------------------------------- |
| `activity_name`       | `str`            | Android Activity 名（可选，仅元数据）。   |
| `activity`            | `dict`           | 包裹层，内含 `root`。                     |
| `activity.root`       | `dict`           | View Hierarchy 根节点。                   |
| `node.bounds`         | `list[int,×4]`   | `[x1, y1, x2, y2]` 绝对像素坐标。         |
| `node.class`          | `str`            | Android 类名（如 `android.widget.Button`）。|
| `node.text`           | `str` 或 `null`  | 显示的文本（可能不存在或为 `null`）。     |
| `node.content-desc`   | `list[str\|null]`| 无障碍描述列表（如 `[null]` 或 `["desc"]`）。|
| `node.clickable`      | `bool`           | 是否可点击。                              |
| `node.visibility`     | `str`            | `"visible"` / `"invisible"` / `"gone"`。 |
| `node.visible-to-user`| `bool`           | 用户实际是否可见（补充过滤器）。          |

> **注意:** 实际的 RICO JSON **不**包含 `screen_id`、`screen_width` 或 `screen_height` 顶层字段。`screen_id` 从 JSON 文件名派生（取 stem），`screen_width` / `screen_height` 从根节点的 `bounds[2]` / `bounds[3]` 推导。

### 3.5.3 GT 提取策略 (Extraction Strategy)

从 RICO View Hierarchy 提取 GT 元素时：

1. **解析根节点**：尝试 `data["activity"]["root"]`，回退到 `data["root"]`（legacy 格式）。
2. **推导屏幕尺寸**：`screen_width = root["bounds"][2]`, `screen_height = root["bounds"][3]`。
3. **递归遍历** root 下的所有节点，只取**叶子节点**（无 children 的节点）。
4. **过滤**：跳过 `visibility != "visible"` 或 `visible-to-user == False` 或 bbox 面积为 0 的元素。
5. **解析 bounds**：`bounds` 为 `[x1, y1, x2, y2]` 整数列表（也兼容旧的字符串格式 `"[x1,y1][x2,y2]"`）。
6. **归一化**：各分量除以 `(screen_width, screen_height)`。
7. **类型映射**：优先使用 `componentLabel`（Semantic 格式），回退到 Android class → 共享分类体系。
8. **文本提取**：优先 `text` → 回退到 `content-desc` 列表（取第一个非 `null` 字符串）。
9. **截图路径**：文件名 stem + `.jpg`（优先）或 `.png`（回退）。

| Android Class Pattern            | Canonical Type |
| -------------------------------- | -------------- |
| `android.widget.Button`          | `button`       |
| `android.widget.ImageButton`     | `icon`         |
| `android.widget.ImageView`       | `image`        |
| `android.widget.TextView`        | `text`         |
| `android.widget.EditText`        | `input`        |
| `android.widget.CheckBox`        | `icon`         |
| `android.widget.Switch`          | `icon`         |
| `android.widget.Spinner`         | `icon`         |
| `android.widget.ProgressBar`     | `icon`         |
| `android.webkit.WebView`         | `container`    |
| `android.widget.ListView`        | `list`         |
| `android.widget.ScrollView`      | `container`    |
| 其它 (any other)                 | `other`        |

### 3.5.4 Semantic Annotations (RICO Semantics)

Google Research 提供的人工精标注版本（~500K 元素），bbox 质量高于自动提取的 View Hierarchy：

- 下载: `semantic_annotations.zip` (150 MB)
- 格式: 每张截图对应一个 JSON 文件，**与 View Hierarchy 使用相同的递归树结构**，但增加了 `componentLabel` 字段
- 覆盖范围: ~500K 个元素（RICO 子集），含 icon shape/semantic 分类

**Semantic Annotation JSON 格式：**

```json
{
  "class": "com.android.internal.policy.PhoneWindow$DecorView",
  "bounds": [0, 0, 1440, 2560],
  "clickable": false,
  "children": [
    {
      "class": "android.widget.Button",
      "bounds": [50, 100, 200, 300],
      "text": "Submit",
      "componentLabel": "Button",
      "clickable": true
    },
    {
      "class": "android.widget.ImageView",
      "bounds": [600, 200, 700, 300],
      "componentLabel": "Icon",
      "clickable": false
    }
  ]
}
```

> **注意:** Semantic Annotations 中根节点直接位于 JSON 顶层（无 `activity.root` 包裹）。`componentLabel` 提供比 Android class 更精细的类型标识。

**`componentLabel` → canonical type 映射：**

| componentLabel      | Canonical Type |
| ------------------- | -------------- |
| `"Icon"`            | `icon`         |
| `"Text"`            | `text`         |
| `"Input"`           | `input`        |
| `"Drawer"`          | `container`    |
| `"Image"`           | `image`        |
| `"Button"`          | `button`       |
| `"List"`            | `list`         |
| `"Checkbox"`        | `icon`         |
| `"Switch"`          | `icon`         |
| `"On/Off"`          | `icon`         |
| `"Radio Button"`    | `icon`         |
| `"Text Button"`     | `button`       |
| `"Toolbar"`         | `container`    |
| 其它 (any other)    | `other`        |

- **建议优先使用** Semantic Annotations 中存在的元素；对未覆盖的部分回退到 View Hierarchy 提取。

---

## 4. Unified GTElement 数据结构 (Unified GTElement Dataclass)

将 GUI-360° 和 ScreenSpot 的标注统一为单一数据结构。所有坐标均为归一化 xyxy，
所有元素类型均已映射到共享分类体系。

Unifies GUI-360° and ScreenSpot annotations into a single data structure.
All coordinates are normalised xyxy; all element types are mapped to the shared taxonomy.

```python
@dataclass
class GTElement:
    element_id: str                             # 元素唯一标识符 (来自原始数据集)
    bbox: tuple[float, float, float, float]      # 归一化 xyxy: (x1, y1, x2, y2) ∈ [0, 1]
    element_type: str                           # 规范元素类型 (canonical key, §4 vlm_format.md)
    text_content: Optional[str] = None           # OCR 文本或描述。无文本时为 None
    source_dataset: str = ""                     # 来源数据集: "gui360" 或 "screenspot"
    metadata: dict = field(default_factory=dict)  # 原始元数据 (group, platform, attributes, etc.)
```

### 字段说明 (Field Descriptions)

| Field           | Type                    | Required | Description                                                       |
| --------------- | ----------------------- | -------- | ----------------------------------------------------------------- |
| `element_id`    | `str`                   | yes      | Unique element identifier carried over from the source dataset.  |
| `bbox`          | `tuple[float×4]`        | yes      | `(x1, y1, x2, y2)` in normalised `[0, 1]` screen coordinates.    |
| `element_type`  | `str`                   | yes      | Canonical element type from the shared taxonomy (§4 `vlm_format.md`). |
| `text_content`  | `Optional[str]`         | no       | OCR text or element description.  `None` when no text is present. |
| `source_dataset`| `str`                   | yes      | Origin dataset identifier: `"gui360"`, `"screenspot"` or `"rico"`.|
| `metadata`      | `dict`                  | yes      | Original metadata merged from the source (see §4.1).              |

### 4.1 元数据合并规则 (Metadata Merging Rules)

| 来源 (Source)   | `metadata` 内容                                         |
| --------------- | ------------------------------------------------------- |
| GUI-360°        | `{"platform": "...", **raw_attributes}`                 |
| ScreenSpot      | `{"group": "...", "instruction": "...", **raw_attributes}` |
| RICO            | `{"app_category": "...", "package_name": "...", "class": "...", **raw_attributes}` |

### 4.2 类型规范化 (Type Normalisation)

加载时对 `type` 字段的处理逻辑与 VLM 输出相同（§4 `vlm_format.md`）：

- 大小写不敏感匹配（`"Button"` → `"button"`）。
- 未识别的类型标签映射到 `"other"` 并记录警告。
- GUI-360° 的类型体系与共享分类体系高度兼容；ScreenSpot 可能使用少量非标准标签。

During loading, the `type` field is processed identically to VLM output (§4 `vlm_format.md`):

- Case-insensitive matching (`"Button"` → `"button"`).
- Unrecognised type labels are mapped to `"other"` with a logged warning.
- GUI-360° types are highly compatible with the shared taxonomy; ScreenSpot may use a
  small number of non-standard labels.

### 4.3 空文本处理 (Empty Text Handling)

| 源值 (Source Value)    | `text_content` 值 |
| ---------------------- | ----------------- |
| `""` (空字符串)         | `None`            |
| `null` / 字段不存在     | `None`            |
| 任意非空字符串          | 直接保留          |

---

## 5. GroundTruth 数据结构 (GroundTruth Dataclass)

单张截屏对应的全部 GT 标注集合。

A collection of all ground-truth annotations for one screenshot.

```python
@dataclass
class GroundTruth:
    elements: list[GTElement]                   # 有序元素列表 (ordered list)
    image_path: str = ""                         # 对应截屏的本地路径 (local path to the screenshot)
    image_width: int = 0                         # 原始图片宽度（像素）
    image_height: int = 0                        # 原始图片高度（像素）
    source: str = ""                             # 来源数据集: "gui360" 或 "screenspot"
```

### 字段说明 (Field Descriptions)

| Field          | Type              | Required | Description                                                   |
| -------------- | ----------------- | -------- | ------------------------------------------------------------- |
| `elements`     | `list[GTElement]` | yes      | Ordered list of ground-truth annotations.                     |
| `image_path`   | `str`             | no       | Local filesystem path to the corresponding screenshot image.  |
| `image_width`  | `int`             | no       | Original image pixel width.  `0` when unknown.                |
| `image_height` | `int`             | no       | Original image pixel height.  `0` when unknown.               |
| `source`       | `str`             | no       | Source dataset identifier: `"gui360"`, `"screenspot"` or `"rico"`.|

> **关系 (Relationship):** `GroundTruth` 与 `VLMOutput`（§3 `vlm_format.md`）结构对称，
> 便于匹配算法需要时将两者并排使用。`image_path` 替代了 `VLMOutput` 中的
> `model_name`/`timestamp` 字段，因为 GT 与模型推理无关。
>
> `GroundTruth` is structurally symmetric to `VLMOutput` (§3 `vlm_format.md`) so
> matching algorithms can handle both side-by-side.  `image_path` replaces the
> `model_name`/`timestamp` fields since ground truth is model-independent.

---

## 6. 匹配策略 (Matching Strategy)

VLM 产生一组预测元素后，需要将其与 GT 元素建立对应关系，以便：

- 正确匹配的预测可用于坐标精化损失 (coordinate-refinement loss)。
- 未匹配的预测计为 **假阳性 (FP)**。
- 未匹配的 GT 元素计为 **假阴性 (FN)**。

After the VLM produces a set of predicted elements, we need to establish
correspondences with ground-truth elements so that:

- Correctly matched predictions can be used for coordinate-refinement loss.
- Unmatched predictions are counted as **false positives (FP)**.
- Unmatched ground-truth elements are counted as **false negatives (FN)**.

### 6.1 IoU 矩阵 (IoU Matrix)

设 `M = len(predictions)`，`N = len(ground_truth_elements)`。

构造代价矩阵 (cost matrix) `C` 形状为 `(M, N)`：

```
C[i, j] = 1 - IoU(pred_i, gt_j)
```

即 IoU 越高 → 代价越低 → 匹配优先级越高。

Build a cost matrix `C` of shape `(M, N)` where each entry is `1 - IoU(pred, gt)`.
Higher IoU → lower cost → higher matching priority.

### 6.2 IoU 定义 (IoU Definition)

对于两个轴对齐边界框 `A = (x₁ᴬ, y₁ᴬ, x₂ᴬ, y₂ᴬ)` 和 `B = (x₁ᴮ, y₁ᴮ, x₂ᴮ, y₂ᴮ)`：

```
intersection_w = max(0, min(x2_A, x2_B) - max(x1_A, x1_B))
intersection_h = max(0, min(y2_A, y2_B) - max(y1_A, y1_B))
intersection   = intersection_w × intersection_h

area_A = (x2_A - x1_A) × (y2_A - y1_A)
area_B = (x2_B - x1_B) × (y2_B - y1_B)
union   = area_A + area_B - intersection

IoU = intersection / max(union, ε)     其中 ε = 1e-8 防止除零
```

IoU ∈ `[0, 1]`，`1.0` 表示完全重合。

### 6.3 阈值过滤 (Threshold Filtering)

**阈值 `τ = 0.5`**：任何 IoU < τ 的匹配对视为无效，对应代价设为 `∞`（或在矩阵中标记为
不可匹配）。这防止了空间上几乎不重叠的框被强行配对。

**Threshold `τ = 0.5`**: any pair with IoU < τ is considered invalid; its cost is set
to `∞` (or flagged as unmatchable).  This prevents boxes with negligible spatial overlap
from being forcibly paired.

```
C[i, j] = ∞    if IoU(pred_i, gt_j) < 0.5
```

### 6.4 匈牙利算法 (Hungarian Algorithm)

对过滤后的代价矩阵 `C` 使用 **匈牙利算法** (Kuhn–Munkres) 求解最优一对一双向匹配：

Apply the **Hungarian algorithm** (Kuhn–Munkres) to the filtered cost matrix `C` to
find the optimal one-to-one bipartite matching:

```
row_ind, col_ind = linear_sum_assignment(C)
```

求解器返回最小化总代价的行-列配对。

- 仅代价为有限值（即 IoU ≥ τ）的配对被视为**已匹配 (matched)**。
- 若 `M > N`，多余的行（预测）无法匹配 → **FP**。
- 若 `N > M`，多余的列（GT）无法匹配 → **FN**。

The solver returns row-column pairs that minimise total cost.

- Only pairs with finite cost (i.e., IoU ≥ τ) are considered **matched**.
- If `M > N`, excess predictions are left unmatched → **FP**.
- If `N > M`, excess ground-truth elements are left unmatched → **FN**.

### 6.5 匹配输出 (Matching Output)

```
matched_pairs: list[tuple[int, int]]   # (pred_idx, gt_idx) 对
fp_indices:    list[int]               # 未匹配的预测索引
fn_indices:    list[int]               # 未匹配的 GT 元素索引
```

### 6.6 类型条件匹配（可选扩展）(Type-Conditioned Matching — Optional Extension)

可选的约束条件：匹配的预测与 GT 元素必须具有**相同的元素类型**（`"other"` 类型忽略此约束）。

An optional constraint requires matched predictions and ground-truth elements to have
the **same element type** (elements of type `"other"` are exempt from this constraint).

启用时：

```
C[i, j] = ∞    if pred_type[i] != gt_type[j] and pred_type[i] != "other" and gt_type[j] != "other"
```

这产生类型纯净 (type-pure) 的匹配，可减少虚假对应 (spurious correspondences)，
但可能会增加 FN 计数。

### 6.7 FP 与 FN 的统计 (Per-Image and Dataset-Level Aggregation)

对单张图片：

```
TP = len(matched_pairs)
FP = M - TP
FN = N - TP
```

数据集级 FP/FN 为所有图片的简单累加和。

---

## 7. 加载函数接口 (Loading Functions)

以下函数将在 Phase 4 中实现，作为 GT 数据加载的公开接口。

The following functions will be implemented in Phase 4 as the public interface for
GT data loading.

### 7.1 `load_gui360_annotation`

```python
def load_gui360_annotation(path: str | Path) -> GroundTruth:
    """从 GUI-360° JSON 标注文件加载并返回统一 GroundTruth。

    Args:
        path: GUI-360° JSON 文件路径。

    Returns:
        GroundTruth 实例，所有 bbox 已归一化，类型已规范化。

    Raises:
        FileNotFoundError: 文件不存在。
        GroundTruthParseError: JSON 格式无效或关键字段缺失。

    Processing steps:
        1. 读取 JSON 文件。
        2. 提取 image_id, image_width, image_height, platform。
        3. 遍历 annotations[]：
           - 复制归一化 bbox（已是 [0,1]，直接使用）。
           - 规范化 type → element_type（case-insensitive match）。
           - text → text_content（空字符串 → None）。
           - 合并 platform + attributes → metadata。
           - 构造 GTElement。
        4. 根据 image_id 推导 image_path。
        5. 返回 GroundTruth(elements=..., source="gui360")。
    """
```

### 7.2 `load_screenspot_annotation`

```python
def load_screenspot_annotation(path: str | Path) -> GroundTruth:
    """从 ScreenSpot JSON 标注文件加载并返回统一 GroundTruth。

    Args:
        path: ScreenSpot JSON 文件路径。

    Returns:
        GroundTruth 实例，所有 bbox 已归一化，类型已规范化。

    Raises:
        FileNotFoundError: 文件不存在。
        GroundTruthParseError: JSON 格式无效、image_width≤0 或关键字段缺失。

    Processing steps:
        1. 读取 JSON 文件。
        2. 提取 image_id, image_width, image_height, group。
        3. 验证 image_width > 0 且 image_height > 0。
        4. 遍历 annotations[]：
           - bbox 归一化：各分量除以 (image_width, image_height)。
           - clamp 到 [0, 1] 范围。
           - 规范化 type → element_type（case-insensitive match）。
           - text → text_content（空字符串 → None）。
           - 合并 group + attributes → metadata。
           - 构造 GTElement。
        5. 根据 image_id 推导 image_path。
        6. 返回 GroundTruth(elements=..., source="screenspot")。
    """
```

### 7.3 `load_rico_annotation`

```python
def load_rico_annotation(path: str | Path, semantic: bool = False) -> GroundTruth:
    """从 RICO View Hierarchy JSON 加载并返回统一 GroundTruth。

    递归遍历 View Hierarchy 树，提取所有可见叶子节点的 bbox 和类型信息。

    Args:
        path: RICO View Hierarchy JSON 文件路径。
        semantic: 是否为 Semantic Annotations 格式（字段名略有差异）。

    Returns:
        GroundTruth 实例，所有 bbox 已归一化，类型已规范化。

    Processing steps:
        1. 读取 JSON 文件，提取 screen_width, screen_height。
        2. 递归遍历 root.children。
        3. 对每个叶子节点（无 children 或 children 为空列表）：
           - 解析 bounds "[x1,y1][x2,y2]" → (x1, y1, x2, y2)。
           - 过滤 visibility != "visible" 或面积为 0 的元素。
           - 归一化：各分量除以 (screen_width, screen_height)。
           - 映射 class → element_type（Android class → canonical type）。
           - text / content-desc → text_content。
           - 构造 GTElement。
        4. 根据 screen_id 推导 image_path。
        5. 返回 GroundTruth(elements=..., source="rico")。
    """
```

### 7.4 工厂函数 (Factory Dispatcher)

```python
def load_ground_truth(path: str | Path, source: str | None = None) -> GroundTruth:
    """根据 source 提示或文件内容自动选择加载器。

    Args:
        path: 标注文件路径。
        source: 可选的数据集标识 (`"gui360"`, `"screenspot"` 或 `"rico"`)。
                为 None 时通过文件内容自动推断。

    Returns:
        GroundTruth 实例。

    Auto-detection heuristics (when source is None):
        - 文件中存在 "platform" 键 → GUI-360°
        - 文件中存在 "group" 键   → ScreenSpot
        - 文件中存在 "root" 键    → RICO (View Hierarchy)
        - 否则 → 抛出 GroundTruthParseError
    """
```

### 7.4 错误处理 (Error Handling)

| 严重度 (Severity) | 条件 (Condition)                                      | 处理 (Action)                                    |
| ----------------- | ----------------------------------------------------- | ------------------------------------------------ |
| **Warning**       | 未知元素类型标签 (unknown element type)                | 映射到 `"other"`，记录警告。                     |
| **Warning**       | 空文本字段 (empty text field)                          | 设为 `None`，静默处理。                          |
| **Warning**       | 坐标略超出 `[0, 1]`（clamp 后可修复）                 | Clamp 到 `[0, 1]`，记录警告。                    |
| **Error**         | 退化 bbox (`x2 <= x1` 或 `y2 <= y1`)                  | 丢弃该元素，记录错误。                           |
| **Fatal**         | JSON 不可解析、缺少 `annotations` key、image_width ≤ 0 | 抛出 `GroundTruthParseError`。                   |

---

## 8. 与其他模块的关系 (Relationship to Other Phases)

| 消费者 (Consumer)   | Phase   | 关系说明                                                       |
| ------------------- | ------- | -------------------------------------------------------------- |
| `vlm_output.py`     | 4.2.1   | 使用相同的分类体系和坐标约定。GT 与 VLM 的数据结构对称。       |
| `preprocess.py`     | 4.2.3   | `CoordinateNormalizer` 消费 GT 坐标和匹配对进行 delta 监督。   |
| `builder.py`        | 4.3.3   | `HeteroGraphBuilder` 同时消费 GT 和 VLM 元素构建训练图。       |
| `metrics.py`        | 4.5.1   | `ElementRecall`、`ElementPrecision` 依赖匹配结果计算指标。    |
| `evaluator.py`      | 4.5.2   | `Evaluator` 调用 `match_predictions_to_ground_truth` 建立对应。 |
| `ground_truth.py`   | 4.2.2   | 本文档的协议在此文件中实现。新增 `load_rico_annotation` 支持 RICO。 |
| `rico_loader.py`    | 新      | RICO View Hierarchy 递归解析器（可选独立模块）。               |

---

## 9. 修订历史 (Revision History)

| Date       | Version | Author | Changes                                                    |
| ---------- | ------- | ------ | ---------------------------------------------------------- |
| 2026-05-25 | 2.0     | —      | 重写为双语格式，新增章节 7（加载函数接口）；与 `vlm_format.md` 结构对齐。 |
| —          | 1.0     | —      | 初始版本。                                                  |
