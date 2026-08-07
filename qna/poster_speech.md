# Poster Speech — 10 分钟展示讲稿

中英对照：**EN** = 上台讲的英文稿（可直接背），**CN** = 中文对照/速记。
展示时间: 2026-08-14 (Fri) 9:30–12:00 · ERBLT foyer (8/F ERB) · 9:20 前签到
**注意:** 10–15 分钟含 Q&A → 讲稿控制在 **8–9 分钟**，留 2–6 分钟提问。

---

## 时间分配 (Timing Plan)

| 段落 | 时长 | 海报位置 | 内容 |
|---|---|---|---|
| 0 开场 | 0:30 | 标题区 | 自我介绍 + 一句话主题 |
| 1 问题与动机 | 1:30 | 左栏 Block 1 | 为什么重要 + VLM 两类错误 |
| 2 现有方案与差距 | 1:00 | 左栏 Block 2 | 三个替代方案都不行 → 换个角度 |
| 3 方法 | 3:00 | 左栏 Block 4 + 中栏 | 图构建 → 两跳消息传递 → 四个头 |
| 4 结果 | 2:00 | 中栏 Block 6 + 右栏 | 三个发现 + 端到端数字 |
| 5 局限与未来 | 1:00 | 右栏 Block 7 | 如实讲局限 + 未来工作 |
| 6 收尾 | 0:30 | 标题区 | 一句话总结 + 邀请提问 |
| **合计** | **~9:30** | | 余量留给 Q&A |

---

## 讲稿 (Speech Script)

### 0. 开场 Opening (0:00–0:30)

**EN:** Good morning. I'm Alex Xie from Information Engineering, supervised by Professor Wing Cheong Lau. Today I'll present our work: *Heterogeneous Bipartite Graph Neural Networks for GUI Structure Error Correction* — a lightweight way to fix the mistakes small vision-language models make when they "read" a phone screen.

**CN:** 早上好。我是信息工程系的 Alex Xie，导师刘永昌教授。今天介绍我们的工作：面向 GUI 结构错误修正的异构二分图神经网络——用轻量方式修正小型视觉语言模型"读"屏幕时的错误。

---

### 1. 问题与动机 Problem & Motivation (0:30–2:00)

**EN:** First, why does this matter? Screen readers, automated UI testers, and software agents all need to know *what* is on a screen and *where* it is. On-device AI forces us to use lightweight VLMs — under three billion parameters — but these models make two systematic mistakes. *(指海报左栏)* First, **element omission**: 38 percent of visible elements are never reported — small icons, dividers, nested containers just disappear. Second, **misalignment**: bounding boxes are off by tens of pixels. And here's the catch: these errors come from compressing a big model into a small one — so the fix is not to train a bigger model; that's exactly what on-device constraints forbid.

**CN:** 首先，为什么重要？屏幕阅读器、自动化 UI 测试、软件智能体都需要知道屏幕上有什么、在哪里。端侧 AI 只能用轻量 VLM（<30 亿参数），但它们有两个系统性错误：**漏报**——38% 的可见元素完全没报出，小图标、分隔线、嵌套容器直接消失；**错位**——框偏移数十像素。关键在于：错误源于"大模型压缩成小模型"——所以解法不是训练更大的模型，那恰恰是端侧不允许的。

---

### 2. 现有方案与差距 Existing Solutions & Gap (2:00–3:00)

**EN:** Existing approaches fall short. Fine-tuning a larger model violates the on-device budget. Adding a second object detector doubles the cost and brings its own errors. Generative layout models are heavy and not designed for correction. So we take a different angle: the VLM stays untouched — we add a tiny post-correction network that exploits what every well-designed interface has: **spatial structure** — alignment, containment, spacing, grid membership.

**CN:** 现有方案都不够好：微调更大模型违背端侧预算；再加检测器成本翻倍还带新错误；生成式布局模型太重、也不是为修正设计的。所以我们换角度：VLM 完全不动——只加一个极小的后修正网络，利用每个好界面都有的东西：**空间结构**——对齐、包含、间距、网格。

---

### 3. 方法 Method (3:00–6:00)

**3a. 图构建 (Graph construction)** *(指左栏 Block 4 流程图与二分图)*

**EN:** We convert the VLM's noisy element list into a heterogeneous bipartite graph. One side is element nodes — buttons, text, icons. The other side is constraint nodes: ten spatial constraint types — six alignment predicates, plus spacing, containment, grid, and same-size — each a pairwise predicate with a tolerance. Edges only connect elements to constraints.

**CN:** 我们把 VLM 的噪声元素列表转成异构二分图。一边是元素节点——按钮、文本、图标；另一边是约束节点：十种空间约束——六种对齐谓词，加间距、包含、网格、等尺寸——都是带容差的两两谓词。边只连接元素与约束。

**3b. 消息传递 (Message passing)** *(指中栏公式)*

**EN:** Then we run two alternating hops of GraphSAGE mean-aggregation. Hop one: each constraint aggregates its participating elements. Hop two: each element aggregates the constraints it shares — forming its spatial neighborhood. Element features are just five numbers: four normalized coordinates plus normalized area. Optionally, we fuse frozen DINOv2 visual features.

**CN:** 然后跑两跳交替的 GraphSAGE 均值聚合。第一跳：约束聚合参与的元素；第二跳：元素聚合共享的约束，形成空间邻域。元素特征只有 5 个数：4 个归一化坐标加归一化面积。可选地融合冻结的 DINOv2 视觉特征。

**3c. 四个预测头 (Four heads)** *(指中栏 Models 段)*

**EN:** The encoder feeds four prediction heads, trained jointly. **Coordinate correction** repairs misaligned boxes. **Violation detection** classifies whether each constraint is broken — this makes the model understand layout rules. **Existence scoring** filters hallucinated detections. And **element completion** is self-supervised: we randomly mask 20 to 80 percent of ground-truth elements; the dangling edges signal what's missing, and the head proposes their boxes and types. The whole network is 57 thousand parameters — under a millisecond per screenshot.

**CN:** 编码器接四个联合训练的预测头。**坐标修正**修复错位框；**违反检测**判断每个约束是否被破坏——让模型"理解"布局规则；**存在性打分**过滤幻觉检测；**元素补全**是自监督的：随机掩码 20–80% 的 GT 元素，悬空边指示缺失，头输出它们的框和类型。整个网络 57K 参数——每图不到 1 毫秒。

---

### 4. 结果 Results (6:00–8:00)

**EN:** Results — three findings. *(指中栏 Block 6 图表)* First, **joint training wins**: single-objective training lets the unsupervised head collapse — violation accuracy drops to 0.489, while joint training keeps both heads at 0.876 and 0.051 proposal MSE. Second, **constraints matter asymmetrically**: containment is the most valuable constraint, while alignment constraints actually hurt — removing them all improves violation accuracy by 3.1 points, because alignment is already recoverable from box coordinates. Third, **element completion**: gains grow with the drop ratio — the more structure is missing, the more the graph exploits it, reaching plus 56 percent IoU over a nearest-neighbor baseline. *(指右栏端到端图)* End to end, on 200 real screenshots with Qwen3-VL Flash: we recover 106 missed elements — recall up 2.2 points, precision up 1.1, F1 up 2.0 — all under a millisecond per screenshot.

**CN:** 结果，三个发现。第一，**联合训练胜出**：单目标训练让无监督头塌缩——违反准确率掉到 0.489；联合训练两者都保持 0.876 / 提议 MSE 0.051。第二，**约束价值不对称**：CONTAINMENT 最有价值，ALIGNMENT 反而有害——全部去掉后违反准确率 +3.1pp，因为对齐信息从框坐标就能得到。第三，**元素补全**：增益随掩码率增大——缺得越多图利用越充分，高掩码率下比最近邻基线 +56% IoU。端到端：200 张真实截图 + Qwen3-VL Flash，恢复 106 个漏检——召回 +2.2pp、精确率 +1.1pp、F1 +2.0pp——每图不到 1 毫秒。

---

### 5. 局限与未来 Limitations & Future Work (8:00–9:00)

**EN:** Honest limitations: the framework inherits the VLM's initial detections — it doesn't fix purely semantic errors. The existence head gives limited filtering on real data, so the gains come mainly from completion. And desktop and web generalization is still untested. That's part of our future work, together with cross-attention visual fusion and temporal context — to distinguish missed elements from elements hidden by UI state.

**CN:** 如实说局限：框架继承 VLM 的初始检测，修不了纯语义错误；存在性头在真实数据上过滤价值有限，增益主要来自补全；桌面/网页泛化尚未验证。这些是未来工作，还有交叉注意力视觉融合、以及用时序上下文区分漏检与 UI 状态隐藏。

---

### 6. 收尾 Close (9:00–9:30)

**EN:** To summarize: a 57-thousand-parameter heterogeneous bipartite graph network that repairs and completes lightweight-VLM GUI detections — 90 to 94 percent violation accuracy on RICO, plus 56 percent completion IoU, plus 2.0 points F1 end to end, at under a millisecond per screenshot. Thank you — I'm happy to take questions.

**CN:** 总结：一个 57K 参数的异构二分图网络，修复并补全轻量 VLM 的 GUI 检测——RICO 上违反检测 90–94%，补全 IoU +56%，端到端 F1 +2.0pp，每图不到 1 毫秒。谢谢，欢迎提问。

---

## 演示技巧 (Delivery Tips)

1. **节奏**: 每段先报结论再展开；数字处放慢、重读（38%、90–94%、+56%、+2.0pp、106、1 ms）。
2. **指图**: 每段开头的"海报位置"就是手势提示——讲哪指哪，不要背对观众。
3. **时间检查**: 3:00 处应已开始讲方法；6:00 处应已进入结果；超过 9:30 就砍 §5 细节直接收尾。
4. **Q&A 承接**: 被问倒时用 "That's a good question — the reason is…" 起头，卡住就回到速查表数字（见 `poster_qna.md` §0）。
5. **开场 10 秒**: 站稳、微笑、报名字和导师名，再开始第一句。
