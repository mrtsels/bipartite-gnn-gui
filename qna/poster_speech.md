# Poster Speech — 10 分钟展示讲稿

中英对照：**EN** = 上台讲的英文稿（可直接背），**CN** = 中文对照/速记。
**受众提醒:** 现场提问者很多不是 GNN 专家 → 讲稿已口语化；必须出现的术语都附了一句通俗解释（讲的时候自然带过即可）。
展示时间: 2026-08-14 (Fri) 9:30–12:00 · ERBLT foyer (8/F ERB) · 9:20 前签到
**注意:** 10–15 分钟含 Q&A → 讲稿控制在 **8–9 分钟**，留 2–6 分钟提问。

---

## 时间分配 (Timing Plan)

| 段落 | 时长 | 海报位置 | 内容 |
|---|---|---|---|
| 0 开场 | 0:30 | 标题区 | 自我介绍 + 一句话主题 |
| 1 问题与动机 | 1:30 | 左栏 Block 1 | 为什么重要 + VLM 两类错误 |
| 2 现有方案与差距 | 1:00 | 左栏 Block 2 | 三个替代方案都不行 → 换个角度 |
| 3 方法 | 3:00 | 左栏 Block 4 + 中栏 | 图构建 → 两轮信息交换 → 四个小模块 |
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

**EN:** First, why does this matter? Screen readers, automated UI testers, and software agents all need to know *what* is on a screen and *where* it is. On-device AI forces us to use lightweight vision-language models — small models that understand both images and text — but these models make two systematic mistakes. *(指海报左栏)* First, **element omission**: 38 percent of the elements actually on screen are never reported — small icons, dividers, nested containers just disappear. Second, **misalignment**: the rectangles the model draws around elements are off by tens of pixels. And here's the catch: these errors come from compressing a big model into a small one — so the fix is not to train a bigger model; that's exactly what running on a phone forbids.

**CN:** 首先，为什么重要？屏幕阅读器、自动化 UI 测试、软件智能体都需要知道屏幕上有什么、在哪里。端侧 AI 只能用轻量视觉语言模型——同时理解图像和文字的小模型——但它们有两个系统性错误：**漏报**——38% 的真实可见元素完全没报出，小图标、分隔线、嵌套容器直接消失；**错位**——模型画出的元素框偏移数十像素。关键在于：错误源于"大模型压缩成小模型"——所以解法不是训练更大的模型，那恰恰是手机端不允许的。

---

### 2. 现有方案与差距 Existing Solutions & Gap (2:00–3:00)

**EN:** Existing approaches fall short. Fine-tuning — continuing to train the model — on a larger model violates the phone's budget. Adding a second object detector — a separate model that finds elements — doubles the cost and brings its own errors. Generative layout models — models that draw layouts from scratch — are heavy and not designed for correction. So we take a different angle: the VLM stays untouched — we add a tiny correction network that exploits what every well-designed interface has: **spatial structure** — alignment, containment, spacing, grid membership.

**CN:** 现有方案都不够好：微调更大模型（在已有模型上继续训练）违背手机预算；再加一个检测模型成本翻倍还带新错误；生成式布局模型（从零画布局的模型）太重、也不是为修正设计的。所以我们换角度：VLM 完全不动——只加一个极小的修正网络，利用每个好界面都有的东西：**空间结构**——对齐、包含、间距、网格。

---

### 3. 方法 Method (3:00–6:00)

**3a. 图构建 (Graph construction)** *(指左栏 Block 4 流程图与二分图)*

**EN:** We turn the VLM's noisy element list into a graph — a bipartite graph, meaning two kinds of nodes, with connections only between the two kinds. One side is element nodes — buttons, text, icons. The other side is rule nodes: layout rules. Ten types — six alignment rules, plus spacing, containment, grid, and same-size — each a pairwise check with a small tolerance, a small allowance so near-misses still count. Edges only connect elements to the rules they participate in.

**CN:** 我们把 VLM 的噪声元素列表转成图——二分图，即两类节点、连线只出现在两类之间。一边是元素节点：按钮、文本、图标；另一边是规则节点：布局规则，共十种——六种对齐规则，加间距、包含、网格、等尺寸——每条都是带小容差的两两检查，允许轻微偏差。连线只连接元素与它参与的规则。

**3b. 信息交换 (Message passing)** *(指中栏公式)*

**EN:** Then we run two rounds of message passing — nodes exchanging information with their neighbors — using GraphSAGE, a simple graph network where each node averages its neighbors' information. Round one: each rule summarizes the elements it involves. Round two: each element collects the rules it shares — forming its spatial neighborhood. Element features are just five numbers: four normalized coordinates plus normalized area. Optionally, we can add visual features from a fixed image model — kept frozen, not updated during training.

**CN:** 然后跑两轮信息交换——节点与邻居交换信息——用 GraphSAGE，一种"每个节点把邻居信息取平均"的简单图网络。第一轮：规则汇总涉及的元素；第二轮：元素收集共享的规则，形成空间邻域。元素特征只有 5 个数：4 个归一化坐标加归一化面积。可选地加入固定图像模型（冻结、不参与训练）提供的视觉特征。

**3c. 四个小模块 (Four heads)** *(指中栏 Models 段)*

**EN:** The network ends in four small output modules, trained together. **Coordinate correction** repairs misplaced boxes. **Violation detection** decides, for each layout rule, whether it is broken — this makes the model understand layout rules. **Existence scoring** filters out hallucinated elements — things the VLM imagined. And **element completion** is self-supervised: it learns from the data itself, with no extra labels. We randomly hide 20 to 80 percent of the real elements; the leftover dangling connections signal what's missing, and this module proposes the missing boxes and types. The whole network is 57 thousand parameters — under a millisecond per screenshot.

**CN:** 网络末端是四个一起训练的小模块。**坐标修正**修复错位框；**违反检测**判断每条布局规则是否被破坏——让模型"理解"布局规则；**存在性打分**过滤幻觉元素——VLM 想象出来的东西；**元素补全**是自监督的：从数据自身学习、不需要额外标注——随机藏起 20–80% 的真实元素，残留的悬空连接指示缺失，该模块补出框和类型。整个网络 57K 参数——每图不到 1 毫秒。

---

### 4. 结果 Results (6:00–8:00)

**EN:** Results — three findings. *(指中栏 Block 6 图表)* First, **training everything together wins**: if we train only one task, the self-supervised part collapses — rule-judging accuracy drops to 0.489, while joint training keeps both at 0.876, with a small average error of 0.051. Second, **rules matter asymmetrically**: containment is the most valuable rule, while alignment rules actually hurt — removing them all improves rule-judging accuracy by 3.1 points, because alignment is already visible from the box coordinates themselves. Third, **element completion**: the gains grow with how much we hide — the more structure is missing, the more the graph exploits it, reaching 56 percent better box overlap than a simple copy-the-nearest-element baseline. *(指右栏端到端图)* End to end — the full pipeline on 200 real screenshots with Qwen3-VL Flash — we recover 106 missed elements. Recall — the share of real elements we catch — is up 2.2 points; precision — the share of our reports that are real — up 1.1; the combined F1 score up 2.0. All under a millisecond per screenshot.

**CN:** 结果，三个发现。第一，**一起训练胜出**：只训一个任务时自监督部分塌缩——规则判断准确率掉到 0.489；一起训练两者都保持 0.876，平均误差 0.051。第二，**规则价值不对称**：CONTAINMENT 最有价值，ALIGNMENT 反而有害——全部去掉后规则判断准确率 +3.1pp，因为对齐信息从框坐标就能直接看出。第三，**元素补全**：增益随隐藏比例增大——缺得越多图利用越充分，高掩码率下框重叠度比"复制最近元素"的简单方法好 56%。端到端：200 张真实截图 + Qwen3-VL Flash，恢复 106 个漏检。召回（真实元素里抓到多少） +2.2pp；精确率（报出的东西里多少是真的） +1.1pp；综合分 F1 +2.0pp。每图不到 1 毫秒。

---

### 5. 局限与未来 Limitations & Future Work (8:00–9:00)

**EN:** Honest limitations: the framework inherits the VLM's initial detections — it doesn't fix purely semantic errors, like confusing one label with another. The existence-scoring module gives limited filtering on real data, so the gains come mainly from completion. And desktop and web generalization is still untested. That's part of our future work, together with attention-based visual fusion and temporal context — using information from past screens — to distinguish missed elements from elements hidden by the current UI state.

**CN:** 如实说局限：框架继承 VLM 的初始检测，修不了纯语义错误（比如认错标签）；存在性打分在真实数据上过滤价值有限，增益主要来自补全；桌面/网页泛化尚未验证。这些是未来工作，还有注意力视觉融合、以及利用时序上下文（历史屏幕信息）区分漏检与 UI 状态隐藏。

---

### 6. 收尾 Close (9:00–9:30)

**EN:** To summarize: a 57-thousand-parameter bipartite graph network that repairs and completes lightweight-VLM GUI detections — 90 to 94 percent accuracy on judging layout rules on RICO, 56 percent better completion overlap, 2.0 points better F1 end to end, at under a millisecond per screenshot. Thank you — I'm happy to take questions.

**CN:** 总结：一个 57K 参数的二分图网络，修复并补全轻量 VLM 的 GUI 检测——RICO 上规则判断准确率 90–94%，补全重叠度 +56%，端到端 F1 +2.0pp，每图不到 1 毫秒。谢谢，欢迎提问。

---

## 演示技巧 (Delivery Tips)

1. **节奏**: 每段先报结论再展开；数字处放慢、重读（38%、90–94%、+56%、+2.0pp、106、1 ms）。
2. **指图**: 每段开头的"海报位置"就是手势提示——讲哪指哪，不要背对观众。
3. **术语带解释**: 讲稿里括号内的通俗解释是给"非 GNN 专家"准备的——讲到术语时自然停顿半秒带一句，不要跳过（例如说完 "bipartite graph" 立刻说 "two kinds of nodes, connections only between the two kinds"）。
4. **时间检查**: 3:00 处应已开始讲方法；6:00 处应已进入结果；超过 9:30 就砍 §5 细节直接收尾。
5. **Q&A 承接**: 被问倒时用 "That's a good question — the reason is…" 起头，卡住就回到速查表数字（见 `poster_qna.md` §0）。
6. **开场 10 秒**: 站稳、微笑、报名字和导师名，再开始第一句。
