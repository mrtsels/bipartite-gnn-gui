# Poster Q&A 准备 — Heterogeneous Bipartite GNN for GUI Structure Error Correction

中英对照 · 问题尽可能发散，回答保持简练（1–2 句）。
**受众提醒:** 提问者很多不是 GNN 专家 → 术语尽量口语化；必须出现时附带一句通俗解释。
展示时间: 2026-08-14 (Fri) 9:30–12:00 · ERBLT foyer (8/F ERB) · 9:20 前签到 · 10–15 分钟 + Q&A

---

## 0. 关键数字速查 (Key Numbers Cheat-Sheet)

| 项目 | 数值 | 通俗解释 |
|---|---|---|
| 模型规模 | 57K 参数，<1 ms/图 | 很小很快，手机上跑得动 |
| 编码器 | 2 层 GraphSAGE，hidden 128，AdamW + cosine | 特征提取部分；hidden 128 = 内部特征 128 维；AdamW = 常用优化器 |
| 特征 | 5-d（4 个归一化坐标 + 归一化面积） | 只用一个框的 4 个角坐标加面积，很省 |
| 约束 | 10 种，平均 37.3 条/图 | 布局规则（对齐、包含、间距等） |
| 数据 | RICO（训练/评估）、ScreenSpot（跨域）；端到端 200 张截图，4,789 GT / 2,947 VLM 预测 | GT = 人工标好的正确答案；匹配阈值 0.1 = 中心距离误差容忍度 |
| VLM 问题 | 38% 元素漏报；框偏移数十像素 | 小模型读屏幕的两种系统性错误 |
| 违反检测 | RICO 90–94%（control 0.908）；去 ALIGNMENT +3.1pp；去 CONTAINMENT −1.9pp | 判断"布局规则是否被破坏"的正确率；pp = 百分点 |
| 训练目标消融 | joint 0.876 / 0.051；violation-only 0.898 / 0.116；proposal-only 0.489 / 0.051 | 四个任务一起训 vs 只训一个的效果对比 |
| 补全 | 高掩码率时较 NN 基线 +56% IoU | 补出的框与真实框的重叠度，比"复制最近元素"的简单方法好 56% |
| 端到端 (200 图) | P 0.382→0.393 (+1.1pp)，R 0.235→0.257 (+2.2pp)，F1 0.291→0.311 (+2.0pp)；TP +106，FP +84，FN −106 | 完整流程在真实截图上的表现：抓到的漏检更多，误报略增，综合分上升 |
| 视觉融合 | cross-attention 较拼接 Proposal MSE −18~22% | 加视觉信息时用注意力机制比简单拼接好，补框误差降约两成 |
| 损失权重 | w_c=1.0, w_v=0.5, w_e=0.5, w_p=0.5 | 四个任务的训练目标按此比例加权 |

**术语通俗版 (Glossary):**
- **VLM（视觉语言模型）**: 能同时"看"图、"读"字的模型，这里指手机上跑得动的轻量版。
- **图 (Graph)**: 由"节点"和"连线"组成的数据结构；本工作里节点是界面元素和布局规则。
- **二分图 / 异构**: 有两类节点（元素、规则），连线只出现在两类之间；异构 = 两类节点性质不同。
- **GraphSAGE / 消息传递**: 一种图神经网络做法——每个节点把邻居的信息取平均，多轮交换后节点就"知道"周围的情况。
- **预测头 (Head)**: 网络末端为每个任务单独设的小输出模块（这里共 4 个）。
- **IoU**: 两个框的重叠程度（0 = 完全不重叠，1 = 完全重合）。
- **MSE**: 平均误差，越小越好。
- **Precision / Recall / F1**: 精确率 = 报出的元素里有多少是对的；召回 = 屏幕上真实存在的元素里抓到多少；F1 = 两者的综合分。
- **TP / FP / FN**: 真阳 = 正确检出的；误报 = 错误报出的；漏检 = 该报没报的。
- **自监督**: 从数据自身学，不需要人工标注。
- **掩码 (Mask)**: 训练时把部分真实元素藏起来，让模型练习补全。
- **消融 (Ablation)**: 每次去掉一个组件，看它对结果的影响。
- **基线 (Baseline)**: 拿来对比的简单参考方法。
- **冻结 (Frozen)**: 训练时保持不变（这里指视觉特征提取器不参与训练）。
- **AUROC**: 衡量模型排序能力的标准分数：1 = 完美，0.5 = 等于瞎猜。
- **端到端 (End-to-end)**: 完整流程在真实数据上的整体表现。

---

## 1. 动机与问题 (Motivation)

**Q1. Why is this problem important? / 这个问题为什么重要？**
EN: Screen readers, automated UI testers and software agents must know what is on screen and where; wrong or missing elements break all of them. On-device constraints force small VLMs, which make systematic detection errors.
CN: 屏幕阅读器、自动化 UI 测试、软件智能体都需要知道屏幕上"有什么、在哪"；漏报和错框会全部破坏。端侧限制只能用轻量 VLM，而它们有系统性检测错误。

**Q2. What exactly goes wrong with lightweight VLMs? / 轻量 VLM 具体错在哪？**
EN: Two failure modes. Element omission: 38% of the elements actually on screen are never reported, small icons, dividers, nested containers. And misalignment: the rectangles the model draws around elements are off by tens of pixels.
CN: 两类失败模式。漏报：38% 的真实可见元素完全没报出，小图标、分隔线、嵌套容器；错位：模型画出的元素框偏移数十像素。

**Q3. Why not just fine-tune a bigger VLM? / 为什么不直接微调更大的 VLM？**
EN: Bigger models are exactly what on-device constraints forbid; the errors come from compressing a large model into a small one, so retraining a small one still leaves the same failure modes.
CN: 端侧约束恰恰不允许更大模型；错误源于"大模型压缩成小模型"，所以再微调小模型仍会残留同类失败模式。

**Q4. Why not add a second object detector as a cascade? / 为什么不再加一个检测器级联？**
EN: A cascade, stacking a second detection model after the first, adds another heavy component with its own training cost and its own false positives; our 57K-parameter graph post-processor is far cheaper and reuses the VLM's output.
CN: 级联，也就是在第一个检测模型后面再叠一个，是又一个重型组件，有自己的训练成本和误报；我们 57K 参数的后处理网络便宜得多，且直接复用 VLM 输出。

**Q5. Why not just prompt-engineer the VLM? / 为什么不做提示词工程？**
EN: Prompting can't repair box coordinates or invent missing elements. These are perception errors, not instruction-following errors.
CN: 提示词修不了框坐标，也变不出漏掉的元素；这是感知错误，不是指令遵循错误。

---

## 2. 方法设计 (Method)

**Q6. Why model a screen as a graph? / 为什么把屏幕建模成图？**
EN: Screens have variable element counts, and the information that matters is relational, how elements relate to each other (aligned, contained, spaced). A graph expresses that structure directly and works for any number of elements.
CN: 屏幕元素数量不定，而有用的信息本质是"元素之间的关系"（对齐、包含、间距）。图能直接表达这种关系，且与元素数量无关。

**Q7. Why heterogeneous and bipartite? / 为什么是异构二分图？**
EN: Two kinds of nodes, elements and rules, with connections only between the two kinds. That lets each rule collect its participating elements, then each element collects the rules it shares, forming its spatial neighborhood.
CN: 两类节点，元素和规则，连线只出现在两类之间。规则先收集参与它的元素，元素再收集共享的规则，形成自己的空间邻域。

**Q8. Why GraphSAGE instead of GAT / GCN / Transformer? / 为什么用 GraphSAGE 而不是 GAT/GCN/Transformer？**
EN: GraphSAGE is a simple graph network where each node averages its neighbors' information, cheap and easy to train. Attention-based models and Transformers, which let every item look at every other item, add capacity we don't need and cost more on-device.
CN: GraphSAGE 就是"每个节点把邻居信息取平均"的简单图网络，便宜好训。注意力模型和 Transformer（让每个元素和所有其他元素互看）增加的容量我们用不上，还更贵。

**Q9. Why two rounds (hops) of message passing? / 为什么只做两轮信息交换？**
EN: Round one (element→rule): each rule summarizes its elements; round two (rule→element): each element learns its spatial neighborhood. That is enough to propagate local structure; more rounds add cost with little gain.
CN: 第一轮让规则汇总元素，第二轮让元素拿到空间邻域；局部结构两轮就够，更多轮只增加开销。

**Q10. What are the ten constraint types and how are they extracted? / 十种约束是什么、怎么提取？**
EN: Six alignment rules (left/right/top/bottom and center alignment), plus spacing, containment, grid membership, and same-size. Each is a pairwise check on normalized boxes with a small tolerance, a small allowance so near-misses still count.
CN: 六种对齐规则（左右上下、中心对齐），加间距、包含、网格、等尺寸。每条都是对归一化框的两两检查，带一个小容差，允许轻微偏差，差不多的也算成立。

**Q11. Why are rules nodes rather than edges? / 为什么规则是节点而不是连线？**
EN: Rule nodes carry learned internal values, receive messages, and let the violation module judge each rule's status, which is richer than static labels on edges.
CN: 规则节点有可学习的内部数值、能收消息，违反检测模块也能逐个判断规则状态，这比静态的连线属性表达力强。

**Q12. Why four heads trained jointly? / 为什么四个输出模块一起训练？**
EN: They share one feature extractor and reinforce each other; the experiment shows joint training keeps all four strong, while training only one task lets the self-supervised part collapse.
CN: 它们共享一个特征提取器、互相促进；实验显示一起训练四个都强，只训一个任务会让自监督部分塌缩。

**Q13. Why self-supervised element completion (masking)? / 为什么用自监督补全（掩码）？**
EN: Missing elements are the core failure mode, but we have no labels for what the VLM should have said, so we randomly hide ground-truth elements during training and let the model propose them, simulating omissions for free.
CN: 漏元素是核心失败模式，但我们没有"VLM 本该说什么"的标签——所以训练时随机藏起真实元素、让模型补全，免费模拟漏报。

**Q14. Why mask 20–80%? / 为什么掩码率 20–80%？**
EN: It trains the model across difficulty levels, and results show the gains grow with the drop ratio: the more structure is missing, the more the graph exploits it (+56% IoU at high drop).
CN: 覆盖各种难度；结果显示增益随掩码率增大——缺得越多，图结构利用得越充分（高掩码率 +56% 重叠度）。

**Q15. What does existence scoring do? / 存在性打分是干什么的？**
EN: A module that scores whether each detection is real or hallucinated, imagined by the VLM, meant to filter false positives. To be fair, on real VLM output its filtering value is limited, so the main end-to-end gains come from completion.
CN: 给每个检测打分，判断是真的还是 VLM 想象出来的幻觉，用于过滤误报。坦白说，在真实输出上过滤价值有限——端到端主要增益来自补全。

**Q16. Why only 5-d structural features? / 为什么只用 5 维结构特征？**
EN: Four normalized coordinates plus normalized area are enough to describe geometric structure, cost almost nothing, and keep the model tiny; visual features are an optional add-on.
CN: 四个归一化坐标加归一化面积足够描述几何结构，几乎零成本，模型保持极小；视觉特征只是可选项。

**Q17. Why frozen DINOv2 and cross-attention fusion? / 为什么用冻结的 DINOv2 + 注意力融合？**
EN: DINOv2 is a pre-trained image feature extractor; keeping it frozen, not updated during training, gives visual grounding for free. Cross-attention, where the structure looks up relevant visual information, beats simple concatenation on proposal error (−18~22%).
CN: DINOv2 是预训练的图像特征提取器；保持冻结（训练时不变）免费获得视觉依据。用注意力机制（结构特征去视觉特征里查相关部分）比简单拼接好，补框误差降约两成。

---

## 3. 实验与结果 (Experiments)

**Q18. What are the main results? / 核心结果是什么？**
EN: 90–94% accuracy on judging whether layout rules are broken (RICO), +56% better box overlap than a copy-the-nearest-element baseline on completion, and +2.0pp F1 on 200 real Qwen3-VL Flash screenshots. 106 missed elements recovered, under 1 ms per screenshot with 57K parameters.
CN: 判断布局规则是否被破坏的正确率 90–94%；补全时框的重叠度比"复制最近元素"的简单方法好 56%；200 张真实 Qwen3-VL Flash 截图综合分 F1 +2.0pp——恢复 106 个漏检元素；57K 参数、每图 <1 ms。

**Q19. Why does removing ALIGNMENT rules help (+3.1pp)? / 为什么去掉 ALIGNMENT 反而提升？**
EN: Alignment is almost visible from the raw box coordinates, so those rules add redundant, often-broken noise; containment is the most valuable because it encodes hierarchy, an element living inside another, which coordinates alone don't show.
CN: 对齐信息从框坐标几乎就能直接看出，这些规则是冗余且常被违反的噪声；CONTAINMENT 最有用，因为它编码了坐标看不出的层级关系（一个元素在另一个里面）。

**Q20. Why does joint training beat single-task training? / 为什么一起训练优于只训一个任务？**
EN: Single-task training lets the self-supervised part collapse: training only the completion task drops rule-judging accuracy to 0.489, while joint training keeps both at 0.876 / 0.051.
CN: 只训一个任务会让自监督部分塌缩——只训补全时，规则判断准确率掉到 0.489；一起训练两者都保持 0.876 / 0.051。

**Q21. What is the nearest-neighbor baseline for completion? / 补全的最近邻基线是什么？**
EN: Copying the box of the nearest kept element as the proposal, a simple but strong heuristic that the graph model still beats at high drop ratios.
CN: 用最近保留元素的框直接复制作为提议——方法简单但很强，高掩码率下图模型仍能胜过它。

**Q22. Why does recall improve more than precision? / 为什么召回提升比精确率多？**
EN: Completion deliberately proposes new elements: it adds correct catches (recall up), but some proposals are wrong (a few new false positives), so precision, the share of our reports that are real, moves less.
CN: 补全刻意新增元素：正确抓到的变多（召回上升），但部分提议是错的（少量新误报），所以精确率（报出的东西里真元素占比）提升较少。

**Q23. Are the +84 new false positives a problem? / 新增 84 个误报是问题吗？**
EN: Recall was the bottleneck (0.235 before), and the net F1 gain is still +2.0pp; false positives are also cheaper to filter downstream than missed elements are to recover.
CN: 召回才是瓶颈（此前仅 0.235），净综合分仍 +2.0pp；而且误报下游过滤的成本比漏检更低。

**Q24. How fast is it, and is it really deployable on-device? / 有多快？真能端侧部署吗？**
EN: 57K parameters and under 1 ms per screenshot. The graph is built from the VLM's own output, no image re-processing needed, so it fits the on-device budget.
CN: 57K 参数、每图 <1 ms——图直接由 VLM 的输出构建，无需重新处理图像，满足端侧预算。

**Q25. Is 500 graphs / 200 screenshots enough training data? / 500 图 / 200 截图够吗？**
EN: Learning structure is data-efficient, and fine-tuning on real data still adds +2.1pp completion F1; more data would help mainly the harder cross-domain cases.
CN: 学结构数据效率高，真实数据微调还能再 +2.1pp 补全分；更多数据主要对更难的跨域场景有帮助。

**Q26. How robust are the numbers? / 数字有多稳？**
EN: Reported with five independent training runs and error bars; the spread is small (e.g., ±0.019 for rule-judging accuracy), and the qualitative before/after visualizations match the metrics.
CN: 用 5 次独立训练并画了误差范围；波动很小（如规则判断准确率 ±0.019），前后对比可视化也与指标一致。

**Q27. How is rule-judging accuracy measured? / 规则判断准确率怎么测？**
EN: Against ground truth: each extracted rule is classified as satisfied or broken, and accuracy is the fraction correct over all rules in the test screenshots.
CN: 对照人工标注：每个提取出的规则被分类为满足或违反，准确率即测试截图全部规则中的正确比例。

---

## 4. 局限与未来 (Limitations & Future Work)

**Q28. What are the main limitations? / 主要局限是什么？**
EN: It inherits the VLM's initial detections, so purely semantic errors are not fixed; the existence-scoring module gives limited filtering on real data; and desktop/web generalization is untested.
CN: 继承 VLM 的初始检测，纯语义错误修不了；存在性打分在真实数据上过滤价值有限；桌面/网页泛化未验证。

**Q29. Does it work on ScreenSpot (cross-domain)? / 跨域（ScreenSpot）能用吗？**
EN: Only the confidence model has been cross-domain tested, and its ranking score drops to 0.554, close to guessing. So cross-domain transfer is currently a limitation and an explicit future-work item, not a claim.
CN: 目前只有置信度模型做过跨域验证，排序分数掉到 0.554，接近瞎猜。所以跨域迁移是当前局限和明确的未来工作项，不作为卖点。

**Q30. What is the future work? / 未来工作？**
EN: Attention-based visual fusion (already −18~22% proposal error in early experiments), adapting to ScreenSpot, and using temporal context, information from past screens, to distinguish missed elements from elements hidden by the current UI state.
CN: 注意力视觉融合（早期实验补框误差已降约两成）、ScreenSpot 域适应、以及利用时序上下文（历史屏幕信息）区分"漏检"和"被当前界面状态隐藏"。

---

## 5. 发散深挖 (Divergent / Tough Questions)

**Q31. Why not just feed all boxes into a Transformer? / 为什么不把所有框直接喂给 Transformer？**
EN: A Transformer, a model where every item attends to every other item, has no built-in layout structure and costs far more parameters; the bipartite graph encodes exactly the structure (rules as nodes) we want the model to reason over.
CN: Transformer（每个元素与所有其他元素互相关注的模型）没有内置布局结构、参数多得多；二分图直接编码了我们想让模型推理的结构（规则即节点）。

**Q32. Could this transfer to web accessibility, documents or CAD? / 能迁移到网页无障碍、文档或 CAD 吗？**
EN: In principle yes, any domain with well-structured layouts shares alignment/containment/grid patterns; the rule vocabulary would need adaptation, which is the ScreenSpot direction.
CN: 原则上可以——任何结构良好的布局都有对齐/包含/网格规律；规则词汇需要调整，这正是 ScreenSpot 方向的思路。

**Q33. How is this different from existing layout/GUI graph works? / 和已有布局/GUI 图网络工作有何不同？**
EN: Most prior work generates or understands layouts; we are the first to frame post-correction of VLM detections as a bipartite constraint graph with self-supervised completion.
CN: 前人大多做布局生成或理解；我们是首个把 VLM 检测的后修正建模为"二分约束图 + 自监督补全"的工作。

**Q34. What is the biggest risk to your results? / 结果最大的风险是什么？**
EN: The existence-scoring module's weak performance on real data and the small end-to-end evaluation set (200 screenshots); both are acknowledged in the report, and the headline gains come from completion, which is consistently measured.
CN: 存在性打分在真实数据上偏弱、端到端评估集只有 200 张；报告里都如实说明了，而核心增益来自补全，这一项测量是一致的。

**Q35. What would you do with three more months? / 再多三个月会做什么？**
EN: ScreenSpot adaptation, temporal context modeling, and shrinking the pipeline further (e.g., exporting to a device-friendly format like ONNX) toward a real on-device deployment.
CN: ScreenSpot 域适应、时序上下文建模、以及进一步压缩管线（如导出成 ONNX 这类设备友好格式）走向真实端侧部署。

**Q36. How does this fit inside an AI agent loop? / 这在智能体流程里处于什么位置？**
EN: A cheap post-processor between the VLM parser and the action planner: agents act on corrected element lists, so fewer missed elements means fewer wrong actions.
CN: 是 VLM 解析与动作规划之间的廉价后处理：智能体基于修正后的元素列表行动，漏检越少，错误动作越少。

**Q37. Why the 0.1 center-distance matching threshold? / 为什么匹配阈值取中心距 0.1？**
EN: A lenient threshold reflects the VLM's real misalignment (tens of pixels); a stricter one would make the already-low recall look even worse and punish the correction unfairly.
CN: 宽松阈值符合 VLM 真实的数十像素偏移；阈值更严会让本就偏低的召回更难看，对修正网络不公平。

**Q38. What is the novelty in one sentence? / 一句话说清创新点？**
EN: A 57K-parameter bipartite constraint graph that repairs and completes lightweight-VLM GUI detections with four jointly trained modules, self-supervised via element hiding.
CN: 用 57K 参数的二分约束图修复并补全轻量 VLM 的 GUI 检测，四个模块联合训练、通过隐藏元素自监督。
