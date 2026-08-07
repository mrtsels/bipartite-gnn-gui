# Poster Q&A 准备 — Heterogeneous Bipartite GNN for GUI Structure Error Correction

中英对照 · 问题尽可能发散，回答保持简练（1–2 句）。
展示时间: 2026-08-14 (Fri) 9:30–12:00 · ERBLT foyer (8/F ERB) · 9:20 前签到 · 10–15 分钟 + Q&A

---

## 0. 关键数字速查 (Key Numbers Cheat-Sheet)

| 项目 | 数值 |
|---|---|
| 模型规模 | 57K 参数，<1 ms/图 |
| 编码器 | 2 层 bipartite GraphSAGE，hidden 128，AdamW + cosine |
| 特征 | 5-d（4 归一化坐标 + 归一化面积），可选 frozen DINOv2 (192-d) |
| 约束 | 10 种，平均 37.3 条/图 |
| 数据 | RICO（训练/评估）、ScreenSpot（跨域）；端到端 200 张截图，4,789 GT / 2,947 VLM 预测，中心距匹配阈值 0.1 |
| VLM 问题 | 38% 元素漏报；框偏移数十像素 |
| 违反检测 | RICO 90–94%（control 0.908）；去 ALIGNMENT +3.1pp；去 CONTAINMENT −1.9pp |
| 训练目标消融 | joint 0.876 / MSE 0.051；violation-only 0.898 / 0.116；proposal-only 0.489 / 0.051 |
| 补全 | 高 drop 时较 NN 基线 +56% IoU |
| 端到端 (200 图) | P 0.382→0.393 (+1.1pp)，R 0.235→0.257 (+2.2pp)，F1 0.291→0.311 (+2.0pp)；TP +106，FP +84，FN −106 |
| 视觉融合 | cross-attention 较拼接 Proposal MSE −18~22% |
| 损失权重 | w_c=1.0, w_v=0.5, w_e=0.5, w_p=0.5 |

---

## 1. 动机与问题 (Motivation)

**Q1. Why is this problem important? / 这个问题为什么重要？**
EN: Screen readers, automated UI testers and software agents must know what is on screen and where; wrong or missing elements break all of them. On-device constraints force small VLMs, which make systematic detection errors.
CN: 屏幕阅读器、自动化 UI 测试和软件智能体都需要知道屏幕"有什么、在哪"；漏报和错框会全部破坏。端侧限制只能用轻量 VLM，而它们有系统性检测错误。

**Q2. What exactly goes wrong with lightweight VLMs? / 轻量 VLM 具体错在哪？**
EN: Two failure modes: element omission (38% of visible elements are never reported — small icons, dividers, nested containers) and misalignment (boxes off by tens of pixels).
CN: 两类失败模式：元素漏报（38% 的可见元素完全未报出——小图标、分隔线、嵌套容器）和错位（框偏移数十像素）。

**Q3. Why not just fine-tune a bigger VLM? / 为什么不直接微调更大的 VLM？**
EN: Bigger models are exactly what on-device constraints forbid; the errors come from compressing a large model into a small one, so retraining a small one still leaves the same failure modes.
CN: 端侧约束恰恰不允许更大模型；错误源于"大模型压缩成小模型"，所以再微调小模型仍会残留同类失败模式。

**Q4. Why not add a second object detector as a cascade? / 为什么不再加一个检测器级联？**
EN: A second detector is another heavy component with its own training cost and its own false positives; our 57K-parameter graph post-processor is far cheaper and reuses the VLM's output.
CN: 第二个检测器是又一个重型组件，有自己的训练成本和误报；我们 57K 参数的后处理网络便宜得多，且直接复用 VLM 输出。

**Q5. Why not just prompt-engineer the VLM? / 为什么不做提示词工程？**
EN: Prompting can't repair box coordinates or invent missing elements; these are perception errors, not instruction-following errors.
CN: 提示词修不了框坐标，也变不出漏掉的元素；这是感知错误，不是指令遵循错误。

---

## 2. 方法设计 (Method)

**Q6. Why model a screen as a graph? / 为什么把屏幕建模成图？**
EN: Screens have variable element counts, and the information that matters is relational (alignment, containment, spacing). A graph expresses that structure directly and is size-agnostic.
CN: 屏幕元素数量不定，而有用的信息本质是关系性的（对齐、包含、间距）。图能直接表达这种结构，且与元素数量无关。

**Q7. Why heterogeneous and bipartite? / 为什么是异构二分图？**
EN: Two distinct node types (elements vs constraints) with edges only across partitions — this lets each constraint node aggregate its participating elements, then elements re-aggregate the constraints they share.
CN: 两类节点（元素 vs 约束），边只跨分区——约束节点先聚合参与的元素，元素再聚合共享的约束，形成"空间邻域"。

**Q8. Why GraphSAGE instead of GAT / GCN / Transformer? / 为什么用 GraphSAGE 而不是 GAT/GCN/Transformer？**
EN: Mean-aggregation GraphSAGE is inductive, cheap and needs no attention machinery for tiny graphs; attention and Transformers add capacity we don't need and hurt the on-device budget.
CN: 均值聚合的 GraphSAGE 是归纳式的、便宜，小图上不需要注意力机制；Transformer 的容量用不上，还破坏端侧预算。

**Q9. Why two hops of message passing? / 为什么只做两跳消息传递？**
EN: Hop 1 (element→constraint) lets each constraint summarize its elements; hop 2 (constraint→element) gives each element its spatial neighborhood. That is enough to propagate local structure; more hops add cost with little gain.
CN: 第一跳让约束汇总其元素，第二跳让元素拿到空间邻域；局部结构两跳就够，更多跳只增加开销。

**Q10. What are the ten constraint types and how are they extracted? / 十种约束是什么、怎么提取？**
EN: Six alignment predicates (ALIGN_LEFT/RIGHT/TOP/BOTTOM, CENTER_X/Y), plus SPACING, CONTAINMENT, GRID and SAME_SIZE — each a pairwise predicate holding within a tolerance ε on normalized boxes.
CN: 六种对齐谓词（左右上下对齐、中心对齐），加间距、包含、网格、等尺寸——都是归一化框上带容差 ε 的两两谓词。

**Q11. Why are constraints nodes rather than edges? / 为什么约束是节点而不是边？**
EN: Constraint nodes carry learnable state, get aggregated messages, and let the violation head classify each constraint's status — richer than static edge attributes.
CN: 约束节点有可学习的状态、能收消息，违反检测头也能对每个约束做分类——比静态边属性表达力强。

**Q12. Why four heads trained jointly? / 为什么四个头联合训练？**
EN: They share one encoder and reinforce each other; the ablation shows joint training keeps all heads strong, while single-objective training lets the unsupervised head collapse.
CN: 共享编码器、互相促进；消融显示联合训练让所有头都强，单目标训练会让无监督头塌缩。

**Q13. Why self-supervised element completion (masking)? / 为什么用自监督补全（掩码）？**
EN: Missing elements are the core failure mode, but we have no labels for "what the VLM should have said" — so we mask ground-truth elements and train the model to propose them, simulating omissions for free.
CN: 漏元素是核心失败模式，但我们没有"VLM 本该说什么"的标签——所以随机掩码 GT 元素、让模型补全，免费模拟漏报。

**Q14. Why mask 20–80%? / 为什么掩码率 20–80%？**
EN: It trains the model across difficulty levels, and results show the gains grow with the drop ratio — the more structure is missing, the more the graph exploits it (+56% IoU at high drop).
CN: 覆盖各种难度；结果显示增益随掩码率增大——缺得越多，图结构利用得越充分（高掩码率 +56% IoU）。

**Q15. What does existence scoring do? / 存在性打分是干什么的？**
EN: A BCE head scoring whether each detection is real or hallucinated, meant to filter false positives. Honest caveat: on real VLM output its filtering value is limited — the main end-to-end gains come from completion.
CN: 用 BCE 头给每个检测打分，判断是真元素还是幻觉，用于过滤误报。如实说：在真实 VLM 输出上过滤价值有限——端到端主要增益来自补全。

**Q16. Why only 5-d structural features? / 为什么只用 5 维结构特征？**
EN: Four normalized coordinates plus normalized area are enough for geometric structure, cost almost nothing, and keep the model tiny; visual features are an optional add-on.
CN: 四个归一化坐标加归一化面积足够表达几何结构，几乎零成本，模型保持极小；视觉特征只是可选项。

**Q17. Why frozen DINOv2 and cross-attention fusion? / 为什么用冻结 DINOv2 + 交叉注意力？**
EN: Frozen features give visual grounding without extra training cost; cross-attention (struct as query, visual as key/value) beats simple concatenation on proposal MSE (−18~22%).
CN: 冻结特征提供视觉依据又不多花训练成本；交叉注意力（结构作 query、视觉作 KV）在提议 MSE 上比简单拼接好 −18~22%。

---

## 3. 实验与结果 (Experiments)

**Q18. What are the headline results? / 核心结果是什么？**
EN: 90–94% violation accuracy on RICO, +56% IoU over nearest-neighbor completion at high drop ratios, and +2.0pp F1 on 200 real Qwen3-VL Flash screenshots — 106 missed elements recovered, under 1 ms per screenshot with 57K parameters.
CN: RICO 上违反检测 90–94%；高掩码率下补全较最近邻 +56% IoU；200 张真实 Qwen3-VL Flash 截图 F1 +2.0pp——恢复 106 个漏检元素；57K 参数、每图 <1 ms。

**Q19. Why does removing ALIGNMENT constraints help (+3.1pp)? / 为什么去掉 ALIGNMENT 反而提升？**
EN: Alignment is almost recoverable from raw box coordinates, so those constraints add redundant, often-violated noise; CONTAINMENT is the most valuable because it encodes hierarchy that coordinates alone don't show.
CN: 对齐信息几乎可从框坐标直接看出，这些约束是冗余且常被违反的噪声；CONTAINMENT 最有用，因为它编码了坐标看不出的层级关系。

**Q20. Why does joint training beat single-objective training? / 为什么联合训练优于单目标？**
EN: Single-objective training lets the unsupervised head collapse — proposal-only training drops violation accuracy to 0.489 while joint keeps both at 0.876 / 0.051.
CN: 单目标会让无监督头塌缩——只训练补全头时违反准确率掉到 0.489，联合训练两者都保持 0.876 / 0.051。

**Q21. What is the nearest-neighbor baseline for completion? / 补全的最近邻基线是什么？**
EN: Copying the box of the nearest kept element as the proposal — a strong, cheap heuristic that the graph model still beats at high drop ratios.
CN: 用最近保留元素的框直接复制作为提议——一个很强且廉价的启发式，但高掩码率下图模型仍能胜过它。

**Q22. Why does recall improve more than precision? / 为什么召回提升比精确率多？**
EN: Completion deliberately proposes new elements: it adds true positives (recall up), but some proposals are wrong (a few new false positives), so precision moves less.
CN: 补全刻意新增元素：真阳增加（召回上升），但部分提议是错的（少量新误报），所以精确率提升较少。

**Q23. Are the +84 new false positives a problem? / 新增 84 个误报是问题吗？**
EN: Recall is the bottleneck (0.235 before), and the net F1 gain is still +2.0pp; false positives are also cheaper to filter downstream than missed elements are to recover.
CN: 召回才是瓶颈（此前仅 0.235），净 F1 仍 +2.0pp；而且误报下游过滤的成本比漏检更难补救。

**Q24. How fast is it, and is it really deployable on-device? / 有多快？真能端侧部署吗？**
EN: 57K parameters and under 1 ms per screenshot — the graph is built from the VLM's own JSON, no image re-processing needed, so it fits the on-device budget.
CN: 57K 参数、每图 <1 ms——图直接由 VLM 的 JSON 构建，无需重新处理图像，满足端侧预算。

**Q25. Is 500 graphs / 200 screenshots enough training data? / 500 图 / 200 截图够吗？**
EN: Structure learning is data-efficient, and real-data fine-tuning still adds +2.1pp completion F1; more data would help mainly the harder cross-domain cases.
CN: 结构学习数据效率高，真实数据微调还能再 +2.1pp 补全 F1；更多数据主要对更难的跨域场景有帮助。

**Q26. How robust are the numbers? / 数字有多稳？**
EN: Reported with 5 seeds and error bars; violation accuracy std is small (e.g., ±0.019 for joint), and the qualitative before/after visualizations match the metrics.
CN: 用 5 个种子并画了误差棒；违反准确率标准差很小（如 joint ±0.019），前后对比可视化也与指标一致。

**Q27. How is violation accuracy measured? / 违反准确率怎么测？**
EN: Against ground-truth constraint labels: each extracted constraint is classified as satisfied or violated, and accuracy is the fraction correct over all constraints in the test screenshots.
CN: 对照 GT 约束标签：每个提取的约束被分类为满足或违反，准确率即测试截图全部约束中的正确比例。

---

## 4. 局限与未来 (Limitations & Future Work)

**Q28. What are the main limitations? / 主要局限是什么？**
EN: It inherits the VLM's initial detections (no purely semantic errors fixed), the existence head gives limited filtering on real data, and desktop/web generalization is untested.
CN: 继承 VLM 的初始检测（纯语义错误修不了）；存在性头在真实数据上过滤价值有限；桌面/网页泛化未验证。

**Q29. Does it work on ScreenSpot (cross-domain)? / 跨域（ScreenSpot）能用吗？**
EN: Only the confidence model has been cross-domain tested, and AUROC drops to 0.554 — so cross-domain transfer is currently a limitation and an explicit future-work item, not a claim.
CN: 目前只有置信度模型做过跨域验证，AUROC 掉到 0.554——跨域迁移是当前局限和明确的未来工作项，不作为卖点。

**Q30. What is the future work? / 未来工作？**
EN: Visual fusion with cross-attention (already −18~22% proposal MSE in early experiments), domain adaptation to ScreenSpot, and temporal context to distinguish missing elements from elements hidden by UI state.
CN: 交叉注意力视觉融合（早期实验提议 MSE 已 −18~22%）、ScreenSpot 域适应、以及用时序上下文区分"漏检"和"被 UI 状态隐藏"。

---

## 5. 发散深挖 (Divergent / Tough Questions)

**Q31. Why not just feed all boxes into a Transformer? / 为什么不把所有框直接喂给 Transformer？**
EN: A Transformer has no relational prior and costs far more parameters; the bipartite graph encodes the exact structure (constraint nodes) we want the model to reason over.
CN: Transformer 没有关系先验、参数多得多；二分图直接编码了我们想让模型推理的结构（约束节点）。

**Q32. Could this transfer to web accessibility, documents or CAD? / 能迁移到网页无障碍、文档或 CAD 吗？**
EN: In principle yes — any domain with well-structured layouts shares alignment/containment/grid patterns; the constraint vocabulary would need adaptation, which is the ScreenSpot direction.
CN: 原则上可以——任何结构良好的布局都有对齐/包含/网格规律；约束词汇需要调整，这正是 ScreenSpot 方向的思路。

**Q33. How is this different from existing layout/GUI GNN works? / 和已有布局/GUI 图网络工作有何不同？**
EN: Most prior work generates or understands layouts; we are the first to frame post-correction of VLM detections as a heterogeneous bipartite constraint graph with self-supervised completion.
CN: 前人大多做布局生成或理解；我们是首个把 VLM 检测的后修正建模为"异构二分约束图 + 自监督补全"的工作。

**Q34. What is the biggest risk to your results? / 结果最大的风险是什么？**
EN: The existence head's weak performance on real data and the small end-to-end evaluation set (200 screenshots); both are acknowledged in the report, and the headline gains come from completion, which is consistently measured.
CN: 存在性头在真实数据上偏弱、端到端评估集只有 200 张；报告里都如实说明了，而核心增益来自补全，这一项测量是一致的。

**Q35. What would you do with three more months? / 再多三个月会做什么？**
EN: ScreenSpot domain adaptation, temporal context modeling, and shrinking the pipeline further (e.g., ONNX export) toward a real on-device deployment.
CN: ScreenSpot 域适应、时序上下文建模、以及进一步压缩管线（如 ONNX 导出）走向真实端侧部署。

**Q36. How does this fit inside an AI agent loop? / 这在智能体流程里处于什么位置？**
EN: A cheap post-processor between the VLM parser and the action planner: agents act on corrected element lists, so fewer missed elements means fewer wrong actions.
CN: 是 VLM 解析与动作规划之间的廉价后处理：智能体基于修正后的元素列表行动，漏检越少，错误动作越少。

**Q37. Why the 0.1 center-distance matching threshold? / 为什么匹配阈值取中心距 0.1？**
EN: A lenient threshold reflects the VLM's real misalignment (tens of pixels); a stricter one would make the already-low recall look even worse and punish the correction unfairly.
CN: 宽松阈值符合 VLM 真实的数十像素偏移；阈值更严会让本就偏低的召回更难看，对修正网络不公平。

**Q38. What is the novelty in one sentence? / 一句话说清创新点？**
EN: A 57K-parameter heterogeneous bipartite constraint graph that repairs and completes lightweight-VLM GUI detections with four jointly trained heads, self-supervised via element masking.
CN: 用 57K 参数的异构二分约束图修复并补全轻量 VLM 的 GUI 检测，四个头联合训练、通过元素掩码自监督。
