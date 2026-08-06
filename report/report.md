# Heterogeneous Bipartite Graph Neural Networks for GUI Structure Error Correction

*A Post-Correction Framework for Lightweight Vision-Language Models*

Alex Licheng Xie, Summer Research Intern Project, August 2026

## Abstract

Computer programs that interpret a phone screen, such as assistive tools for visually impaired users, automated UI testers, and software agents, all need to know what is on the screen and where. Small vision-language models (VLMs) are attractive for running such tasks directly on a phone because they are fast and memory-efficient, but they make mistakes: they miss 38% of the visible interface elements, and the bounding boxes they draw around the elements they do find are often misaligned with reality. This report presents a post-correction framework that fixes these mistakes without retraining the VLM or adding a second detector. The underlying assumption is that well-designed interfaces obey spatial rules: elements align, sit inside containers, and follow consistent spacing. We encode these rules as a heterogeneous bipartite graph: one set of nodes for the detected elements, one set for the spatial constraints among them, and edges only between the two sets. A GraphSAGE network passes messages along this graph so that every element's prediction is informed by its spatial neighbours, then predicts per-element coordinate corrections, flags violated constraints, scores whether each detection is real or hallucinated, and proposes elements the VLM missed entirely. On the RICO dataset of real Android screenshots, the model detects violated constraints at 90–94% accuracy and, when a large fraction of elements is missing, completes the layout with up to 56% better IoU than a nearest-neighbor baseline. End-to-end on 200 real screenshots with a Qwen3-VL Flash front end, the corrected pipeline recovers 106 previously missed elements, raising F1 by 2.0 points and recall by 2.2 points. The correction module runs after the VLM, does not modify the underlying model, and adds less than one millisecond of inference per screenshot.

## 1. Introduction

To understand a smartphone interface, a program must first answer a basic question: *what is on this screen, and where is it?* Every application (from screen readers for visually impaired users, to automated UI testers, to software agents that operate a phone on a user's behalf) depends on accurate knowledge of the interface elements (buttons, text fields, icons, images) and their positions.

### 1.1 Why Lightweight Vision-Language Models

Vision-language models (VLMs) are neural networks that can describe images in natural language and, given a screenshot, can list the interface elements and their bounding boxes. Large VLMs with 7 billion parameters or more are quite accurate at this task, but they are too slow and too memory-hungry to run on a phone. Lightweight VLMs under 3 billion parameters (such as Qwen3-VL Flash [1] and MiniMax-VL-01) are fast enough for on-device use, which makes them attractive for real-time applications.

### 1.2 Two Systematic Failure Modes

Evaluation on real screenshots shows that lightweight VLMs make two kinds of mistakes:

- **Element omission.** 38% of the visible elements are never reported: on 200 real RICO screenshots, the front-end VLM reports only 2,947 of 4,789 ground-truth elements (Section 4.2). Small icons, dividers, and nested containers are missed most often.
- **Misalignment.** The bounding boxes around detected elements can deviate by tens of pixels from where the element actually is, which breaks down downstream reasoning about layout.

These are not random glitches; they are systematic consequences of compressing a large model into a small one. The natural response, training a bigger model, is precisely what on-device constraints forbid.

### 1.3 Our Approach in One Paragraph

Early experiments shaped this design. We first attempted to correct coordinates directly, training an MLP on per-element features. The results were consistently poor: on simulated Gaussian noise the model never beat the no-op baseline, and on real VLM output (Qwen3-VL Plus and Flash) the positional errors were already small enough that there was little to correct. A local LLaVA-7B model produced too few detections per screenshot for the graph to carry useful structure. These failures pointed to a common cause: single-element features carried too little context for coordinate prediction, regardless of how the regressor was trained.

The alternative explored in this report is to exploit structure. GUI screenshots are laid out according to spatial rules such as alignment, containment, spacing, and grid membership. These rules are independent evidence that a VLM looking at pixels alone cannot use: if every element in a row is left-aligned except one, the odd one out is probably misplaced, regardless of what the pixels look like. We formalize these rules as a graph: element nodes on one side, constraint nodes on the other, and edges connecting an element to every constraint it participates in. A graph neural network (GraphSAGE) then reasons over this structure and outputs corrected coordinates, violated constraints, per-element confidence, and (for the omission problem) entirely new element proposals. Because the correction runs after the VLM, it does not modify the underlying model: the same correction network works behind any VLM that outputs elements with bounding boxes.

The rest of this report is organized as follows. Section 2 summarizes existing solutions and their limitations. Section 3 describes the proposed method in detail. Section 4 presents experiments and results. Section 5 discusses limitations and future work, and Section 6 concludes.

## 2. Existing Solutions and Their Limitations

### 2.1 Fine-Tuning Larger Models

The most direct remedy for VLM mistakes is to fine-tune a larger VLM (7B+) on GUI data, or to fine-tune the lightweight model on domain-specific screenshots. This improves accuracy but has two costs: it requires labeled GUI data, and (more fundamentally for the on-device use case) the improved model is larger, slower, or both. Fine-tuning also bakes in a specific VLM; swapping the front-end model means retraining everything.

### 2.2 Object-Detector Cascades

A different approach adds a dedicated object detector (for example, a DETR [2] model) in front of or alongside the VLM. Detectors are accurate but add a second network to run on-device, increasing latency and memory, and require their own training data. The cascade approach treats the GUI as a generic object-detection problem, discarding the strong spatial structure that all well-designed interfaces share.

### 2.3 Generative Layout Models

LayoutGAN [3] and LayoutTransformer [4] learn layout priors and can *generate* new layouts. They are generative: given random noise or partial structure, they produce a layout. Our problem is different: we receive a noisy layout and must *repair* it, but the idea of a learned layout prior is shared. We make the prior explicit and interpretable (ten spatial constraint types) rather than implicit in a generative model.

### 2.4 Why a Correction Network

The framework proposed here differs from the approaches above in one respect: it never generates pixels or detections from scratch, and it never modifies the VLM. It *post-processes* the VLM's noisy output using structural reasoning. This choice was motivated by the on-device constraint: the VLM is the expensive component, and any fix that forces a larger model or an additional detector undoes the reason for using a lightweight VLM in the first place. Post-processing keeps the VLM unchanged, adds a small graph network (57K parameters), and works with any VLM that outputs a list of elements with bounding boxes. The trade-off is that the correction is limited by what the VLM initially detects; elements missed entirely can only be recovered through the completion head, which is evaluated in Section 4.

## 3. Proposed Method

### 3.1 Overview

A screenshot is fed to a lightweight VLM, which produces a noisy JSON list of detected elements. We convert that list into a heterogeneous bipartite graph, run two hops of GraphSAGE message passing, and read out corrections from four prediction heads. The output is a corrected JSON with refined boxes, new proposals for missed elements, and per-element reliability scores.

### 3.2 Formalizing a Screenshot as a Graph

#### 3.2.1 Elements

Let a screenshot yield N detected elements $\mathcal{E} = \{e_i\}_{i=1}^{N}$, each with a normalized bounding box $\mathbf{b}_i = (x_1^{(i)}, y_1^{(i)}, x_2^{(i)}, y_2^{(i)}) \in [0,1]^4$ (the box coordinates, scaled so the image is a unit square) and a type label $t_i \in \mathcal{T}$ from the element taxonomy (button, text field, icon, image, etc.).

#### 3.2.2 Spatial Constraints

From these elements we extract M spatial constraints $\mathcal{C} = \{c_j\}_{j=1}^{M}$, where each constraint is a typed relationship between elements. The ten constraint types are: ALIGN_LEFT, ALIGN_RIGHT, ALIGN_TOP, ALIGN_BOTTOM, CENTER_X, CENTER_Y, SPACING, CONTAINMENT, GRID, and SAME_SIZE. Each constraint $c_j$ has a type $\tau_j$, a source index set $S_j \subseteq \{1, \dots, N\}$, and a target index set $T_j \subseteq \{1, \dots, N\}$; the constraint exists when its predicate $P_\tau(\mathbf{b}_{S_j}, \mathbf{b}_{T_j})$ holds within a tolerance $\varepsilon$. For example, ALIGN_LEFT asserts $|x_1 - x_1'| < \varepsilon$ (buttons share a left edge), CONTAINMENT asserts that one box sits inside another (an icon inside a container), and SPACING asserts consistent gaps between items.

#### 3.2.3 The Bipartite Graph

The GUI structure is a *heterogeneous bipartite graph* $G = (\mathcal{V}, \mathcal{E}_{\text{edge}}, \phi, \psi)$, where $\mathcal{V} = \mathcal{E} \cup \mathcal{C}$ is the node set, partitioned into element nodes $V_e$ ($|V_e| = N$) and constraint nodes $V_c$ ($|V_c| = M$); $\phi: \mathcal{V} \to \{0,1\}$ labels each node's partition; edges exist only between partitions; an edge $(e_i, c_j)$ connects element $e_i$ to constraint $c_j$ iff $e_i \in S_j \cup T_j$; and $\psi$ weights each edge by the normalized distance between the element and the constraint's subspace. There are no element–element or constraint–constraint edges: elements never connect directly and communicate only through the constraints they share.

### 3.3 Message Passing

A graph neural network propagates information along the edges. We use two alternating hops of GraphSAGE convolution [5], which aggregates neighboring features with a mean operation.

#### 3.3.1 Hop 1: Element to Constraint

Each constraint node collects information from the elements it involves:

$$\mathbf{h}_{c_j}^{(1)} = \sigma\!\left( \mathbf{W}_1 \cdot \operatorname{MEAN}\!\left( \left\{\mathbf{h}_{e_i}^{(0)} : (e_i, c_j) \in \mathcal{E}_{\text{edge}}\right\} \right) + \mathbf{b}_1 \right),$$

where $\mathbf{h}_{e_i}^{(0)}$ is the initial element feature vector and $\sigma$ is the ReLU activation. Intuitively, the constraint node now "knows" the aggregate state of the elements that share it.

#### 3.3.2 Hop 2: Constraint to Element

Each element node collects the updated constraint states back:

$$\mathbf{h}_{e_i}^{(2)} = \sigma\!\left( \mathbf{W}_2 \cdot \operatorname{MEAN}\!\left( \left\{\mathbf{h}_{c_j}^{(1)} : (e_i, c_j) \in \mathcal{E}_{\text{edge}}\right\} \right) + \mathbf{b}_2 \right).$$

After this hop, every element's representation carries information from all elements that share a constraint with it, its "spatial neighbourhood." Elements that share no constraint never exchange messages, which keeps the inductive bias strong and computation local.

### 3.4 Node Features

Each element's initial feature $\mathbf{h}_{e_i}^{(0)}$ is 5-dimensional: the four normalized box coordinates plus the normalized area $a_i = (x_2 - x_1)(y_2 - y_1)$. An optional frozen visual feature $\mathbf{v}_i \in \mathbb{R}^d$ (192-d ViT-Tiny or 768-d DINOv2 [6]) can be concatenated. Constraint node features embed the constraint type as a 10-d one-hot vector plus spatial statistics of the participating elements (mean pairwise distance, containment overlap ratio, alignment residual).

### 3.5 Four Prediction Heads

After message passing, four heads read out the result. The total training objective is a weighted sum

$$\mathcal{L} = w_c\,\mathcal{L}_{\text{coord}} + w_v\,\mathcal{L}_{\text{vio}} + w_e\,\mathcal{L}_{\text{exist}} + w_p\,\mathcal{L}_{\text{prop}},$$

with default weights $w_c = 1.0$, $w_v = 0.5$, $w_e = 0.5$, $w_p = 0.5$.

#### 3.5.1 Coordinate Correction

The first head predicts a per-element delta vector $\Delta\mathbf{x}_i = \operatorname{MLP}_{\text{coord}}(\mathbf{h}_{e_i}^{(2)})$, the amount by which the VLM's box should move, optimized with a smooth L1 loss:

$$\mathcal{L}_{\text{coord}} = \frac{1}{N}\sum_{i=1}^{N} \operatorname{smooth}_{L_1}(\Delta\mathbf{x}_i - \Delta\mathbf{x}_i^*),$$

where $\Delta\mathbf{x}_i^*$ is the ground-truth correction.

#### 3.5.2 Violation Detection

The second head classifies whether each constraint is violated (broken), with $\hat{v}_j = \sigma(\operatorname{MLP}_{\text{vio}}(\mathbf{h}_{c_j}^{(1)}))$:

$$\mathcal{L}_{\text{vio}} = -\frac{1}{M}\sum_{j=1}^{M}\left[ v_j^* \log \hat{v}_j + (1 - v_j^*) \log(1 - \hat{v}_j) \right],$$

where $v_j^* \in \{0,1\}$ indicates whether constraint $c_j$ is genuinely violated in the ground-truth layout. This head is what makes the model "understand" layout rules: it must learn what a broken alignment or containment looks like from the graph alone.

#### 3.5.3 Existence Scoring

The third head scores whether each detected element is real or hallucinated, predicting $\hat{e}_i$ with a binary classifier on the element embedding and training with binary cross-entropy:

$$\mathcal{L}_{\text{exist}} = -\frac{1}{N}\sum_{i=1}^{N}\left[ e_i^* \log \hat{e}_i + (1 - e_i^*) \log(1 - \hat{e}_i) \right],$$

where $e_i^*$ is the ground-truth existence label. Downstream, this score can filter false positives or rank elements by reliability.

#### 3.5.4 Element Completion

The fourth head, the one that addresses the omission problem, proposes *missing* elements. During training, we randomly drop a fraction $\rho \in [0.2, 0.8]$ of the ground-truth elements. A constraint that referenced a dropped element now has a "hole" (a dangling edge), which is exactly the signature a missing element leaves in the graph. The head predicts the missing element's box and type from the aggregated constraint embedding:

$$\hat{\mathbf{b}}_k, \hat{t}_k = \operatorname{MLP}_{\text{proposal}}(\mathbf{h}_{c_j}^{(1)}), \qquad \mathcal{L}_{\text{prop}} = \frac{1}{K}\sum_{k=1}^{K}\left[ \mathcal{L}_{\text{IoU}}(\hat{\mathbf{b}}_k, \mathbf{b}_k^*) + \alpha\,\mathrm{CE}(\hat{t}_k, t_k^*) \right],$$

where $K$ is the number of violated constraints with a missing target and $\mathcal{L}_{\text{IoU}}$ is the IoU-based box loss. Note that the training signal is derived purely by masking ground-truth layouts; the completion head is *self-supervised*: it needs no additional human annotation.

## 4. Experiments and Results

### 4.1 Experimental Setup

All models use AdamW with cosine annealing, a 128-d hidden dimension, and two-layer bipartite GraphSAGE, trained with 5 random seeds where reported. We evaluate on the RICO dataset [7] (real Android screenshots) and ScreenSpot [8]. Implementation uses PyTorch [9], PyTorch Geometric [10], NumPy [11], SciPy [12], and Matplotlib [13]; the front-end VLM is Qwen3-VL Flash [1].

### 4.2 End-to-End on Real VLM Output

We deploy the trained model behind Qwen3-VL Flash on 200 real RICO screenshots (4,789 ground-truth elements, 2,947 VLM predictions). The GNN receives the VLM's noisy elements, detects violated constraints, and proposes missing elements; proposals are merged with non-maximum suppression. We evaluate with center-distance matching at threshold 0.1 against ground truth.

| Metric | VLM only | VLM + GNN | Δ |
|--------|:--------:|:---------:|:-:|
| Precision | 0.382 | 0.393 | +1.1 pp |
| Recall | 0.235 | 0.257 | +2.2 pp |
| F1 | 0.291 | 0.311 | +2.0 pp |
| TP / FP / FN | 1,126 / 1,821 / 3,663 | 1,232 / 1,905 / 3,557 | +106 TP |

The corrected pipeline recovers 106 previously missed elements (+106 TP, +2.2 pp recall) at a cost of 84 additional false positives, improving F1 by 2.0 points. Precision also improves slightly (+1.1 pp), unlike earlier results reported with incorrectly loaded checkpoints, a subtle but important evaluation pitfall. An earlier evaluation loaded a checkpoint with mismatched hidden dimensions under non-strict weight loading, silently dropping 89% of the weights and producing a spurious +2.9 pp F1 gain from near-random weights. All results reported here use shape-filtered checkpoint loading, which only keeps weights whose shapes match, and we recommend this as a standard practice.

### 4.3 Constraint-Type Ablation

Which spatial rules matter most? We ablate the ten constraint types on the violation-detection task (n=500, drop=0.6). Removing containment causes the largest accuracy drop (−1.9 pp), confirming that parent–child containment is the strongest structural signal. Removing *all* alignment types *increases* accuracy to 93.9%: with far fewer constraints per graph, the remaining violations are easier to classify. Keeping only alignment types hurts most (−3.0 pp): alignment alone cannot support violation detection.

| Constraint set | Violation acc. |
|----------------|:--------------:|
| All 10 types (control) | **0.908** |
| Remove `CONTAINMENT` | 0.889 (−1.9 pp) |
| Remove `SPACING` | 0.903 (−0.5 pp) |
| Remove `GRID` | 0.916 (+0.8 pp) |
| Remove all `ALIGNMENT` | 0.939 (+3.1 pp) |
| Only `ALIGNMENT` | 0.878 (−3.0 pp) |

### 4.4 Training-Objective Ablation

We compare joint training against single-objective training (5 seeds). Joint training achieves the best balance: violation accuracy 0.876 with proposal MSE 0.051. Violation-only training reaches higher violation accuracy (0.898) but poor proposal MSE (0.116); proposal-only training reaches good proposal MSE (0.051) but near-chance violation accuracy (0.489). The joint objective trades a modest violation-accuracy loss for a large proposal-quality gain, which matters for completion.

### 4.5 Element Completion

We compare the GNN proposal head against a nearest-neighbor baseline that copies the box of the closest surviving element. At low drop ratios (0.2–0.4) the baseline wins because many survivors are nearby. At high drop ratios (0.6–0.8) the GNN wins: at drop 0.6 the GNN reaches IoU 0.122 vs. 0.088 for NN (+39%); at drop 0.8, 0.097 vs. 0.062 (+56%). With fewer survivors, structural reasoning matters more than proximity, which is consistent with the model using structural priors rather than simple interpolation.

### 4.6 Performance

The correction network is small (57K parameters). Graph construction takes about 5 ms and inference about 0.53 ms (p50) per screenshot on a CPU-only M3 MacBook Pro, negligible compared with VLM inference at roughly two seconds per image. Training 2,000 samples × 50 epochs completes in under five minutes on CPU.

## 5. Discussion

### 5.1 When Does Structural Reasoning Help?

Across the experiments, structural reasoning helped most when the VLM output was sparse (many elements missing) and less when the VLM was already accurate. This mirrors the sequence of negative results that motivated the graph formulation in the first place. On simulated Gaussian noise, the model never beat the no-op baseline; Qwen3-VL Plus and Flash produced positional errors around 0.01 or smaller, leaving nothing to correct; LLaVA-7B detected only a few elements per screenshot, so the graph carried little structure; and a reweighted existence-only objective had no false positives to filter. Only when a substantial fraction of elements was missing did the constraint graph contain information that a per-element regressor could not access. For lightweight VLMs on realistic screenshots (which omit 38% of elements), this regime is the relevant one for practical use.

### 5.2 Limitations

The main limitations are as follows. First, *type prediction is weak*: predicting an element's semantic type from constraint context alone caps at 62% accuracy; constraints carry spatial but not semantic information. Second, *domain transfer is limited*: the confidence model's AUROC drops from 0.703 on RICO to 0.554 on ScreenSpot, indicating that false-positive patterns learned on mobile screens do not fully transfer to mixed mobile/PC/web layouts. Third, *dense toolbars are hard*: a toolbar with 8+ tiny icons creates a dense constraint graph in which individual elements are hard to distinguish, and free-form layouts (maps, drawings) offer few constraints to reason from.

### 5.3 Future Work

Visual feature fusion with cross-attention is a candidate next step (early experiments show 18–22% proposal-MSE improvement), as is domain adaptation to ScreenSpot. Extending the graph with temporal context could help the model distinguish missing elements from elements hidden by UI state.

## 6. Conclusion

This report presented a heterogeneous bipartite GraphSAGE framework for correcting noisy GUI element predictions from lightweight VLMs. Ten spatial constraint types are encoded as a bipartite graph, so elements exchange messages only through shared constraints; the multi-task objective couples coordinate correction, violation detection, existence scoring, and self-supervised element completion. On RICO, the model achieves 90–94% violation accuracy, a +56% IoU gain over nearest-neighbor completion at high drop ratios, and a +2.0 pp F1 improvement on 200 real Qwen3-VL Flash screenshots with shape-filtered checkpoint loading.

The results are encouraging but bounded. The gains appear when the VLM output is sparse, and the framework inherits the VLM's initial detections; it does not fix errors that are purely semantic, and its confidence scores transfer only partially across domains. These limits, along with the negative results reported in the discussion, suggest that structural post-correction is best viewed as one component of a GUI-understanding pipeline rather than a complete solution. Whether the approach generalizes beyond mobile layouts to desktop and web interfaces remains to be tested.

## References

[1] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, et al. Qwen Technical Report. arXiv preprint arXiv:2309.16609, 2023.

[2] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-End Object Detection with Transformers. European Conference on Computer Vision (ECCV), pages 213–229, 2020.

[3] Jianan Li, Jimei Yang, Aaron Hertzmann, Jianming Zhang, and Tingfa Xu. LayoutGAN: Generating Graphic Layouts with Wireframe Discriminators. International Conference on Learning Representations (ICLR), 2019.

[4] Kamal Gupta, Justin Lazarow, Alessandro Achille, Larry S. Davis, and Dhruv Mahajan. LayoutTransformer: Layout Generation and Completion with Self-Attention. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 1004–1014, 2021.

[5] William L. Hamilton, Rex Ying, and Jure Leskovec. Inductive Representation Learning on Large Graphs. Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[6] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, et al. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research (TMLR), 2024.

[7] Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. RICO: A Mobile App Dataset for Building Data-Driven Design Applications. Proceedings of the 30th Annual ACM Symposium on User Interface Software and Technology (UIST), pages 845–854, 2017.

[8] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. SeeClick: Harnessing GUI Grounding for Advanced Visual GUI Agents. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), pages 9033–9049, 2024.

[9] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems (NeurIPS), volume 32, 2019.

[10] Matthias Fey and Jan Eric Lenssen. Fast Graph Representation Learning with PyTorch Geometric. ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[11] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, et al. Array Programming with NumPy. Nature, volume 585, pages 357–362, 2020.

[12] Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, et al. SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, volume 17, pages 261–272, 2020.

[13] John D. Hunter. Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, volume 9, number 3, pages 90–95, 2007.
