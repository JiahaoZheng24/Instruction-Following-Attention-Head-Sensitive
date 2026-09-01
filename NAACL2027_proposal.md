# NAACL 2027 研究方案：指令关键注意力头与保护性量化

> 基于 ACL ARR 2026-01 被拒稿（Submission 7892）的复盘 + 截至 2026-08-21 的文献核查（4 路并行检索，约 80 次搜索、120+ 篇论文；五篇高危近邻已于 2026-08-21 逐篇精读验证）。
> 目标：ARR 2026-10-12 截稿 → NAACL 2027（2027-06，旧金山）。

---

## 🎯 投稿转向（2026-09-01）— 本节优先于全文

**新目标：ICLR 2027**（摘要 9/18、全文 9/25 AoE；结果约 2027-01 下旬）→ 被拒则转 **ACL 2027**（ARR ~2027-02 cycle，届时手握 ICLR 审稿意见修补）。不再投 NAACL 10/12。

为此**解冻实验**，W10 批补 ICLR 审稿人的两个必杀题（其余维持冻结，不加新基准/新机制实验）：

1. **旋转臂**（"GPTQ 太老"防线）：QuaRot 式 R1 残差流旋转（随机正交阵离线融合，`src/rotate_model.py`，代数已在 fp64 numpy 全栈复刻中验证恒等）。臂：llama-rot-none（崩塌是否在旋转基下幸存？）、llama-rot-tacq（旋转基显著性）、qwen-rot-none、rotfp 恒等性对照。**两种结局都预先框入故事**：崩塌幸存 → "旋转不能替代临界集保护"；崩塌消失 → "旋转是换 regime 的杠杆"，与 regime=模型×量化器交互的定律一致，优雅态两大不变量不受影响。
2. **规模臂**（"只有 7-8B"防线）：Qwen2.5-14B 全套（fp16/salience/none/tacq@70M）+ Qwen2.5-32B regime 检查（fp16/none）+ Mistral-Small-24B regime 检查（W10c，单卡；Llama 家族无 30B 档，70B 需 2 卡难排 → 降为 rebuttal 可选）。规模证据形态 = 两个家族各一条族内缩放线（Qwen 7→14→32B，Mistral 7→24B）。若 24B 崩塌则加跑 salience+tacq（规模化救援头条）。

作业：`jobs/w10a_rotation_scale.sh`（9 臂）→ `jobs/w10b_rot_tacq.sh`（hold_jid，2 臂）；`jobs/w10c_mistral24b.sh`（2 臂）；`jobs/w11_gsm8k.sh`（6 臂，GSM8K）。日程：9/4–9/10 跑 W10 并行写作，9/18 摘要，9/25 全文；11 月 rebuttal 可补实验改稿。

**W10/W11 裁决（2026-09-02，IFEval avg4）**：①**旋转消除 Llama 崩塌**——rotfp 0.7848≈fp16 0.7683（旋转无损）；rot-none 0.6406 ≈ 未旋转 tacq 救援 0.6426（两条路同一天花板）；rot-tacq 0.6675（优雅态 +2.7，第四个"优雅态保护无用"实例）；qwen rot-none 0.6884≈0.6717（优雅不变）。写法按预框定结局 B："旋转是换 regime 的杠杆"。②**Qwen-14B@3bit 半崩塌**：fp16 0.8203→0.4116（−40.9），而 7B −9.4、32B −7.6 均优雅——**regime 随规模非单调**，"不可预测、可检测"升格为核心主张；Mistral-24B 优雅（0.7574→0.6978）。③**GSM8K 复现 regime 定律**：llama none 0.0015→tacq 0.6164（fp16 0.8431）；qwen 优雅 −10.5、tacq +4.3。五线证据闭环（IFEval/Multi-IF/MMLU/GSM8K/PPL）。④bug：torch.quantile 16M 上限在 14B 显著性上爆掉（已修：common.safe_quantile 子采样）；14B-tacq 臂随 **W12**（`jobs/w12_qwen14b_anatomy.sh`，5 臂）重跑：14B 崩塌解剖 = tacq 救援 + RTN 对照 + 旋转杠杆 + randw 对照 + GSM8K v2none 参照。

**ICLR 化定位（2026-09-01 定稿）**：IF 从"研究对象"降级为"主探针"，标题与 RQ 讲通用能力（"When Does Protecting Weights Protect Capabilities?"）。依据是自家数据：临界集任务无关（c4≈IF 掩码）、崩塌 PPL 可见、MMLU 复现。IF 保留为差异化显微镜（逐约束损伤解剖、Multi-IF 轮次复利、localization≠protection 电路线）。补 **W11**（`jobs/w11_gsm8k.sh`，6 臂）：GSM8K 推理跑在与 Multi-IF 相同的六个 checkpoint 上，凑齐 IF/知识/推理/PPL 四线 regime 证据，无新量化成本。

---

## ⚡ 主轴修订（2026-08-27，W1/W2 实验裁决后）— 本节优先于下文所有章节

W1 诊断与 W2 保护实验（16 臂，全量 541 IFEval）已完成，数据对原假设 H1 做出了部分裁决，论文主轴随之修订。

### 已确立的实验事实（干净 checkpoint，gptqmodel 5.6.12，TORCH backend）

1. **Bit 叙事**：fp16 avg4=0.7652；8-bit 无损；4-bit −0.8 pt 但 74/541 条判定翻转（39 降 35 升，平均分掩盖尾部）；**3-bit −8.2 pt（真实主战场）**；2-bit 崩塌（0.12，仅边界分析）。旧"3-bit 掉 15 分"证实为损坏 checkpoint 伪影（gptqmodel 5.4.2 packing bug，#1278）。
2. **损伤表达是结构化的**：3-bit 逐约束退化极不均匀——language −22.6（自 100%）、combination −16.9、punctuation −12.1，而 detectable_content −1.9。
3. **IF 因果电路是局部的**：ablate top-32 quant-fragility heads 选择性砸 keywords/language（Δavg4 −3.2 vs 随机 −0.4，8 倍选择性）；top32-act ablation 伤得弥散（通用损伤对照）。
4. **但损伤位置是弥散的（W2 核心负结果）**：在 gptq3 上事后恢复权重切片至 fp16，16 个臂（dev3-top16/32/64、随机×3、act、layer01、qkv/o 分解、同预算 MLP×3、noop）**全部落在 −0.7～+1.8 pt 噪声带内**，无一臂产生可辨认的恢复。noop 与基线逐位一致（harness 确定性验证）；selftest 数值校验通过。**0.5% 参数预算下的头级/通道级恢复已证伪。**
5. **Dissociation**：梯度显著性（TaCQ 的 MSG 项）与 fragility/激活排序近乎正交（Spearman ≤0.16, Jaccard@32 ≤0.05）；act ≈ dev3（ρ=0.94，dev 排序受激活尺度混淆，最终认定以因果 ablation 为准）。

### 修订后的核心科学问题

> **量化对指令遵循的损伤住在哪里？任何权重级保护方案能否找到并保护它？**
> 即：salient-weight 保护纲领（TaCQ/OWQ/SPQR/PAHQ 一系）的隐含前提——"能力损伤可在权重层面定位"——对 IF 是否成立。

三个子问题与状态：
- **SQ1 功能定位的结构（IF-heads）承载损伤吗？** 已答：**否**。因果必要（ablation 8× 选择性）但保护无效——**causal-yet-unprotectable**，功能定位 ≠ 损伤定位。这是论文的概念贡献。
- **SQ2 敏感度定位的散点权重（TaCQ 判据 `|W|·|∇L|·|ΔW|`，0.35% 预算，环内豁免）承载吗？** 待答（v2b，稳赢设计：救回 = 首个把 salient-weight 保护带到 IF + 与功能头的 dissociation；救不回 = IF 是独特弥散能力，保护纲领对 IF 整体失效，接上 Alignment Collapse 未修复的 IFEval 缺口）。
- **SQ3 浓度曲线：要恢复多少才够？** 待答（w2c：0.5%→2%→5%→11%（全部 attention）→25%；若需 ≥25% 才回血则比特数已不如直接 4-bit——保护纲领的定量墓志铭，论文主图）。

### 证伪的完备性条件（缺一即 incremental）

1. **v2 环内豁免必须跑**（TaCQ 是量化环内持出，我们 v1 是事后恢复——不堵此混淆，负结果可被"手术做错了"一句推翻）：v2a 环内头切片 + v2b 环内 TaCQ 判据散点，同预算对照。
2. **TaCQ 判据臂真实现**（代码公开可移植；梯度显著性 dissociation.py 已算 MSG 项，补 |ΔW| 因子即可）。
3. **≥2 模型**（Llama-3.1-8B-Instruct 验证，管线零改动）。

### 查重增补（2026-08-27 针对新主轴）

- [TaCQ 全文](https://arxiv.org/html/2504.07389v2)：非结构化 0.35% 散点、环内豁免、**从未测 IF**、未分析权重分布、模拟精度；3-bit 通用校准 MMLU 仍留 10 pt 缺口。
- [OWQ](https://arxiv.org/abs/2306.02272)（AAAI24）：列结构化 Hessian 弱列保护，3.1-bit≈4-bit——结构化保护可行，但有效结构是"敏感列"非"功能头"。
- [Critical Weight Protection (2601.12033)](https://arxiv.org/html/2601.12033v2)：capability 定向（fairness/safety）权重保护先例——但**需保留 60% FP16**、无 IFEval、无随机对照、无定位分析；其 60% 预算反向支持弥散结论。"首个 capability 定向保护"不可再主张，改为"首个在 IF 上系统检验保护可行性 + 首个功能/损伤定位 dissociation"。
- [FAQ (2601.11200)](https://arxiv.org/html/2601.11200v1)：校准数据侧修复，IFEval 仅 INT4 浅水区 +0.5~0.7；[llama.cpp 量化评测 (2601.14277)](https://arxiv.org/html/2601.14277v1)：描述性记录 IFEval 掉 20%，无机制。均为引用非威胁。

### 修订后的结局矩阵

| v2 环内头保护 | TaCQ 判据（散点）救 IF？ | 论文形态 |
|---|---|---|
| 有效 | — | 回到方法论文：环内头结构化保护（v1 负结果成为"手术时机很重要"的消融）|
| 无效 | 救回 | **Dissociation 论文**：损伤稀疏但非结构化；可解释性引导的保护瞄错对象；首个 IF 上的 salient-weight 保护 |
| 无效 | 救不回 | **边界论文**：IF 损伤对权重级保护整体免疫（浓度曲线定量化）；弥散累积机制；对保护纲领的结构性批评 + 实践指引（IF 场景直接 4-bit/QAT）|

三个出向都可发表；负结果风险以三样带走的东西对冲：language 崩塌发现、浓度曲线、实践指引。

### 修订后的时间表（替代 §5 52 天计划的 W2 之后部分）

| 周 | 内容 |
|---|---|
| ~8/31 | w2c 浓度曲线（v1 代码零改动）；v2 环内量化实现（gptq_core 带 mask 持出）+ TaCQ 显著性计算 |
| 9/1–7 | v2a/v2b 环内臂 + 对照跑完；**Go 检查点（9/7）：按结局矩阵定论文形态** |
| 9/8–21 | Llama-3.1-8B 全线复制；确定形态后的补充实验（per-head 因果地图 / 约束类型矩阵 / Multi-IF）|
| 9/22 起 | 与原计划 W5–W7 相同（消融图表 → 写作挂 arXiv → 内审提交）|

---

## 0. TL;DR（2026-08-21 版，其量化×保护主轴已被上节修订取代；查重与背景仍有效）

**把论文从「运行时修复」翻转为「量化时保护」。**

- **研究问题（一句话）**：量化导致的指令遵循（IF）退化，是由少数可识别、跨输入稳定的 attention heads 的损伤所介导的**局部现象**，还是**弥散的全局现象**？若是前者，在量化时以 head 为单位保护这些组件（约 1–3% 参数保高精度），能否以可忽略的代价**预防性地**保住指令遵循——不需要运行时干预、不需要 FP16 参考模型、天然 prompt 无关？
- **查重结论（经五篇高危近邻逐篇精读 + 对抗性扫描验证，2026-08-21）**：核心组合「IF 关键 head 识别 → head 结构化权重精度保护 → IF 基准验证」**确认无人占坑，值得做**。但精读后两处主张必须收窄：①「首个 IF 的 head 级因果地图」**不可再主张**——Weight Patching（2026-04）已在 6 类 IFEval 格式约束上做了单 head 粒度的因果分析；我们的诊断新颖性缩至"全约束 taxonomy（含语言/长度/关键词，他们做不了）+ 稀疏度曲线 + 稳定性 + 免配对模型的单模型方法"。②「可解释性→量化」不可主张为首创（TaCQ 已用此框架、Capability-Guided Compression 已做 SAE 密度分配）——收窄为「首个**因果验证的、能力定向的、head 结构化**的 PTQ 保护，在 IF 基准上评测」。论文主轴必须是**量化×保护**，诊断降级为地基。
- **影响力**：实用面——量化 instruct 模型是开源部署的默认形态，IF 失败（格式破坏、语言漂移）是用户与下游系统最直接感知的失败，且被 perplexity 和平均基准系统性掩盖；方法是 GPTQ/AWQ 上一个配置级改动，可被工具链直接吸收。科学面——为「能力局部化 vs 弥散」辩论补上"控制类能力"这块拼图，并正面检验"机制发现能否转化为工程决策"。
- **风险对冲**：实验设计让"局部化成立/不成立"两种结局都有可发表产出；不成立时降级为系统分析论文（方案三），前半段工作不浪费。

---

## 1. 我们到底在研究一个什么问题

### 1.1 现象（已确立的事实，不是我们的贡献）

1. 指令微调 LLM 的实际部署几乎默认伴随量化（边缘设备、本地推理、服务成本）。
2. 现行 PTQ 方法（GPTQ / AWQ / RTN / SmoothQuant）的目标函数是**能力无关**的：最小化逐层权重或激活的重构误差（或校准集困惑度）。
3. 大规模评测已确证：量化后**指令遵循是掉得最不成比例的能力之一**。[IJCAI 2025 (2409.11055)](https://arxiv.org/abs/2409.11055) 在 1B–405B instruct 模型上发现量化模型几乎全面胜过更小的 FP16 模型、唯独 IF 与幻觉两项例外；[UniComp (2602.09130)](https://arxiv.org/abs/2602.09130) 在 40 个数据集上确认压缩后事实记忆保留、而 IF/推理/多语言不成比例受损。
4. 退化模式是"**平均分几乎不动、尾部灾难**"：我们自己的数据（Qwen2.5-7B-Instruct）显示 GPTQ-4bit 的 IFEval 平均分与 FP16 持平（0.760 vs 0.759），但个别样本出现德语输出、退化重复。⚠ 注意：旧数据中 "gptq3 掉 15 分、gptq2 掉 82 分"来自已删除的旧 checkpoint，且 2026-08-25 冒烟测试证实 3-bit Triton kernel 存在机械性故障——**3-bit 真实掉分以新量化基线（jobs/w1_quant_baselines.sh）为准**。
5. 能力在参数中的编码是**不均匀**的：安全能力集中在约 2.5–3% 的参数区域（[Wei et al., ICML 2024](https://arxiv.org/abs/2402.05162)）；长上下文检索集中在 <5% 的 attention heads（[Retrieval Heads, ICLR 2025](https://arxiv.org/abs/2404.15574)）；SpQR 观察到量化敏感权重"聚集成部分行或 attention heads"（[ICLR 2024](https://arxiv.org/abs/2306.03078)）却没有沿 head 行动。

**推论（问题的来源）**：能力无关的均匀误差最小化，会系统性牺牲那些"编码稀疏、误差敏感"的能力；IF 恰好是这样的能力。这是一个**目标函数与保护对象错位**的问题，不是某个量化算法的 bug。

### 1.2 核心科学问题与假设

**核心问题 RQ**：
> 量化引起的指令遵循退化，是被少数 attention heads 的损伤所**介导（mediated）**的，还是全模型弥散损伤的累积结果？

这正是上一稿提出但没能回答的问题（Reviewer VsXW 的原话："If 80% of the cases fail to recover via DAS, it suggests that the degradation may indeed be a global phenomenon rather than a localized one"）。上一稿失败的原因不是问题不好，而是证据形态错了：10 个人工样本 + prompt 特定修复回答不了这个问题。新的证据形态是因果实验 + 大规模自动评测。

**工作假设 H1**（可证伪）：存在一小撮（<5%）attention heads，满足：
- (a) **因果必要**：在全精度模型上 ablate 它们会选择性破坏可验证指令约束的遵循（而基本不伤一般语言能力）；
- (b) **跨输入稳定**：这个集合在不同 prompt、不同约束类型、不同随机种子下高度重合；
- (c) **量化脆弱**：低比特量化对它们的功能损伤不成比例，且损伤程度可预测 IF 掉分；
- (d) **可保护**：量化时把它们对应的权重切片（W_Q/W_K/W_V 的列块 + W_O 的行块）保在高精度，即可恢复大部分 IF 损失。

**对立假设**：
- H0-弥散：IF 退化是全局的，保护任何小子集都无效；
- H0-不稳定：IF-heads 存在但随输入漂移（呼应 [Retrieval Heads are Dynamic (2602.11162)](https://arxiv.org/pdf/2602.11162) 对静态 head 画像的批评）；
- H0-非注意力：IF 主要由 MLP/neuron 承载（[SPARCOM (2505.21191)](https://arxiv.org/abs/2505.21191) 已发现 instruction-specific neurons），head 保护不是正确的粒度。

### 1.3 子问题分解（论文的三段结构）

**RQ1（定位，science）**：在指令微调模型中，哪些 attention heads 对可验证指令约束是因果必要的？稀疏吗？跨 prompt / 约束类型 / 模型稳定吗？
- 方法：mean-ablation + activation patching（遵循 [Best Practices of Activation Patching, ICLR 2024 (2309.16042)](https://arxiv.org/abs/2309.16042) 协议——直接回应上轮"未与因果中介分析文献对比"的批评），全量 IFEval（541 prompts）自动评测，3+ 模型。
- 我们已有的起点：现有 pipeline 的 per-head 指令注意力（ISI）是相关性证据，升级为因果证据即可复用全部基础设施。
- **精读修正（2026-08-21）**：Weight Patching（2604.13694）已在 6 类 IFEval 格式约束上给出 head 级因果分析，故 RQ1 **不再是独立头牌**，定位为诊断地基；其增量卖点收窄为：全约束 taxonomy（补上他们做不了的语言/长度/关键词三类）、稀疏度曲线、跨 prompt 稳定性、免配对模型的单模型方法。诊断设计还应吸收两点方法学差异（对 CASIA 整层 patching 的超越）：①**双向 patching**（FP head→量化模型证充分性修复 + 量化 head→FP 模型证损伤充分性，他们只做修复方向）；②在**多 token 可验证约束输出**上打分（用官方 IFEval checker 作 restoration 度量，参照 2606.09662 的协议），而非单 token 答案概率。

**RQ2（机制，science）**：量化损伤与 IF-heads 的重叠度如何？head 级损伤能否预测 (a) 哪些样本失败、(b) 哪类约束先失败、(c) IFEval 总分掉多少？
- 我们已有的起点：ISI 与 IFEval drop 在 4 个 bit 档上 r≈0.94（n=4，需扩展为逐样本、逐约束类型的预测实验）。
- 这一问自带一个独立卖点：**一个无需跑全量评测的 IF 退化预测器**（对量化发布者是实用工具）。

**RQ3（干预，engineering）**：量化时按 head 结构化保护能否恢复 IF？代价多大？相比两类现有范式的优势：
- vs **事后修复**范式（EoRA 低秩补偿、steering、LoRA 恢复训练）：零运行时开销、无额外模块、不需要 FP16；
- vs **能力无关保护**范式（离群值/Hessian 显著性、随机对照）：证明"针对性"是必要的，这一步是把 RQ1 的因果结论转化为工程价值的关键对照。

### 1.4 结局矩阵（为什么这个问题怎么答都有产出）

| RQ1：稀疏且稳定？ | RQ3：保护有效？ | 论文形态 |
|---|---|---|
| 是 | 是 | **方法论文（主推故事）**：诊断 → 保护，capability-aware quantization 在 IF 上的首例 |
| 是 | 否 | 分析论文：IF-heads 存在但精度保护不足以挽救 → 说明误差经由其他路径（逐层传播/MLP）汇入，对 localization 辩论仍是实质贡献 |
| 否 | — | 分析论文：**"IF 退化是全局的"** 是反直觉、可引用的结论；配合失败分类学与退化预测器降级为方案三 |

注意：这是内部的风险对冲逻辑，论文写作时仍然主推 H1，不写成"怎样都行"。

### 1.5 为什么是 attention head 这个粒度（而不是 neuron / channel / layer）

1. **计算职能对口**：指令遵循的计算特征是"生成全程持续 attend 回指令 token 并据此调制输出"——这正是注意力的职能。行为学证据：对指令 span 加注意力偏置能直接提升 IF（[SpotLight (2505.12025)](https://arxiv.org/html/2505.12025)、[InstABoost (2506.13734)](https://arxiv.org/abs/2506.13734)，IFEval 提升约 26%）；参数空间证据：[Weight Patching (2604.13694)](https://arxiv.org/abs/2604.13694) 恢复出中层 attention heads 聚合"指令方向"的回路。
2. **工程对口**：head 对应权重矩阵中的**连续行/列块**，结构化混合精度对 kernel 和内存布局友好；对比 TaCQ 的非结构化散点 16-bit 权重需要稀疏索引、难以高效执行。这是我们相对最强近邻的一个可正面攻击的差异点。
3. **诚实的对照**：H0-非注意力必须被实验排除——设一个"同参数量 MLP 神经元保护"对照组；若 MLP 同样有效，叙事从"IF-heads"调整为"IF 回路保护"，方法结论不变、机制结论修正。

---

## 2. 这个问题有没有人做过（逐条查重）

结论先行：**组合未被占，但每个部件都有必须引用并差异化的邻居。** 按贡献主张逐条核查：

| # | 主张 | 有没有人做过 | 最近邻及差异 |
|---|---|---|---|
| C1 | 量化伤害 IF（现象） | **做过** —— 不作为贡献，只作动机 | [2409.11055](https://arxiv.org/abs/2409.11055)（IJCAI 2025）、[UniComp](https://arxiv.org/abs/2602.09130)、[Give Me BF16](https://arxiv.org/abs/2411.02355)（ACL 2025）均报告现象，无一问机制 |
| C2 | IF 的 head 级**因果**定位（全精度） | **已被部分做掉（精读修正，见 §2.1）** | [Weight Patching](https://arxiv.org/abs/2604.13694)（2026-04）实际**就是 IFEval**（6 类格式约束）+ **单 head 粒度**（W_Q/W_O 交换 + head 级 activation patching）+ knockout/restoration 验证。我们剩余的空间：全约束 taxonomy（他们因 anchor 向量不稳定**缺语言、长度、关键词三类**——语言漂移恰是我们的招牌失败模式）、稀疏度曲线、跨 prompt 稳定性、以及无需配对 base 模型的单模型诊断（量化模型正是他们权重交换法失效的场景）。另参 [SPARCOM](https://arxiv.org/abs/2505.21191)（neuron 级）、[Instruction Anchor (2602.03677)](https://arxiv.org/abs/2602.03677)（MLLM 中 IF head 稀疏性旁证） |
| C3 | **量化模型**内部的 head 级分析 | **没人做过** | 现有量化内部分析全在其他粒度：neuron 级（[AACL 2025 (2508.16785)](https://arxiv.org/abs/2508.16785)）、channel 级（[Alignment Collapse (2606.09864)](https://arxiv.org/pdf/2606.09864)）、层级/SNR（CASIA：[Two Failure Modes (2604.19884)](https://arxiv.org/abs/2604.19884)，ACL 2026 Findings；[Signal–Noise (2608.08188)](https://arxiv.org/abs/2608.08188)）、残差流方向（[AIRD (2504.04215)](https://arxiv.org/abs/2504.04215)） |
| C4 | head 粒度的**权重**混合精度 | **没人做过** | head 级精度分配全在 KV cache/激活侧（[RateQuant](https://arxiv.org/abs/2605.06675)、[MixKVQ](https://arxiv.org/pdf/2512.19206)、[Block-GTQ](https://arxiv.org/abs/2606.24033)、TurboAttention）；权重侧混合精度全在散点/列/组/层/expert 粒度（AWQ、OWQ、SpQR、[SliM-LLM](https://arxiv.org/abs/2405.14917)、LLM-MQ、[TAQ](https://arxiv.org/pdf/2511.06516)）。三组独立检索零命中 |
| C5 | **IF 作为被保护能力**的量化 | **没人做过** | "能力感知量化"已被占的能力槽：安全（[AAQ](https://www.arxiv.org/pdf/2511.07842)、[Q-resafe (ICML 2025)](https://arxiv.org/pdf/2506.20251)）、公平（[Critical Weight Protection (2601.12033)](https://arxiv.org/pdf/2601.12033)）、推理（[Quantization Meets Reasoning (2505.11574)](https://arxiv.org/abs/2505.11574)，DPO 修复）、通用任务（[TaCQ (COLM 2025)](https://arxiv.org/abs/2504.07389)）。IF 槽位为空 |
| C6 | 可解释性诊断 → PTQ 决策的闭环 | **框架被部分占用，收窄后成立（精读修正）** | TaCQ 已用"知识定位/可解释性→量化"框架（但仅梯度归因，**无任何因果验证**）；[Capability-Guided Compression (2603.16440)](https://arxiv.org/abs/2603.16440) 做了 SAE 密度→预算分配（但 GPT-2 规模、无具体量化器、无具名能力、自报负结果）；[PAHQ (2510.23264)](https://arxiv.org/pdf/2510.23264) 做了 head 级混合精度（但用途是加速 circuit discovery，非部署 PTQ）。收窄后的主张：**首个因果验证的、能力定向的、head 结构化的部署级 PTQ 保护**。KV 驱逐侧先例（HeadKV、DuoAttention、RLKV）照旧引用 |
| C7 | 指令数据校准 vs 结构保护的对比 | **没人系统做过** | [ACL 2024 (2311.09755)](https://aclanthology.org/2024.acl-long.544/) 证明校准数据影响下游能力；Qwen 官方文档把 chat 格式校准当 folklore；无人在 IFEval 上把两种 IF 保全手段对比——论文里的免费 ablation |

### 2.1 五篇高危近邻：逐篇精读核对结论（2026-08-21 全部验证完毕）

五篇论文全部**真实存在**、ID/作者/venue 无误；此前引用的数字经核对准确（个别语境需修正，见下）。

1. **[TaCQ（COLM 2025）](https://arxiv.org/abs/2504.07389)** ✅ 不挡路，但要小心措辞。核实：完全**非结构化**逐权重挑选（saliency = |W|·|∂L/∂W|·|W_quant−W|，保 ~0.35% 权重于 16-bit）；全文含附录**无任何 IF 评测**、无 head/结构分析；混合精度是**模拟的**（无 kernel、无延迟/内存实测，这是它自认未解决的弱点，恰是我们结构化方案的动机）。但它的标题与 related work 已占"知识定位/可解释性→量化"框架（仅梯度归因、无因果验证）→ 我们必须表述为"首个**因果验证的**组件级诊断驱动量化"。作为 baseline 可行：代码开源（Apache-2.0），梯度捕获峰值 **93GB 显存**、约 3.5h/7B（H200 单卡可跑）；最有说服力的对比 = 用 IF 数据做 task-conditioning 的 TaCQ vs 我们的 head 结构化保护。
2. **[Weight Patching（2604.13694）](https://arxiv.org/abs/2604.13694)** ⚠️ **比预想近得多——推翻了检索代理的两处描述**。核实：它**就是用 IFEval**（6 类可验证约束：大写、无逗号、标题、分节、引号、高亮分节），**就是单 head 粒度**（交换每个 head 的 W_Q/W_O + head 级 activation patching 对照），有 knockout/restoration 因果验证。因此"首个 head 级 IF 因果地图"不可主张。我们剩余的实空间：①量化连接（他们全文含 future work 零量化）；②全 taxonomy——他们因 anchor 向量不稳定**做不了语言、长度、关键词约束**，语言漂移恰是我们的招牌；③稀疏度曲线与跨 prompt 稳定性（他们没有）；④单模型激活空间诊断，无需配对 base 模型——量化模型正是他们权重交换法失效的场景。写作时必须与其"head=路由器、neuron=源头"的层级结论对表（我们的保护故事=保住路由通路，兼容）。
3. **CASIA 双雄（[2604.19884](https://arxiv.org/abs/2604.19884) ACL 2026 Findings + [2608.08188](https://arxiv.org/abs/2608.08188)）** ✅ 主张完好。核实：两篇均无 IF/IFEval（Paper 1 锚定 Pararel 事实回忆；Paper 2 十一个基准全是 loglikelihood 式），**无任何 head 级分析**（Paper 1 是整层残差流 patching，两级粗于 head；Paper 2 是七个线性模块的 SNR）；Paper 1 的修复是"前 2 层保 8-bit"或按峰度挑权重——统计驱动、能力盲。**他们反而递刀**：Paper 2 的 Discussion 明确写"混合精度应瞄准被任务激活放大最强的误差"——可直接引用此句把我们的方法定位为其开放问题的答案；"4-bit 信号退化可修 / 2-bit 计算坍塌"边界（核实：3-bit 未被归类，Paper 2 称 3-bit 方差最大）为我们锁定 3–4 bit 主战场提供依据。**关键对比实验**：同平均比特下，我们的 head-slice 保护 vs 他们的"前 2 层 8-bit"层级保护，在 3-bit 处比 IF 恢复。Scoop 风险：保护轴中等（他们下一步最可能是模块级任务感知混合精度）、IF-head 轴低（其评测设施全是 loglikelihood，无约束验证器）。
4. **[Alignment Collapse（2606.09864）](https://arxiv.org/abs/2606.09864)** ✅ 主张完好，且是强力动机引文。核实：**纯 KV cache 量化，权重全程 FP16**（§2 原文可引）；IFEval 数字精确（Pass_strict 69.50→59.89→16.82，但注意：在**附录 B.13**、是 **KV 比特**、只测 Qwen2.5-7B 一个模型，4-bit KV 也崩至 16.64）；61 页中 "head" 仅两次且均为张量形状记账；**其 PCR 修复只在安全基准上评测，IFEval 崩塌被记录但从未被修复**——明说这是我们填的缺口。Future work 只列剪枝/驱逐/低秩/旋转，scoop 风险低。定位措辞修正：不可说他们"纯现象学"（他们有层/channel 级因果消融），应表述为"他们把安全崩塌因果定位到 KV 的层/通道；我们把 IF 退化因果定位到 attention head 并在权重量化时预防"。
5. **TPQA（[FGCS 2025-12](https://www.sciencedirect.com/science/article/abs/pii/S0167739X25006466)）** ✅ 差异化表述准确。核实（经摘要，全文被墙）：确为 per-head 精度 + 任务感知 attention 模式 + 专用加速器，目标是注意力**计算**吞吐，非 LLM 权重 PTQ、无 IF 内容。一个软点：\[CLS\]/\[SEP\]/标点 head 的细节未能从摘要逐字确认，引用时软化为"task-dependent attention patterns"。

### 2.2 反面证据与怀疑派文献（先发制人）

- [SpotLight](https://arxiv.org/html/2505.12025) 消融明确报告**单个选定 head 的 steering 不稳定**（他们因此选择全 head 均匀偏置）→ 我们用 head **集合**（k=8–64 sweep）而非单头，并做 selected vs random vs all-heads 消融正面回应。
- [Retrieval Heads are Dynamic (2602.11162)](https://arxiv.org/pdf/2602.11162)：静态 head 画像会漏掉长尾关键 head → RQ1 的稳定性实验（跨 prompt 子集/约束类型/种子的集合重合度）就是为这条准备的。
- [Interpretability without Actionability (2603.18353)](https://arxiv.org/pdf/2603.18353)、[Navigating by Old Maps (2605.06076)](https://arxiv.org/abs/2605.06076)：机制发现未必可转化为干预、定位随训练漂移 → 我们的 RQ3 本身就是对 actionability 的正面检验，写进 discussion。

### 2.3 精读过程中新扫出的近邻（对抗性扫描结果，均已确认）

对抗性扫描（换措辞重搜 2026-03 至 08）**未发现直接碰撞**——「IF 关键 head 识别 → head 结构化权重精度保护 → IFEval 评测」仍无人占。但以下新发现须进 related work：

- **[PAHQ (2510.23264)](https://arxiv.org/pdf/2510.23264)** —— 名字就叫 per-attention-head quantization（被研究的 head 保高精度），但用途是**加速 ACDC circuit discovery** 的解释性工具，非部署 PTQ、无 IF。必引，堵住"head 级混合精度已存在"的质疑。
- **[Quantization Damage Is Multiplicative (2608.06564)](https://arxiv.org/abs/2608.06564)**（2026-08，workshop preprint）—— 权重量化 × 行为破坏（安全拒绝、tool-calling）的"margin shrinkage"统计解释，16 模型。无 head、无因果消融、无 IFEval 主线、无保护方法。轻度挤压"权重量化悄悄破坏行为"叙事，按机制+预防差异化。
- **[Capability-Guided Compression (2603.16440)](https://arxiv.org/abs/2603.16440)**（2026-03，单作者）—— SAE 密度→组件级压缩预算分配框架，含 head 级（GPT-2 Medium 的 384 heads）。但：GPT-2 规模、无具体量化算法、无具名能力、无 IF、自报 PPL 对比负结果。引用并以"因果 vs SAE 密度、部署级 instruct LLM vs GPT-2、具名能力 vs 抽象密度"切割。
- **[TASA (2607.00908)](https://arxiv.org/abs/2607.00908)**（2026-07）—— 任务感知的校准数据 + 层级/层内混合比特联合优化。无 head、无 IF。归入 baseline 家族叙述。
- **[2606.09662](https://arxiv.org/abs/2606.09662)**（2026-06）—— 对 Qwen3 IFEval 约束失败做 activation patching，但**只到层级**、无 head 地图；其"用官方 IFEval checker 做 restoration 评分"的协议值得借用并引用。
- 外围：APTQ（DAC 2024，注意力感知 Hessian、层级）、HydraHead（2606.20097，解释性选 head 做注意力压缩）、Attn-QAT（4-bit 注意力 QAT、报过 IFEval 但无 head 选择）、Complementary Attention Head Pruning（2606.19150，剪枝）。

---

## 3. 这个问题的影响力

### 3.1 实用影响：谁会用、怎么用

- **部署现实**：HuggingFace 上 instruct 模型的主流分发形态就是 GPTQ/AWQ/GGUF 量化版；llama.cpp、vLLM、AutoGPTQ 的用户直接消费这些产物。IF 失败是这些用户**最直接感知**的失败形态——"要求输出 JSON 却给了散文"会直接打断下游 agent/工具链（[ACBench (ICML 2025)](https://arxiv.org/abs/2505.19433)：压缩后结构化输出与 function-calling 掉得最多），语言漂移在多语言产品里是事故级问题。
- **被指标掩盖 → 价值在"揭示+预防"**：这类失败对 perplexity 与平均基准不可见（[Accuracy is Not All You Need (NeurIPS 2024)](https://arxiv.org/abs/2407.09141) 的 flips 现象；[多语言评测证明自动指标低估损伤约 10 倍](https://arxiv.org/abs/2407.03211)）。我们同时交付一个 head 级退化预测器（RQ2），让量化发布者不跑全量评测就能预估 IF 风险。
- **采纳成本低**：方法对 GPTQ/AWQ 是配置级改动（per-head 精度映射，复用现有 group-wise kernel），内存代价约 1–3%，推理零开销。这决定了它可能被量化工具链直接吸收——工具链吸收是效率类论文引用的主要放大器。
- **评测规范推动**：论文的一个副产品主张是"量化模型发布应报告 IFEval 而不只是 PPL/MMLU"——IFEval 正在成为量化论文的质量指标（[LFQ](https://arxiv.org/pdf/2605.29756)、[FAQ](https://arxiv.org/html/2601.11200v1) 已采用），我们是给这个趋势提供机制依据的一方。

### 3.2 科学影响

- **能力局部化辩论**：检索（retrieval heads）、安全（safety-critical 参数）之后，为"控制类能力"（IF 是对输出的约束控制，不是知识存取）补上局部化证据——无论正反结论都进入这条被两个社区（interpretability + efficiency）共同引用的证据链。
- **Actionability 检验**：2026 年可解释性社区正在争论"机制发现是否可操作"。「head 诊断 → 量化决策 → 能力恢复」是一个干净的正面案例（或干净的反例），比多数纯分析工作更有立场价值。
- **范式推动**：把量化的目标函数从"重构误差"推向"能力保持"（capability-aware compression）。这个转向已在安全上发生（AAQ、Q-resafe），我们做 IF 上的首例；如果转向成立，早期占位论文会被后续所有 capability-aware 工作引用。

### 3.3 时机（为什么是现在，而不是明年）

- 能力感知量化在 2025H2–2026H1 密集成型（安全/公平/推理各就各位），IF 槽位的空窗**正在关闭**；
- KV 侧的同款现象已被 [Alignment Collapse](https://arxiv.org/pdf/2606.09864)（2026-06）占掉，权重侧是最后的空地；
- CASIA 组以约 4 个月一篇的节奏在量化机制线上推进；
- 判断：这个交点在 6–12 个月内大概率被人占。**52 天内成稿 + 提交时同步挂 arXiv（NAACL 无匿名期）是必要动作。**

### 3.4 影响力的上限与诚实的风险

- **4-bit 悖论**：若主流部署点是 4-bit 而多数模型 4-bit 平均分不掉（我们的 Qwen 数据如此），则方法的实用主张主要落在 (a) 2–3 bit 极限压缩、(b) 4-bit 的尾部灾难样本（用 flips/CondFlip 类风险指标量化，而非平均分）、(c) 更小或更脆弱的模型（文献显示模型间差异极大：[长上下文研究](https://arxiv.org/abs/2505.20276)中 BNB-nf4 在 Llama-70B 掉 32% 而 Qwen-72B 无损）。论文叙事要主动把"平均分掩盖尾部风险"立为靶子，而不是回避它。
- **引用面预期**：交点位于两个社区之间，天然读者是量化工具作者 + 机制解释研究者，中等偏上的引用预期；不是 scaling law 级的爆点，但是扎实的、可长期被引的"拼图型"工作。

---

## 4. 被拒复盘 → 新方案的结构性消解

| 审稿意见 | 新方案中的消解 |
|---|---|
| 修复信号 prompt-specific，诊断与评测同 prompt | head 画像在校准集上离线算一次，保护写进量化权重，对任意输入生效；全部评测在 held-out 基准上 |
| 推理时需要 FP16 参考模型 | 保护发生在量化那一刻；部署产物是普通量化模型，无 hook、无参考模型 |
| 10 个人工样本、2/10 恢复 | 全量 IFEval + Multi-IF + FollowBench 自动评测；"局部化"由因果实验直接检验 |
| α 手动调且非单调 | 无 α；超参只剩保护 head 数 k 与保护精度，作为 sweep 呈现 |
| 未对比因果中介分析文献；两条引用疑似不存在 | 诊断直接采用 activation patching 规范（2309.16042）；**删除/替换两条被点名引用（grokking、evaluating-IF）并全文复核**——当下审稿环境里这是一票否决项 |
| Datasets/Software 双 1 分 | 投稿附匿名代码仓 + head 画像数据 |
| （未点名）DAS 与 Distributed Alignment Search 撞名 | 改名。备选：IPQ（Instruction-Preserving Quantization）、IF-Heads |

---

## 5. 实验设计（方案一）

- **模型**：Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct、Mistral-7B-Instruct（可选 +Qwen2.5-14B 规模检查）——直接回应"单模型"批评。
- **量化**：GPTQ / AWQ / RTN × {2, 3, 4}-bit，全部自己重新量化（GPTQModel，协议固定：c4-128×2048 校准、g128、sym、desc_act；`QUANT_PROTOCOL.json` 随 checkpoint 保存）。**GPTQ 推理一律用 TORCH backend**（2026-08-25 冒烟测试证实 Triton 3-bit kernel 机械性损坏）。主战场 3-bit（待新基线复核）；4-bit 讲尾部灾难（flips 类指标）；2-bit 结合 CASIA 的"计算坍塌"结论作边界分析。AWQ 生态仅 4-bit，低比特主线 GPTQ+RTN。
- **评测**：IFEval（主）+ Multi-IF（多轮，零先行者）+ FollowBench 或 InFoBench + MMLU/GSM8K（证明不伤通用能力）+ 内存开销核算。
- **方法实现**：IF-heads 的 W_Q/W_K/W_V 列块与 W_O 行块保 8-bit（或 FP16、或更细 group size），其余按目标位宽量化；GPTQ 与 AWQ 各做一版证明量化器无关。
- **Baselines**（精读后更新）：均匀量化；随机 k head 保护（关键对照）；离群值/Hessian 显著性保护（AWQ/SpQR 信号，同预算）；**CASIA 层级保护**（"前 2 层保 8-bit"，同平均比特对照——精读确认这是最有说服力的单个对比：在他们未测、方差最大的 3-bit 区间，head-slice vs 层级保护比 IF 恢复）；**TaCQ**（用 IF 数据做 task-conditioning 后作为最强 baseline；梯度捕获峰值 93GB 显存，H200 单卡可跑）；指令数据校准；EoRA（事后修复系代表）；LFQ（若可复现）。
- **消融**：k ∈ {8, 16, 32, 64}；保护精度 {FP16, 8-bit, 6-bit}；selected vs random vs all-heads；**同预算 MLP 神经元保护对照**（排除 H0-非注意力）；head 画像跨 prompt/约束类型/种子稳定性。
- **机制图**：IF-heads 与 retrieval heads / 离群 head 的重叠矩阵（分离性分析）；约束类型 × 失败模式 × 恢复率矩阵；ISI→IF 掉分预测曲线（逐样本级）。

### 52 天计划

| 周 | 日期 | 内容 |
|---|---|---|
| W0/W1 | 8/25–31 | 重新量化（w0_quantize ✅ 已跑，3-bit kernel 问题已定位）；量化基线重测（w1_quant_baselines）；因果诊断管线（calib/screen/ablate）；分离性分析首跑 |
| W2 | 9/1–7 | head 稳定性实验；实现 head 保护版 GPTQ/AWQ；内存开销核算。**Go/No-Go 检查点（9/7）** |
| W3–4 | 9/8–21 | 主网格（3 模型 × 3 量化器 × 3 bit × 全套评测 + baselines）；集群排队是最大变数，W2 末开始滚动提交 |
| W5 | 9/22–28 | 消融 + 机制图 + MLP 对照 |
| W6 | 9/29–10/5 | 写作、图表、匿名代码仓；同步挂 arXiv |
| W7 | 10/6–12 | 内审、Limitations、checklist（含引用真实性复查）、提交 |

**Go/No-Go 检查点（W2 末，9/7）**：若 Qwen 上 IF-heads 因果实验显示弥散（ablate top-k 无选择性效应），立即切换方案三叙事（系统分析 + 失败分类学 + 退化预测器），已有管线与数据全部复用。

---

## 6. 备选方案（简述）

- **方案二：免 FP16 的按约束类型 steering 修复**（DAS 正统续作）。按 Stolfo（[ICLR 2025](https://arxiv.org/abs/2410.12877)）模板按约束类型离线算差分向量——最优雅版本在量化模型自身上算 with/without-instruction 差分（彻底摆脱 FP16），或测试从未被做过的 FP16→量化向量迁移。**风险**：必须与 InstABoost/SpotLight 直接施加于量化模型对比，若全 head 均匀偏置就够用则 head 级机制失去卖点。定位：方案一的对比方法，或第二篇论文。
- **方案三：系统研究 + 失败分类学**（保底）。权重 PTQ × 完整 IF 基准 × 统一失败分类学（语言漂移/格式违规/退化重复/约束遗忘，文献中从未统一）× Multi-IF 多轮（零先行者）× ISI 退化预测器。管线现成、52 天最稳；风险是主会对纯评测卷执行力要求高、入场门槛低易被 scoop。定位：方案一的降级出口。

---

## 7. 精读后的最终判断：值不值得做

**值得做，判断置信度经五篇逐篇精读 + 对抗性扫描后从"检索代理的二手结论"升级为"一手核实"。** 但有三条修正必须落实到写作里：

1. **主轴必须是「量化 × head 保护」，不能是「IF 的 head 级机制」。** Weight Patching 已把后者做掉了六成（IFEval + head 粒度 + 因果验证）；我们真正独占的是量化连接——五篇里没有任何一篇、对抗性扫描里没有任何一篇碰过它。诊断章节的卖点是"补全 + 更稳的方法"（全 taxonomy、稀疏度、稳定性、免配对模型），不是"首个"。
2. **"首个"类主张只剩一个可以安全使用**：首个**因果验证的、能力定向（IF）的、head 结构化的**部署级权重 PTQ 保护，在 IF 基准上评测。其余"首个可解释性驱动量化"（TaCQ 占）、"首个 head 级混合精度"（PAHQ/TPQA 占）、"首个 head 级 IF 因果地图"（Weight Patching 占）全部不可用。
3. **论文的两个最强实验已经明确**（都来自精读，不是臆测）：① 同平均比特下 head-slice 保护 vs CASIA 的"前 2 层 8-bit"层级保护，在 3-bit（他们自称方差最大且未归类的区间）比 IF 恢复；② 用 IF 数据 task-conditioning 的 TaCQ vs 我们的结构化保护——非结构化 vs 结构化在同一能力目标下的正面对决，顺带打其"模拟混合精度、无硬件故事"的软肋。

**风险面（诚实）**：CASIA 组的 2608.08188 是 8 月刚挂的无 venue 新稿，大概率与我们同投 ARR 10 月轮——他们是诊断论文、我们是方法论文，不冲突但会分走"量化机制"的注意力；他们的下一步（模块级任务感知混合精度）与我们的保护故事有 6 个月碰撞窗口。结论不变：**这个窗口是真实的，10 月这轮必须上，且成稿即挂 arXiv。**

---

## 8. 提交前硬性清单

1. 引用整改：删除/替换被点名的两条引用（"A mechanistic interpretability analysis of grokking""Evaluating instruction following in language models"），全文逐条复核真实性。
2. 改名：弃用 DAS（与 Distributed Alignment Search 撞名）。备选：IPQ（Instruction-Preserving Quantization）、IF-Heads。
3. 开源：匿名代码仓 + per-head 画像数据。
4. 叙事纪律：不再出现"10 个精选样本"式表述；怀疑派文献写进 related work 主动接招。
5. 滚动查新：每两周复查 TaCQ / Alignment Collapse / RateQuant / CASIA 组（尤其 2608.08188 的 venue 去向）/ 2608.06564 的新动向与引用者。
6. 引用核实状态：五篇高危近邻 + 八篇次级引用（RateQuant、HeadKV、Stolfo、SpotLight、InstABoost、SPARCOM、Best-Practices、Retrieval Heads）已逐篇验证无误（2026-08-21）；其余 2026 年 arXiv 编号仍来自搜索代理，落笔引用前复核。注意 HeadKV/InstABoost 是方法名而非论文标题，按真实标题引用。TPQA 的 \[CLS\]/\[SEP\] 细节未经全文确认，引用时软化表述。
7. ARR research area：Efficient/Low-Resource Methods 为主、Interpretability 为辅（与上轮对调）。

---

## 附：时间线事实

- ARR 提交：**2026-10-12**（AoE）｜ meta-review：2026-12-18 ｜ NAACL commit：2026-12-23 ｜ 录用通知：2027-02-10 ｜ 会议：2027-06-01~05，旧金山。
- 也接受更早 ARR cycle 的完整审稿结果 commit，但上一轮 2/2/2 的分数不可用，实际目标就是 10 月 cycle。
- NAACL 2027 无匿名期限制（可挂 arXiv）；不允许 dual submission。

来源：[NAACL 2027 官方 CFP](https://2027.naacl.org/calls/main_conference_papers/)。
