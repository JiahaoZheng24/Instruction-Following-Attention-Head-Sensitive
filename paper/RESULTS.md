# Master Results v2（按论文 story 重组，2026-09-04）

> 全部数字取自 runs/*.csv，与 v1（`RESULTS_v1_archive_2026-09-03.md`，按实验批次 W1–W19 记录）一一对应；本版只改组织方式和措辞，不改任何数值。**Yi-1.5-9B 已按作者决定整体移出论文口径**（其选择题协议在 fp16 即乱猜级），下文任何表格都不含 Yi；原始数据仍在 runs/ 存档。
>
> 目标：ICLR 2027（摘要 9/18、全文 9/25 AoE）。story 诊断见 `STORY_DIAGNOSIS_2026-09-04.md`。

---

## 0. 论文一句话与主张清单

**一句话**：低比特 GPTQ 存在一种独特的灾难模式，起因是误差补偿把极少数临界权重的舍入误差扩散进整个网络；同一模型用不做补偿的 RTN 反而高 30–40 分。理解这个原因，就能解释"保护显著权重"什么时候值钱、为什么定位不等于保护、为什么 PPL/MMLU 看不见它。

| # | 主张 | 关键证据（节） |
|---|---|---|
| C1 | 3-bit 下存在与优雅退化不同的**崩塌模式**，2/11 模型出现，不按家族、规模或任何 fp16 统计量出现 | §2 |
| C2 | 崩塌不是信息损失而是**补偿之害**：RTN 反超 GPTQ；2×2 因子；零化探针；事后 vs 环内；保护不足更差；旋转消崩塌 | §3 |
| C3 | **保护的价值有条件**：≈拦下的补偿之害；优雅态、4-bit、2-bit 全部 CI 跨零 | §4 |
| C4 | **定位 ≠ 保护**：因果头、超权重、事后恢复（哪怕 11% 参数）全失败 | §5 |
| C5 | **似然指标探不到崩塌**：PPL/MMLU 漏报 Q14 崩塌；RTN 符号或小型生成探针即可检出 | §6 |

**待补的洞（W20–W25，见 §9）**：配置混淆（damping / asym / 校准语料 / AWQ）、方程级机制、group-scale 伪影、普查扩展、无梯度判据。

---

## 1. 协议

- **量化**：GPTQ，g128 / sym / desc_act / c4 校准 128×2048 / percdamp 0.05（gptq_core.py，IST-DASLab 循环的 faithful port）；fake-quant fp 检查点。RTN 为无补偿对照（同 group / sym）。两套实现：gptqmodel packed（W0，TORCH backend）与自研 v2；主线全部用 v2，packed 仅作复现。
- **保护**：TaCQ 判据 s=|W|·|∇L|·|ΔW_rtn|（2504.07389 faithful 复现，IF 数据梯度，全局分位数阈值）环内豁免；预算 ≈ 线性层参数 0.55%（7B=37.6M，8B=40.2M，14B=70M）= 32 个注意力头的参数量（公平安慰剂对照），另有 1e4–1e7 扫描。
- **旋转**：QuaRot 式 R1（随机正交，离线融合）；恒等性 rotfp_llama 0.7848 ≈ fp16 0.7683。
- **评测**：IFEval 541 条 avg4（四指标均值）为默认分数；GSM8K；Multi-IF；MMLU（5-shot 选项 logit）；WikiText-2 PPL。生成：贪心，max_new_tokens 1280，chat template。
- **统计**：逐题配对 bootstrap 5000 次，95% CI（src/stats_tests.py）。
- **代码语义要点（写作用）**：被保护权重在轮到它之前照常吸收前面列的补偿，轮到时豁免舍入并注入零误差，即"自由 fp16 吸收器"，不是冻结原始值。group scale 在 v2 中由组内最大 |W|（含被保护项）决定，W22 检查此项。

---

## 2. 现象：3-bit 崩塌存在且不可预测（C1）

### 2.1 比特阶梯（同配方 8/4/3/2）

| bits | Qwen2.5-7B | Llama-3.1-8B |
|---|---|---|
| 16 | 0.765 | 0.768 |
| 8 | 0.770 | — |
| 4 | 0.758 | 0.745 |
| 3 | 0.672 | **0.150** |
| 2 | 0.121 | — |

8/4-bit 无损；2-bit 终末（Qwen-7B：none/tacq/randw/sw/2%/5% 全部 0.117–0.139，保护到 5% 预算也无用）；3-bit 是分水岭。4-bit Llama 各臂全在噪声带（none 0.745 / tacq 0.753 / randw 0.768 / heads32 0.755）。8/2/Qwen-4bit 点来自早期管线，冻结协议复跑差 ≤1.5。

### 2.2 11 模型普查（3-bit，同配方）

| 模型 | fp16 | RTN3 | GPTQ3 | Δ vs fp16 | 复读循环 % | regime |
|---|---|---|---|---|---|---|
| Mistral-Small-24B | 0.757 | — | 0.698 | −6.0 | ≈7 | 优雅 |
| OLMo-2-7B | 0.750 | — | 0.688 | −6.3 | 1.5 | 优雅 |
| Qwen2.5-32B | 0.838 | — | 0.762 | −7.6 | ≈7 | 优雅 |
| Mistral-7B | 0.552 | 0.420 | 0.468 | −8.5 | ≈7 | 优雅 |
| Qwen2.5-7B | 0.765 | 0.500 | 0.672 | −9.4 | ≈7 | 优雅 |
| Llama-3-8B | 0.758 | — | 0.611 | −14.7 | 8.3 | 中间 |
| Llama-3.2-3B | 0.750 | — | 0.577 | −17.3 | 11.3 | 中间 |
| Mistral-Nemo-12B | 0.651 | — | 0.476 | −17.5 | 4.3 | 中间 |
| Qwen2.5-3B | 0.658 | 0.130 | 0.411 | −24.7 | 13.7 | 中间 |
| **Qwen2.5-14B** | 0.820 | 0.697 | **0.412** | **−40.9** | **69.1** | **崩塌** |
| **Llama-3.1-8B** | 0.768 | 0.565 | **0.150** | **−61.9** | **92.4** | **崩塌** |

- 掉分连续排开，但复读循环率有断层（中间档最高 13.7%，崩塌 55–92%；fp16 0.7%）。
- 两个崩塌都种子复现：Llama none 两套实现 0.150 / 0.127；Q14 none cs0 0.412 / cs1 0.508（崩塌臂种子方差 ±10，优雅臂 ±1.5，如实以区间报告）。
- **不是尺寸效应**：小模型确实压得更惨（3B −17/−25），但两个崩塌显著偏离趋势线（Llama-3.2-3B −17 vs 同族更大的 3.1-8B −62）。
- **不可从家族/规模预测**：Qwen 7B 优雅 → 14B 崩塌 → 32B 优雅；Llama-3-8B −15 vs Llama-3.1-8B −62（同尺寸隔代翻转）。
- Qwen-7B 校准对照：instruct 校准反而更差（0.6633 vs c4 0.6717）；g64 packed 0.7193。

### 2.3 四态谱系

| regime | IFEval 掉分 | 循环 | 补偿效应 (GPTQ−RTN) | 保护增益 | 成员 |
|---|---|---|---|---|---|
| 优雅 | −2…−9 | ≤7% | 帮忙 +5…+17 | +0.8…+3.3，CI 全跨零 | 5 + Llama@4bit |
| 中间 | −15…−25 | 4–14% | 帮忙 +12…+28 | +7.4（Qwen-3B，p=4e-4）| 4（多为小模型）|
| **崩塌** | −41 / −62 | 55–92% | **行凶 −29 / −41** | **+37 / +49（p<1e-4）** | 2 |
| 终末（2-bit）| −64 | — | — | 任何预算为 0 | 全部 |

判定签名（机制级，不是掉分）：**补偿效应符号为负**。中间档 = 被小尺寸放大的普通退化（补偿仍在帮忙）。

### 2.4 fp16 预测器全部证伪（六模型）

| 模型 | 最大激活尖峰 | 显著性集中度@1e5 | 激活峰度 | regime |
|---|---|---|---|---|
| Qwen2.5-7B | 11,584 | 0.0126 | 15,309 | 优雅 |
| Qwen2.5-32B | 19,072（>50 的通道 1,632 个）| — | — | 优雅 |
| Mistral-7B | 212 | 0.0414 | 23,344 | 优雅 |
| Mistral-24B | 244 | — | — | 优雅 |
| Llama-3.1-8B | 324 | 0.0150 | 21,091 | 崩塌 |
| Qwen2.5-14B | 10,880 | 0.0078 | 4,908 | 崩塌 |

崩塌集 {324, 10880} 与优雅集 {212, 244, 11584, 19072} 完全交错，跨族与族内均成立；显著性集中度与峰度反向预测（Super Weight 论文的尖峰信号也反向）。**推论**：崩塌是 (模型 × 校准 Hessian × 量化器) 三元组的性质，预测量应在量化过程中读（W23），不在 fp16 模型上读。

---

## 3. 原因：补偿是凶器（C2）

### 3.1 RTN 对照：补偿符号翻转

| 模型@3bit | RTN（无补偿）| GPTQ（补偿）| 补偿效应 | regime |
|---|---|---|---|---|
| Qwen2.5-3B | 0.130 | 0.411 | +28.1 帮 | 中间 |
| Qwen2.5-7B | 0.500 | 0.672 | +17.2 帮 | 优雅 |
| Mistral-7B | 0.420 | 0.468 | +4.8 帮 | 优雅 |
| Qwen2.5-14B | 0.697 | 0.412 | **−28.6 害** | 崩塌 |
| Llama-3.1-8B | 0.565 | 0.150 | **−41.5 害** | 崩塌 |

崩塌不是"3 bit 装不下信息"：笨办法在两个崩塌模型上到 0.57–0.70。**这个反转在文献里无先例**（2404.14047、2608.08188 均报告 GPTQ 在所有设置赢 RTN；见 §8）。

### 3.2 2×2 因子表：补偿 × 保护（决定性实验）

| 模型@3bit | RTN | RTN+保护 | GPTQ | GPTQ+保护 |
|---|---|---|---|---|
| Llama-8B（预算 1e5）| 0.565 | 0.601 | 0.150 | 0.671（@40.2M 0.643）|
| Qwen-14B（预算 1e6）| 0.697 | 0.720 | 0.412 | 0.734（@70M 0.782）|

| 衍生量 | Llama | Q14 | 读法 |
|---|---|---|---|
| 补偿效应（无保护）| −41.5 | −28.6 | 补偿是元凶 |
| RTN 下保护增益 | +3.6（p=0.03 [+0.3,+6.7]）| +2.2 | 没有元凶时保护值几分 |
| GPTQ 下保护增益 | +49.3 | +32.2（@70M +37.0）| ≈ 拦下的害 |
| GPTQ+保护 − RTN+保护 | +4.2 | +1.4 | 屏蔽后补偿净有益 |

**机制句**：保护拆掉了补偿唯一的作案工具，而不是关掉补偿；临界集被屏蔽后 GPTQ 反超保护过的 RTN。范围：Llama 1e5 单种子 p=0.03 压线；14B 方向一致幅度更温和，如实报。

### 3.3 Llama-3.1-8B 全干预矩阵（fp16 0.768）

| 基线与保护 | avg4 | 对照与变体 | avg4 |
|---|---|---|---|
| RTN3 | 0.565 | randw 同预算 0.55% | 0.181 |
| GPTQ3 none | 0.150（packed 0.127）| heads32（因果头）| 0.165 |
| +tacq 0.55%（40.2M）| 0.643（cs1 0.654）| super weights only | 0.163 |
| +tacq@1e5 | **0.671** | no-desc_act | 0.156 |
| +tacq@1e4 | 0.246 | g64 none | 0.145 |
| +tacq@1e6 / 1e7 | 0.665 / 0.638 | g64 + tacq | **0.703**（全场最佳）|
| +cols 40M（列结构化）| 0.641 | RTN + tacq@1e5 | 0.601 |
| +c4 通用掩码@1e5 | 0.643 | 旋转 R1 + GPTQ | 0.641 |
| | | 旋转 + GPTQ + tacq | 0.668 |
| | | rotfp（旋转恒等）| 0.785 |

读法：救援判据特异（tacq +49 vs 随机/头/超权重 +1–3）；预算便宜（1e5 = 0.001% 够，1e4 不够）；几何手段（g64、旋转）单独不如靶向保护，但与 tacq 相乘。

### 3.4 Qwen2.5-14B 复刻（fp16 0.820）

| 臂 | avg4 |
|---|---|
| RTN3 | 0.697 |
| GPTQ3 none（s0 / s1）| 0.412 / 0.508 |
| +tacq 0.55%（70M）| **0.782**（fp16 的 95%）|
| +tacq@1e6 | 0.734 |
| +tacq@3e5 | 0.689 |
| +tacq@1e5（s0 / s1）| **0.216 / 0.204（比 none 更差）** |
| +cols 70M | 0.227（Llama 上有效，14B 无效）|
| +randw 70M | 0.480 |
| RTN + tacq@1e6 | 0.720 |
| 旋转 R1 + GPTQ | 0.665 |

GSM8K：fp16 0.926 / none 0.129 / tacq 0.894（恢复 97%）。

### 3.5 预算相变：临界集尺度随模型变化

| 预算（权重数）| Llama-8B | Qwen-14B |
|---|---|---|
| 0 | 0.150 | 0.412 / 0.508 |
| 1e4 | 0.246 | — |
| 1e5 | 0.671 | 0.216 / 0.204 |
| 3e5 | — | 0.689 |
| 1e6 | 0.665 | 0.734 |
| 1e7 | 0.638 | — |
| 0.55%（4e7 / 7e7）| 0.643 | 0.782 |

相变窗口 Llama (1e4, 1e5]、14B (1e5, 3e5]。**保护不足主动致害**（0.216 < 0.412，双种子，有害臂方差仅 ±0.6 vs none 臂 ±10）：只有"豁免改变补偿轨迹"的交互框架能解释。⚠️ W22 需排除 group-scale 伪影后此主张才能入文。实际必需预算 0.001–0.01%，0.55% 是超配 50–500 倍的公平对照设定。

### 3.6 六个机制探针（Llama 除注明外）

| 探针 | 结果 | 对照 | 推论 |
|---|---|---|---|
| fp16 下把临界 1e5 置零 | 0.188 | 随机 1e5 置零 0.776 | 集合 fp16 致命且特异 |
| GPTQ 之后事后恢复临界集 | 0.162 | 同集环内 0.671 | 损伤已扩散到集合之外 |
| 事后恢复 8.22 亿参数（11%，Qwen）| 0.710 | noop 0.684 | 事后天花板 ≈ +3，与恢复量无关 |
| desc_act 关闭 | 0.156 | none 0.150 | 量化顺序不是原因 |
| 权重位移分析 | 临界权重位移是随机的 4–6× | GPTQ ≈ RTN 位移 | 位移幅度不是原因 |
| c4 通用掩码（身份重叠 Jaccard 0.155）| 0.643 | 任务掩码 0.671 | 救援靠结构类型不靠权重身份 |

合成图景：一小撮 fp16 下即超敏感的权重存在；GPTQ 的补偿把它们的量化误差扩散进网络其余部分，所以事后修复不可能，豁免必须发生在环内。**尚缺**：方程级证据（补偿更新 `W[:, j:] -= err_j · Hinv[j, j:]` 对这些权重做了什么）→ W23。

### 3.7 两个临界集的解剖

| 性质 | Llama-3.1-8B（@1e5，101,476 个）| Qwen2.5-14B（@1e6，95.7 万个）|
|---|---|---|
| 相变尺度 | ≈1e5（0.001%）| ≈3e5–1e6 |
| top-1% 输入通道集中度 | 69.7% | 52.8% |
| 主导模块 | 注意力（k_proj 38k、q_proj 26k = 63%）| MLP（gate 221k、down 219k、up 135k = 60%）|
| 与激活离群通道重叠 | 42.0% | 39.0% |
| 列结构化保护 | 有效 0.641 | 失败 0.227 |
| 与超权重关系 | 包含之；sw_only 失败 0.163 | — |

同一机制，无普适解剖学；防御必须逐模型定位。Qwen-7B 优雅臂掩码@37.6M 通道集中度仅 14.9%（散点型、MLP 为主）。

### 3.8 旋转：缓解而非替代

| 臂@3bit | Llama-8B | Qwen-14B | Qwen-7B |
|---|---|---|---|
| GPTQ none | 0.150 | 0.412 | 0.672 |
| 旋转 + GPTQ | 0.641 | 0.665 | 0.688 |
| 旋转 + GPTQ + tacq | 0.668 | — | — |
| GPTQ + tacq（无旋转）| 0.643 | 0.782 | 0.702 |

旋转消除两个崩塌（与补偿 × 临界集框架一致：摊平了离群结构），但 14B 上输靶向保护 11.7 分（[+8.3, +15.2]，p<1e-4）。

---

## 4. 后果一：保护的价值有条件（C3）

### 4.1 剂量-响应：阶跃，不是斜坡

| 模型 / 设置 | 裸量化掉分 | 保护增益 | 95% CI | p |
|---|---|---|---|---|
| Llama-8B @4bit | −2.3 | +0.8 | [−2.2, +3.8] | 0.61 |
| Mistral-7B @3bit | −8.5 | +3.3 | [−0.0, +6.6] | 0.056 |
| Qwen-7B @3bit | −9.4 | +3.0 | [−0.4, +6.4] | 0.085 |
| Llama-8B 旋转 @3bit | −14.4 | +2.7 | [−0.9, +6.3] | 0.13 |
| Qwen-3B @3bit（中间）| −24.7 | +7.4 | [+3.5, +11.1] | 4e-4 |
| Qwen-14B @3bit（崩塌）| −40.9 | +37.0 | [+32.7, +41.2] | <1e-4 |
| Llama-8B @3bit（崩塌）| −61.9 | +49.3 | [+45.5, +53.2] | <1e-4 |

近零平台 + 补偿翻负处的不连续跳变。定量观察（谨慎陈述）：增益 ≈ 补偿之害 + 小常数（害 41.5 → 增 49.3；害 28.6 → 增 37.0；补偿帮忙处增益仅 2–7）。

### 4.2 优雅态：一切选择性保护 ≈ null

| 臂@3bit | Qwen-7B（种子 cs0/cs1/cs2）| Mistral-7B | Llama-8B@4bit |
|---|---|---|---|
| none | 0.6717（.6547/.6709）| 0.4677 | 0.7454 |
| tacq | 0.7018（.6988/.6829）| 0.5003 | 0.7534 |
| randw | 0.6846（.7143/.6673）| — | **0.7684（"最佳"=噪声）** |
| heads32 | 0.6746 | — | 0.7553 |
| g64 / g64+tacq | 0.7193 / 0.7230 | — | — |

四个优雅实例的 tacq 增益符号一致为正但 CI 全跨零 → 论文措辞："优雅态中没有任何选择性保护产生统计显著收益"。散点吸收器机制（事后散点恢复 null 0.6628/0.6699 vs 环内 +3.0）只作点趋势解释。Qwen tacq035 预算变体 0.6882。

### 4.3 与 TaCQ 的和解

TaCQ 主模型 Llama-3-8B-Instruct 在我们普查里是中间档（−14.7）；其 3-bit 增益（GSM8K 52→67，MMLU 56→63）与"增益随补偿之害增长"一致。**待调和**：TaCQ 2-bit 大增益 vs 我们 2-bit 终末（差异疑在真混合精度存储、平均 3.1/2.1 bit、预算格式）；主张限定为"匹配预算与本协议下"。

---

## 5. 后果二：定位 ≠ 保护（C4）

| 候选"重要权重"（Llama@3bit）| 预算 | avg4 | 判决 |
|---|---|---|---|
| 随机权重（安慰剂）| 0.55% | 0.181 | 失败 |
| 32 个因果头 | 0.55% | 0.165 | 失败 |
| 超权重（Yu et al.）| <10 个标量 | 0.163 | 失败 |
| 真临界集，事后恢复 | 0.001% | 0.162 | 失败 |
| TaCQ 显著性，环内 | 0.001% | 0.671 | 救援 |

- 头是因果真实的：Qwen-7B dev 协议 screen_base 0.7434，消融 top-32 fragility 头 → 0.7117（−3.2），随机 32 头三种子 0.7342/0.7502/0.7350（均值 −0.4）→ 8× 选择性；top-16 0.7459、top-64 0.6666；按激活排序 32 头 0.6732（弥散损伤对照）。但保护它们无用；功能重要性与量化临界性是两套排序（ρ≤0.16，Jaccard@32 ≤0.05）。
- 事后恢复天花板（gptq3-Qwen）：noop 0.6835；32 头 0.7010；整个注意力栈（784 头、8.22 亿=11%）0.7095；同预算 MLP 通道 0.7009；MLP 2%/5%/25% → 0.689/0.706/0.714。**上限 ≈ +3，与恢复量无关**。W2 十六臂（dev3-top16/32/64、随机×3、act、layer01、qkv/o 分解、MLP×3、noop）全部落在 −0.7～+1.8 噪声带；prot32_qkv/o 0.6767/0.6914。
- heads32 保护的 MMLU/PPL = 0.6691/9.44 ≈ none。

---

## 6. 后果三：似然指标探不到崩塌（C5）

### 6.1 检测地图（11 点）

| 模型@3bit | IFEval Δ | 循环 % | PPL fp16→量化（比值）| MMLU Δ |
|---|---|---|---|---|
| Mistral-24B | −6.0 | ≈7 | 5.6→6.6（1.17×）| −5.2 |
| Qwen-32B | −7.6 | ≈7 | 5.3→7.1（1.33×）| −4.3 |
| Mistral-7B | −8.5 | ≈7 | 5.5→6.6（1.20×）| −4.7 |
| Qwen-7B | −9.4 | ≈7 | 7.5→9.7（1.30×）| −7.5 |
| Llama-3-8B | −14.7 | 8.3 | 8.3→17.4（2.10×）| −9.9 |
| Llama-3.2-3B | −17.3 | 11.3 | 11.1→1749（**158×**）| −10.0 |
| Mistral-Nemo-12B | −17.5 | 4.3 | 6.1→9.1（1.49×）| −10.2 |
| Qwen-3B | −24.7 | 13.7 | 8.6→12.0（1.40×）| −13.2 |
| **Qwen-14B** | −40.9 | 69.1 | 5.7→7.7（**1.35×**）| **−6.4** |
| **Llama-3.1-8B** | −61.9 | 92.4 | 7.2→59.2（8.2×）| −12.2 |

OLMo 未跑 PPL/MMLU。Qwen 各臂 MMLU/PPL：fp16 .7421/7.46、v2none .6668/9.71、v2tacq .6683/9.35、packed .6544；Llama v2l_tacq .5882/9.64。

### 6.2 Qwen-14B：teacher-forced 完好，自由生成崩塌（主线主张）

- cs0 PPL 7.713 / MMLU 0.7349，cs1 7.739 / 0.7367（种子稳健），IFEval 0.412 / 0.508。
- 逐响应审计：q14-none IFEval **69.1% 重复循环**（fp16 0.7%、tacq 4.1%、优雅带 4–7%、llama-none 92.4%），平均输出 4325 字符（≈打满上限）；GSM8K 55.3% 循环；1149 个错答中 0 个在 #### 后含正确数（无提取 bug）；非退化的 31% 响应通过率也只有 34.1%（fp16 77.6%）。
- 措辞（收窄版）："在我们的一个崩塌实例中（n=1 模型、2 种子），teacher-forced 指标完全探测不到自由生成的灾难性退化；可靠检测需要小规模生成探针（几十条；输出长度打顶 + 重复率即廉价检测器）或一次 RTN 对照。"

### 6.3 Llama-3.2-3B 反向解离（→ 附录）

fp16 PPL 11.05 → 量化 1748.9（158×），比 Llama-8B 崩塌的 8.2× 大 20 倍，但生成仅中间档（−17.3，循环 11.3%，MMLU −9.9）。W19 微观：量化后 wikitext 中位 NLL 7.23（fp16 1.36）、top-1 一致率 0.116（fp16 0.508）、剔除最差 1% token 后 PPL 仍 1158 → 均匀的裸文本似然崩溃；同 checkpoint MMLU 0.508 正常 → 领域选择性损伤（裸文本续写崩、对话格式完好）。Llama-8B 恰为反象限（wikitext 中度 55.6 / top-1 0.40，对话崩 0.15）。定稿句："固定语料 PPL 度量的是该领域的损伤，不是部署能力。"论文正文只留一句，细节进附录。

### 6.4 跨基准（同一批检查点）

| GSM8K acc | fp16 | GPTQ3 | +tacq |
|---|---|---|---|
| Llama-8B | 0.843 | 0.002（可解析率 8%）| 0.616 |
| Qwen-14B | 0.926 | 0.129 | 0.894 |
| Qwen-7B | 0.901 | 0.778（packed 0.795）| 0.839 |

Multi-IF（prompt_strict t1/t2/t3）→ 附录：Llama fp16 .722/.559/.409，none .086/.008/.001，tacq .592/.372/.215；Qwen fp16 .721/.465/.306，gptq3(packed) .638/.283/.105（v2none t1 .633），tacq .705/.352/.152。优雅损伤随轮次复利（Qwen 相对差 −8% → −20%）。

---

## 7. 统计总表（配对 bootstrap，5000 次）

| 对比（IFEval avg4）| Δ | 95% CI | p |
|---|---|---|---|
| Llama@3bit tacq−none（崩塌）| +49.3 | [+45.5, +53.2] | <1e-4 |
| Q14@3bit tacq−none（崩塌）| +37.0 | [+32.7, +41.2] | <1e-4 |
| Q14 tacq−旋转 | +11.7 | [+8.3, +15.2] | <1e-4 |
| Qwen-3B tacq−none（中间）| +7.4 | [+3.5, +11.1] | 4e-4 |
| Llama RTN tacq−none | +3.6 | [+0.3, +6.7] | 0.03 |
| Mistral-7B tacq−none | +3.3 | [−0.0, +6.6] | 0.056 |
| Qwen-7B tacq−none | +3.0 | [−0.4, +6.4] | 0.085 |
| Llama 旋转 tacq−none | +2.7 | [−0.9, +6.3] | 0.13 |
| Llama@4bit tacq−none | +0.8 | [−2.2, +3.8] | 0.61 |
| Qwen-7B heads32−none | +0.3 | [−2.7, +3.2] | 0.87 |
| 旋转恒等 rotfp−fp16 | +1.7 | [−0.4, +3.7] | 0.11 |

---

## 8. 文献：定位、引用与措辞（写作直接用）

### 8.1 最近邻（必须逐篇划界）

| 论文 | 类型 | 他们占的地 | 我们的差异 / 措辞 |
|---|---|---|---|
| **TaCQ**（Xiao, Stengel-Eskin, Bansal，UNC，arXiv 2504.07389，COLM 2025）| 方法 | 判据 s=\|W\|·\|∇L\|·\|ΔW\|；混合精度真存储（~0.35% 16-bit）；3-bit Llama-3-8B-Instruct GSM8K 67 vs GPTQ 52，MMLU 63 vs 56；2-bit 大增益；从未测 IF；无 RTN 对照；无机制实验 | 我们借其判据为仪器，回答 when/why。**红线**：救援数字永远写 "consistent with TaCQ"；明说 TaCQ 从未做 RTN 对照，故其增益无法区分"保信息"与"拆补偿" |
| **Super Weight**（Yu et al.，ND+Apple，arXiv 2411.07191）| 分析 | 1–6 个标量剪掉即 PPL 爆 1000×；data-free 定位；保 SW + 激活处理即可做好 RTN 量化 | "极小 fp16 致命集"概念的 lineage，零化探针大方引用。差异：临界集 1e5 级、通道结构化、显著性定义；sw_only 救不活补偿崩塌（0.163）；SW 尖峰信号反向预测 regime。"两种病两副药" |
| **HeRo-Q**（arXiv 2601.21626，2026-01）| 方法 | **独立报告 Llama-3.1-8B W3A16 GPTQ 灾难**（WikiText PPL 20.13，GSM8K 26.36 vs fp16 77.17）与 Llama-3.2-1B W3 GSM8K 15.19；诊断为 Hessian 各向异性（"low-error, high-loss 悖论"）；用对角平滑 + 学习旋转做 Hessian 条件化，GSM8K 回到 70.15 | Intro 引为独立复现。划界：他们无 RTN 基线、无生成式评测、不解释为什么是这个模型。若 W20 damping 扫描单调恢复，写 "consistent with the Hessian-conditioning account"，然后给他们没有的逐模型判别量 |
| **Llama-3 量化实证**（arXiv 2404.14047）| 分析 | Llama-3-8B base W3 g128：GPTQ PPL 8.2 / acc 61.7，RTN 27.9 / 40.2，AWQ 8.2 / 64.4，QuIP 7.5；per-channel W3 GPTQ PPL 13.0；原文 "under 2-3 bits, GPTQ causes severe accuracy collapse" 主要指 2-bit | 与我们 Llama-3-8B-Instruct 中间档一致；说明 RTN 反超是 3.1 代的新现象 |
| **Signal–Noise**（CASIA，arXiv 2608.08188）| 分析 | Llama-2/3、Mistral、Qwen2.5-7B、Qwen3；W4→W3 中位掉分 RTN 43.7 vs GPTQ 12.7；**RTN 从未赢 GPTQ**；SNR 分解与跨层累积 | 我们的反转与其相反 → 配置对照（W21）必须补；其 Discussion "混合精度应瞄准被任务激活放大最强的误差" 可引 |
| **Two Failure Modes**（CASIA，arXiv 2604.19884，ACL 2026 Findings）| 分析 | 主模型 Llama-3.1-8B，GPTQ；4-bit 信号退化 vs 2-bit 计算坍塌（FFN 门符号翻转 >30%、注意力熵坍塌、CKA 拓扑丢失）；MLP 比注意力脆弱；EoRA 修不了坍塌 | 3-bit 未归类、无 RTN；我们的崩塌是他们两模式之间的第三种：不由比特数决定而由补偿决定 |
| **GPTQ = Babai 最近平面**（arXiv 2507.18553）| 理论 | GPTQ 等价于格上 CVP 的 Babai 算法；误差界**仅在无裁剪时成立**；提出避免裁剪的变体 | 直接支持"裁剪普查"诊断（gptq_core 在补偿后 clamp）；数学合作者的切入点 |
| **GPTAQ / GPTQv2**（arXiv 2504.02692）、**QEP**（2504.09629）、**First-Order Error Matters**（2507.11017）| 方法 | GPTQ 逐层 / 跨层误差累积，非对称校准、误差传播修正 | 机制章 lineage："补偿误差累积"已知，但无人报告它能反超 RTN |
| **LLM-KICK**（ICLR 2024）| 分析/基准 | PPL 掩盖压缩损伤，须任务评测 | 互补：我们发现 MMLU（选项似然）也掩盖崩塌，PPL 还会假报警 |
| **Scaling Laws for Precision**（2411.04330）、**低比特偏爱欠训练模型**（2411.17691，ACL 2025）| 分析 | 精度当作连续折损；PTQ 损伤随训练 token 单调恶化 | 我们的 7B/14B/32B 非单调翻转不被任何连续律预测；离散 regime vs 连续律 |
| OWQ / SpQR / SqueezeLLM / SliM-LLM | 方法 | 敏感权重混合精度保 PPL（全部建于 GPTQ 之上）| 无 regime 概念、无机制、无安慰剂；cols 臂引 OWQ 存储格式 |
| AWQ | 方法 | 激活感知缩放保护 1% 通道，非补偿 | 预测其不崩（W21 验证）|
| QuIP / QuaRot / SpinQuant | 方法 | 旋转去离群 | 我们把旋转当 regime 杠杆测：消崩塌但 14B 输 tacq 11.7 |
| massive activations / attention sinks | 分析 | 激活离群结构 | 其标记不预测 regime（§2.4）|
| Hase et al. 2023（NeurIPS）| 分析 | 定位 ≠ 编辑 | 我们是其量化域对应物（heads null 全 regime）|
| CWP（2601.12033）、FAQ（2601.11200）| 方法 | 关键权重保护需 ~60% FP16；校准侧修复 IFEval +0.5 | 预算差 3 个量级 |
| 一篇 2025 评测（疑为 2509.03054 或 2507.17417，**落笔前核实**）| 分析 | GPTQ 在 Llama-3.2-1B 多份实现上均显著退化，"不太可能是单一仓库或 checkpoint 的伪影"，归因校准敏感 | 与我们"两套实现都崩"同一逻辑 |

### 8.1b 第三轮近邻（2026-09-05，story 改为"校准集过拟合"后新增；★ 本轮核对过）

| 论文 | 他们有的 | 我们有而他们没有的 / 措辞 |
|---|---|---|
| ★ **Williams & Aletras**, On the Impact of Calibration Data in PTQ and Pruning（ACL 2024，2311.09755）| 首个系统研究校准数据对 GPTQ/SpQR/SparseGPT/Wanda 的影响；4-bit；9 个模型；**明确写 GPTQ 会过拟合校准集**（PTB 校准时退化）；下游任务方差"substantial"但属中等幅度；建议公开校准集、多集评测 | **最需要小心的一篇。** 他们发现的是敏感性，没有崩塌、没有 RTN 反超、没有机制量、没有检测器、没有治愈等价。Intro 必须主动引用："calibration sensitivity is known (Williams & Aletras); we show when it becomes catastrophic, why, and that three cures coincide." |
| ★ **KronQ**（2607.07964）| "LLaMA-3-70B 的列离群权重让 GPTQ/GPTAQ 的 OBS 更新产生退化解"，W3 PPL 2600、GPTAQ NaN；归因权重侧、Llama-3 家族特异 | 只有 70B、只测 PPL、无 RTN、无 8B、无生成评测；把它当方法动机。我们的反证：若只是权重离群，换校准语料不应治愈（Llama wikitext .624、Q14 instruct .754）。 |
| ★ The Uniqueness of LLaMA3-70B Series with Per-Channel Quantization（2408.15301）| Llama-3-70B 在 W8A8 per-channel 下独特退化，归因权重分布；明说 3.1-8B 稳健 | 他们的现象在 8-bit、我们在 3-bit；引为"Llama-3 家族权重分布特殊"的旁证 |
| ★ DASH-Q, Robust Ultra Low-Bit PTQ via Stable Diagonal Curvature（2604.13806）| Hessian 方法在低比特退化的原因是"小校准集导致曲率估计噪声"；对角近似 + IRLS | 与我们"过拟合"同源，但是方法论文、无 RTN 反超、无分布偏移量。引用为同方向 |
| Coverage-Based Calibration via Weighted Set Cover over Outlier Channels（2604.24008）| 校准集失败是因为没覆盖离群通道；AWQ/GPTQ INT4 选样方法 | 校准选样线；我们的 chat 校准治愈可以用他们的框架解释一部分 |
| Understanding and Selecting Calibration Data for LLM Quantization（OpenReview pfw3saHzGU，**未能打开，投稿前必须读**）| 校准敏感性分析 + 基于激活的选样 | 潜在最近邻，需确认是否有 GPTQ vs RTN、instruct 校准 |
| Beware of Calibration Data for Pruning LLMs（2410.17711）；Self-calibration（2410.17170）；Preserving LLM Capabilities through Calibration Data Curation（2510.10618）| 剪枝/量化的校准数据合成与筛选 | 校准数据线的其余成员，一句话引 |
| Training Dynamics Impact PTQ Robustness（2510.06213）| 训练超参（学习率）决定 checkpoint 对 PTQ 的稳健性 | "为什么是这个模型"的训练侧解释候选；与我们的"量化时可预测、fp16 不可预测"互补 |
| GPTQ-intrinsic LoRA（2606.01412）；QAM-W（2605.26339）；OffQ（2606.07116）| 提到用 10× damping 避免大模型数值崩溃；结构化离群处理 | damping 被当作数值稳定手段而非正则化——我们的差异点 |
| GPTQModel 库（默认 damp_percent 0.1，推荐 0.1；AutoGPTQ/原始 GPTQ 0.01）| — | **我们 Llama ρ=0.2 仍崩（.189）、Q14 ρ=0.1 仍崩（.422）**：社区推荐值救不了，必须写明 |

### 8.2 现象与动机引文

2409.11055（IJCAI 2025，1B–405B 量化 instruct 模型评测：IF 与幻觉是例外掉分项）；UniComp 2602.09130；Give Me BF16 2411.02355（ACL 2025）；Accuracy is Not All You Need 2407.09141（flips）；Alignment Collapse 2606.09864（KV 量化侧 IFEval 崩塌，未修复）；Quantization Meets Reasoning 2505.11574；ACBench 2505.19433；llama.cpp 量化评测 2601.14277（Llama-3.1-8B-Instruct IFEval 描述性记录）。

### 8.3 ICLR 同类先例

LLM-KICK（ICLR'24）、Scaling Laws for Precision 线、层剪枝无效性分析线；方法侧 GPTQ（ICLR'23）、OmniQuant/SpQR（ICLR'24）、LeanQuant（ICLR'25）。我们相对已录取分析论文的超配项：因果机制对照全套 + 配对统计 + 可操作决策规则；短板：fake-quant、GPTQ 家族、≤32B、崩塌 n=2。

---

## 9. W20–W27 裁决（2026-09-04 晚，56 臂全部回收；IFEval avg4）

> 复读循环率此节用 src 外的临时度量（40 字符块重复 ≥4 次或长度 ≥3500），比 v1 的人工审计口径略高（v2l_none 96% vs 92%；q14 78% vs 69%），只作相对比较。

### 9.1 W20 damping 扫描：**一个超参数治愈两个崩塌**

| percdamp | Llama-8B（fp16 .768，RTN .565）| 循环 | Qwen-14B（fp16 .820，RTN .697）| 循环 |
|---|---|---|---|---|
| 0.01（GPTQ 默认）| 0.128 | 51% | 0.298 | 77% |
| 0.05（冻结协议）| 0.150 | 96% | 0.412 / 0.508 | 78% |
| 0.2 | 0.189 | 68% | **0.707** | 16% |
| 1 | 0.152 | 44% | 0.748 | — |
| 5 | **0.642** | 11% | **0.786** | 6% |

- Q14 单调恢复，阈值在 (0.05, 0.2]；Llama 在 (1, 5] 之间跳变。两者在 damp 5 都**超过 RTN** 并**追平最佳保护臂**（Llama tacq 0.643；Q14 tacq@70M 0.782）。
- 冻结协议 0.05 已比 GPTQ 默认 0.01 保守，审稿人不能把崩塌归咎于 damping 选择。
- Llama damping 臂的 PPL / MMLU：0.01 → 21.8 / 0.551；0.2 → 81.8 / 0.546；1 → 85.8 / 0.559；5 → 23.0 / 0.533。**PPL 21.8 的模型 IFEval 只有 0.128**，与 HeRo-Q 报告的 W3 GPTQ PPL 20.13 吻合（他们大概率用了默认 damping 且只测了似然）；治愈臂 damp5 的 PPL/MMLU 反而不比崩塌臂好。似然解离再添四点。

### 9.2 W21 配置混淆：**校准语料单独就能治愈 Llama；asym 不能；AWQ 不崩**

| Llama-8B 臂 | avg4 | 循环 |
|---|---|---|
| c4 sym（冻结）| 0.150 | 96% |
| asym | 0.212 | 99% |
| **instruct 校准** | **0.611** | 26% |
| **wikitext 校准** | **0.624** | 12% |
| AWQ 式缩放 + RTN，sym | 0.597 | 15% |
| AWQ 式，asym | 0.663 | — |

Q14：AWQ sym 0.678 / asym 0.755（无崩塌）。GSM8K：AWQ Llama 0.425、Q14 0.839（fp16 0.843 / 0.926）；MMLU/PPL：AWQ Llama 0.504 / 12.1，Q14 0.720 / 8.68。
- **崩塌是 c4 校准特异的**（c4 两个种子都崩；wikitext 与 instruct 都不崩）。混淆待拆：c4 臂的 token 数（128 篇短文 ≈ 6 万 token）比 wikitext（128×2048 ≈ 26 万）少 4 倍，而 instruct 只有几千 token 却也不崩（它是同分布）。W28 用 c4×512、wikitext×32、c4 种子 2 拆开"语料 vs 数量 vs 分布"。
- AWQ 预测兑现：无补偿的量化器不崩，补偿特异性坐实。

### 9.3 W22 group-scale 伪影：**"保护不足有害"主张作废**

| 臂 | 冻结 scale（含被保护项）| 排除被保护项 |
|---|---|---|
| Q14 tacq@1e5 cs0 / cs1 | 0.216 / 0.204 | **0.609 / 0.386** |
| Llama tacq@40.2M | 0.643 | 0.684 |
| Llama tacq@1e5 | 0.671 | 0.653 |

修正后 1e5 臂落在 none（0.412 / 0.508）的种子噪声带附近（一上一下），"主动致害"消失。§3.5 的相变窗口改写为"1e5 下保护近零，3e5 起有效"；救援臂本身不受影响（±3 内）。

### 9.4 W23 机制日志：**GPTQ 的解在 chat 输入下输给 RTN，且严重度按崩塌排序**

| 模型 | H 对角线相关（c4 vs chat）| chat-Hessian 目标比 GPTQ/RTN 中位 | GPTQ 输给 RTN 的模块数 | c4 目标比 | 裁剪比例 | 条件数最大 |
|---|---|---|---|---|---|---|
| Llama-8B（崩）| **0.543** | **1.02** | **124 / 224** | 0.40 | 0.20% | 2.9e5 |
| Qwen-14B（崩）| 0.721 | 0.91 | 88 / 336 | 0.36 | 0.20% | 2.8e5 |
| Qwen-7B（优雅）| 0.740 | 0.86 | 34 / 196 | 0.38 | 0.20% | 3.4e5 |

- 在自己的校准目标上三者都大幅赢 RTN（比值 0.36–0.40）；换成 chat 输入，Llama 的 GPTQ 解整体不比 RTN 好（中位 1.02，5–28 层 1.0–1.5），Q7 仍赢（0.86）。**机制 = 补偿过拟合了一个不迁移到部署分布的校准 Hessian**，与 W21 的语料治愈、W20 的正则化治愈自洽。
- **裁剪不是机制**（三模型都是 0.2%，Babai 裁剪假说否定）；条件数、发送质量集中度、位移比都不分离两类。**唯一分离量是 chat-Hessian 目标比**，这是量化时可算的预测器候选（需在 15 模型上验证）。
- 最差模块与临界集解剖对上：Llama 是 5–10 层的 k_proj / q_proj（比值 1.3–1.5，注意力主导）；Q14 是第 4 层 down_proj（比值 **257**，MLP 主导）。
- 分歧起点：Llama 残差流余弦在第 3 层 0.53、第 4 层 0.23，prompt token top-1 一致率 0.038 —— 前四层就被毁；Q14 渐进（第 4 层 0.99 → 末层 0.65，top-1 0.535）；Q7 渐进（末层 0.76，top-1 0.571）。

### 9.5 W25 无梯度判据：**|W|·√H_ii 与 TaCQ 等效**

Llama hmag@1e5 0.629、@40M 0.657（tacq 0.671 / 0.643）；Q14 hmag@1e6 0.750（tacq 0.734）。实用修复不需要梯度。

### 9.6 W26 诱导崩塌：**优雅模型推不崩，补偿符号始终为正**

| 臂 | Qwen-7B（冻结 0.672，RTN 0.500）| Mistral-7B（冻结 0.468，RTN 0.420）|
|---|---|---|
| percdamp 0.001 | Cholesky 失败（H 非正定）| 0.434 |
| n_calib 8 | 0.611 | 0.417 |
| per-channel g=−1 | 0.264（RTN g=−1 0.176；循环 64% vs 100%）| 0.295（RTN g=−1 0.182）|

极端配置下分数可以掉到崩塌区间，但 GPTQ 仍高于同配置 RTN → 这是信息损失，不是补偿崩塌。临界结构是模型侧必要条件。

### 9.7 W24 普查扩展：15 模型，仍 2 崩塌；符号判据 9/9 成立

| 模型 | fp16 | GPTQ3 | RTN3 | Δ | 补偿效应 | 循环（GPTQ）|
|---|---|---|---|---|---|---|
| Falcon3-7B | 0.770 | 0.671 | 0.608 | −9.9 | +6.3 | — |
| Mistral-7B-v0.2 | 0.569 | 0.493 | 0.440 | −7.6 | +5.3 | — |
| Llama-3.2-1B | 0.526 | 0.309 | 0.259 | −21.8 | +5.0 | 35% |
| SmolLM2-1.7B | 0.543 | 0.205 | 0.185 | −33.9 | +2.0 | 37%（RTN 94%）|

gemma-2 两个模型 403（HF 许可未接受）。SmolLM −34 超过原中间档上限，"掉分断层"措辞作废；**分型只用补偿符号**：15 模型中有 RTN 参照的 9 个，符号为负的仍只有 Llama-3.1-8B 与 Qwen-14B。HeRo-Q 报告的 Llama-3.2-1B W3 失败在我们这里是小模型的普通退化（GPTQ > RTN）。

### 9.8 对论文主张的影响（必须改的）

1. **C2 机制句改写**："补偿 × 临界集交互" → "**补偿过拟合校准 Hessian**：GPTQ 在 c4 上的最优解不迁移到 chat 输入；三种互相独立的正则化（加大 damping、换校准语料、环内豁免高显著性权重）都能治愈，无补偿的量化器（RTN/AWQ）不会得病"。临界集保留为"过拟合最严重的模块"（Llama 早期 k/q_proj，Q14 第 4 层 down_proj）。
2. **C3 改写**：保护的价值 = 正则化的价值。damp 5 与 tacq 在两个崩塌上打平（0.642 vs 0.643；0.786 vs 0.782），优雅态两者都近零（Q7 待 W28 验证）。这是对 TaCQ/SpQR 一系的更强解释：**它们的低比特增益里，有多少只是把补偿正则化了？**
3. **删除**："保护不足主动致害"（W22 伪影）；"掉分断层"（SmolLM）。
4. **新增**：damping 阶梯图（论文 Figure 2）；chat-Hessian 目标比作为量化时预测器（W28 后扩到 15 模型可成正结果）；c4 特异性（写明并拆混淆）。
5. **HeRo-Q 对接**：他们的 PPL 20.13 ≈ 我们 damp 0.01 臂的 21.8，而该臂 IFEval 0.128 —— 引用为独立复现的同时指出似然指标没让他们看到生成侧崩塌。

### 9.9 W28 + gemma 裁决（2026-09-05）

**damping 阶梯完整版（IFEval avg4；括号内为临时口径循环率）**

| percdamp | Llama-8B（RTN .565）| Qwen-14B（RTN .697）| Qwen-7B 优雅（RTN .500）|
|---|---|---|---|
| 0.01 | .128（51%）| .298（77%）| — |
| 0.05 冻结 | .150（96%）| .412（78%）| .672（10%）|
| 0.1 | — | .422（61%）| — |
| 0.2 | .189（68%）| **.707**（16%）| — |
| 1 | .152（44%）| .748 | — |
| 2 | **.578**（11%）| — | — |
| 3 | .652（12%）| — | — |
| 5 | .642（11%）| **.786**（6%）| **.702**（8%）|
| 10 | .656 | — | — |
| 20 | .676 | .757 | .683 |
| 50 / 100 | .673 | .761 | — |

- 阈值：Llama (1, 2]，Q14 (0.1, 0.2]。阈值之上是宽平台（Llama .64–.68，Q14 .75–.79），到 damp 50/100 仍高于 RTN 约 10 分：正则化后的补偿始终是净资产。
- **Qwen-7B damp5 = .702 = tacq .702（+3.0，同为不显著）**：保护 ≈ 正则化在优雅态也成立。

**校准语料 × 数量 × 分布**

| 臂 | Llama-8B | Q14 |
|---|---|---|
| c4 ×128（seed 0 / 1 / 2）| .150 / .127（packed）/ .158 | .412 / .508 |
| c4 ×512 | **.151，但失败模式变了：拒答/极短回复（中位 53 字符，循环 1.8%）** | — |
| wikitext ×128 | .624 | **.197（循环 94%）** |
| wikitext ×32（token 数 ≈ c4×128）| .246（连贯但退化，循环 16%）| — |
| instruct（chat 格式，仅几千 token）| .611 | **.754** |

- 哪种裸文本语料有毒是模型特异的（Llama：c4；Q14：wikitext），但 **chat 格式校准治愈两者**，且用的 token 最少 → 不是数量问题，是分布问题。c4 加到 512 篇不治愈，只把循环换成拒答：**崩塌至少有两种表型（循环、拒答），循环率单独不是充分检测器，需要"长度异常（双向）+ 重复 + 小规模任务检查"的生成探针**。
- 优雅模型 Qwen-7B 的 instruct 校准略差（.663 vs .672）：分布偏移只在有临界结构的模型上致命。

**机制闭环（Llama，chat-Hessian 目标比 GPTQ/RTN）**

| 臂 | H 对角线相关 c4/chat | 中位比 | GPTQ 输给 RTN 的模块 | 残差流余弦 L4 | prompt top-1 一致率 |
|---|---|---|---|---|---|
| c4 冻结（崩）| .543 | 1.02 | 124/224 | .23 | .038 |
| wikitext（愈）| .648 | .92 | 61/224 | .84 | .616 |
| damp5（愈）| .691 | .82 | **0/224** | .86 | .628 |
| RTN | — | — | — | .85 | .500 |

治愈臂的分歧曲线与 RTN 重合并在末层反超（.68/.67 vs .58）。**GPTQ 崩塌 = 校准集过拟合**：damping 是层级最小二乘的岭正则；chat 校准消除偏移；显著权重豁免删掉过拟合最重的坐标。三条路殊途同归。

**gemma（普查 17 模型）**：gemma-2-9b .767 → GPTQ .725（RTN .711，+1.4）；gemma-2-2b .583 → .504（RTN .385，+11.9）。均优雅。**17 模型、11 个 RTN 符号点，负号仍只有 Llama-3.1-8B 与 Qwen-14B。**

**遗留**：Q14 RTN checkpoint 已被删除（W28.19 失败，Q14 RTN 分歧曲线缺）；comp_stats 汇总未含新臂（数字系手工汇总）。

### 9.10 W29（jobs/w29_predictor.sh，20 臂）

14 个普查模型的量化时统计（chat-Hessian 目标比是否只在两个崩塌模型 ≈1）；damp5 臂的 GSM8K / MMLU（治愈是否全能力）；2-bit damping（"终末"是否只是 damping 不足：Qwen-7B 2-bit damp5，Llama 2-bit none/damp5）。

### 9.10b W29 前 15 臂裁决（2026-09-05；磁盘写满，16–20 待重跑）

**量化时预测器：17 模型全部算完，两个崩塌排名第 1、第 2。**

| 模型 | regime（Δ）| chat-H 目标比中位 | GPTQ 输给 RTN 的模块占比 | 最大模块比 | H 对角相关 c4/chat |
|---|---|---|---|---|---|
| **Llama-3.1-8B** | **崩塌 −62** | **1.02** | **0.55** | 1.5 | **0.54** |
| **Qwen2.5-14B** | **崩塌 −41** | **0.91** | **0.26** | **256.8** | 0.72 |
| Qwen2.5-7B | 优雅 −9 | 0.86 | 0.17 | 1.7 | 0.74 |
| Mistral-7B-v0.2 | 优雅 −8 | 0.86 | 0.15 | 1.3 | 0.67 |
| Llama-3.2-3B | 中间 −17 | 0.84 | 0.12 | 1.2 | 0.69 |
| Mistral-7B | 优雅 −9 | 0.83 | 0.12 | 1.2 | 0.68 |
| Qwen2.5-32B | 优雅 −8 | 0.82 | 0.17 | 3.8 | 0.62 |
| Llama-3.2-1B | 中间 −22 | 0.80 | 0.05 | 1.1 | 0.68 |
| SmolLM2-1.7B | 中间 −34 | 0.78 | 0.10 | 1.4 | 0.72 |
| Mistral-24B | 优雅 −6 | 0.77 | 0.20 | 1.5 | 0.80 |
| Llama-3-8B | 中间 −15 | 0.76 | 0.06 | 1.2 | 0.68 |
| OLMo-2-7B | 优雅 −6 | 0.76 | 0.05 | 1.1 | 0.80 |
| Qwen2.5-3B | 中间 −25 | 0.75 | 0.14 | 6.1 | 0.65 |
| Falcon3-7B | 优雅 −10 | 0.73 | 0.12 | 1.4 | 0.73 |
| Mistral-Nemo-12B | 中间 −18 | 0.60 | 0.04 | 1.2 | 0.72 |
| gemma-2-9b | 优雅 −4 | 0.57 | 0.03 | 1.2 | 0.81 |
| gemma-2-2b | 优雅 −8 | 0.56 | 0.00 | 0.9 | 0.76 |

- 排序变量 = "GPTQ 的解在 chat 输入下相对 RTN 的目标比"，只需一次 GPTQ 过程加 64 条 chat prompt 的第二个 Hessian，零部署成本。fp16 统计量全部失败之后，这是第一个在 17 模型上把两个崩塌排到最前的量。
- 两种崩塌指纹：Llama 是**弥散过拟合**（55% 模块输给 RTN，H 相关性全场最低 0.54）；Q14 是**单模块灾难**（第 4 层 down_proj 比值 257，其余正常）。检测器用"中位比 + 最大模块比"两个信号。
- 措辞留意：中位比与掉分不是单调的（gemma-9b 0.57 掉 4，nemo 0.60 掉 18），它预测的是**补偿是否有害**，不是掉多少分，与符号判据同一含义。
- GSM8K Llama damp5 = 0.583（none 0.002，tacq 0.616，fp16 0.843）：damping 治愈是全能力的。

W29 任务 16–20（Q14 damp5 GSM8K/MMLU、三个 2-bit damping 臂）因 /store01 写满未完成，清理后 `qsub -t 16-20 jobs/w29_predictor.sh` 重跑。

### 9.10c W29 后 5 臂裁决（2026-09-05）——**实验三次冻结**

- **治愈是全能力的**：Q14 damp5 GSM8K 0.892（none 0.129，tacq 0.894，fp16 0.926）；MMLU 0.745 / PPL 7.62（none 0.735 / 7.71）。Llama damp5 GSM8K 0.583（none 0.002，tacq 0.616，fp16 0.843）。
- **2-bit 仍是终末，damping 救不了**：Qwen-7B 2-bit damp5 0.123（none 0.136，tacq 0.133，循环 90%）；Llama 2-bit none 0.150、damp5 0.117。2-bit 是信息损失，不是补偿过拟合；"上有 4-bit 无病、下有 2-bit 无药、3-bit 是唯一有病也有药的档"保留。
- 磁盘清理：41 个 checkpoint 列为可删，103 份协议 JSON 归档于 runs/protocols。

🔒🔒🔒 **第三次冻结（2026-09-05）：W1–W29，17 模型，~230 臂。此后只写不跑；rebuttal 储备：70B、TaCQ 2-bit 真混合精度复现、Q14 RTN 分歧曲线（需重建 checkpoint）、damping 阈值臂种子副本。**

### 9.11 论文主张终版草案

- **C1** 3-bit GPTQ 存在补偿崩塌模式，17 模型中 2 个，不可从 fp16 统计量预测，但可在量化时用 chat-Hessian 目标比检测（W29 验证）。
- **C2** 崩塌是**校准集过拟合**：补偿把层级最小二乘拟合到不迁移的校准 Hessian 上。证据：OOD 语料致崩、chat 校准治愈、岭正则（damping）治愈、GPTQ 解在 chat 输入下输给 RTN、无补偿量化器免疫、优雅模型无法诱导。
- **C3** "保护显著权重"的价值 = 正则化的价值：崩塌态 +37/+49 与 damp5 打平；优雅态两者同为 +3 不显著；4-bit 零。
- **C4** 定位 ≠ 保护（不变）。
- **C5** 似然指标探不到崩塌，且崩塌有循环与拒答两种表型；一次 RTN 对照或 50 条生成探针即可检出。damp 0.01 臂 PPL 21.8 / IFEval .128 = HeRo-Q 报告的 PPL 20.13 的另一面。

（原 W28 计划文字见下）

#### （存档） W28 收口批（jobs/w28_closure.sh，19 臂）

Llama damping {2,3,10,20,50}、Q14 {0.1,20,100}、Q7 {5,20}（优雅态 damping 是否也近零）；校准拆混淆（Llama c4×512、c4 种子 2、wikitext×32；Q14 wikitext / instruct）；重建被删的 Llama RTN checkpoint 并跑 RTN / damp5 / calwiki 的分歧曲线；damp5 与 wikitext 的机制日志（chat 目标比应降到 1 以下）；Q14 RTN 分歧。

---

## 9b. 原计划与预注册读法（存档，2026-09-04 上午）


| 作业 | 目的 | 臂数 | 预注册读法 |
|---|---|---|---|
| **W20 damping 扫描** | 补偿强度连续旋钮：percdamp {0.01, 0.2, 1, 5}（0.05 已有），Llama & Q14 | 8 | damp→∞ 时 GPTQ→RTN。单调恢复 → 单模型内相变曲线、机制 = 病态补偿、与 HeRo-Q 对话；不单调 → 看裁剪（W23）。0.01 更差 → 冻结协议已保守 |
| **W21 配置混淆 + AWQ** | asym；instruct 校准；wikitext 校准；AWQ 式缩放+RTN（sym/asym，Llama & Q14）| 7 | 任一臂消崩塌 → 主张改"补偿 × 校准分布交互"；AWQ 不崩 → 补偿特异坐实；AWQ 也崩 → story 改写 |
| **W22 group-scale 伪影** | 掩码权重排除出 scale，重跑 Q14 tacq@1e5（两种子）、Llama tacq、Llama tacq@1e5 | 4 | 0.216 消失 → 删"保护不足有害"；仍在 → 该主张防弹；救援臂可能被低估 |
| **W23 机制日志 + 分歧起点** | 每模块补偿位移、逐列发送质量、裁剪比例、GPTQ vs RTN 层目标（c4 与 chat 两个 Hessian）、条件数；逐层隐状态与 fp16 的余弦（fp16 vs none/rtn）| 6 | GPTQ 在 c4 目标赢、chat 目标输 → 分布偏移；发送质量集中在临界模块 → 结构对应；任一统计量分离 {llama, q14} vs q7 → 量化时预测器 |
| **W24 普查扩展** | gemma-2-9b/2b、Llama-3.2-1B、Mistral-7B-v0.2、Falcon3-7B、SmolLM2-1.7B，各 fp16/GPTQ/RTN | 6×3 | 2/11 → x/17；第三个崩塌则 n=2 问题消失 |
| **W25 无梯度判据** | \|W\|·√H_ii 环内豁免：Llama 1e5 / 40M，Q14 1e6 | 3 | 救援等效 → 实用修复 = GPTQ 里一行，机制 Hessian 侧 |
| **W26 诱导崩塌** | 优雅模型 Qwen-7B / Mistral-7B 各三杠杆：percdamp 0.001、n_calib 8、per-channel g=−1（+ g=−1 的 RTN 参照）| 8 | 优雅模型崩 → 崩塌是 (模型 × 配置) 空间的阈值现象，2/17 只是默认配置所在；不崩 → 临界结构是模型侧必要条件，机制更强 |
| **W27 后续评测**（hold 于 W20/W21/W23）| Llama damping 阶梯的 PPL/MMLU；AWQ 臂的 PPL/MMLU/GSM8K；W23 日志汇总 | 9 | 似然是否跟随 IFEval 恢复；AWQ 是否在每个轴上都优雅 |

汇总命令：`python src/comp_stats.py --runs llama=runs/stats/llama31-8b q7=runs/stats/qwen25-7b q14=runs/stats/qwen25-14b --summary runs/comp_stats_summary.csv --per-layer runs/comp_stats_layers.csv`。

**决策点 9/10**：W21 消崩塌 → 第 3 节重写；W22 让 0.216 消失 → 删主张；其余不动。Rebuttal 储备：70B、TaCQ 2-bit 和解、14B 旋转+tacq、逐约束 IF 解剖、Multi-IF。

---

## 10. 写作纪律与诚实项

- 每个 headline 数字带 CI；不出现"精选样本"式表述。
- 引用逐条核实（上轮 oL3r 点名两条不存在的引用）；§8 里标"落笔前核实"的条目未核实前不引。
- 崩塌 n=2（各有种子副本 + 全干预矩阵 + HeRo-Q 外部复现 Llama）；Q14 外部 n=0。
- fake-quant 无部署内核；AWQ 臂是 per-linear 缩放 + RTN、无 clip 搜索（协议偏差写明）；崩塌仅 GPTQ 家族；≤32B。
- Multi-IF / GSM8K 的 Qwen "gptq3" 参照有 packed 与 v2 两版（IFEval 差 ~1；GSM8K 已补 v2none 0.7779，Multi-IF v2none t1 0.6326 ≈ packed 0.6381）。
- Mistral IF 基线本身低（0.552），属模型能力。
- 24B/32B 只有 regime 检查臂（none）。
- heads 排序来自因果 ablation dev 集；act ≈ dev3（ρ=0.94，dev 排序受激活尺度混淆，最终以因果消融为准）。
- detect_super_weights 在 Qwen 上有归因 bug（未用于任何最终主张；Llama 检出正常并被 sw_only 臂消费）。
- 0.5% 预算出处：判据公式属 TaCQ；预算数值属我们（= 32 头参数量，恰与 TaCQ ~3.1bit 开销档同级）；预算扫描证明实际必需 0.001–0.01%。
- 代码语义：被保护权重是"自由 fp16 吸收器"，不是冻结原始值（§1）。
- 匿名代码仓 + 45 个 PROTECT_PROTOCOL.json 随投稿放出（上轮 Datasets/Software 双 1 分）。

---

## 附录 A. 归档的支线数据（不进主线）

- **逐约束损伤**（Qwen-7B 3-bit）：language −22.6（自 100%）、combination −16.9、punctuation −12.1、detectable_content −1.9；4-bit 平均 −0.8 但 74/541 条判定翻转（39 降 35 升）。
- **W1 因果消融变体**：rand16/rand64/rand32_qkv、prot32_qkv/o（0.6767/0.6914）均在噪声带。
- **Dissociation**：梯度显著性（TaCQ MSG 项）与 fragility/激活排序近乎正交（Spearman ≤0.16，Jaccard@32 ≤0.05）。
- **Yi-1.5-9B**（已移出口径）：fp16 0.594 → gptq3 0.443（−15.1，循环 21.6%）；RTN 0.3201 vs GPTQ 0.4426（补偿帮 +12.2）；tacq +1.97 [−1.3, +5.2] p=0.258；PPL 6.11→9.42；MMLU fp16 = 量化 = 0.2295（选项 logit 协议与其 tokenizer 不兼容）。
- **中间档 PPL/MMLU 量化侧**：nemo 9.06/0.583、q3 11.96/0.531、l3 17.37/0.567、l32 1748.9/0.508；fp16 参照 l32 MMLU 0.607、l3 0.666/8.29、nemo 0.685/6.09、q3 0.664/8.56。
- **Llama 临界集 vs c4 掩码**：Jaccard 0.155；与激活离群通道重叠 42.0%；k_proj 38k > q_proj 26k。
- **2-bit 终末**（Qwen-7B）：none 0.136 / tacq 0.133 / tacq 2% 0.126 / 5% 0.127 / randw 0.139 / sw 0.117（fp16 0.765）。

## 附录 B. 文献 URL 清单（2026-09-04 汇总；★ = 本轮实际打开核对过，其余 ID 凭记忆，落笔前核实）

### B.1 最近邻与机制线
- ★ TaCQ — https://arxiv.org/abs/2504.07389
- ★ Super Weight — https://arxiv.org/abs/2411.07191
- ★ HeRo-Q（Hessian 条件化；独立报告 Llama-3.1-8B W3 GPTQ 灾难）— https://arxiv.org/abs/2601.21626
- ★ How Good Are Low-bit Quantized LLaMA3 Models — https://arxiv.org/abs/2404.14047
- ★ Quantization Degradation: A Signal–Noise Perspective（CASIA）— https://arxiv.org/abs/2608.08188
- ★ From Signal Degradation to Computation Collapse（CASIA，ACL 2026 Findings）— https://arxiv.org/abs/2604.19884
- ★ The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm — https://arxiv.org/abs/2507.18553
- ★ GPTAQ / GPTQv2: Asymmetric Calibration — https://arxiv.org/abs/2504.02692
- ★ Quantization Error Propagation — https://arxiv.org/abs/2504.09629
- First-Order Error Matters: Accurate Compensation for Quantized LLMs — https://arxiv.org/abs/2507.11017
- Calibration and Transformation-Free Weight-Only Quantization via Dynamic Grouping（疑为"GPTQ 在 Llama-3.2-1B 多实现退化"出处）— https://arxiv.org/abs/2509.03054
- A Comprehensive Evaluation on Quantization Techniques for LLMs（同上候选出处）— https://arxiv.org/abs/2507.17417
- GPTQ — https://arxiv.org/abs/2210.17323
- QuaRot — https://arxiv.org/abs/2404.00456
- AWQ — https://arxiv.org/abs/2306.00978
- SpQR — https://arxiv.org/abs/2306.03078
- OWQ — https://arxiv.org/abs/2306.02272
- SqueezeLLM — https://arxiv.org/abs/2306.07629
- SliM-LLM — https://arxiv.org/abs/2405.14917
- QuIP — https://arxiv.org/abs/2307.13304
- SpinQuant — https://arxiv.org/abs/2405.16406
- OmniQuant — https://arxiv.org/abs/2308.13137
- LeanQuant（ICLR'25，ID 待核）— https://arxiv.org/abs/2407.10032
- Massive Activations — https://arxiv.org/abs/2402.17762
- Attention Sinks / StreamingLLM — https://arxiv.org/abs/2309.17453
- Hase et al., Does Localization Inform Editing（NeurIPS 2023）— https://arxiv.org/abs/2301.04213
- Weight Patching（IFEval head 级因果）— https://arxiv.org/abs/2604.13694
- Retrieval Heads — https://arxiv.org/abs/2404.15574
- Critical Weight Protection（CWP）— https://arxiv.org/abs/2601.12033
- FAQ — https://arxiv.org/abs/2601.11200

### B.1b 第三轮近邻
- ★ Williams & Aletras, ACL 2024 — https://arxiv.org/abs/2311.09755
- ★ KronQ — https://arxiv.org/abs/2607.07964
- ★ Uniqueness of LLaMA3-70B with per-channel quantization — https://arxiv.org/abs/2408.15301
- ★ DASH-Q — https://arxiv.org/abs/2604.13806
- Coverage-based calibration (set cover over outlier channels) — https://arxiv.org/abs/2604.24008
- Understanding and Selecting Calibration Data for LLM Quantization — https://openreview.net/forum?id=pfw3saHzGU
- Beware of Calibration Data for Pruning LLMs — https://arxiv.org/abs/2410.17711
- Self-calibration for LM Quantization and Pruning — https://arxiv.org/abs/2410.17170
- Preserving LLM Capabilities through Calibration Data Curation — https://arxiv.org/abs/2510.10618
- Training Dynamics Impact PTQ Robustness — https://arxiv.org/abs/2510.06213
- GPTQ-intrinsic LoRA — https://arxiv.org/abs/2606.01412 ; QAM-W — https://arxiv.org/abs/2605.26339 ; OffQ — https://arxiv.org/abs/2606.07116
- GPTQModel (damp_percent default) — https://github.com/ModelCloud/GPTQModel ; ModelCloud GPTQ-v2 Llama-3.1-8B-Instruct checkpoint — https://huggingface.co/ModelCloud/GPTQ-v2-Llama-3.1-8B-Instruct

### B.2 评测 / scaling / 现象线
- LLM-KICK（ICLR 2024）— https://arxiv.org/abs/2310.01382
- ★ Scaling Laws for Precision（δ_PTQ = C·D^γD/N^γN·e^(−P/γ)；≤1.7B；"跨量化器只差常数"）— https://arxiv.org/abs/2411.04330
- ★ Low-Bit Quantization Favors Undertrained LLMs（ΔqLoss = k·D^0.53/(N^0.23·P^5.5)；GPTQ 主拟合；ACL 2025）— https://arxiv.org/abs/2411.17691
- A Comprehensive Evaluation of Quantized Instruction-Tuned LLMs up to 405B（IJCAI 2025）— https://arxiv.org/abs/2409.11055
- UniComp — https://arxiv.org/abs/2602.09130
- Give Me BF16 or Give Me Death（ACL 2025）— https://arxiv.org/abs/2411.02355
- Accuracy is Not All You Need（flips）— https://arxiv.org/abs/2407.09141
- Alignment Collapse Under KV Cache Quantization — https://arxiv.org/abs/2606.09864
- Quantization Meets Reasoning — https://arxiv.org/abs/2505.11574
- Extreme Low-Bit Inference in Reasoning Models: Failure Modes（循环 = 失败签名，可引为 loop-rate 检测器先例）— https://arxiv.org/abs/2606.02011
- Through a Compressed Lens: Quantization and Factual Recall（含 Qwen2.5-14B）— https://arxiv.org/abs/2505.13963
- ACBench（ICML 2025）— https://arxiv.org/abs/2505.19433
- Which Quantization Should I Use? llama.cpp on Llama-3.1-8B-Instruct（IFEval 描述性）— https://arxiv.org/abs/2601.14277
- Quantization Damage Is Multiplicative — https://arxiv.org/abs/2608.06564

### B.3 非论文来源
- GPTQModel issue #1278（3-bit 回归，我们早期 checkpoint 的 bug 出处）— https://github.com/ModelCloud/GPTQModel/issues/1278
- 实践者报告 "Avoid Quantizing Llama 3 8B with GPTQ" — https://medium.com/data-science/quantize-llama-3-8b-with-bitsandbytes-to-preserve-its-accuracy-e84283b233f7
- STORM（写作方法论，不入文）— https://arxiv.org/abs/2402.14207 ，代码 https://github.com/stanford-oval/storm

### B.4 数学合作者的起点包
Babai（2507.18553，Theorem 5 与无裁剪假设）；HeRo-Q（Hessian 各向异性）；GPTAQ / QEP / First-Order（误差累积的三种修正）；两篇 scaling 律（作为"连续律产生不了翻转"的对照，不是研究对象）；src/gptq_core.py 的更新方程 `err = (w−q)/Hinv_ii；W[:, j:] −= err·Hinv[j, j:]` 与 damping `H += percdamp·mean(diag H)·I`。
