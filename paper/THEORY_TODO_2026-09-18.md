# Theory hand-off, 2026-09-18 (supersedes THEORY_TODO_2026-09-15.md)

Deadline: full paper 2026-09-25 AoE. Files: `ICLR/ICLR_quantization/sections/05_theory.tex` (main text, complete with proofs) and `sections/appendix/A_proofs.tex` (yours; seven subsections, each a `\todo`). Notation: H_λ = x₀x₀ᵀ + M, v = M⁻¹x₀, κ = x₀ᵀv, m_jk = [M⁻¹]_jk.

## 1. Check in the main text (nothing to write, only to verify)

| Statement | What to check |
|---|---|
| Lemma 1 (two-sided curvature) | Gauss–Newton form; the sentence "ignoring cross-position blocks and the first-order term" is the assumption. |
| Prop. 1 (position mismatch) | Numbers only; nothing to prove. |
| Lemma 2 (gradient attenuation) | Currently one sentence with a citation; the proof is yours (A.2). |
| Assumption 1 (attention-mediated readout) | Labelled as an assumption on purpose; do not turn it into a theorem unless A.2 gives the first-order form. |
| Theorem 1 (a)(b)(c) | Your (7)–(12); the paper's (c) keeps the exact denominators λ + ‖x₀,−j‖². |
| Corollary 1 (i)(ii)(iii) and its proof | (iii) says κ is rotation-invariant; check the one-line argument. |
| Prop. 3 (a)(b)(c) and its proof | (a) is your PSD-G version with the u / z decomposition; (b) has the new sentence on w_t = g_t alone (BOS share = A·G share). |

## 2. Write in Appendix A (in order of importance)

1. **A.3 direction along the loop under general M.** (Update 2026-09-19: the main text now has Lemma 3, "the decomposition survives the loop": the Schur complement of x₀x₀ᵀ + M is x₀'x₀'ᵀ + M/j with x₀' = √c (x₀,−j − (x₀j/M_jj) M_{−j,j}), c = M_jj/(M_jj + x₀j²), so Theorem 1 is exact at every step; for M = λI the direction is preserved exactly. What remains for you is the angle between x₀' and x₀ for general M, and the point below.) The empirical result you must account for: the sending pattern of the sink column (a row of the upper Cholesky factor of H_λ⁻¹ in act-order) has cosine .52–.93 with x₀ in 14 of 17 sink matrices and .16–.18 on Mistral-7B and Qwen2.5-7B, and it fits x₀ better than v = M⁻¹x₀ in 11 of 13 (e.g. Llama .84 vs .34, granite .91 vs .03, nemo .82 vs .05; Qwen2.5-14B .58 vs .56 is the exception). The general-M form of Theorem 1(a) predicts v. Candidate reason: the first column's pattern is −(m_j· − v_j v/(1+κ)) with m_j· = M⁻¹e_j, and the ordinary tokens with mass on channel j are the secondary sinks, whose activations lie along x₀, so M⁻¹e_j is itself along x₀. Data to use: `runs/theory/INDEX_theory.csv`, columns `pat_cos_x0_bos`, `pat_cos_v_bos`, `nob_pat_cos_x0_bos`.
2. **A.4 clipping.** New fact: with the BOS token in H the sink column's compensation is 4× (Llama) to 470× (Qwen2.5-14B) larger than without it, and the largest quantised entry reaches 5–12× the largest original entry (without BOS ≤ 1.7×). The statement to make: direction from the channel geometry (shared with the BOS-free H), amplitude 1/(1+κ₋ⱼ)-suppression at BOS pushing O(δ_j) onto the other sink coordinates, hence overshoot when x₀ is concentrated (it is: 1–2 columns hold 90 % of ‖x₀‖² in 15 of 17 models). Columns `pat_norm_bos`, `nob_pat_norm_bos`, `maxabs_Qg`, `maxabs_Qnob`, `n_bos_cols90_J`.
3. **A.2 Lemma 2, both paths** (RMSNorm Jacobian O(1/rms); attention path transports a near-zero value vector at the primary sink) and the first-order form of Assumption 1 (‖∂L/∂y_t‖ ∝ attention mass received × transported norm). Empirical support now causal: injection at template vs ordinary positions (.291 vs .558 at 8× the norm), restoring the chat scaffold alone repairs both collapses (Llama .636, Qwen .755; user-text positions .141 / .435).
4. **A.5 the floor and the two readouts.** Row readout δ⁽ʳ⁾ᵀx_t is the one the data select (recall .54 / .55 vs whitened .22 / .33); state it as Corollary 1(ii)'s second form and give the order of η. Data: `runs/sink/f_l_readout.csv`, `f_q14_readout.csv`.
5. **A.6 levers table** (your §1.5), with two corrections: rotation leaves κ unchanged and removes the coordinate concentration; long windows do not lower κ for Llama (c4 1.2e5, c4win 1.3e5, pile 6.4e4) and the protocol dependence is in the background. κ values per model and protocol: `INDEX_theory.csv`, column `kappa`.
6. **A.7 Prop. 3(c) approximation error** and the remark that the token axis is the only place the output side enters without changing the column-wise update (boundary with KronQ / GuidedQuant).
7. **A.1 Lemma 1** Gauss–Newton derivation and the measurement of g_t (one backward pass, summed token cross-entropy, activation gradients only).

## 3. Facts that constrain what you may claim

- κ does not order collapse: Qwen collapses at κ = 2.6e5 (short docs) and 87 (WikiText windows), is healthy at 2.7e5 (C4 windows). Cor. 1(iii) is about amplitude only.
- Capping g_t at 100× its median inside the corrected objective changes nothing (.629 / .751 / .425): do not argue from the dynamic range of g_t.
- A gradient-free surrogate works: normalise and up-weight the first 8 positions of each document (Llama .623, Mistral .405/.418). Prop. 3 explains the normalisation half; the prior's role is empirical.
- Collapse is model × protocol × sample: Llama collapses in 4 of 5 draws, Qwen 4/4, Mistral (C4 windows) 2/2; Pile short documents collapse both (.154 / .332). The theory explains the lesion, not the collapse; the paper says so in §5 and §8.
- Mistral under C4 windows is a different phenotype (270× sink-direction error on every ordinary token); do not build on it.

Provenance for every number: `paper/RESULTS.md` §9.10bo–bs.
