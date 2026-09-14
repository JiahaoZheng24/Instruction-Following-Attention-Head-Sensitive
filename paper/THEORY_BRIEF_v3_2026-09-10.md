# Theory note v3: the collinear-channel trade at the sink-forming matrix

Supersedes v2 (2026-09-09). v2's §1 (setup) and the off-distribution identity (v2 §2, P1) stand. v2's factorisation $\|e\|^2\approx\|\delta(b)\|^2 R(H,\lambda,z)$ (v2 §3, P3) is **refuted** by W49–W51: at the sink-forming matrix GPTQ's template-token error does not scale with the rounding step. This note replaces §3–4 with the mechanism the diagnostics established and restates the propositions. Data: RESULTS.md §9.10w–z.

---

## 0. Facts the theory must reproduce (Llama-3.1-8B, layer-1 `down_proj`, template token `<|end_header_id|>`)

| recipe | GPTQ error | RTN error | residual error after layer 2 | collapse |
|---|---|---|---|---|
| 3-bit g128 | 2.56 | 0.28 | 3.85 | yes |
| 3-bit g32 | 1.77 | 0.17 | 2.43 | yes |
| 3-bit per-channel | 2.98 | 0.40 | 2.95 | yes |
| 3-bit, no act-order | **7.16** | 0.19 | 6.76 | yes |
| 4-bit g128 | **1.68** | **0.08** | 1.33 | no |
| 4-bit per-channel | 1.50 | 0.14 | 1.27 | no |
| 3-bit ρ=1 / 2 / 5 | 1.34 / 1.20 / 1.00 | 0.17 | 1.54 / 1.37 / 1.20 | yes / no / no |
| 3-bit token-normalised H | 0.27 | 0.18 | 0.45 | no |
| 3-bit c4chat | 1.68 | 0.18 | 1.36 | no |
| 3-bit, 5 super-weight rows fp16 | ≈1.3 (est.) | — | 1.78 | yes (.204) |

1. RTN's error scales with the step ($\times\tfrac12$ per bit); GPTQ's does not (1.5–1.8 at 3- and 4-bit).
2. GPTQ's error is insensitive to the size of the compensation displacement (`comp_disp_rel`: 0.305 default, 0.043 at 4-bit, 0.0014 at ρ=5) and to adding the template tokens to the calibration text (c4chat).
3. Inside the matrix the error is concentrated in a few output rows — the rows with large weights on the BOS channels (super-weight rows 788, 1384, 4062: 26–40 % of the energy) — and comes from the BOS input channels (198, 2427, 6412, 12638, 12657: 54–93 %).
4. In those rows GPTQ moves near-zero weights on the BOS channels to the **edge of the quantisation range** (3-bit: ±0.53–0.71; 4-bit: ±0.52–0.59; without act-order: −4.4 … +6.8, inflating the group scale), while RTN leaves them near zero. Damping 5 removes the >1 outliers but not the range-edge trades; token normalisation removes the trades.
5. In every other matrix of blocks 0–1, GPTQ's template error ≈ RTN's and is spread over rows (top-row share ≤ 3 %).
6. Keeping the five worst rows at fp16 halves the layer-2 residual error (3.85 → 1.78) but does not cure (still above the 1.4–1.5 threshold).
7. Qwen2.5-14B, layer-4 `down_proj`, sink token (first newline): GPTQ 536 vs RTN 25; worst row 3094 has max|Q| 7.9 vs max|W| 0.82 and its small weights are annihilated (rounded to 0).
8. Collapse across 13 recipes is separated (threshold 8.6–9.5) by the product of the down_proj template error and the RSS of the other 13 matrices' template errors; three pre-registered predictions held.

---

## 1. Setup (unchanged)

Row $w\in\mathbb{R}^n$ of `down_proj`; calibration inputs $x_t$, $H=\sum_t x_tx_t^\top$, $H_\lambda=H+\lambda I$. The calibration set is dominated by the BOS token: at this matrix its input norm is 481 vs 1–2.5 for other tokens, so $H\approx\kappa^2 N\,bb^\top + H_{\rm bulk}$, $\kappa\approx481$. Let $C=\{c_1,\dots,c_m\}$ be the input channels that carry the BOS massive activation (198, 2427, 6412, 12638, 12657, …). Within the calibration distribution these channels fire together and only at BOS: their columns of $H$ are (nearly) proportional, i.e. $H_{CC}\approx \kappa^2 N\, u u^\top$ with $u$ the BOS-proportion vector on $C$.

Grid: per row, per group of columns, step $s=2\max_{j\in g}|w_j|/(2^{b-1}-1)$; values are clipped to $\pm\max_{j\in g}|w_j|$. In `gptq_core.py` the group maximum is taken over the *current* (already compensated) weights when the group is reached.

## 2. The trade (why compensation is large and range-bound)

Take a row with one large weight $w_{c_1}$ on a BOS channel (a super weight, |w| ≈ 0.6) and near-zero weights on the other BOS channels. Rounding $w_{c_1}$ leaves $\delta_1$ (≤ s/2 ≈ 0.09 at 3-bit, 0.04 at 4-bit). OBS compensation solves for the free coordinates $F$
$$\Delta w_F = -(H_{FF}+\lambda I)^{-1}H_{F c_1}\,\delta_1 .$$
Restricted to $C$, $H_{CC}$ is rank-one up to the bulk, so $(H_{CC}+\lambda I)^{-1}H_{C c_1}$ is the ridge regression of channel $c_1$ on the other BOS channels — a coefficient vector of size $O(u_j/\!\sum u^2)$ per channel, of order 1 (the channels are interchangeable for the calibration objective). Hence
$$|\Delta w_{c_j}| \approx |\delta_1|\cdot\frac{\kappa^2 N u_1u_j}{\kappa^2 N\|u\|^2+\lambda}\;=\;O(|\delta_1|)\quad\text{for every }c_j\in C,$$
and, summed over the many rounding errors of the $|C|$ large-diagonal columns that act-order processes first, the free BOS-channel weights of the row receive displacements of order $|C|\cdot s/2$, i.e. **larger than the row's quantisation range** $\max|w|\approx0.6$ for 3-bit ($|C|\ge 5$, $s/2\approx0.09$) and for 4-bit ($s/2\approx0.04$, borderline but observed). The subsequent rounding of those weights therefore lands at the range edge (clip) or, if their group has not been reached yet, inflates the group maximum (the 6.8 / 7.9 / 3.2 values). Either way the final displacement is set by the **range** $\max_{g}|w|$, not by the step $s$:
$$\boxed{\;\Delta w_{c_j}^{\rm final}\approx \pm\max_g|w|\ \ (\text{or larger, if the scale is inflated}),\quad\text{independent of }b\;}$$
This is fact 1 and fact 4. Damping enters through $\lambda$ in the denominator and only reduces the trade below the range for the smaller coefficients (fact 4, ρ=5). c4chat adds eight template tokens of norm ~1 to an $H$ whose BOS entry is $481^2$: $H_{CC}$ stays rank-one, the trade is unchanged (fact 2). Token normalisation sets $\kappa=1$: $H_{CC}$ is no longer rank-one relative to the bulk, the regression coefficients collapse, no trade (fact 4).

## 3. Why the trade is invisible to the objective and visible at template tokens

For a calibration input (BOS proportion $u$ on $C$), the traded weights cancel by construction: $\sum_j \Delta w_{c_j}u_j\approx-\delta_1u_1$. For a deployment input $z$ whose activations on $C$ are **not** in BOS proportion — the template tokens, where $x_C = (2.1, 1.0, 0.3, 0.2, \dots)$ instead of $\kappa u$ — the same weights produce
$$e(z)=\sum_j \Delta w_{c_j} z_{c_j} = O(\max_g|w|\cdot\|z_C\|)\approx 0.6\times2 \approx 1,$$
which is the observed row error (1.3 at 3-bit, 1.05 at 4-bit) in the super-weight output rows, i.e. in the sink output channel. RTN has $\Delta w_{c_j}=\text{rnd}(w_{c_j})-w_{c_j}\approx w_{c_j}\lesssim0.08$ there, hence 0.12 / 0.002. This is the "spurious sink component written into every non-BOS token".

## 4. From the matrix to the collapse

The residual-stream error at the template token after block 1 is not additive in the matrices; empirically it is predicted by the product of the down_proj template error and the block's ordinary template error (fact 8, threshold 8.6–9.5), and it collapses the model once it exceeds a relative error of 1.4–1.5 at that position (13 recipes, no exception). Both thresholds are measured, not derived. Keeping only the five worst rows halves the down_proj error and leaves the product near the boundary (fact 6, collapse persists): the trade happens in every row with sizeable BOS-channel weights, the super-weight rows are merely its largest instances.

## 5. Propositions (revised)

**P1 (off-distribution identity)** — as v2: $e_{\rm GPTQ}(z)-e_{\rm RTN}(z)=-\delta^\top H_{SF}(H_{FF}+\lambda I)^{-1}z_F$.

**P2′ (rank-one channel block ⇒ order-one trade).** If $H_{CC}=\alpha uu^\top+E$ with $\|E\|\ll\alpha$ and $c_1\in C$ is rounded with error $\delta_1$, the OBS update on $C\setminus\{c_1\}$ satisfies $\Delta w_{c_j}= -\delta_1\,\frac{\alpha u_1u_j}{\alpha\|u_{-1}\|^2+\lambda}+O(\|E\|/\alpha)$; in particular $|\Delta w_{c_j}|\to|\delta_1|\,|u_1u_j|/\|u_{-1}\|^2$ as $\alpha\to\infty$ — independent of $\alpha$, i.e. the trade does not shrink as the dominant token grows, and $\lambda$ reduces it only by the factor $\alpha\|u_{-1}\|^2/(\alpha\|u_{-1}\|^2+\lambda)$, which needs $\lambda\sim\alpha$ (ρ of order 1 in units of the BOS-inflated mean diagonal) to bite. *Short proof via Sherman–Morrison.*

**P3′ (range-bound displacement).** With per-group scale $\max_g|w|$ and clipping, any compensated weight with $|w+\Delta w|>\max_g|w|$ is stored at $\pm\max_g|w|$; if the group scale is recomputed after compensation it becomes $\max(|w|,|w+\Delta w|)$ and the other weights of the group round with a step inflated by the same factor (annihilation below $s/2$). Hence for rows in which P2′ applies, the stored displacement on $C$ is $\Theta(\max_g|w|)$ whenever $|C|\,s/2\gtrsim\max_g|w|$, which holds at 3-bit and 4-bit for $|C|\ge5$. *Consequence: the template-token error at this matrix is bit-independent, and RTN's is $O(s)$.*

**P4′ (token normalisation removes the trade).** Under $x_t\to x_t/\|x_t\|$, $\alpha$ falls from $\kappa^2N_{\rm BOS}$ to $N_{\rm BOS}$ while the bulk keeps its scale $\sim N$; the premise $\|E\|\ll\alpha$ of P2′ fails and the regression coefficients on $C$ are $O(N_{\rm BOS}/N)$ — a factor $\kappa^2$ smaller. *This is the one-line fix.*

**P5 (sequential exactness)** — as v1 P6.

**P6′ (invisibility).** The calibration objective restricted to $C$ is $\alpha(\Delta w_C^\top u)^2+O(\|E\|)$; any $\Delta w_C$ with $\Delta w_C^\top u=-\delta_1u_1$ is optimal, so the objective (and any token-averaged score) cannot distinguish the traded solution from the harmless one. Detection therefore needs inputs off the BOS proportion — the per-position detector.

## 6. What the numbers in §0 test

P2′/P3′: bit-independence (facts 1, 4), damping's partial effect (ρ=5 keeps range-edge trades), the range-edge values themselves (±0.53–0.71 vs ±0.52–0.59), the no-act-order blow-up (more columns free when the BOS columns are rounded ⇒ scale inflation, 6.8). P4′: token-normalised row values ≈ W. P6′: c4chat leaves the matrix unchanged. Fact 6 (five rows are not enough) and fact 8 are the boundary of what §2–4 explain: the propagation thresholds are empirical.

**Addendum 2026-09-11 (W55, RESULTS §9.10ae).** The propagation threshold is *model-specific*. Six models with per-position detector 30–69 were quantised with per-channel grouping (g=−1, ×1.3 background error) and with the original GPTQ damping ρ=0.01; none fell below RTN (GPTQ leads RTN by +5 to +41 at g=−1, because per-channel 3-bit destroys RTN first), and ρ=0.01 left the healthy models unchanged while it hurt the two already-collapsed ones (Llama .150→.128, Q14 .412→.298). So the detector ranks risk but does not place a model on a common distance-to-threshold scale; the product law of §0 is calibrated within one model (13 Llama recipes) and must not be read as a cross-model formula. Nothing to prove here beyond P2′/P3′; it bounds the claim.

## 7. Known vs new

Known: OBS/GPTQ closed form; group-wise scaling and clipping; massive activations at the first token (2402.17762); super weights and their input spikes (2411.07191). New: the observation that the BOS channels are collinear in the calibration Hessian so that OBS trades weight among them; that the trade is range-bound and therefore bit-independent; that it lands in the sink output channel of every token not in BOS proportion; that super weights are the largest but not the only instances; and the two repairs that follow (RTN on the matrix; normalise tokens).


## 7. Addendum 2026-09-14: the OBS compensation under a rank-one-dominated Hessian (closed form)

At the sink-forming down_proj, H = Σ_t x_t x_tᵀ + λI ≈ x_0 x_0ᵀ + λI with ‖x_0‖ ≫ ‖x_t‖ (Llama 481 vs 1–2.5). Sherman–Morrison gives H⁻¹ = (1/λ)[I − x_0x_0ᵀ/(λ+‖x_0‖²)], hence the OBS compensation of the quantisation error δ_j on column j is

  Δw_k = −δ_j (H⁻¹)_{jk}/(H⁻¹)_{jj} ≈ δ_j x_{0j} x_{0k} / ‖x_0‖²   (k ≠ j),

i.e. every compensation step moves weight along the BOS activation pattern x_0. For any token t the output error of row r is therefore

  e_t^{(r)} ≈ c_r · (x_0ᵀ x_t)/‖x_0‖²,   c_r = Σ_j δ_j^{(r)} x_{0j}.

Consequences. (i) Across rows the error is dominated by the rows with the largest weights on the BOS channels — the super-weight rows (Llama row 788); the output error direction is that channel, i.e. the BOS hidden-state direction u (measured: cos .90 at the template position, W66). (ii) The error at token t scales with the overlap x_0ᵀx_t of its MLP-intermediate activation with the BOS pattern, independent of position: template markers and newlines (semi-sink tokens) are hit at every occurrence, in the prompt and throughout the generated response; ordinary tokens are hit ∝ their small overlap. This is why a fixed-position synthetic perturbation of the prompt (W66/W67) does not reproduce the collapse. (iii) Range clipping (P3′) and damping only rescale c_r. Cures: token normalisation removes the rank-one dominance so Δw no longer lies along x_0; output-side weighting g_t puts a price on the high-overlap tokens inside the objective; their product is the W61 objective. Tested by W68 (RESULTS §9.10bg): per-token ‖e_t‖ vs |x_0ᵀx_t| on the real checkpoint, and single-matrix transplants.

**Correction 2026-09-14 (W68/W69 outcome).** Consequence (ii) holds only for the extreme-overlap tokens (top quintile: 2.3× Llama, 4× Q14) and, per token class, for the prompt's template markers (Llama 1.34 vs 0.62) and Q14's first newline (0.95 vs 0.17); response tokens are barely affected (Llama 0.34–0.41), so the damage does NOT accumulate over the generated tokens. The lesion is a portable matrix (W68 transplants), and the fatal amplification is a property of the whole short-calibration GPTQ solution rather than of layers 2–4 (W69: swapping layers 2–4 for the non-collapsing c4win solution does not cure, and the reverse does not collapse). Mechanism of the amplification: open.
