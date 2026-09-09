# Theory note v2: one token dominates the layer-wise objective — a closed-form account of the 3-bit collapse and of why 4-bit is safe

Supersedes `THEORY_BRIEF_2026-09-05.md` (whose notation and P1–P6 remain valid; this note narrows the target to three propositions and attaches the numbers that must come out). Everything measured is in `paper/RESULTS.md` §9.10i–u and `runs/stats/*/stats.csv`.

---

## 0. What must be explained (facts, IFEval avg4)

1. Llama-3.1-8B, 3-bit g128, c4 calibration: RTN .565, GPTQ **.150**. Qwen2.5-14B: .697 vs **.412**. 15 other models: GPTQ ≥ RTN.
2. At the layer-1 `down_proj` of Llama, the input vector of the BOS token has norm **481**; every other token 1–2.5. (Mistral-7B: 807 vs 1; Llama-3.2-3B 634; Qwen2.5-14B: the first newline, 70 vs 4–25 at layer 4.) 14 of 16 models have such a token at an early `down_proj`; gemma-2 does not.
3. On the same matrix, measured on chat prompts, GPTQ's output error divided by RTN's is **0.17 at BOS** and **32 / 85 / 102** at the three template tokens that follow (`system`, `<|end_header_id|>`, `\n\n`); the token-averaged ratio (GPTQ's own objective) is 0.17.
4. Residual-stream error after layer 2 at `<|end_header_id|>`, ten Llama recipes:

| recipe | error | IFEval | collapse |
|---|---|---|---|
| 3-bit g128 (default) | 3.85 | .150 | yes |
| 3-bit g32 | 2.43 | .178 | yes |
| 3-bit ρ=1 | 1.54 | .152 | yes |
| 3-bit ρ=2 | 1.37 | .578 | no |
| 4-bit per-channel | 1.27 | .693 | no |
| 4-bit g128 | 1.33 | .745 | no |
| 3-bit ρ=5 | 1.20 | .642 | no |
| 3-bit, RTN on layer-0/1 down_proj | 0.50 | .607 | no |
| 3-bit, token-normalised Hessian | 0.45 | .648 | no |

   One threshold (1.4–1.5) separates all ten.
5. Cures, all zero extra bits: token-normalised Hessian (Llama .648/.668, Q14 .720/.724, Mistral act-order collapse .174→.404); RTN on the single flagged matrix (Llama .630, Q14 .751); calibration text containing the sink-forming token (c4 wrapped in the model's template: .644 / .772; a foreign template cures Qwen but not Llama). Damping ρ ≥ 2 cures Llama.
6. Median-capped re-weighting (BOS to 10× typical instead of 481×) does **not** cure Llama (.153) but cures Q14 (.742, whose sink token was only 10× to begin with) and improves Nemo (+6.3).

---

## 1. Setup (as in v1)

Row $w\in\mathbb{R}^n$ of a linear layer; calibration inputs $x_t$ ($t=1..N$), $H=\sum_t x_t x_t^\top$ (the $2/N$ factor is irrelevant below); dampened $H_\lambda = H+\lambda I$, $\lambda=\rho\cdot\mathrm{mean\,diag}H$. Two-block relaxation: coordinates split into a rounded block $S$ and a free block $F$; rounding error $\delta = q_S - w_S$ (coordinate-wise $|\delta_j|\le s_{g(j)}/2$, $s_g$ the group step). OBS/GPTQ sets

$$w_F^\star = w_F + A_\lambda\delta,\qquad A_\lambda = -(H_{FF}+\lambda I)^{-1}H_{FS}.$$

This closed form is standard (OBS 1993; GPTQ; Babai view 2507.18553). **Nothing in §1 is new.**

---

## 2. The new step: evaluate the closed form on a deployment input outside the calibration span

Let $z\in\mathbb{R}^n$ be the input vector of one deployment token (e.g. the `<|end_header_id|>` token entering layer-1 `down_proj`). The output error of the compensated row on $z$ is

$$e_{\mathrm{GPTQ}}(z) = \delta^\top z_S + (A_\lambda\delta)^\top z_F = \delta^\top\big(z_S - H_{SF}(H_{FF}+\lambda I)^{-1} z_F\big),$$

while RTN gives $e_{\mathrm{RTN}}(z)=\delta^\top z_S$. Define the **residual vector** $r_\lambda(z) = z_S - H_{SF}(H_{FF}+\lambda I)^{-1}z_F$: the error of predicting $z$'s rounded coordinates from its free coordinates with the calibration-optimal ridge regressor. Then

$$\boxed{\;e_{\mathrm{GPTQ}}(z)=\delta^\top r_\lambda(z),\qquad e_{\mathrm{RTN}}(z)=\delta^\top z_S\;}$$

Reading. For $z$ typical of the calibration set, $r_\lambda(z)$ is small — that is the whole benefit of compensation. For $z$ with a large component in directions where $H_{FF}$ has little curvature, $(H_{FF}+\lambda I)^{-1}z_F$ is large and $r_\lambda(z)$ can exceed $z_S$ by orders of magnitude: compensation *amplifies* the rounding error on that input. The measured amplification (fact 3) is $\|r_\lambda(z)\|/\|z_S\|\approx 85$–$102$ at the template tokens and $0.17$ at BOS.

Why BOS makes the amplification large. With one token of norm $\kappa$ times the others, $H \approx \kappa^2 b b^\top + H_{\mathrm{bulk}}$, where $b$ is the BOS direction and $H_{\mathrm{bulk}}$ has its energy spread over $\sim n$ directions. The eigenvalue along $b$ is $\sim\kappa^2 N$; the bulk eigenvalues are $\sim N/n$ each (Llama: $n=14336$, $\kappa=481$: ratio $\sim 3\cdot10^9$). GPTQ's act-order and its regression are both driven by this spectrum: the regressor fits $b$ essentially exactly and treats every other direction as nearly free. A template token $z$ has a small component along $b$ and lives in the bulk, so $r_\lambda(z)$ is governed by $(H_{\mathrm{bulk},FF}+\lambda I)^{-1}$ — small eigenvalues, large inverse.

---

## 3. Factorisation: bits act on $\delta$, everything else acts on $r_\lambda$

Write the error energy at the template token as

$$\|e_{\mathrm{GPTQ}}(z)\|^2 \;\approx\; \underbrace{\mathbb{E}\|\delta\|^2}_{\text{rounding, bits \& group}}\;\cdot\;\underbrace{R_\lambda(z)}_{\text{amplification, Hessian only}},\qquad R_\lambda(z)=\frac{\|r_\lambda(z)\|^2}{|S|}\ \text{(isotropic-}\delta\text{ approximation)}.$$

- **Bits.** $s_g = 2\max_{j\in g}|w_j|/(2^{b-1}-1)$, so $\mathbb{E}\|\delta\|^2\propto 4^{-b}$: one bit less multiplies the error energy by 4 and the error norm by 2. $R_\lambda$ does not depend on $b$.
- **Group size.** Only $\max_{j\in g}|w_j|$ changes; from $g=128$ to $g=32$ the step shrinks by a modest factor (measured: layer-2 error 3.85 → 2.43).
- **Damping.** $r_\lambda$ contains $(H_{FF}+\lambda I)^{-1}$; along a bulk eigen-direction with eigenvalue $\mu$ the regressor is scaled by $\mu/(\mu+\lambda)$. Since $\lambda=\rho\cdot\mathrm{mean\,diag}H$ and the mean diagonal is itself inflated by the BOS token ($\approx\kappa^2/n$ per coordinate), $\rho$ of order 1 already dominates the bulk eigenvalues: $R_\lambda\to 0$ smoothly, and $\rho\to\infty$ gives RTN.
- **Token normalisation.** Replacing $x_t$ by $x_t/\|x_t\|$ sets $\kappa=1$: the BOS eigenvalue drops to the bulk level, the spectrum flattens, $R$ collapses toward 1. The 10×-cap keeps $\kappa=10$, i.e. an eigenvalue ratio $\sim 10^2\cdot n/1$ — still dominant, which is why it does not cure Llama (fact 6) but does cure Qwen-14B, whose $\kappa$ was already 10.
- **Calibration text containing $z$'s direction** (chat template, c4chat): $z_F$ enters $H_{FF}$, the regressor learns it, $r_\lambda(z)$ shrinks (fact 5).
- **RTN on that matrix.** $A=0$: $e=\delta^\top z_S$, $R\equiv 1$.

---

## 4. Collapse as a threshold on the amplified error

The linear-layer error above is what enters the block's non-linearity. Empirically (fact 4) the residual-stream error at that token after layer 2 orders the recipes exactly as $\|\delta\|\cdot\sqrt{R_\lambda}$ would, and one threshold $\theta\approx1.4$–$1.5$ (relative error) separates collapse from non-collapse. Beyond $\theta$ the error is amplified layer by layer (non-sink positions inflate 3.3× by layer 4); below it, it is absorbed (norm ratio 1.0). The existence of $\theta$ is attributed to the softmax/normalisation non-linearity (attention re-allocation once the anchor token's key is corrupted); its value is **not** derived here and is reported as measured.

Consequence, the paper's sentence: *collapse iff $\|\delta(b,g)\|\cdot\sqrt{R_\lambda(z)} > \theta$.* Bits move $\|\delta\|$ by 2× per bit and are the strong lever (4-bit: 1.3, below $\theta$; 3-bit: 3.85); group size is a weak lever (2.4, still above); damping, token normalisation, calibration and per-matrix RTN act on $R$ and can bring a 3-bit model below $\theta$ without spending bits.

---

## 5. Propositions to prove (in order of value)

**P1 (off-distribution error identity).** In the two-block model, $e_{\mathrm{GPTQ}}(z)-e_{\mathrm{RTN}}(z) = -\delta^\top H_{SF}(H_{FF}+\lambda I)^{-1}z_F$ for every $z$, and for $z$ drawn from the calibration distribution $\mathbb{E}\,e_{\mathrm{GPTQ}}^2\le\mathbb{E}\,e_{\mathrm{RTN}}^2$ (v1's P1). Short.

**P2 (amplification bound under one dominant token).** Let $H = \kappa^2 N\, bb^\top + H_{\mathrm{bulk}}$ with $\lambda_{\max}(H_{\mathrm{bulk}})\le \beta$ and $\lambda_{\min}$ of the bulk restricted to $F$ equal to $\mu_{\min}$. For a unit deployment input $z$ with $|b^\top z|\le\epsilon$, show
$$\frac{\|r_\lambda(z)\|}{\|z_S\|}\;\le\; 1 + \frac{\|H_{SF}\|}{\mu_{\min}+\lambda}\cdot\frac{\|z_F\|}{\|z_S\|}\quad\text{and a matching lower bound of the same order when } z_F \text{ aligns with the } \mu_{\min} \text{ eigenvector},$$
and that the ratio is decreasing in $\lambda$ and tends to 1 as $\lambda\to\infty$. The point of the statement: the amplification is controlled by the *bulk* spectrum, which the dominant token makes irrelevant to GPTQ; $\kappa$ enters only through $\lambda=\rho\cdot\mathrm{mean\,diag}H$ (which scales like $\kappa^2/n$) — this is the precise sense in which damping "adds curvature where the sink left none".

**P3 (bit factorisation).** For a symmetric uniform grid with $b$ bits and group step $s_g$, $\mathbb{E}\|\delta\|^2 = \tfrac{1}{12}\sum_j s_{g(j)}^2$ under the usual uniform-rounding-error assumption, and $s_g\propto 2^{-(b-1)}$; hence $\|e_{\mathrm{GPTQ}}(z)\|$ scales as $2^{-b}\sqrt{R_\lambda(z)}$ with $R_\lambda$ independent of $b$. (Standard; state it so that the table in §4 is a corollary.)

**P4 (token normalisation flattens the spectrum).** With $\tilde x_t = c\,x_t/\|x_t\|$, $\tilde H = c^2\sum_t \hat x_t\hat x_t^\top$ has $\lambda_{\max}(\tilde H)\le c^2 N\cdot\max_t\|\hat x_t\|^2 = c^2 N$ and the BOS direction's eigenvalue falls from $\kappa^2 N$ to at most $c^2 N_{\mathrm{BOS}}$; the ratio to the bulk drops from $\sim\kappa^2 n$ to $\sim n\cdot N_{\mathrm{BOS}}/N$. Combine with P2 to show $R_{\tilde H}(z)\ll R_H(z)$. Also state the capped variant ($\kappa\to\min(\kappa,K)$) to explain fact 6: the ratio is $K^2 n$, still $\gg 1$ for $K=10$.

**P5 (sequential exactness, as v1 P6).** The real algorithm nests the two-block step; the identity in P1 holds step-wise with $\delta$ replaced by the running error. Needed so that P1–P4 apply to GPTQ as implemented.

Optional: **P6** — a statement of when the token-averaged objective ranks a matrix as "good" while the template-token error is large: with $H\approx\kappa^2 N bb^\top$, the objective is $\kappa^2 N(\delta^\top b)^2+O(N/n)$, so any solution that fits $b$ wins the objective regardless of $r_\lambda$ on the bulk (the double dissociation of RESULTS 9.10k).

---

## 6. Data that tests each proposition (batch W49, Llama, no checkpoints)

`runs/stats/theory-l-*/stats.csv`, layer-1 `down_proj`, columns `tplpos_err_gptq`, `tplpos_err_rtn`, `tplpos_ratio`, `xnormpos`:

| recipe | predicted | checks |
|---|---|---|
| 3-bit g128 vs 4-bit g128 | ratio ($R$) equal; absolute error halves | P3 |
| 3-bit g32, 3-bit per-channel, 4-bit per-channel | ratio unchanged; absolute error tracks the group step | P3 |
| ρ = 1, 2, 5 | ratio decreasing in ρ; absolute error crosses $\theta$ between 1 and 2 | P2 |
| token-normalised | ratio ≈ 1 | P4 |
| c4chat | ratio ≈ 1 (z in span) | P1 |

Figure for the paper: x = $\|\delta\|\sqrt{R}$ from the stats (linear layer), y = measured residual-stream error after layer 2 (RESULTS 9.10q), one point per recipe, the threshold as a horizontal line, collapse/no-collapse as colour. If the ten points are monotone, the table in §4 is a prediction of §2–3 rather than an observation.

## 7. What is known vs. new (for the related-work paragraph)

Known: the OBS closed form and its in-distribution optimality; Babai-type bounds on the calibration objective (2507.18553); calibration-set sensitivity of GPTQ (2311.09755); Hessian anisotropy as a source of "low error, high loss" (HeRo-Q 2601.21626); massive activations at the first/delimiter tokens (2402.17762). New: evaluating the closed form off-distribution at the sink-forming layer; the identification of the dominant-token spectrum as the source of the amplification; the $\delta(b)\times R(H,\lambda,z)$ factorisation that separates bits from every other lever; the threshold that turns it into a collapse; the two zero-bit repairs that follow.
