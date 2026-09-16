# Theory hand-off (2026-09-15, revised the same day)

Files, in this order:

1. `ICLR/ICLR_quantization/sections/05_mechanism.tex` — Section 5. Statements and the proofs of Theorem 1(a)–(c) and Proposition 3(a)–(c) are written; red `\todo{collaborator: ...}` marks what is missing.
2. `ICLR/ICLR_quantization/sections/appendix/A_proofs.tex` — Appendix A, one subsection per remaining item.
3. `paper/THEORY_BRIEF_v3_2026-09-10.md` §7 — earlier version of the derivation (superseded where it assumed `‖E‖ ≪ λ`; that assumption is false in the data, λ is ~10⁻⁶ ‖x_0‖²).

Notation (fixed in the paper): `W ∈ R^{d_out × d_in}`, inputs `x_t`, `x_0` = BOS input, `H = Σ_t x_t x_tᵀ`, `H_λ = H + λI = x_0 x_0ᵀ + M` with `M = Σ_{t≠0} x_t x_tᵀ + λI ≻ 0`, `v = M⁻¹ x_0`, `κ = x_0ᵀ v`. Rounding error of column `j`: `δ_j = w_j − q_j` (row vector over d_out). OBS update: `Δw_k = −δ_j [H_λ⁻¹]_{jk} / [H_λ⁻¹]_{jj}`. `g_t = ‖∂L/∂y_t‖²`.

## Already proved in the main text (please check)

- Theorem 1(a): exact single-step update `Δw_k = δ_j (v_j v_k/(1+κ) − [M⁻¹]_{jk}) / ([M⁻¹]_{jj} − v_j²/(1+κ))` (Sherman–Morrison).
- Theorem 1(b): for `M = λI`, `Δw_k = δ_j x_{0j} x_{0k} / (λ + ‖x_0‖² − x_{0j}²)`; output change at BOS `→ 0`.
- Theorem 1(c): output change at position `t`, leading order: `e_t = δ_j x_{0j} x_0ᵀx_t/‖x_0‖² − δ_j x_{tj}` (overlap term + plain rounding term); summed over rounded columns `J`: `e_t^{(r)} = c_r x_0ᵀx_t/‖x_0‖² + η_t^{(r)}`, `c_r = Σ_j δ_j^{(r)} x_{0j}`, `η_t^{(r)} = −Σ_j δ_j^{(r)} x_{tj}`.
- Proposition 3(a): a diagonal output-side factor `G` leaves the column-wise update unchanged (rows decouple). (b): BOS share of `tr H̃` equals `G_bos`. (c): `H̃` is the per-token-whitened diagonal-`G_t` approximation.

## What is needed

| # | Item | Where |
|---|---|---|
| 1 | Lemma 2: gradient attenuation through RMSNorm at the sink, with the residual path; proof; cite arXiv 2603.17771 | §5.1, App. A.2 |
| 2 | Theorem 1, successive steps under act-order (largest `x_{0j}²` first): the accumulated update over the sink columns stays in `span{x_0}` up to a controlled remainder; give the remainder | App. A.3 (1) |
| 3 | Theorem 1, general `M`: bound the difference between the exact form and the `M = λI` form in terms of `κ` and the off-diagonal mass of `M⁻¹`; the direction becomes `v = M⁻¹x_0`, the overlap `x_0ᵀ M⁻¹ x_t` | App. A.3 (2) |
| 4 | Order of the noise term `η` for grid step `Δ` (the floor seen for positions with small overlap) | App. A.3 (3) |
| 5 | Corollary 1(iii): one table — each remedy (damping, token normalisation, chat calibration, windows, re-rounding) → effect on `κ` or on the update → residual failure case | App. A.4 |
| 6 | Proposition 3(c) with an explicit approximation error; remark on the boundary with KronQ (arXiv 2607.07964) and GuidedQuant (arXiv 2505.07004): a token-axis weight is the only way the output side enters without changing the column-wise update | App. A.5 |
| 7 | Lemma 1: short Gauss–Newton proof and the measurement protocol of `g_t` | App. A.1 |

Priority: 2, 3, 6, then the rest.

## Measured facts the proofs may use

- `A_bos ≥ 0.998` (Llama), `1.000` (Qwen); `G_bos ≤ 0.008`; `κ` to be read from `runs/stats/f-l-3b` when W72 finishes (expected ≫ 10²).
- Three channels carry 99 % of `u` (Llama); the hidden-state error has cosine 0.90 with `u` at the lesion position.
- Overlap–error proportionality holds at the extremes (top vs bottom quintile 2–4×); the middle 60 % sits at the noise floor.

## Not asked

The amplification of the lesion by the rest of the network is an open question in the paper.

Deadline 2026-09-25 AoE. Statements may be adjusted only in `05_mechanism.tex`; labels `lem:twosides`, `lem:rmsnorm`, `thm:rankone`, `cor:shape`, `prop:tokenlevel` are referenced elsewhere.
