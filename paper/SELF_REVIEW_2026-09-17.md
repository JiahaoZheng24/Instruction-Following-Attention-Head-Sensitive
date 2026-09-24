# Self-review of the ICLR draft (2026-09-17)

Written as a reviewer of `ICLR/ICLR_quantization` (sections as of commit c29bd74 plus the §7 edit) and of the collaborator's draft `ICLR/ICLR_Math_Roy.pdf`. Three questions from the advisor meeting are answered first, then the collaborator's draft, then the paper as a whole. Every claim below was checked against `paper/RESULTS.md` and the run files; where a number is quoted it is the fixed-code value.

## 1. Does every experimental finding have a theoretical statement behind it?

No. The ledger below lists each finding in the paper, the statement that is supposed to cover it, and how far the coverage actually goes. "Derived" means the statement predicts the finding; "consistent" means the statement allows it but does not predict it; "none" means the paper currently has no mathematics for it.

| # | Finding (section) | Statement | Coverage | Action |
|---|---|---|---|---|
| 1 | Sink-forming down-projection has A_BOS ≥ .5, G_BOS ≤ .008 in 26/26 sink models, 0/2 without (§5, §7) | Prop. 1 | Measured premise, not derived. Lemma 2 covers only the G_BOS ≈ 0 half. | Collaborator: Lemma 2 with both paths (RMSNorm attenuation and the near-zero value vector at the sink). |
| 2 | g_t concentrates on the first positions of each sample, 10³–10⁵× the median (§5) | Prop. 1, second half | None. The text asserts "because the response attends to them through the sink"; nothing derives it. | State as an explicit Assumption (attention-mediated readout: ‖∂L/∂y_t‖ grows with the attention mass later positions place on t, except at the sink whose value vector is ≈ 0). Collaborator can give the first-order chain-rule form. |
| 3 | Damage sits in the sink-forming down-projection and nowhere else (§4) | Prop. 1 + Thm. 1 | Derived: only that matrix has a dominant token. | None. |
| 4 | Re-rounding that matrix with RTN cures, no extra bits (§4) | Thm. 1 | Derived: the error is in the compensation, which RTN lacks. | None. |
| 5 | Error concentrated in row 788, largest entry 5.4 vs 0.62 (§4); row concentration 14/17, overshoot 16/17 (§7) | Cor. 1(i) | Direction derived; the magnitude and the clipping are not. | Collaborator: a corollary with the grid step Δ of the super-weight row showing the update to near-zero weights on BOS columns is O(Δ_row), hence exceeds the group range. |
| 6 | Hidden-state error direction cos .88 with the sink direction (§4) | Cor. 1(i) | Derived (rows with largest c_r carry the error, that row is the sink output channel). | None. |
| 7 | Template tokens damaged (.97), response spared (.34) (§4) | Cor. 1(ii) overlap term | **Partly contradicted.** The overlap quintile test (W68, fixed code) shows the top quintile at 2.5× (Llama) and 3.9× (Qwen) the bottom, but template tokens are not the high-overlap tail: in Llama they make up 3% of the top quintile and 12% of the bottom one. The measured overlap is the isotropic x₀ᵀx_t; the whitened overlap x₀ᵀM⁻¹x_t of the collaborator's Corollary 2 has never been measured. | Experiment E1 below. Until it runs, §4 and §5 must not say that (ii) "is" the template-versus-response contrast. |
| 8 | Overlap quintiles: extremes scale, middle three flat (§5, end) | Cor. 1(ii) + noise floor η | Derived and confirmed. | None. |
| 9 | Protocol dependence: short docs collapse Llama, windows do not; Qwen collapses under Pile and WikiText windows; Mistral under c4win only (§3) | Cor. 1(iii) via κ | **Not supported by our own data.** Llama's BOS price ratio is 26k under short docs and 24k under c4win (W59), and the two lesions are of similar size (W57); the transplants show the protocol dependence lives in the background. κ has never been computed. | Experiment E2 (κ per protocol). Rewrite (iii) as a statement about the amplitude of the deterministic term, not a collapse criterion. |
| 10 | Rotation (QuaRot) removes the collapse (§3, §5) | Cor. 1(iii) "lowers κ" | **Wrong as stated.** κ = x₀ᵀM⁻¹x₀ is invariant under an orthogonal rotation of the inputs. What rotation removes is the coordinate-wise lever: x₀ becomes dense, so x₀ⱼ²/‖x₀‖² is small for every column and the per-entry update no longer exceeds the grid. The collaborator's table (§1.5 of his draft) says this correctly. | Fix §5 (iii). |
| 11 | Lesion harmless in fp16 (Llama .764), costs 4 points in RTN, fatal in any GPTQ background (§4) | none | Open problem, stated as such. Note Qwen's lesion alone costs 18 points in fp16 (.637 vs .820); the text says only "almost nothing". | Fix §4 text. |
| 12 | Injecting the sink direction at 4× amplitude does not hurt fp16 (§8) | none | Excludes fragility; no statement needed. | None. |
| 13 | g_t alone fails on Llama (.329) and Qwen (.580) (§6) | Prop. 3(b) | **Derivable but not stated.** With w_t = g_t the BOS share of tr H̃ is the A·G share, which is .984 (Llama) and .993 (Qwen): the dominant token survives. | Add one sentence to Prop. 3(b) or §6. |
| 14 | 1/‖x_t‖² alone repairs Llama and Qwen but not Mistral (.189); g_t alone repairs Mistral (.475) (§6) | Prop. 3(b) | Half derived (normalisation removes the dominant token). Why Mistral needs the output side is not explained. | Honest sentence in §6: the theory predicts both factors are needed, and the two models on which each half fails are its evidence, not a derivation of which half fails where. |
| 15 | Capping g_t at 100× median kills the repair (.160) (§6) | none | None. | Remark: the cap lowers the template weight below the level at which the price of trading BOS against a template token changes the OBS solution; a quantitative version needs κ. |
| 16 | Corrected objective repairs all 3-bit collapses; bounded −4.6 vs RTN on 18 models (§6, §7) | Prop. 3(b)(c) | Motivation derived; no guarantee. | State as such. |
| 17 | MMLU and PPL do not see the collapse (§3) | none | Not mathematical; the metrics never present the token class that carries the damage (WikiText has no chat template, MMLU few-shot has none). | Say this in §3 instead of the teacher-forcing argument alone. |
| 18 | AutoRound shares the collapse (§3, §5) | Prop. 1 | **Inconsistent with the paper's own logic.** §5 says the mismatch alone does not imply collapse and the compensation freedom is needed; Thm. 1 is about OBS. AutoRound has no OBS step. | Either add a sentence that learned rounding offsets are another free variable fitted to the same A-weighted block loss (consistent, not derived), or drop the claim that §5 "is why" AutoRound fails. |
| 19 | 4-bit: GPTQ does not beat RTN on average; corrected recovers Mistral and Hermes (§6) | none in the current draft | Earlier RESULTS argued the clipped displacement is set by the group range and therefore bit-independent; the paper no longer says this. | Optional: reinstate as a remark under Cor. 1(i) once item 5 is derived. |
| 20 | Lesion signature 17/17 on unseen models (§7) | Thm. 1, Cor. 1(i) | Derived prediction, confirmed. | None. |
| 21 | Lesion size does not rank losses (Spearman −0.35, n.s.) (§7) | consistent with "lesion × background" | Fine. | None. |
| 22 | κ ≫ 1 (§5) | Thm. 1 | Asserted, never measured (D.2 is a todo). | Experiment E2. |

Summary for the advisor: items 3, 4, 6, 8, 20 are derived and confirmed; 1, 13, 16 are derivable with a sentence each; 5, 7, 9, 10, 18 are the ones where the draft currently claims more than the mathematics gives, and 11 is the open problem the paper already declares. The two things the theory is silent on and the experiments carry alone are the collapse itself (11) and the protocol dependence (9).

## 2. Are the format tokens significant beyond BOS and the sink, and on what assumption?

What we have:

- Output-side: on the chat set, template tokens carry g_t 30–200× the median while BOS is 0.03× (Llama, W59); on the C4 calibration set the same role is played by the first tokens of each document. The sensitivity is a property of the position class (attended by everything after it), not of the token identity. The paper's Prop. 1 currently writes "the template tokens" for a measurement made on C4 documents; the two must be distinguished.
- Damage: the largest hidden-state error after the lesion sits at position 4 (`<|end_header_id|>`, 3.59 vs .58), and template tokens carry .97 against .34 on the response.
- Causal, negative: a perturbation along the sink direction at the template position, at 4× the lesion's amplitude, does not hurt the fp16 model; in the RTN background it costs 8 points at 4×, in the cured GPTQ background 22 points at 8×. So the format tokens are where the damage lands, but a perturbation there is fatal only inside a quantised background.

What is missing: a position-matched control (the same perturbation energy at ordinary positions) and a derivation. The assumption to state in §5 is: the loss is read out from each prompt position in proportion to the attention mass that later positions place on it, so ‖∂L/∂y_t‖² is large at the positions that act as secondary sinks (template tokens, newlines) and small elsewhere; the primary sink is the exception because its value vector is near zero and RMSNorm attenuates its gradient (Lemma 2). This is an assumption with a measurement behind it, not a theorem, and should be labelled Assumption 1.

## 3. Why is the template-token error larger: overlap with BOS?

Tested (W68, fixed code), and the answer is "only partly":

| | Llama L1 | Qwen L4 |
|---|---|---|
| mean ‖e_t‖, top vs bottom overlap quintile | 1.76 / 0.70 (2.5×) | 22.4 / 5.7 (3.9×) |
| middle three quintiles | 0.59 / 0.61 / 0.55 | 5.5 / 5.7 / 7.3 |
| template share of the top quintile | 0.03 | 0.14 |
| template share of the bottom quintile | 0.12 | 0.03 |

The overlap term explains the extreme tail and the flat middle, as Theorem 1(c) says. It does not explain why template tokens are damaged: in Llama they mostly sit in the low-overlap quintile. The candidate explanation inside the theory is the row, not the overlap: the compensated error is concentrated in the sink output row on the columns where x₀ is large, so any token with activation on those columns, in whatever sign pattern, receives an error in the sink output channel; the relevant readout is δ⁽ʳ⁾ᵀx_t, or in the collaborator's general form x₀ᵀM⁻¹x_t, neither of which has been measured. Experiment E1 settles which.

## 4. The collaborator's draft (ICLR_Math_Roy.pdf)

The narrative of §1 (objective mismatch → compensation freedom → misallocation → fingerprints) is the right one and matches the paper. Checked derivations: Theorem 1(a) eq. (7), (b) eqs. (8)–(9), (c) eqs. (10)–(12) are correct and (8) is the general-M output error the paper's todo asked for. Proposition 2(a) is stronger than the paper's (which assumed diagonal G) and correct for any positive semidefinite G: with the column constraint Δw⁽ʳ⁾ⱼ = c_r, write Δw⁽ʳ⁾ = c_r u + z_r with u = H⁻¹e_j/[H⁻¹]_jj and z_r,j = 0; then uᵀHz_r = 0, so the cross terms vanish and z = 0 is optimal. Adopt it.

To adopt into §5: eq. (8) as Theorem 1(b); Corollary 2 (whitened overlap x₀ᵀM⁻¹x_t) in place of the paper's Cor. 1(ii); Prop. 2(a) for general G; the remedies table into Appendix A.

Missing or to change:

1. **Lemma 2 is absent.** The draft takes G_BOS ≈ 0 as measured. The advisor wants it derived: Jacobian of h ↦ h/rms(h) has operator norm O(1/rms(h)), plus the value path (attention to the sink transports a value vector of near-zero norm). Both halves needed for the statement "the sink receives almost no gradient".
2. **Successive steps.** "Sequential interactions remain" is a sentence, not a bound. Under act-order the BOS columns are rounded first; needed: the accumulated update over those columns stays in span{v} up to a remainder, with the remainder's order.
3. **Row concentration and clipping** are called fingerprints but not derived (item 5 above).
4. **Corollary 2's direct test** is the whitened overlap; say explicitly that the isotropic overlap has been tested and does not place template tokens in the tail, so the general-M form is the one that must be measured (E1).
5. **§1.5, long windows**: "reduce the relative dominance of the BOS direction" is not what the data show for Llama (price ratio unchanged between short docs and c4win); keep the row but say the effect is on the background, per the transplants.
6. **Rotation row** is correct and should replace the paper's κ argument for rotation.
7. Numbers in Prop. 1 (0.998, 0.008) should come from the paper's macros so they cannot drift.
8. Notation: the paper uses H_λ, M, v, κ as in the draft; the draft's "ρ tr H/d_in" and the paper's "ρ mean diag H" are the same and should read the same.

## 5. Experiments to add before 09-25 (stats mode, no checkpoints, one job)

W76, pre-registered:

- **E1 whitened overlap.** At the sink down-projection of Llama, Qwen-14B and Mistral, from the actual calibration Hessian: M = H_λ − x₀x₀ᵀ, v = M⁻¹x₀; per token on the chat set, |vᵀx_t| and the row-readout |δ⁽⁷⁸⁸⁾ᵀx_t|; quintiles of each against ‖e_t‖, and the template share of each quintile. Prediction: template share of the top quintile ≥ 0.5 for at least one of the two statistics; if neither, Cor. 2 is demoted to "extremes only" and the row-readout becomes the stated mechanism for template damage.
- **E2 κ.** κ = x₀ᵀM⁻¹x₀ at the sink down-projection for the 17 sink models under short docs, and for Llama/Qwen/Mistral under every protocol in Table 1. Prediction: κ ≫ 10² everywhere; κ does not order the protocols by collapse (Llama short vs c4win within 2×). Fills D.2 and item 22.
- **E3 rank-one test on the actual update.** For the same matrices, ΔW = Q_GPTQ − Q_RTN (the compensation alone); share of ‖ΔW‖²_F captured by projecting each row onto x₀/‖x₀‖ and onto v/‖v‖; same share on a non-sink matrix as control. Prediction: ≥ 0.5 on the sink matrix along v, ≤ 0.05 on the control. This is the direct test of Theorem 1(a) on the real sequential output and is currently absent.
- **E4 position-matched perturbation.** In the RTN3 Llama background, inject the same energy along the sink direction at the template positions (2–7) and at six ordinary prompt positions; IFEval. Prediction: template arm ≥ 5 points below the ordinary arm. Answers the advisor's question causally.
- **E5 (W77) format tokens as carriers.** Activation patching of the collapsed checkpoint: restore the fp16 hidden states after the lesion layer at the template positions only (control: the same number of ordinary positions; BOS only; whole prompt) during prefill; IFEval. Prediction: template ≥ .45 from .155, ordinary ≤ .25. Calibration side: token normalisation plus a constant weight on the first eight positions (no gradient) should cure if the position class is what matters; the corrected objective with those positions' weight removed should fail. `jobs/w77_format_tokens.sh`, RESULTS §9.10bp.

## 6. The paper as a whole

**Narrative.** The chain is sound: silent failure → one matrix → why the objective produces it → repair → how far it goes. Two places break the "why before what" rule and one breaks consistency:

- §3 paragraphs "Collapse that likelihood does not show" and "Dependence on how the calibration text is cut" are result narrations. Each should open with the question the experiment answers (why compare against RTN: because RTN has no compensation, so any gap is the compensation's; why change the sampling: because the Hessian is the only thing the sampling touches).
- §6 "Ablation" lists outcomes. It should open with the prediction (Prop. 3 says both factors are necessary; the two halves are run to find the failure case of each), then the outcomes.
- §4 says the lesion "costs almost nothing" in fp16 while Table 6 shows Qwen's lesion costing 18 points; §5 says (ii) "is" the template-versus-response contrast while its own last paragraph says (ii) holds only at the extremes; §5 attributes rotation to κ, which rotation does not change.

**Structure.** The collaborator's "messy" impression has a specific cause: the theory's predictions are verified in §4, before the theory, and §5 then points back to §4 for each corollary ("(i) is the row-788 concentration of §4"). Proposed change, with the rationale that each section should answer one question:

- §3 What fails: keep the setup, collapse and protocol paragraphs; move "Which quantisers share it" to §7.
- §4 Where: keep "Where it is" (localisation, excision) and "Necessary, not sufficient" (transplants). Drop "What it looks like".
- §5 Why: after Corollary 1, one paragraph "The lesion as predicted" holding row 788, clipping, the cosine, the quintiles and the template/response contrast, with Figure 3 moved here. Predictions and their confirmation are then read in one place, and every "derived next" / "of §4" cross-reference disappears.
- §6 Repair: unchanged, plus the g_t-alone sentence (item 13).
- §7 Generality: across models (present content) and across quantisers (the paragraph from §3).

This keeps eight sections, moves two paragraphs and one figure, and removes four forward references.

**Wording to fix now (independent of new experiments).**

1. §4, transplants: "Inside the full-precision network it costs almost nothing for Llama (.764) and 18 points for Qwen (.637), short of collapse in both."
2. §4 and §5: replace "read out wherever a token overlaps that pattern" and "(ii) is the template-versus-response contrast" with: the overlap term accounts for the extreme quintiles; the template-token damage is carried by the concentrated row and is the subject of E1.
3. §5 Cor. 1(iii): rotation does not change κ; it removes the coordinate lever. Long windows: the data do not show a κ change for Llama; the protocol dependence is in the background (§4).
4. §5 Prop. 1: "the first positions of each calibration sample (document-initial tokens under C4, template tokens under chat prompts)".
5. §5: the AutoRound sentence becomes "consistent with", not "which is why".
6. §6: add "with w_t = g_t alone the BOS share of the weighted trace is the A·G share, .984 for Llama, so the dominant token survives; this is why output weighting alone fails there."
7. §5: label the template-sensitivity claim as Assumption 1 with its measurement.

**Missing pieces for a submission.** κ (E2), Appendix A (collaborator), Appendix C rows, Appendix D tables D.1–D.6, all four figures, the AI-use statement, and one seed replicate of the Mistral c4win collapse (only one seed of GPTQ; the corrected arm has two).


## 7. Status after W76–W80 (2026-09-18)

Ledger update, same numbering as section 1. Derived and confirmed: 3, 4, 6, 8, 20, plus the new amplitude statement (BOS sets the size of the compensation on the sink column, 4–500×, and the overshoot 5–12× |W|; direction is the sink-channel geometry shared with the BOS-free Hessian). Assumed, measured and causally tested: 1, 2 (Assumption 1; injection at template vs ordinary positions .291 vs .558 at β=8; restoring the chat scaffold alone repairs both collapses, .636 / .755, user-text positions .141 / .435). Theory offered forms, data chose: 7 (row readout, recall .54 / .55; whitened worst), 13 (Prop 3(b)), 14 (position prior replaces the gradient on Mistral, .405 / .418). Silent: 9 (collapse frequency 4/5, 4/4, 2/2; corpus-independent, Pile short docs .154 / .332), 11 (background), Mistral's global phenotype, 4-bit non-gain, MMLU cost. Removed from the theory after refutation: κ as a collapse criterion, extreme weights as the working part of the repair, whitened overlap as the placement of template tokens. Open for the collaborator: why the sending pattern follows x₀ rather than v; Lemma 2 both paths; act-order accumulation; clipping corollary.
