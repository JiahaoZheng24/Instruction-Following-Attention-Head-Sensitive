# figures/

Expected files (PDF, referenced without extension from sections/):

| file | where | content |
|---|---|---|
| `fig1_mechanism.pdf` | Figure 1, Introduction | the mechanism: three panels, spec below |
| `fig2_bars.pdf` | Figure 2, Section 3 | Llama five bars (fp16 / RTN3 / GPTQ3 / corrected / GPTQ4) with MMLU and PPL markers |
| `fig2_lesion.pdf` | Figure 3, Section 4 | (a) row-wise error of the sink down-projection, GPTQ vs RTN; (b) hidden-state error by token class along prompt and response; (c) cosine with u by position |
| `fig3_ag.pdf` | Figure 4, Section 5 | A_bos vs G_bos scatter, 26 sink models + 2 gemma-2 |

`fig1_schematic.tex` is a TikZ draft of Figure 1 kept for reference; it is not `\input` anywhere.

## Figure 1 specification (draw in PowerPoint, export PDF, full text width, ~14 x 7 cm)

Three panels left to right, (a) narrow, (b) wide, (c) medium. Colours used throughout the paper's figures:
sink token red, template tokens orange, instruction tokens grey, response tokens blue, GPTQ dark, RTN light, repair green.

### (a) Where in the block — a pre-norm transformer block, vertical, like Figure 1 of "When Attention Sink Emerges"
- Residual stream as a vertical line with two additions (⊕).
- Branch 1: RMSNorm → attention (Q/K/V, O) → ⊕.
- Branch 2: RMSNorm → MLP: gate/up → SiLU·(·) → **down_proj** → ⊕.
- Highlight `down_proj` in red; label its input arrow `x_t` ("the calibration Hessian is built from these") and its output arrow "writes the sink: ‖h_BOS‖ ≈ 475 vs ≈ 1".
- On the RMSNorm at the top of the NEXT block, a small backward arrow labelled "∂L/∂y_t attenuated by 1/rms(h) at BOS (Lemma 2)".
- Caption inside panel: "layer 1 of Llama-3.1-8B-Instruct, layer 4 of Qwen2.5-14B-Instruct".

### (b) Two sides on different tokens — token axis with two bar rows
- Token axis (11 boxes, left to right): `BOS` (red), `<|start_header_id|>`, `user`, `<|end_header_id|>`, `\n\n` (orange), `Write`, `a`, `poem` (grey), `<|eot_id|>` (orange), `Roses`, `are` (blue). Under the axis: "prompt" bracket over the first nine, "response" over the last two.
- Row 1 (below the axis, red bars): input side ‖x_t‖². BOS bar full height, every other bar ~2 % of it. Right label: "A_BOS = .998 → H ≈ x₀x₀ᵀ". Left label: "what the objective weights by".
- Row 2 (above the axis, orange bars): output side g_t. BOS ~0; `<|end_header_id|>` tallest, `<|start_header_id|>` and `\n\n` about 70 %, `user` 40 %, `<|eot_id|>` 60 %; instruction tokens ~15 %; response ~5 %. Right label: "G_BOS = .000, mass on the first 1.7 % of positions". Left label: "what the loss depends on".
- Two thin arcs from the response tokens: to BOS (red, "attention sink") and to `<|end_header_id|>` (orange).
- One line between the rows, italic: "the objective sees only the lower row".

### (c) From blind spot to collapse — four small items in a row, arrows between
1. The matrix W (down_proj) as a rectangle with one highlighted row "row 788 (super weight)".
2. "OBS update  Δw_k = δ_j x₀ⱼ x₀ₖ / ‖x₀‖²  — rank one, along x₀ (Theorem 1)".
3. Small bar chart "error e_t ∝ x₀ᵀx_t along u, by token class": template tall (orange), instruction short (grey), response short (blue), BOS ~0 (red).
4. Red box: "collapse in free generation: IFEval .768 → .155; MMLU and perplexity unchanged".
- Below, a green box with a dashed arrow back to item 2: "repair: w_t = g_t / ‖x_t‖² puts the upper row of (b) into H".

Source data for (b): `runs/sides/*.csv` (A/G shares by position class); for (c) bars: `runs/sink/f_l_artifact.csv` (rel_err by class: prompt_template .97, prompt_ordinary .56, resp_ordinary .34).
