# ICLR/figure — the paper's figures

Everything figure-related lives here; the LaTeX project only receives the exported PDFs.

```
ICLR/figure/
  fig1_mechanism.pptx   Figure 1  (14.0 x 5.0 cm)   one slide per revision, VERSION in the notes
  fig2_bars.pptx        Figure 2  (14.0 x 4.8 cm)
  fig3_ag.pptx          Figure 3  ( 8.6 x 5.4 cm)
  fig4_lesion.pptx      Figure 4  (14.0 x 4.6 cm)
  fig*_preview.png      preview of the LAST slide of each deck (what the paper shows)
  code/
    pptfig.py           shared helpers: Fig (slides, text, boxes, lines, polygons, axes), palette, heat map
    make_fig1.py ... make_fig4.py   generators (data -> editable PowerPoint shapes)
    render.py           export the last slide: preview PNG here, PDF into ../../ICLR_quantization/figures/
    fig1_schematic.tex  old TikZ draft of Figure 1, not used
ICLR/ICLR_quantization/figures/
  fig1_mechanism.pdf fig2_bars.pdf fig3_ag.pdf fig4_lesion.pdf   what \fig{} includes; do not edit by hand
```

## Workflow

From the repo root (base `python`: python-pptx, pywin32, pymupdf; PowerPoint installed):

```
python ICLR/figure/code/make_fig2.py                     # appends a slide (a new revision)
FIG_REPLACE=1 python ICLR/figure/code/make_fig2.py       # overwrites the last slide while iterating
python ICLR/figure/code/render.py ICLR/figure/fig2_bars.pptx
```

`render.py` exports the last slide to `fig2_bars_preview.png` (~450 dpi, for checking) and to
`ICLR/ICLR_quantization/figures/fig2_bars.pdf` (one page, vector, 1:1 with the slide size).
Slides are the printed size, so nothing is scaled in LaTeX (`\fig{name}{\linewidth}`).
Every deck keeps its earlier revisions as earlier slides; the paper always shows the last one.

## Design rules (2026-09-23, v13 / v4 / v7 / v6)

Modelled on two ICLR 2025 mechanism papers: An et al., *Systematic Outliers in LLMs* (Fig. 2:
pre-norm block, pale-blue rounded boxes with dark-blue text, dashed MLP / attention groups, the one
matrix in saturated red, red labels on red arrows, residual line on the right) and Gu et al., *When
Attention Sink Emerges* (few large elements, symbols not sentences, prose in the caption).

- Fig. 1 keeps light-grey rounded containers (Gu et al.) because its three panels are heterogeneous; the data figures have none. Panel titles are formal noun phrases in sentence case (`(b) Error by token class`).
- Token names are spelled out (`start_header`, `end_header`, backslash-n backslash-n for the double newline, `eot`), never symbols.
- No text below 6 pt at print size; axis titles 6.5 pt; panel titles 7.5 pt.
- One palette, one meaning everywhere (`pptfig.py`): red = sink token / BOS / the short-document
  GPTQ collapse; orange = format (template) tokens; grey = user text / RTN; blue = response / fp16;
  green = corrected objective; purple / teal = other families in Fig. 3.
- Every number drawn is read from the run files or `00_numbers.tex`, never typed.
- No text touches a box edge, a dashed group or a line (v12 of Fig. 1 widened panel (a) to 4.2 cm for this); every arrow is a full segment with one head, never a bare head (boxes 0.27 cm high, every gap at least 0.2 cm, v13); one arrow into each adder; the residual line stays clear of the dashed groups.

| figure | content | data |
|---|---|---|
| Fig. 1 | (a) pre-norm block after An et al., the matrix in red, 475 = BOS input norm of the matrix (00_numbers LbosNorm) on the input arrow; (b) mirrored per-token bars on one token axis, g_t up and ‖x_t‖² down, with the consequence (GPTQ ÷ RTN output error, header 2-5x, BOS 0.4x) noted beside each side | `runs/sides/f_l_fig1_tokens.csv` (W86), `runs/stats/f-l-3b/stats.csv` |
| Fig. 2 | (a) dot plot: fp16 / RTN / GPTQ per benchmark (IFEval, Multi-IF, GSM8K, MMLU) for Llama and Qwen, RTN–GPTQ joined; (b) IFEval of 3-bit GPTQ by calibration protocol, three marker shapes, filled red when below the model's RTN by more than the seed noise | macros of `00_numbers.tex` |
| Fig. 3 (Section 7, 0.62 linewidth) | A_BOS (linear) against G_BOS (log), 28 models by family, A = G curve, blind-spot box, rings on the two collapses | `runs/sides/<model>_c4.csv` |
| Fig. 4 | (a) layer × position heat map of relative error with the origin cell boxed; (b) relative error by token class; (c) error by quintile of the lesion row's readout with the template share | `runs/div_f_l_none.pos.csv`, `runs/sink/f_l_artifact.csv`, `runs/sink/f_l_readout.csv` |

Why Fig. 4(a) is a heat map and not a 3-D bar plot like Fig. 1 of *Systematic Outliers*: that plot
shows magnitude over token × channel, where one spike stands alone; here the tall layer-4 wave would
occlude the layer-2 origin cell, which is the point of the panel.

## Earlier revisions

Slides 1-5 of `fig1_mechanism.pptx` are v1-v6 (grey containers, prose inside the figure, schematic
bars before W86); slide 1 of the other decks is the first drawing (parallel coordinates for Fig. 2,
log-log for Fig. 3). Kept for the record; nothing in the paper points at them.
