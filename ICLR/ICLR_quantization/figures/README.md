# figures/

Expected files (PDF preferred; referenced without extension from sections/):

- `fig1_teaser.pdf`  Llama five bars (fp16 / RTN3 / GPTQ3 / corrected / GPTQ4) + PPL and MMLU flat lines from the same checkpoints
- `fig2_lesion.pdf`  row 788 error, per-position error by token class, transplant 2x2
- `fig3_ag.pdf`      A vs G share by position class, 26 sink models + 2 gemma-2

Source data: `runs/INDEX_scores.csv`, `runs/sides/*.csv`, `runs/sink/*.csv`.
Plot scripts go in `src/plots/` (to be written).
