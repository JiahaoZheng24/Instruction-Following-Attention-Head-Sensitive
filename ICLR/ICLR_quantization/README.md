# ICLR 2027 draft layout

- `main.tex`            preamble (unchanged ICLR style) + `\input` of every section; compile this file
- `sections/`           one file per section, numbered in paper order:
  `01_introduction`, `02_background`, `03_failure`, `04_lesion`, `05_theory`, `06_repair`, `07_generality`, `08_conclusion`, `10_statements`
  `00_macros.tex` holds notation, `00_numbers.tex` holds every number quoted in the paper (one macro per value, with its run tag)
- `sections/appendix/`  A proofs (collaborator), B protocols and the loop correction, C failed pre-registrations, D extra tables
- `tables/`             one `table` environment per file; only `tab_ablation` is in the main text, the rest are `\input` from Appendix D
- `figures/`            exported PDFs only (`fig1_mechanism`, `fig2_bars`, `fig3_ag`, `fig4_lesion`); sources, generators and previews live in `../figure/` (see `../figure/code/README.md`)
- `template_original_iclr2027.tex`  the untouched ICLR shell, kept for reference
- style files (`*.sty`, `*.bst`, `math_commands.tex`) untouched

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (or `latexmk -pdf main`).
Every `\todo{...}` prints in red; the source pointer to RESULTS is in a `%` comment at the top of each file.
