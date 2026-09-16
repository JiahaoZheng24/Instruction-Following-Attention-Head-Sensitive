# ICLR 2027 draft layout

- `main.tex`            preamble (unchanged ICLR style) + `\input` of every section; compile this file
- `sections/`           one file per section, numbered in paper order; `00_macros.tex` holds notation
- `sections/appendix/`  A proofs (collaborator), B protocols, C failed pre-registrations, D extra tables
- `tables/`             one `table` environment per file, `\input` from the section that discusses it
- `figures/`            PDF artwork; see `figures/README.md` for the expected files
- `template_original_iclr2027.tex`  the untouched ICLR shell, kept for reference
- style files (`*.sty`, `*.bst`, `math_commands.tex`) untouched

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main` (or `latexmk -pdf main`).
Every `\todo{...}` prints in red; the source pointer to RESULTS_v3 is in a `%` comment at the top of each file.
