"""Figure 3 (fig3_ag): the blind spot, 8.6 x 5.4 cm (Section 7). Every model's most BOS-dominated
down-projection, input-side share A_BOS (linear) against output-side share G_BOS (log). The
dashed curve is A = G (the objective weights the token as much as the loss depends on it); the
shaded box is the blind spot. Data: runs/sides/<model>_c4.csv (fp16, 128 C4 documents).
  python ICLR/figure/code/make_fig3.py    (from the repo root; FIG_REPLACE=1 overwrites the last slide)
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptfig import *  # noqa: E402,F401,F403
from pptfig import Fig, Axis, PP_ALIGN, SUP  # noqa: E402

OUT = 'ICLR/figure/fig3_ag.pptx'
W, H = 8.6, 5.4
VERSION = 'v6 2026-09-22: as v5, shorter labels (blind spot, collapses)'

FAMILY = {
    'l': ('Llama-3.1-8B', 'Llama'), 'l3': ('Llama-3-8B', 'Llama'), 'l32': ('Llama-3.2-3B', 'Llama'),
    'l32_1b': ('Llama-3.2-1B', 'Llama'), 'hermes3': ('Hermes-3', 'Llama'), 'hermes2pro': ('Hermes-2-Pro', 'Llama'),
    'tulu3': ('Tulu-3', 'Llama'), 'r1_llama_8b': ('R1-Llama-8B', 'Llama'),
    'q14': ('Qwen2.5-14B', 'Qwen'), 'q7': ('Qwen2.5-7B', 'Qwen'), 'q32': ('Qwen2.5-32B', 'Qwen'), 'q3': ('Qwen2.5-3B', 'Qwen'),
    'qwen2_7b': ('Qwen2-7B', 'Qwen'), 'qwen3_14b': ('Qwen3-14B', 'Qwen'), 'qwen3_8b': ('Qwen3-8B', 'Qwen'),
    'qwen3_4b': ('Qwen3-4B', 'Qwen'), 'r1_qwen_14b': ('R1-Qwen-14B', 'Qwen'),
    'm7': ('Mistral-7B-v0.3', 'Mistral'), 'm7v02': ('Mistral-7B-v0.2', 'Mistral'), 'nemo': ('Nemo-12B', 'Mistral'),
    'ministral': ('Ministral-8B', 'Mistral'),
    'f3': ('Falcon3-7B', 'other'), 'falcon3_10b': ('Falcon3-10B', 'other'), 'sm': ('SmolLM2-1.7B', 'other'),
    'smollm3': ('SmolLM3-3B', 'other'), 'granite': ('granite-3.1-8B', 'other'),
    'g29': ('gemma-2-9b', 'gemma'), 'g22': ('gemma-2-2b', 'gemma'),
}
COL = {'Llama': (RED_S, RED_T), 'Qwen': (BLU_S, BLU_T), 'Mistral': (ORA_S, ORA_T), 'other': (PUR_S, PUR_T), 'gemma': (TEA_S, TEA_T)}
COLLAPSE = {'l', 'q14'}

pts = []
for fpath in sorted(glob.glob('runs/sides/*_c4.csv')):
    key = os.path.basename(fpath)[:-7]
    if key not in FAMILY:
        continue
    rows = [r for r in csv.DictReader(open(fpath)) if r['proj'] == 'down_proj']
    best = max(rows, key=lambda r: float(r['A_bos']))
    pts.append((key, float(best['A_bos']), max(float(best['G_bos']), 2e-5)))
print(len(pts), 'models')

fig = Fig(OUT, W, H, VERSION)
L, R, T, B = 1.0, 8.35, 0.30, 4.55
ax = Axis(-0.03, 1.10, L, R)
ay = Axis(1e-5, 1.0, B, T, log=True)
fig.frame(L, T, R, B)
fig.xticks(ax, B, [0, 0.25, 0.5, 0.75, 1.0], fmt='%g')
fig.yticks(ay, L, [10 ** e for e in range(-5, 1)], [SUP[e] for e in range(-5, 1)], grid_to=R)
fig.text(L, B + 0.36, R - L, 0.26, 'A_BOS:  input-side share of BOS', 6.5, INK, align=PP_ALIGN.CENTER)
fig.vtext(0.05, T, 0.3, B - T, 'G_BOS:  output-side share of BOS', 6.5, INK)

# blind-spot box and the A = G curve
fig.rect(ax(0.5), ay(0.01), R - ax(0.5), B - ay(0.01), RGBColor(0xFD, 0xEE, 0xEC))
fig.text(ax(0.5) + 0.10, B - 0.28, 4.6, 0.24, 'blind spot', 6.5, RED_T, bold=True)
grid = [10 ** (-5 + 5 * i / 40) for i in range(41)]
prev = None
for v in grid:
    if v > 1.0:
        continue
    pt = (ax(v), ay(v))
    if prev:
        fig.line(prev[0], prev[1], pt[0], pt[1], DASH, 0.6, dash=True)
    prev = pt
fig.text(ax(0.36), ay(0.36) + 0.06, 3.2, 0.22, 'no mismatch:  A_BOS = G_BOS', 6, DASH)

# points
for key, a, g in pts:
    fill, tcol = COL[FAMILY[key][1]]
    x, y = ax(a), ay(g)
    if key in COLLAPSE:
        fig.dot(x, y, 0.44, None, RED_T, 1.0)
    fig.dot(x, y, 0.22, fill, WHITE, 0.4)
P = {p[0]: (p[1], p[2]) for p in pts}
for key, ga in (('l', 1.4e-3), ('q14', 1.1e-4)):
    a, g = P[key]
    fig.line(ax(0.87), ay(ga), ax(a) - 0.24, ay(g), RULE, 0.5)
    fig.text(ax(0.55), ay(ga) - 0.11, ax(0.87) - ax(0.55), 0.22, FAMILY[key][0], 6.5, COL[FAMILY[key][1]][1], align=PP_ALIGN.RIGHT)
for key in ('g22', 'g29'):
    a, g = P[key]
    fig.text(ax(a) + 0.18, ay(g) - 0.11, 1.5, 0.22, FAMILY[key][0], 6.5, TEA_T)
a, g = P['f3']
fig.text(ax(a) - 0.7, ay(g) - 0.38, 2.0, 0.22, 'Falcon3-7B / 10B', 6.5, PUR_T, align=PP_ALIGN.CENTER)

# legend, lower left (empty)
lx, ly = ax(0.04), ay(2.6e-3)
for i, (fam, label) in enumerate([('Llama', 'Llama family'), ('Qwen', 'Qwen family'), ('Mistral', 'Mistral family'),
                                  ('other', 'Falcon, SmolLM, granite'), ('gemma', 'gemma-2 (no sink)')]):
    fig.dot(lx + 0.11, ly + 0.12 + i * 0.27, 0.18, COL[fam][0], WHITE, 0.4)
    fig.text(lx + 0.32, ly + i * 0.27, 2.8, 0.24, label, 6.5, COL[fam][1])
fig.dot(lx + 0.11, ly + 0.12 + 5 * 0.27, 0.28, None, RED_T, 1.0)
fig.text(lx + 0.32, ly + 5 * 0.27, 3.6, 0.24, 'collapses (short documents)', 6.5, RED_T)
fig.save()
