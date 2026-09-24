"""Figure 2 (fig2_bars): the collapse and its silence, two dot plots, 14 x 4.8 cm.
 (a) 3-bit GPTQ (short documents) against RTN on the same grid, by benchmark, Llama and Qwen:
     fp16 / RTN / GPTQ as three dots per row, RTN and GPTQ joined.
 (b) IFEval of 3-bit GPTQ under seven calibration protocols, Llama / Qwen / Mistral as three
     marker shapes; a marker is filled red when it falls more than the seed noise below the
     model's RTN score (dashed lines).
Values are the macros of 00_numbers.tex.
  python ICLR/figure/code/make_fig2.py    (from the repo root; FIG_REPLACE=1 overwrites the last slide)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptfig import *  # noqa: E402,F401,F403
from pptfig import Fig, Axis, PP_ALIGN, MSO_ANCHOR  # noqa: E402

OUT = 'ICLR/figure/fig2_bars.pptx'
W, H = 14.0, 4.8
VERSION = 'v4 2026-09-22: as v3, one-line note under (b), letters on the RTN lines'

num = {m.group(1): m.group(2) for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}',
                                                   open('ICLR/ICLR_quantization/sections/00_numbers.tex', encoding='utf-8').read())}


def f(k):
    v = num[k]
    return float(v.split('/')[0]) if '/' in v else float(v)


NOISE = f('SeedNoise') / 100.0
fig = Fig(OUT, W, H, VERSION)

# ============================================================== (a) by benchmark
AX = 0.10
fig.panel_label(AX, 0.06, 'a', '3-bit GPTQ against RTN, by benchmark')
BENCH = [('IFEval', 'IF'), ('Multi-IF, turn 1', 'MIF'), ('GSM8K', 'GSM'), ('MMLU', 'MMLU')]
MODELS = [('L', 'Llama-3.1-8B'), ('Q', 'Qwen2.5-14B')]
L, R = AX + 2.05, 6.85
ax = Axis(0.0, 1.0, L, R)
rh, gg = 0.34, 0.14
y = 0.72
rows = []
for bi, (bname, bk) in enumerate(BENCH):
    for mi, (mk, mname) in enumerate(MODELS):
        rows.append((y, bname if mi == 0 else '', mname, mk, bk))
        y += rh
    y += gg
B = y - gg + 0.04
T = 0.60
fig.frame(L, T, R, B)
fig.xticks(ax, B, [0, 0.25, 0.5, 0.75, 1.0], fmt='%g')
fig.text(L, B + 0.34, R - L, 0.24, 'accuracy', 6.5, INK, align=PP_ALIGN.CENTER)
for v in [0.25, 0.5, 0.75]:
    fig.line(ax(v), T, ax(v), B, GRID, 0.4)
for (yy, bname, mname, mk, bk) in rows:
    cy = yy + rh / 2
    if bname:
        fig.text(AX, cy - 0.11, 1.25, 0.24, bname, 6.5, INK, bold=True, align=PP_ALIGN.RIGHT)
    fig.text(AX + 1.30, cy - 0.11, 0.72, 0.24, mname.split('-')[0] if mk == 'L' else 'Qwen', 6, GRY_T, align=PP_ALIGN.RIGHT)
    vf, vr, vg = f(mk + 'fp' + bk), f(mk + 'rtn' + bk), f(mk + 'gptq' + bk)
    fig.line(ax(vr), cy, ax(vg), cy, RED_S if vg < vr else BLU_S, 1.2)
    fig.dot(ax(vf), cy, 0.18, WHITE, BLU_S, 0.9)
    fig.dot(ax(vr), cy, 0.18, GRY_S, WHITE, 0.4)
    fig.dot(ax(vg), cy, 0.18, RED_S, WHITE, 0.4)
# legend (top, inside the frame's header line)
lx, ly = AX + 0.55, 0.36
for i, (lab, fill, edge, tcol) in enumerate([('fp16', WHITE, BLU_S, BLU_T), ('RTN 3-bit', GRY_S, GRY_S, GRY_T),
                                             ('GPTQ 3-bit (short documents)', RED_S, RED_S, RED_T)]):
    fig.dot(lx + 0.09, ly + 0.11, 0.15, fill, edge, 0.8)
    fig.text(lx + 0.26, ly, 2.9, 0.22, lab, 6, tcol)
    lx += 0.95 if i == 0 else 1.3

# ============================================================== (b) by protocol
BX = 7.30
fig.panel_label(BX, 0.06, 'b', '3-bit GPTQ IFEval, by calibration protocol')
PROT = [('C4 short documents', {'L': 'LgptqIF', 'Q': 'QgptqIF', 'M': 'MshortIF'}),
        ('Pile short documents', {'L': 'LpileShortIF', 'Q': 'QpileShortIF'}),
        ('C4 windows', {'L': 'LcwinIF', 'Q': 'QcwinIF', 'M': 'MgptqIF'}),
        ('Pile windows', {'L': 'LpileIF', 'Q': 'QpileIF'}),
        ('WikiText windows', {'L': 'LwikiIF', 'Q': 'QwikiIF'}),
        ('C4 in chat template', {'L': 'LcchatIF', 'Q': 'QcchatIF', 'M': 'McchatIF'}),
        ('UltraChat', {'L': 'LuchatIF', 'Q': 'QuchatIF', 'M': 'MuchatIF'})]
RTN = {'L': f('LrtnIF'), 'Q': f('QrtnIF'), 'M': f('MrtnIF')}
NAME = {'L': 'Llama-3.1-8B', 'Q': 'Qwen2.5-14B', 'M': 'Mistral-7B-v0.3'}
L2, R2 = BX + 2.35, 13.85
ax2 = Axis(0.0, 0.9, L2, R2)
rh2 = 0.36
T2 = 0.78
B2 = T2 + 0.08 + len(PROT) * rh2 + 0.04
fig.frame(L2, T2, R2, B2)
fig.xticks(ax2, B2, [0, 0.2, 0.4, 0.6, 0.8], fmt='%g')
fig.text(L2, B2 + 0.34, R2 - L2, 0.24, 'IFEval', 6.5, INK, align=PP_ALIGN.CENTER)
for v in [0.2, 0.4, 0.6, 0.8]:
    fig.line(ax2(v), T2, ax2(v), B2, GRID, 0.4)
for mk in 'LQM':
    fig.line(ax2(RTN[mk]), T2, ax2(RTN[mk]), B2, DASH, 0.6, dash=True)
    fig.text(ax2(RTN[mk]) - 0.3, T2 - 0.20, 0.6, 0.2, mk, 6, DASH, align=PP_ALIGN.CENTER)


def mark(mk, x, cy, collapsed):
    fill = RED_S if collapsed else WHITE
    edge = RED_S if collapsed else DARK
    if mk == 'L':
        fig.dot(x, cy, 0.19, fill, edge, 0.8)
    elif mk == 'Q':
        fig.square(x, cy, 0.17, fill, edge, 0.8)
    else:
        fig.diamond(x, cy, 0.22, fill, edge, 0.8)


for i, (pname, arms) in enumerate(PROT):
    cy = T2 + 0.08 + i * rh2 + rh2 / 2
    fig.text(BX, cy - 0.11, 2.28, 0.24, pname, 6, INK, align=PP_ALIGN.RIGHT)
    for mk, key in arms.items():
        v = f(key)
        mark(mk, ax2(v), cy, v < RTN[mk] - NOISE)
# legend of shapes, bottom right inside the frame (empty region: high IFEval, last rows)
lx, ly = L2 - 0.6, 0.34
for i, mk in enumerate('LQM'):
    mark(mk, lx + 0.09, ly + 0.11, False)
    fig.text(lx + 0.26, ly, 1.6, 0.22, {'L': 'Llama (L)', 'Q': 'Qwen (Q)', 'M': 'Mistral (M)'}[mk], 6, INK)
    lx += 1.5
fig.text(BX, B2 + 0.62, R2 - BX, 0.24, 'dashed: RTN of each model;   red: collapsed', 6, GRY_T, align=PP_ALIGN.CENTER)
fig.save()
