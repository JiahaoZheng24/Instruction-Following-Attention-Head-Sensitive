"""Figure 4 (fig4_lesion): anatomy of the lesion, Llama-3.1-8B-Instruct, 14 x 4.8 cm.
 (a) relative hidden-state error of the collapsed short-document GPTQ3 model against fp16,
     by layer and prompt position (runs/div_f_l_none.pos.csv): one hot cell, then a wave.
 (b) relative error of the block output after the sink-forming layer by token class
     (runs/sink/f_l_artifact.csv).
 (c) that error by quintile of the lesion row's readout, with the share of template tokens
     each quintile captures (runs/sink/f_l_readout.csv).
  python ICLR/figure/code/make_fig4.py    (from the repo root; FIG_REPLACE=1 overwrites the last slide)
"""
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptfig import *  # noqa: E402,F401,F403
from pptfig import Fig, Axis, heat, PP_ALIGN, MSO_ANCHOR  # noqa: E402

OUT = 'ICLR/figure/fig4_lesion.pptx'
W, H = 14.0, 4.6
VERSION = 'v6 2026-09-22: as v5, one-line callout in (a), notes under (b)/(c) removed, (c) x-axis title'
NL = chr(92) + 'n'
fig = Fig(OUT, W, H, VERSION)

# ============================================================== (a) heatmap
AX, AW = 0.10, 5.85
fig.panel_label(AX, 0.06, 'a', 'Error by layer and prompt position')
rows = list(csv.DictReader(open('runs/div_f_l_none.pos.csv')))
layers = [int(r['layer']) for r in rows]
POS = [('pos0', 'BOS'), ('pos1', 'BOS'), ('pos2', 'start_header'), ('pos3', 'system'), ('pos4', 'end_header'),
       ('pos5', NL + NL), ('pos6', 'Cutting'), ('pos7', 'Knowledge'), ('rest', 'rest')]
gx, gy = AX + 1.55, 0.50
gw = AW - 1.70
cw, ch = gw / len(layers), 0.30
VMAX = 4.0
for j, (key, lab) in enumerate(POS):
    y = gy + j * ch
    fig.text(AX + 0.18, y + 0.04, 1.30, 0.22, lab, 6, INK, align=PP_ALIGN.RIGHT)
    for i, r in enumerate(rows):
        fig.rect(gx + i * cw, y, cw + 0.006, ch + 0.006, heat(float(r[f'rel_err_{key}']), VMAX))
fig.rect(gx, gy, gw, len(POS) * ch, None, RULE, 0.5)
io, jo = layers.index(2), 4
vo = float(rows[io][f'rel_err_{POS[jo][0]}'])
fig.rect(gx + io * cw, gy + jo * ch, cw, ch, None, INK, 1.0)
fig.line(gx + (io + 1) * cw + 0.02, gy + jo * ch + ch / 2, gx + 5.2 * cw, gy + 1.2 * ch + 0.08, INK, 0.5)
fig.text(gx + 5.3 * cw, gy + 0.75 * ch, 4.0, 0.24, f'first at layer 2, end_header ({vo:.1f})', 6, INK)
for i in range(0, len(layers), 8):
    fig.text(gx + i * cw + cw / 2 - 0.3, gy + len(POS) * ch + 0.05, 0.6, 0.2, str(layers[i]), 6, GRY_T, align=PP_ALIGN.CENTER)
    fig.line(gx + i * cw + cw / 2, gy + len(POS) * ch, gx + i * cw + cw / 2, gy + len(POS) * ch + 0.05, RULE, 0.5)
fig.text(gx, gy + len(POS) * ch + 0.28, gw, 0.22, 'layer', 6.5, INK, align=PP_ALIGN.CENTER)
fig.vtext(AX - 0.02, gy, 0.16, len(POS) * ch, 'prompt position', 6.5, INK)
# colour bar (right of the grid, vertical)
cbx, cbh = gx + gw + 0.10, len(POS) * ch
for k in range(40):
    fig.rect(cbx, gy + cbh - (k + 1) * cbh / 40, 0.16, cbh / 40 + 0.006, heat(VMAX * k / 39, VMAX))
fig.text(cbx + 0.20, gy - 0.05, 0.6, 0.2, f'≥{VMAX:.0f}', 6, GRY_T)
fig.text(cbx + 0.20, gy + cbh - 0.16, 0.6, 0.2, '0', 6, GRY_T)
fig.vtext(cbx + 0.30, gy + 0.25, 0.16, cbh - 0.5, 'rel. error', 6, GRY_T)

# ============================================================== (b) relative error by token class
BX, BW = 6.60, 3.55
fig.panel_label(BX, 0.06, 'b', 'Error by token class')
art = {}
for r in csv.DictReader(open('runs/sink/f_l_artifact.csv')):
    art[r['class'].split('(')[0]] = float(r['rel_err'])
CLS = [('bos', 'BOS', RED_S, RED_T), ('prompt_template', 'template', ORA_S, ORA_T),
       ('prompt_newline', 'newline', ORA_S, ORA_T), ('prompt_ordinary', 'user text', GRY_S, GRY_T),
       ('resp_ordinary', 'response', BLU_S, BLU_T)]
L, R, T, B = BX + 1.05, BX + BW - 0.45, 0.60, 3.20
ax = Axis(0.0, 1.1, L, R)
fig.frame(L, T, R, B)
fig.xticks(ax, B, [0, 0.5, 1.0], fmt='%.1f')
for v in [0.5, 1.0]:
    fig.line(ax(v), T, ax(v), B, GRID, 0.4)
fig.text(L, B + 0.34, R - L, 0.24, 'relative error', 6.5, INK, align=PP_ALIGN.CENTER)
slot = (B - T) / len(CLS)
for i, (key, lab, col, tcol) in enumerate(CLS):
    v = art[key]
    y = T + slot * (i + 0.5)
    fig.rect(L, y - slot * 0.32, ax(v) - L, slot * 0.64, col)
    fig.text(ax(v) + 0.06, y - 0.11, 0.5, 0.22, f'{v:.2f}', 6, tcol)
    fig.text(BX, y - 0.11, 1.0, 0.22, lab, 6, tcol, align=PP_ALIGN.RIGHT)

# ============================================================== (c) row-readout quintiles
CX, CW = 10.35, 3.55
fig.panel_label(CX, 0.06, 'c', 'Error by readout quintile')
q = {}
for ln in open('runs/sink/f_l_readout.csv', encoding='utf-8'):
    m = re.match(r'"?row_q(\d)\(n=\d+,tpl_frac=([\d.]+),tpl_recall=([\d.]+)\)"?,[^,]*,([\d.]+),', ln)
    if m:
        q[int(m.group(1))] = (float(m.group(4)), float(m.group(3)))
L, R, T, B = CX + 0.62, CX + CW - 0.55, 0.55, 3.55
ay = Axis(0.3, 3.0, B, T, log=True)
ar = Axis(0.0, 0.6, B, T)
fig.frame(L, T, R, B)
fig.line(R, T, R, B, ORA_S, 0.6)
fig.yticks(ay, L, [0.3, 0.5, 1, 2, 3], fmt='%g', grid_to=R)
for v in [0, 0.2, 0.4, 0.6]:
    y = ar(v)
    fig.line(R, y, R + 0.07, y, ORA_S, 0.5)
    fig.text(R + 0.10, y - 0.11, 0.45, 0.22, f'{v:.1f}', 6, ORA_T)
slot = (R - L) / 5
for k in range(1, 6):
    err, rec = q[k]
    x = L + slot * (k - 0.5)
    fig.rect(x - slot * 0.32, ay(err), slot * 0.64, B - ay(err), GRY_S)
    fig.text(x - slot / 2, B + 0.06, slot, 0.22, f'Q{k}', 6, GRY_T, align=PP_ALIGN.CENTER)
prev = None
for k in range(1, 6):
    err, rec = q[k]
    x = L + slot * (k - 0.5)
    if prev:
        fig.line(prev[0], prev[1], x, ar(rec), ORA_S, 1.1)
    prev = (x, ar(rec))
for k in range(1, 6):
    fig.dot(L + slot * (k - 0.5), ar(q[k][1]), 0.18, ORA_S, WHITE, 0.5)
fig.vtext(CX - 0.02, T, 0.3, B - T, 'relative error (bars)', 6.5, INK)
fig.vtext(CX + CW - 0.22, T, 0.3, B - T, 'share of template tokens', 6.5, ORA_T)
fig.text(L, B + 0.34, R - L, 0.24, 'readout quintile', 6.5, INK, align=PP_ALIGN.CENTER)
fig.save()
