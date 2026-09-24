"""Figure 1 (fig1_mechanism), 14 x 5.0 cm, two panels.
 (a) the sink-forming layer as a pre-norm block after An et al. (2025) Fig. 2, the one matrix in red.
 (b) one chat prompt at that matrix, mirrored bars on a shared token axis:
     up   g_t      the sensitivity of the loss (what the loss depends on)
     down ‖x_t‖²   the input energy (what the objective weights by)
     with the consequence of the compensation written next to each side (GPTQ error over RTN error,
     runs/stats/f-l-3b/stats.csv, same matrix).
Data: runs/sides/f_l_fig1_tokens.csv (W86).
  python ICLR/figure/code/make_fig1.py     (from the repo root; FIG_REPLACE=1 overwrites the last slide)
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pptfig import *  # noqa: E402,F401,F403
from pptfig import Fig, PP_ALIGN, MSO_ANCHOR, MSO_SHAPE  # noqa: E402

OUT = 'ICLR/figure/fig1_mechanism.pptx'
W, H = 14.0, 5.0
VERSION = ('v13 2026-09-23: (a) boxes 0.27 cm high and every gap >= 0.2 cm so each arrow has a visible shaft; (b) contrast titles and verbs restored. v12: 475 moved to the input arrow (it is the BOS input norm of down_proj, 00_numbers LbosNorm), output arrow says writes the sink; height 5.4 -> 5.0; (a) widened (3.6 -> 3.9 cm), blocks moved so no label touches the '
           'dashed groups, residual line clear of the attn group, gate/up fed by real arrows from a split line, '
           'one arrow into each adder; (b) narrower, fewer words (side titles, notes, header label beside the bracket)')
NL = chr(92) + 'n'
fig = Fig(OUT, W, H, VERSION)
PANE = RGBColor(0xF6, 0xF6, 0xF6)
PALE = RGBColor(0xDD, 0xE8, 0xF5)
PTXT = RGBColor(0x1F, 0x3A, 0x5C)
AX, AW = 0.10, 4.20
BX, BW = 4.55, 9.35
for (px, pw) in ((AX, AW), (BX, BW)):
    fig.box(px - 0.06, 0.02, pw + 0.12, H - 0.04, '', 6, PANE, None, radius=0.05)

# =============================================================================== (a) the layer
fig.panel_label(AX, 0.06, 'a', 'The sink-forming layer')
bw, bh = 1.30, 0.27
bx = 1.92
cx = bx + bw / 2
rx = AX + AW - 0.10                 # residual line
gL, gR = bx - 0.24, bx + bw + 0.48  # dashed group boxes
sp = 0.75


def spine(y1, y2, color=DASH, w=sp, head=True):
    return fig.line(cx, y1, cx, y2, color, w, head=head)


def plus(y):
    fig.dot(cx, y, 0.24, WHITE, DASH, 0.75)
    fig.ctext(cx, y, 0.3, 0.3, '+', 7, DASH)


def group(y_top, y_bot, label):
    fig.box(gL, y_top, gR - gL, y_bot - y_top, '', 6, None, DASH, shape=MSO_SHAPE.RECTANGLE, dash=True, edge_w=0.5)
    fig.vtext(bx + bw + 0.10, y_top, 0.34, y_bot - y_top, label, 6.5, DASH, bold=True)


y_out, y_plus2, y_attn, y_ln2, y_sep = 0.30, 0.60, 1.24, 1.955, 2.22
y_plus1, y_down, y_act, y_gate, y_split, y_ln1, y_in = 2.46, 2.925, 3.415, 3.905, 4.24, 4.535, 4.92

# input -> RMSNorm -> split -> gate | up
spine(y_in, y_ln1 + bh / 2 + 0.02)
fig.box(bx, y_ln1 - bh / 2, bw, bh, 'RMSNorm', 6.5, PALE, None, PTXT)
spine(y_ln1 - bh / 2, y_split, head=False)
gate_x, up_x, gw = bx - 0.16, bx + bw - 0.62, 0.78
fig.line(gate_x + gw / 2, y_split, up_x + gw / 2, y_split, DASH, sp)
for x0, lab in ((gate_x, 'gate'), (up_x, 'up')):
    fig.box(x0, y_gate - bh / 2, gw, bh, lab, 6.5, PALE, None, PTXT)
    fig.line(x0 + gw / 2, y_split, x0 + gw / 2, y_gate + bh / 2 + 0.02, DASH, sp, head=True)
    fig.line(x0 + gw / 2, y_gate - bh / 2, x0 + gw / 2, y_act + bh / 2 + 0.02, DASH, sp, head=True)
fig.box(bx, y_act - bh / 2, bw, bh, 'SiLU  ×', 6.5, PALE, None, PTXT)
spine(y_act - bh / 2, y_down + bh / 2 + 0.02, RED_S, 1.0)
fig.box(bx, y_down - bh / 2, bw, bh, 'down_proj', 6.5, RED_S, None, WHITE, bold=True)
spine(y_down - bh / 2, y_plus1 + 0.14, RED_S, 1.0)
plus(y_plus1)
group(y_down - bh / 2 - 0.08, y_gate + bh / 2 + 0.11, 'MLP')
# residual line
fig.line(cx, y_in - 0.05, rx, y_in - 0.05, DASH, sp)
fig.line(rx, y_in - 0.05, rx, y_plus2, DASH, sp)
fig.line(rx, y_plus1, cx + 0.12, y_plus1, DASH, sp, head=True)
fig.line(rx, y_plus2, cx + 0.12, y_plus2, DASH, sp, head=True)
fig.dot(rx, y_plus1, 0.07, DASH)
# next layer: RMSNorm -> attention -> adder -> out
spine(y_plus1 - 0.12, y_ln2 + bh / 2 + 0.02)
fig.line(AX + 0.34, y_sep, AX + AW, y_sep, DASH, 0.5, dot=True)
fig.box(bx, y_ln2 - bh / 2, bw, bh, 'RMSNorm', 6.5, PALE, None, PTXT)
spine(y_ln2 - bh / 2, y_attn + 0.22 + 0.02)
fig.box(bx, y_attn - 0.22, bw, 0.44, 'attention\nQ K V · O', 6.5, PALE, None, PTXT)
group(y_attn - 0.22 - 0.14, y_attn + 0.22 + 0.14, 'attn')
spine(y_attn - 0.22, y_plus2 + 0.14)
plus(y_plus2)
spine(y_plus2 - 0.12, y_out)
fig.vtext(AX, y_sep + 0.06, 0.26, y_in - y_sep - 0.06, 'current layer', 6, DASH)
fig.vtext(AX, y_out, 0.26, y_sep - y_out - 0.06, 'next layer', 6, DASH)
# the two red annotations, right-aligned, ending before the dashed group
lx0, lx1 = AX + 0.34, gL - 0.08
ya = (y_act + y_down) / 2
fig.text(lx0, ya - 0.26, lx1 - lx0, 0.72, 'xₜ → H\n‖xₜ‖ = 475\nat BOS', 6.5, RED_T, bold=True, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
fig.line(lx1 + 0.02, ya, cx - 0.06, ya, RED_S, 0.75, head=True)
yo = y_plus1 + 0.17
fig.text(lx0 - 0.04, yo - 0.12, lx1 - lx0 + 0.04, 0.24, 'writes sink', 6, RED_T, bold=True, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
fig.line(lx1 + 0.02, yo, cx - 0.06, yo, RED_S, 0.75, head=True)

# =============================================================================== (b) the two sides
fig.panel_label(BX, 0.06, 'b', 'Loss sensitivity against objective weight')
SHORT = {'<|begin_of_text|>': 'BOS', '<|start_header_id|>': 'start\nheader', '<|end_header_id|>': 'end\nheader',
         '<|eot_id|>': 'eot', 'ĊĊ': NL + NL, 'Ċ': NL}
rows = list(csv.DictReader(open('runs/sides/f_l_fig1_tokens.csv', encoding='utf-8')))
xs_all = [float(r['xnorm2']) for r in rows]
gs_all = [float(r['g']) for r in rows]
x_share, g_share = xs_all[0] / sum(xs_all), gs_all[0] / sum(gs_all)
hdr_share = sum(gs_all[1:4]) / sum(gs_all)
N = 10
TOK = []
in_hdr = False
for r in rows[:N]:
    t = r['token']
    if t == '<|start_header_id|>':
        in_hdr = True
    cls = 'bos' if int(r['pos']) == 0 else ('fmt' if (t.startswith('<|') or t in ('ĊĊ', 'Ċ') or in_hdr) else 'usr')
    if t == '<|end_header_id|>':
        in_hdr = False
    TOK.append((SHORT.get(t, t.replace('Ġ', '')), cls, float(r['xnorm2']) / sum(xs_all), float(r['g']) / sum(gs_all)))
COL = {'bos': (RED_S, RED_T), 'fmt': (ORA_S, ORA_T), 'usr': (GRY_S, GRY_T)}
st = next(r for r in csv.DictReader(open('runs/stats/f-l-3b/stats.csv')) if r['layer'] == '1' and r['proj'] == 'down_proj')
eg = [float(v) for v in st['tplpos_err_gptq'].split(',')]
er = [float(v) for v in st['tplpos_err_rtn'].split(',')]
r_bos = eg[0] / er[0]
r_hdr = [eg[p] / er[p] for p in (2, 3, 4)]


def pct(v):
    if v >= 0.999:
        return f'{100 * v:.2f}%'
    if v < 0.01:
        return f'{100 * v:.1g}%'
    return f'{100 * v:.0f}%'


L, R = BX + 0.35, BX + BW - 0.25
n, gap = len(TOK), 0.16
cw = (R - L - gap * (n - 1)) / n
xs = [L + i * (cw + gap) for i in range(n)]
y_mid_top, y_mid_bot = 2.42, 2.92
g_top, x_bot = 0.92, 4.22
gmax, xmax = max(t[3] for t in TOK), max(t[2] for t in TOK)
fig.text(L, 0.40, 3.6, 0.26, 'gₜ :  what the loss depends on', 6.5, INK, bold=True)
fig.text(L, 4.30, 4.8, 0.26, '‖xₜ‖² :  what the objective weights', 6.5, INK, bold=True)
fig.line(L - 0.1, y_mid_top, R + 0.1, y_mid_top, RULE, 0.5)
fig.line(L - 0.1, y_mid_bot, R + 0.1, y_mid_bot, RULE, 0.5)
for i, (lab, cls, a, g) in enumerate(TOK):
    x = xs[i]
    sat, tcol = COL[cls]
    h = max(g / gmax * (y_mid_top - g_top), 0.03)
    fig.rect(x, y_mid_top - h, cw, h, sat)
    h2 = max(a / xmax * (x_bot - y_mid_bot), 0.03)
    fig.rect(x, y_mid_bot, cw, h2, RED_S if cls == 'bos' else RED_F)
    fig.ctext(x + cw / 2, (y_mid_top + y_mid_bot) / 2, cw + gap, 0.5, lab, 6.5, tcol, bold=(cls == 'bos'))
# the numbers: BOS on each side, one bracket for the header tokens
fig.ctext(xs[0] + cw / 2, y_mid_top - 0.20, 1.2, 0.24, pct(g_share), 6.5, RED_T, bold=True)
xb0, xb1 = xs[1], xs[3] + cw
yb = g_top - 0.12
fig.line(xb0, yb, xb1, yb, ORA_T, 0.6)
fig.line(xb0, yb, xb0, yb + 0.08, ORA_T, 0.6)
fig.line(xb1, yb, xb1, yb + 0.08, ORA_T, 0.6)
fig.text(xb1 + 0.12, yb - 0.12, 3.6, 0.24, f'{pct(hdr_share)} on the header tokens', 6.5, ORA_T, bold=True)
fig.text(xs[1] + 0.05, y_mid_bot + 0.12, 3.2, 0.24, f'BOS {pct(x_share)}', 6.5, RED_T, bold=True)
fig.text(xs[1] + 0.05, y_mid_bot + 0.36, 3.2, 0.24, 'every other token below 0.01%', 6, GRY_T)
# the consequence at this matrix (GPTQ error over RTN error), one line per side, arrow to the bar it concerns
y1, y2 = 1.62, 3.72
fig.text(xs[4], y1 - 0.38, R - xs[4], 0.26, f'GPTQ writes its error here: {min(r_hdr):.0f} to {max(r_hdr):.0f} × RTN', 6.5, ORA_T, bold=True, align=PP_ALIGN.RIGHT)
fig.line(R - 0.15, y1, xs[3] + cw + 0.12, y1, ORA_T, 0.6, head=True)
fig.text(xs[4], y2 - 0.38, R - xs[4], 0.26, f'GPTQ protects BOS: error {r_bos:.1f} × RTN', 6.5, RED_T, bold=True, align=PP_ALIGN.RIGHT)
fig.line(R - 0.15, y2, xs[0] + cw + 0.14, y2, RED_T, 0.6, head=True)
# legend and the repair
ly = 4.62
for x0, (name, cls) in zip((L, L + 1.45, L + 3.20), [('sink token', 'bos'), ('format tokens', 'fmt'), ('user text', 'usr')]):
    fig.rect(x0, ly + 0.05, 0.26, 0.16, COL[cls][0])
    fig.text(x0 + 0.34, ly, 1.4, 0.24, name, 6, COL[cls][1])
fig.text(R - 4.0, ly, 4.0, 0.24, 'repair:  weight tokens by  gₜ / ‖xₜ‖²', 6.5, GRN_T, bold=True, align=PP_ALIGN.RIGHT)
fig.save()
print('BOS ratio %.2f, header ratios %s' % (r_bos, [round(v, 1) for v in r_hdr]))
