"""Masked GPTQ core: standard GPTQ (act-order, grouped, sym) with an optional
per-weight protection mask held out INSIDE the column loop.

Protected entries keep their (compensated) fp value and propagate ZERO error —
the compensation never "repairs around" them and never assumes they were
quantized. This is the same surgery TaCQ/SPQR perform (in-loop exemption),
as opposed to v1's post-hoc restoration (protect_eval.py), which conflicts
with compensation already distributed across columns.

NOTE on semantics (paper wording): a protected weight is NOT frozen at its
original fp16 value. Until its column is reached it keeps absorbing the
compensation updates of earlier columns like any other weight; when its
column is reached it is exempted from rounding and injects zero error. It is
therefore a free fp16 absorber, which is exactly the OBS-consistent behavior.

Faithful port of the IST-DASLab GPTQ fasterquant loop (Apache-2.0), with:
  - mask: bool [rows, cols], True = keep fp16 (exempt from quantization)
  - fake-quant output: returns the dequantized fp weight matrix (values lie
    exactly on the b-bit grid except protected entries) — saved as a normal
    HF checkpoint; packing is mechanical and orthogonal to the science.

W20+ additions (2026-09-04, all default-off so the frozen arms reproduce):
  - percdamp        Hessian dampening as a CONTINUOUS "compensation strength"
                    knob: damp -> inf turns GPTQ into RTN. (Frozen protocol
                    used 0.05; GPTQ's reference default is 0.01.)
  - sym=False       asymmetric grid (per-group zero point) — config-confound arm.
  - scale_excl_mask exclude protected entries from the group scale (SpQR/TaCQ
                    treat outliers as removed from the group). With the frozen
                    behaviour a protected large |W| stretches the grid of its
                    127 neighbours — potential artifact behind "under-budgeted
                    protection hurts" (14B tacq@1e5 = 0.216 < none 0.412).
  - stats           per-module mechanism log: compensation displacement,
                    per-column "sent" compensation mass, clipping fraction,
                    layer-wise objective tr(dH d^T) for GPTQ vs RTN under the
                    calibration Hessian AND (optionally) a second Hessian from
                    chat-formatted inputs (H_alt) — the distribution-shift test.
  - awq_quantize    AWQ-style activation-aware scaling + RTN (no compensation),
                    per-linear, Hessian-form MSE search over alpha. Tests the
                    prediction "scaling quantizers do not collapse".

Known deviation from gptqmodel's pipeline (documented in the paper):
module Hessians within a layer are captured in ONE pre-quantization forward
(gptqmodel re-captures o_proj/down_proj after quantizing earlier modules).
All v2 arms therefore compare against the v2 no-protection arm, not against
the W0 gptqmodel checkpoints.
"""
import math

import torch


# ------------------------------------------------------------ grid helpers

def group_scale(g: torch.Tensor, bits: int, sym: bool,
                gmask: torch.Tensor | None = None):
    """Per-row scale (and zero point) for one [rows, group] block.

    gmask: bool block of protected entries; when given they are EXCLUDED from
    the range statistics (scale_excl_mask semantics). Rows whose group is
    entirely protected get a harmless dummy scale (nothing in them is rounded).
    Returns (scale [rows,1], zero [rows,1]) with dq = scale * (q - zero).
    """
    maxq = 2 ** bits - 1
    if sym:
        a = g.abs()
        if gmask is not None:
            a = a.masked_fill(gmask, 0.0)
        xmax = a.max(dim=1).values.clamp(min=1e-8)
        scale = (2.0 * xmax / maxq).unsqueeze(1)
        zero = torch.full_like(scale, (maxq + 1) / 2)
        return scale, zero
    gg = g
    if gmask is not None:
        big = torch.finfo(g.dtype).max
        gmin = g.masked_fill(gmask, big).min(dim=1).values
        gmax = g.masked_fill(gmask, -big).max(dim=1).values
        allm = gmask.all(dim=1)
        gmin = torch.where(allm, torch.zeros_like(gmin), gmin)
        gmax = torch.where(allm, torch.ones_like(gmax) * 1e-8, gmax)
    else:
        gmin = gg.min(dim=1).values
        gmax = gg.max(dim=1).values
    gmin = torch.minimum(gmin, torch.zeros_like(gmin))
    gmax = torch.maximum(gmax, torch.zeros_like(gmax))
    scale = ((gmax - gmin) / maxq).clamp(min=1e-8).unsqueeze(1)
    zero = torch.round(-gmin / scale.squeeze(1)).unsqueeze(1)
    return scale, zero


def quant_col(w: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int):
    """Quantize one column w [rows] given per-row scale/zero [rows,1].
    Returns (dq [rows], clipped bool [rows])."""
    maxq = 2 ** bits - 1
    qraw = torch.round(w.unsqueeze(1) / scale) + zero
    clipped = ((qraw < 0) | (qraw > maxq)).flatten()
    q = torch.clamp(qraw, 0, maxq)
    return (scale * (q - zero)).flatten(), clipped


def rtn_grouped(W: torch.Tensor, bits: int, group_size: int, sym: bool = True,
                mask: torch.Tensor | None = None, scale_excl_mask: bool = False):
    """Grouped RTN of a full matrix (contiguous groups in the given column
    order). Masked entries keep their fp value. Returns (Q, n_clipped)."""
    Q = W.clone()
    n_clip = 0
    maxq = 2 ** bits - 1
    gs = group_size if group_size != -1 else W.shape[1]
    for c in range(0, W.shape[1], gs):
        g = W[:, c:c + gs]
        gm = mask[:, c:c + gs] if mask is not None else None
        scale, zero = group_scale(g, bits, sym, gm if scale_excl_mask else None)
        qraw = torch.round(g / scale) + zero                 # vectorised block
        clipped = (qraw < 0) | (qraw > maxq)
        dq = scale * (torch.clamp(qraw, 0, maxq) - zero)
        if gm is not None:
            dq = torch.where(gm, g, dq)
            clipped = clipped & ~gm
        Q[:, c:c + gs] = dq
        n_clip += int(clipped.sum())
    return Q, n_clip


def layer_objective(delta: torch.Tensor, H: torch.Tensor) -> float:
    """tr(delta H delta^T) = sum over calibration tokens of ||delta x||^2
    (up to the 2/n normalisation used in add_batch). delta [rows, cols]."""
    return float(((delta @ H) * delta).sum())


# ------------------------------------------------------------ main class

class MaskedGPTQ:
    def __init__(self, layer: torch.nn.Linear, name: str = ""):
        self.layer = layer
        self.name = name
        W = layer.weight.data
        self.rows, self.columns = W.shape
        self.dev = W.device
        self.H = torch.zeros((self.columns, self.columns), device=self.dev,
                             dtype=torch.float32)
        self.nsamples = 0
        self.H_alt = None      # optional second Hessian (e.g. chat-format inputs)
        self.n_alt = 0

    @torch.no_grad()
    def _accum(self, H, n_prev, inp):
        x = inp.reshape(-1, self.columns).t().float()  # [cols, N]
        n = x.shape[1]
        H *= n_prev / (n_prev + n)
        n_new = n_prev + n
        x = x * (2.0 / n_new) ** 0.5
        H += x @ x.t()
        return n_new

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor):
        """inp: [..., columns] input activations of this linear."""
        self.nsamples = self._accum(self.H, self.nsamples, inp)

    @torch.no_grad()
    def add_batch_alt(self, inp: torch.Tensor):
        if self.H_alt is None:
            self.H_alt = torch.zeros_like(self.H)
        self.n_alt = self._accum(self.H_alt, self.n_alt, inp)

    # -------------------------------------------------------------- GPTQ
    @torch.no_grad()
    def quantize(self, bits=3, group_size=128, sym=True, actorder=True,
                 percdamp=0.05, blocksize=128, mask: torch.Tensor | None = None,
                 scale_excl_mask: bool = False, stats: dict | None = None,
                 stats_eig: bool = False):
        """In-place GPTQ. If `stats` is a dict it is filled with the mechanism
        log for this module (see module docstring) and per-column sent-mass
        is stored under stats['send_mass'] (cpu tensor, original column order).
        """
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()
        M = mask.to(self.dev) if mask is not None else None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0
        Hraw = H.clone() if stats is not None else None   # undamped, original order
        W_orig = W.clone() if stats is not None else None  # original order

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            if M is not None:
                M = M[:, perm]

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        cond = None
        if stats is not None and stats_eig:
            try:
                ev = torch.linalg.eigvalsh(H)
                cond = float(ev.max() / ev.min().clamp(min=1e-12))
            except Exception:  # noqa: BLE001
                cond = float("nan")
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)

        Q = torch.zeros_like(W)
        scale = zero = None
        W_pre = W.clone() if stats is not None else None   # permuted, pre-loop
        n_clip = 0
        n_quant = 0
        comp_disp = 0.0                                     # sum ||w_at_quant - w_orig||^2
        send_mass = torch.zeros(self.columns, device=self.dev) if stats is not None else None
        hinv_row_sq = (Hinv ** 2).flip(1).cumsum(1).flip(1) if stats is not None else None
        # hinv_row_sq[i, j] = sum_{k>=j} Hinv[i,k]^2  -> sent mass uses k > i

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                col = i1 + i
                w = W1[:, i]
                d = Hinv1[i, i]
                if group_size != -1 and col % group_size == 0:
                    g = W[:, col:col + group_size]
                    # NB: W here already carries compensation from earlier
                    # blocks, but not from earlier columns of the same block
                    # (those live in W1) — identical to the reference loop.
                    gm = (M[:, col:col + group_size]
                          if (M is not None and scale_excl_mask) else None)
                    scale, zero = group_scale(g, bits, sym, gm)
                elif group_size == -1 and scale is None:
                    gm = M if (M is not None and scale_excl_mask) else None
                    scale, zero = group_scale(W, bits, sym, gm)
                q, clipped = quant_col(w, scale, zero, bits)
                if M is not None:
                    m = M[:, col]
                    q[m] = w[m]              # exempt: keep (compensated) fp, zero error
                    clipped = clipped & ~m
                if stats is not None:
                    n_clip += int(clipped.sum())
                    n_quant += int((~M[:, col]).sum()) if M is not None else self.rows
                    comp_disp += float(((w - W_pre[:, col]) ** 2).sum())
                Q1[:, i] = q
                err = (w - q) / d
                if stats is not None and col + 1 < self.columns:
                    send_mass[col] = float((err ** 2).sum()) * float(hinv_row_sq[col, col + 1])
                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            if send_mass is not None:
                send_mass = send_mass[invperm]

        if stats is not None:
            self._fill_stats(stats, W_orig, Q, Hraw, bits, group_size, sym,
                             mask, scale_excl_mask, n_clip, n_quant,
                             comp_disp, send_mass, cond, dead)

        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        del H, Hinv
        return Q

    # -------------------------------------------------------------- AWQ arm
    @torch.no_grad()
    def awq_quantize(self, bits=3, group_size=128, sym=True, grid=20,
                     mask: torch.Tensor | None = None, stats: dict | None = None):
        """AWQ-style per-input-channel scaling + grouped RTN, NO compensation.

        s_x = sqrt(diag H)  (RMS activation per input channel on calibration)
        for alpha in linspace(0,1,grid):
            s = s_x^alpha / sqrt(max(s)*min(s));  Q = RTN(W * s) / s
            loss = tr((W-Q) H (W-Q)^T)      (exact calibration MSE, Hessian form)
        Keep the best alpha. The stored fp matrix Q'/s computes the same
        function as a deployed AWQ model (scale folded into the previous op),
        so the fake-quant checkpoint is function-equivalent. AWQ's optional
        clipping search is NOT included (documented protocol deviation).
        """
        W = self.layer.weight.data.clone().float()
        H = self.H.clone()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1.0
        W[:, dead] = 0
        M = mask.to(self.dev) if mask is not None else None
        s_x = torch.sqrt(torch.diag(H).clamp(min=1e-10))
        best = (float("inf"), None, None, 0)
        for alpha in torch.linspace(0, 1, grid).tolist():
            s = s_x.clamp(min=1e-5) ** alpha
            s = s / torch.sqrt(s.max() * s.min())
            Ws = W * s.unsqueeze(0)
            Qs, n_clip = rtn_grouped(Ws, bits, group_size, sym, M, scale_excl_mask=False)
            Q = Qs / s.unsqueeze(0)
            if M is not None:
                Q[M] = W[M]
            loss = layer_objective(W - Q, H)
            if loss < best[0]:
                best = (loss, Q, alpha, n_clip)
        loss, Q, alpha, n_clip = best
        if stats is not None:
            Q_rtn, _ = rtn_grouped(W, bits, group_size, sym, M, False)
            stats.update({"quantizer": "awq", "awq_alpha": alpha,
                          "obj_awq": loss, "obj_rtn": layer_objective(W - Q_rtn, H),
                          "clip_frac": n_clip / max(W.numel(), 1)})
            if self.H_alt is not None:
                stats["obj_awq_alt"] = layer_objective(W - Q, self.H_alt)
                stats["obj_rtn_alt"] = layer_objective(W - Q_rtn, self.H_alt)
        self.layer.weight.data = Q.to(self.layer.weight.dtype)
        return Q

    # -------------------------------------------------------------- stats
    @torch.no_grad()
    def _fill_stats(self, stats, W_orig, Q, Hraw, bits, group_size, sym, mask,
                    scale_excl_mask, n_clip, n_quant, comp_disp, send_mass,
                    cond, dead):
        M = mask.to(self.dev) if mask is not None else None
        Q_rtn, n_clip_rtn = rtn_grouped(W_orig, bits, group_size, sym, M, scale_excl_mask)
        d_g = W_orig - Q
        d_r = W_orig - Q_rtn
        stats.update({
            "quantizer": "gptq",
            "rows": self.rows, "cols": self.columns,
            "n_dead_cols": int(dead.sum()),
            "clip_frac": n_clip / max(n_quant, 1),
            "clip_frac_rtn": n_clip_rtn / max(W_orig.numel(), 1),
            "comp_disp": comp_disp,                       # sum ||w_quant_time - w_orig||^2
            "comp_disp_rel": comp_disp / max(float((W_orig ** 2).sum()), 1e-12),
            "wdisp_gptq": float((d_g ** 2).sum()),        # ||W - Q_gptq||_F^2
            "wdisp_rtn": float((d_r ** 2).sum()),         # ||W - Q_rtn||_F^2
            "obj_gptq": layer_objective(d_g, Hraw),       # calibration-Hessian objective
            "obj_rtn": layer_objective(d_r, Hraw),
            "send_total": float(send_mass.sum()),
            "send_top1pct": _top_frac(send_mass, 0.01),
            "send_top16_cols": ",".join(map(str, torch.topk(send_mass, min(16, self.columns)).indices.tolist())),
            "hdiag_max_over_mean": float(torch.diag(Hraw).max() / torch.diag(Hraw).mean()),
            "cond_damped": cond if cond is not None else "",
        })
        if self.H_alt is not None:
            Ha = self.H_alt.clone()
            Ha[dead, dead] = 1.0
            stats["obj_gptq_alt"] = layer_objective(d_g, Ha)
            stats["obj_rtn_alt"] = layer_objective(d_r, Ha)
            stats["hdiag_alt_corr"] = float(torch.corrcoef(
                torch.stack([torch.diag(Hraw), torch.diag(Ha)]))[0, 1])
        stats["send_mass"] = send_mass.cpu()

    def free(self):
        self.H = None
        self.H_alt = None
        torch.cuda.empty_cache()


def _top_frac(v: torch.Tensor, frac: float) -> float:
    k = max(1, int(math.ceil(frac * v.numel())))
    return float(torch.topk(v, k).values.sum() / v.sum().clamp(min=1e-20))
