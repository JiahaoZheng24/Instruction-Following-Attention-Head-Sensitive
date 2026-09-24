"""W76: direct tests of Theorem 1 on the real calibration Hessian and the real
GPTQ output (E2 and E3 of paper/SELF_REVIEW_2026-09-17.md).

At the sink-forming down_proj (auto-detected as the down_proj with the largest
BOS share of tr H, or --layer) and at control matrices, from the SAME
calibration protocol the quantiser uses:

  H      = (2/N) sum_t x_t x_t^T                    (gptq_core scaling)
  B      = (2/N) sum_samples x_0 x_0^T               (BOS block; rank <= n_calib)
  x_0    = top eigencomponent of B  (rank-one share of B reported)
  lambda = percdamp * mean diag H
  M      = H + lambda I - x_0 x_0^T   (PSD: H >= B >= x_0 x_0^T)
  v      = M^{-1} x_0,   kappa = x_0^T v,   kappa_iso = ||x_0||^2 / lambda

  E2  kappa (Theorem 1 premise "kappa >> 1"; Corollary 1(iii) across protocols)
  E3  The SENDING PATTERN of each column, the deterministic object GPTQ uses:
      with U = chol(H_lambda^{-1}) in act-order, column j compensates its
      rounding error on the later columns along p_j = U[j, j+1:] / U[j, j]
      (gptq_core: W1[:, i:] -= err * Hinv1[i, i:]).  Theorem 1 says that on
      the BOS columns J (the columns holding 90% of ||x_0||^2) p_j is the sink
      pattern: proportional to x_0 (isotropic M) or v (general M) on the
      remaining columns, with its mass on the other BOS columns.  Reported,
      averaged over j in J and over the next 64 act-order columns (ordinary):
      |cos(p_j, x_0)|, |cos(p_j, v)|, |cos(p_j, random)|, share of ||p_j||^2
      on the other columns of J; the same under H with the BOS block removed (H_nob).
      Prediction: on the sink matrix cos with v (or x_0) >= 0.7 and BOS-column
      mass >= 0.5 for j in J; <= 0.2 for ordinary columns, under H_nob and at
      the control matrices.
      Also reported, as references: D = accumulated compensation under H
      (gptq_core stats['disp_matrix']), D_nob under H_nob, S = D - D_nob, and
      C = E_gptq - E_rtn, each with its share along x_0 / v / random, top-row
      share and BOS-column share.  (These matrix differences mix in the
      divergence of two rounding sequences and are not expected to be
      rank-one; they document that the lesion's shape must be read from the
      sending pattern and from the row / clipping statistics.)

--save-whiten FILE stores {x0, v, lambda, kappa, layer} for the per-token
readout test (E1, src/inject_sink.py artifact --whiten FILE).

  python src/theory_tests.py --model meta-llama/Llama-3.1-8B-Instruct --calib c4 \
      --tag f_l_c4 --save-whiten runs/theory/f_l_c4_whiten.pt
Writes runs/theory/<tag>.json and appends rows to runs/theory/INDEX_theory.csv.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model  # noqa: E402
from gptq_core import MaskedGPTQ, rtn_grouped  # noqa: E402
from quantize_gptq import load_calib  # noqa: E402

COLS = ["tag", "model", "calib", "n_calib", "seqlen", "role", "layer", "d_in", "d_out", "n_tokens",
        "A_bos", "rank1_share_B", "norm_x0_sq", "lambda", "kappa", "kappa_iso", "sm_check",
        "n_bos_cols90_J", "actorder_topJ_overlap",
        "pat_cos_x0_bos", "pat_cos_v_bos", "pat_cos_rand_bos", "pat_bosmass_bos", "pat_norm_bos",
        "pat_cos_x0_ord", "pat_cos_v_ord", "pat_bosmass_ord", "pat_norm_ord",
        "nob_pat_cos_x0_bos", "nob_pat_cos_v_bos", "nob_pat_bosmass_bos", "nob_pat_norm_bos",
        "share_S_x0", "share_S_v", "share_S_rand", "share_S_top1", "cos_top1_x0", "cos_top1_v",
        "n_bos_cols90", "colshare_S_bos", "colshare_Dnob_bos", "normS_over_normDnob", "toprow_share_S", "toprow_S",
        "share_D_x0", "share_D_v", "share_D_rand", "toprow_share_D", "toprow_D",
        "share_C_x0", "share_C_v", "share_C_rand", "share_Eg_x0", "share_Eg_v", "share_Er_x0", "share_Er_v",
        "normD_over_normEr", "normC_over_normEr", "toprow_share_C", "toprow_C", "toprow_share_Eg", "toprow_Eg",
        "toprow_share_Er", "maxabs_W", "maxabs_Qg", "maxabs_Qnob", "n_clip_g", "n_clip_nob", "n_clip_r", "seconds"]


def _down(model, li):
    return model.model.layers[li].mlp.down_proj


@torch.no_grad()
def detect_sink_layer(model, tok, texts, seqlen):
    """BOS share of sum_t ||x_t||^2 at every down_proj input; returns per-layer A_bos."""
    n_layers = len(model.model.layers)
    e_bos = torch.zeros(n_layers, dtype=torch.float64)
    e_all = torch.zeros(n_layers, dtype=torch.float64)

    def hook(li):
        def f(_m, a):
            x = a[0].reshape(-1, a[0].shape[-1]).float()
            e_bos[li] += float((x[0] ** 2).sum())
            e_all[li] += float((x ** 2).sum())
        return f

    hs = [_down(model, li).register_forward_pre_hook(hook(li)) for li in range(n_layers)]
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seqlen).to(model.device)
        model(**ids, use_cache=False)
    for h in hs:
        h.remove()
    return (e_bos / e_all.clamp(min=1e-30)).tolist()


@torch.no_grad()
def accumulate(model, tok, texts, seqlen, layers):
    """H via MaskedGPTQ.add_batch (exact quantiser scaling) and the BOS inputs X0 [n, d]."""
    g = {li: MaskedGPTQ(_down(model, li), name=f"layers.{li}.down_proj") for li in layers}
    g_nob = {li: MaskedGPTQ(_down(model, li), name=f"layers.{li}.down_proj.nob") for li in layers}
    for li in layers:
        g_nob[li].drop_pos = 1          # the same Hessian without position 0 (accumulated, not subtracted:
                                        # W76 computed H - B in float32 and lost positive-definiteness on Llama)
    X0 = {li: [] for li in layers}
    ntok = {li: 0 for li in layers}

    def hook(li):
        def f(_m, a):
            x = a[0]
            g[li].add_batch(x)
            g_nob[li].add_batch(x)
            xf = x.reshape(-1, x.shape[-1]).float()
            X0[li].append(xf[0].clone())
            ntok[li] += xf.shape[0]
        return f

    hs = [_down(model, li).register_forward_pre_hook(hook(li)) for li in layers]
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seqlen).to(model.device)
        model(**ids, use_cache=False)
    for h in hs:
        h.remove()
    return g, g_nob, {li: torch.stack(X0[li]) for li in layers}, ntok


@torch.no_grad()
def row_share(E, u):
    """share of ||E||_F^2 carried by the rows' projection onto unit vector u."""
    u = u / u.norm().clamp(min=1e-30)
    return float(((E @ u) ** 2).sum() / (E ** 2).sum().clamp(min=1e-30))


@torch.no_grad()
def sending_patterns(H, percdamp, J, x0, v, rnd, n_ord=64):
    """Rows of the upper Cholesky factor of the damped inverse Hessian in act-order,
    i.e. the pattern along which GPTQ compensates column j on the later columns.
    Returns mean statistics over the columns in J (BOS support) and over the next
    n_ord act-order columns, plus the overlap of the first |J| act-order columns with J."""
    d = H.shape[0]
    perm = torch.argsort(torch.diag(H), descending=True)
    Hp = H[perm][:, perm].clone()
    lam = percdamp * torch.mean(torch.diag(Hp))
    Hp.diagonal().add_(lam)
    Lc = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(Lc)
    U = torch.linalg.cholesky(Hinv, upper=True)
    del Lc, Hinv, Hp
    pos_of = torch.empty(d, dtype=torch.long, device=H.device)
    pos_of[perm] = torch.arange(d, device=H.device)
    Jset = torch.zeros(d, dtype=torch.bool, device=H.device)
    Jset[J] = True
    Jp = Jset[perm]                                     # BOS membership in permuted order
    x0p, vp, rp = x0[perm], v[perm], rnd[perm]
    kJ = int(Jset.sum())
    overlap = float(Jp[:kJ].float().mean())

    def stats_for(positions):
        cx, cv, cr, bm, nm = [], [], [], [], []
        for i in positions:
            i = int(i)
            if i + 1 >= d:
                continue
            pj = U[i, i + 1:] / U[i, i]
            n = pj.norm().clamp(min=1e-30)
            cx.append(float((pj @ x0p[i + 1:]).abs() / n / x0p[i + 1:].norm().clamp(min=1e-30)))
            cv.append(float((pj @ vp[i + 1:]).abs() / n / vp[i + 1:].norm().clamp(min=1e-30)))
            cr.append(float((pj @ rp[i + 1:]).abs() / n / rp[i + 1:].norm().clamp(min=1e-30)))
            bm.append(float((pj[Jp[i + 1:]] ** 2).sum() / n ** 2))
            nm.append(float(n))
        m = lambda z: (sum(z) / len(z)) if z else float("nan")  # noqa: E731
        return m(cx), m(cv), m(cr), m(bm), m(nm)

    bos_pos = pos_of[J]
    ord_pos = [i for i in range(d) if not bool(Jp[i])][:n_ord]
    return overlap, stats_for(bos_pos.tolist()), stats_for(ord_pos)


@torch.no_grad()
def analyse(g, X0, ntok, W, bits, group_size, percdamp, role, li, tag_meta, save_whiten=None, g_nob=None):
    t0 = time.time()
    dev = g.H.device
    H = g.H.clone()
    d = H.shape[0]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    # --- BOS block, rank-one component -------------------------------------
    N = ntok
    X0 = X0.to(dev)                                    # [n, d]
    G = (X0 @ X0.t()).double()                         # Gram [n, n]
    ev, U = torch.linalg.eigh(G)
    rank1_share = float(ev[-1] / ev.clamp(min=0).sum().clamp(min=1e-30))
    u1 = U[:, -1].float()
    x0 = (X0.t() @ u1)                                 # direction, norm = sqrt(ev[-1])
    x0 = x0 / x0.norm().clamp(min=1e-30) * float((ev[-1] * 2.0 / N).sqrt())   # gptq scaling of B
    A_bos = float((X0 ** 2).sum() * 2.0 / N / torch.trace(H).clamp(min=1e-30))
    # --- damping, M, v, kappa (float64) --------------------------------------
    lam = float(percdamp * torch.mean(torch.diag(H)))
    Hl = H.double()
    Hl.diagonal().add_(lam)
    x0d = x0.double()
    M = Hl - torch.outer(x0d, x0d)
    try:
        Lm = torch.linalg.cholesky(M)
    except Exception:  # noqa: BLE001  (numerical PSD failure: add jitter)
        M.diagonal().add_(1e-6 * lam)
        Lm = torch.linalg.cholesky(M)
    v = torch.cholesky_solve(x0d.unsqueeze(1), Lm).squeeze(1)
    kappa = float(x0d @ v)
    kappa_iso = float((x0d @ x0d) / lam)
    # Sherman-Morrison check: H_lambda^{-1} x0 == v / (1 + kappa)
    Lh = torch.linalg.cholesky(Hl)
    w = torch.cholesky_solve(x0d.unsqueeze(1), Lh).squeeze(1)
    sm_check = float((w - v / (1.0 + kappa)).norm() / w.norm().clamp(min=1e-30))
    del M, Lm, Lh, Hl
    v = v.float()
    # --- GPTQ and RTN on this matrix -----------------------------------------
    W = W.to(dev).float()
    st = {"want_disp": True}
    Qg = g.quantize(bits=bits, group_size=group_size, sym=True, actorder=True, percdamp=percdamp, stats=st)
    D = st.pop("disp_matrix").to(dev)                  # accumulated compensation, original column order
    n_clip_g = int((Qg.abs() >= 0.999 * Qg.abs().max(dim=1, keepdim=True).values).sum())
    # counterfactual: the same matrix quantised with the BOS block removed from H
    g.layer.weight.data = W.to(g.layer.weight.dtype).clone()
    g.H = g_nob.H.clone()                              # BOS-free Hessian, accumulated directly
    g_nob.free()
    st2 = {"want_disp": True}
    Qnob = g.quantize(bits=bits, group_size=group_size, sym=True, actorder=True, percdamp=percdamp, stats=st2)
    D_nob = st2.pop("disp_matrix").to(dev)
    S_ = D - D_nob                                     # what the sink token adds to the compensation
    n_clip_nob = int((Qnob.abs() >= 0.999 * Qnob.abs().max(dim=1, keepdim=True).values).sum())
    Qr, n_clip_r = rtn_grouped(W, bits, group_size, True, None, False)
    Eg = W - Qg
    Er = W - Qr
    C = Eg - Er                                        # = Qr - Qg: the net effect incl. rounding flips
    torch.manual_seed(0)
    rnd = torch.randn(d, device=dev)
    # top right-singular vector of S
    try:
        _, sv, Vh = torch.linalg.svd(S_, full_matrices=False)
        top1 = Vh[0]
        share_top1 = float(sv[0] ** 2 / (sv ** 2).sum().clamp(min=1e-30))
        cos_top1_x0 = float((top1 @ x0).abs() / x0.norm().clamp(min=1e-30))
        cos_top1_v = float((top1 @ v).abs() / v.norm().clamp(min=1e-30))
    except Exception:  # noqa: BLE001
        share_top1 = cos_top1_x0 = cos_top1_v = float("nan")
    # columns holding 90% of ||x0||^2 (the sink's support) and the share of S / D_nob on them
    order = torch.argsort(x0 ** 2, descending=True)
    cum = torch.cumsum((x0[order] ** 2), 0) / (x0 @ x0).clamp(min=1e-30)
    k90 = int((cum < 0.9).sum()) + 1
    bos_cols = order[:k90]
    # E3 proper: the sending patterns under H and under H without the BOS block
    ov_J, (pcx, pcv, pcr, pbm, pnm), (ocx, ocv, _ocr, obm, onm) = sending_patterns(H, percdamp, bos_cols, x0, v, rnd)
    _, (ncx, ncv, _ncr, nbm, nnm), _ = sending_patterns(g.H, percdamp, bos_cols, x0, v, rnd)
    colshare_S = float((S_[:, bos_cols] ** 2).sum() / (S_ ** 2).sum().clamp(min=1e-30))
    colshare_Dnob = float((D_nob[:, bos_cols] ** 2).sum() / (D_nob ** 2).sum().clamp(min=1e-30))
    rowsS = (S_ ** 2).sum(1)
    rowsD = (D ** 2).sum(1)
    rowsC = (C ** 2).sum(1)
    rowsEg = (Eg ** 2).sum(1)
    rowsEr = (Er ** 2).sum(1)
    rec = dict(tag_meta, role=role, layer=li, d_in=d, d_out=W.shape[0], n_tokens=N,
               A_bos=A_bos, rank1_share_B=rank1_share, norm_x0_sq=float(x0 @ x0), **{"lambda": lam},
               kappa=kappa, kappa_iso=kappa_iso, sm_check=sm_check,
               n_bos_cols90_J=k90, actorder_topJ_overlap=ov_J,
               pat_cos_x0_bos=pcx, pat_cos_v_bos=pcv, pat_cos_rand_bos=pcr, pat_bosmass_bos=pbm, pat_norm_bos=pnm,
               pat_cos_x0_ord=ocx, pat_cos_v_ord=ocv, pat_bosmass_ord=obm, pat_norm_ord=onm,
               nob_pat_cos_x0_bos=ncx, nob_pat_cos_v_bos=ncv, nob_pat_bosmass_bos=nbm, nob_pat_norm_bos=nnm,
               share_S_x0=row_share(S_, x0), share_S_v=row_share(S_, v), share_S_rand=row_share(S_, rnd),
               share_S_top1=share_top1, cos_top1_x0=cos_top1_x0, cos_top1_v=cos_top1_v,
               n_bos_cols90=k90, colshare_S_bos=colshare_S, colshare_Dnob_bos=colshare_Dnob,
               normS_over_normDnob=float(S_.norm() / D_nob.norm().clamp(min=1e-30)),
               toprow_share_S=float(rowsS.max() / rowsS.sum().clamp(min=1e-30)), toprow_S=int(rowsS.argmax()),
               share_D_x0=row_share(D, x0), share_D_v=row_share(D, v), share_D_rand=row_share(D, rnd),
               toprow_share_D=float(rowsD.max() / rowsD.sum().clamp(min=1e-30)), toprow_D=int(rowsD.argmax()),
               share_C_x0=row_share(C, x0), share_C_v=row_share(C, v), share_C_rand=row_share(C, rnd),
               share_Eg_x0=row_share(Eg, x0), share_Eg_v=row_share(Eg, v),
               share_Er_x0=row_share(Er, x0), share_Er_v=row_share(Er, v),
               normD_over_normEr=float(D.norm() / Er.norm().clamp(min=1e-30)),
               normC_over_normEr=float(C.norm() / Er.norm().clamp(min=1e-30)),
               toprow_share_C=float(rowsC.max() / rowsC.sum().clamp(min=1e-30)), toprow_C=int(rowsC.argmax()),
               toprow_share_Eg=float(rowsEg.max() / rowsEg.sum().clamp(min=1e-30)), toprow_Eg=int(rowsEg.argmax()),
               toprow_share_Er=float(rowsEr.max() / rowsEr.sum().clamp(min=1e-30)),
               maxabs_W=float(W.abs().max()), maxabs_Qg=float(Qg.abs().max()), maxabs_Qnob=float(Qnob.abs().max()),
               n_clip_g=n_clip_g, n_clip_nob=n_clip_nob, n_clip_r=int(n_clip_r), seconds=round(time.time() - t0, 1))
    if save_whiten:
        os.makedirs(os.path.dirname(save_whiten) or ".", exist_ok=True)
        torch.save({"layer": li, "x0": x0.cpu(), "v": v.cpu(), "lambda": lam, "kappa": kappa,
                    "toprow_C": int(rowsEg.argmax()), "delta_gptq_toprow": Eg[int(rowsEg.argmax())].cpu(),
                    "meta": tag_meta}, save_whiten)
    g.free()
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", default="c4")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.05)
    ap.add_argument("--layer", type=int, help="sink-forming layer (default: argmax BOS share over down_proj)")
    ap.add_argument("--controls", default="+3,mid", help="control down_proj layers relative to the sink layer")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="runs/theory")
    ap.add_argument("--save-whiten", help="store x0, v for src/inject_sink.py artifact --whiten")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, tok = load_model(args.model)
    texts = load_calib(args.calib, tok, args.n_calib, args.seqlen, seed=args.calib_seed)
    n_layers = len(model.model.layers)
    print(f"[theory] {args.model} calib={args.calib} n={len(texts)} seqlen={args.seqlen}", flush=True)
    a_bos = detect_sink_layer(model, tok, texts, args.seqlen)
    L = args.layer if args.layer is not None else max(range(n_layers), key=lambda i: a_bos[i])
    print("[theory] A_bos per down_proj: " + " ".join(f"L{i}:{a:.3f}" for i, a in enumerate(a_bos))
          + f" -> sink layer L{L}", flush=True)
    layers = [L]
    for c in [s for s in args.controls.split(",") if s]:
        li = (L + int(c) if c.startswith(("+", "-")) else (n_layers // 2 if c == "mid" else int(c)))
        if 0 <= li < n_layers and li not in layers:
            layers.append(li)
    W = {li: _down(model, li).weight.data.clone().float().cpu() for li in layers}
    g, g_nob, X0, ntok = accumulate(model, tok, texts, args.seqlen, layers)
    meta = dict(tag=args.tag, model=args.model, calib=args.calib, n_calib=len(texts), seqlen=args.seqlen)
    recs = []
    for li in layers:
        role = "sink" if li == L else f"control{li - L:+d}"
        rec = analyse(g[li], X0[li], ntok[li], W[li], args.bits, args.group_size, args.percdamp,
                      role, li, meta, save_whiten=args.save_whiten if li == L else None, g_nob=g_nob[li])
        rec["A_bos_all_layers"] = ",".join(f"{a:.4f}" for a in a_bos)
        recs.append(rec)
        print(f"[theory] {role} L{li}: A_bos {rec['A_bos']:.4f} rank1(B) {rec['rank1_share_B']:.3f} "
              f"kappa {rec['kappa']:.3g} (iso {rec['kappa_iso']:.3g}, SM check {rec['sm_check']:.1e}) | "
              f"PATTERN on {rec['n_bos_cols90_J']} BOS cols (act-order overlap {rec['actorder_topJ_overlap']:.2f}): "
              f"cos x0 {rec['pat_cos_x0_bos']:.3f} v {rec['pat_cos_v_bos']:.3f} rand {rec['pat_cos_rand_bos']:.3f} "
              f"BOS-mass {rec['pat_bosmass_bos']:.3f} | ordinary cols: cos x0 {rec['pat_cos_x0_ord']:.3f} "
              f"BOS-mass {rec['pat_bosmass_ord']:.3f} | H_nob: cos x0 {rec['nob_pat_cos_x0_bos']:.3f} "
              f"BOS-mass {rec['nob_pat_bosmass_bos']:.3f} | "
              f"S along v {rec['share_S_v']:.3f} |S|/|D_nob| {rec['normS_over_normDnob']:.3f} | "
              f"Er along x0 {rec['share_Er_x0']:.4f} | toprow S {rec['toprow_S']} {rec['toprow_share_S']:.3f} | "
              f"|Q| {rec['maxabs_Qg']:.3g} vs |W| {rec['maxabs_W']:.3g}", flush=True)
    with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
        json.dump(recs, f, indent=2)
    # rebuild the index from every json in out_dir (idempotent under re-runs)
    idx = os.path.join(args.out_dir, "INDEX_theory.csv")
    allrecs = []
    for jf in sorted(glob.glob(os.path.join(args.out_dir, "*.json"))):
        try:
            allrecs.extend(json.load(open(jf)))
        except Exception:  # noqa: BLE001
            pass
    with open(idx, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in allrecs:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[theory] -> {args.out_dir}/{args.tag}.json (+ INDEX_theory.csv)")


if __name__ == "__main__":
    main()
