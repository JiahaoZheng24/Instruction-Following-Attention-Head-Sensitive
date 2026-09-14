"""W59/W60: the OUTPUT side of the layer Hessian.

GPTQ minimises ||(W - Q) X||^2 with the input covariance A = X X^T of the
calibration set and treats every output direction / position as equally
important, i.e. it uses A (x) I where the loss Hessian is ~ A (x) G (K-FAC).
This module measures the dropped factor per token and, for W60, turns it into
a per-token weight for the calibration Hessian:

    g_t = || dL / dy_t ||^2      (y_t = output of the linear at token t,
                                  L = summed next-token loss of the sample)
    H_G = sum_t g_t x_t x_t^T    (instead of sum_t x_t x_t^T)

Both quantities come from ONE forward+backward pass per calibration sample on
the un-quantised model; only activation gradients are needed (parameters are
frozen), so an 8B model with 128 x 2048 tokens takes minutes.

W59 (measurement):  python src/grad_weights.py --model M --calib c4 --out runs/sides/<tag>.csv
    per module: share of A (sum ||x_t||^2), G (sum g_t) and A.G (sum g_t ||x_t||^2)
    carried by position 0 (BOS), positions 1-7 (template tokens of chat
    prompts / first tokens of a document) and the rest.
W60 (method): quantize_protected.py --hess-grad-weight [--hess-grad-power a]
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DEFAULT_MODEL, load_model  # noqa: E402
from quantize_gptq import load_calib  # noqa: E402

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


def _module(layer, p):
    return getattr(layer.self_attn, p) if p in ATTN else getattr(layer.mlp, p)


def collect_token_sides(model, tok, texts, max_len, loss_scale=1024.0, verbose=True):
    """Returns (xin, gout): dict[(layer, proj)] -> list over samples of
    1-D CPU float tensors [T]: ||x_t||^2 (input side) and ||dL/dy_t||^2
    (output side). Positions align with the tokenisation used by
    capture_layer0_inputs (same tokenizer call)."""
    layers = model.model.layers
    keys = [(li, p) for li in range(len(layers)) for p in ATTN + MLP]
    xin = {k: [] for k in keys}
    gout = {k: [] for k in keys}

    def fwd_hook(key):
        def hook(_m, inp, out):
            x = inp[0].detach().float()
            xin[key].append((x.reshape(-1, x.shape[-1]) ** 2).sum(1).cpu())
            if out.requires_grad:
                def ghook(g):
                    gg = g.detach().float()
                    gout[key].append((gg.reshape(-1, gg.shape[-1]) ** 2).sum(1).cpu())
                out.register_hook(ghook)
        return hook

    handles = [_module(layers[li], p).register_forward_hook(fwd_hook((li, p))) for li, p in keys]
    req = [(prm, prm.requires_grad) for prm in model.parameters()]
    for prm, _ in req:
        prm.requires_grad_(False)
    emb = model.get_input_embeddings()
    h_emb = emb.register_forward_hook(lambda _m, _i, o: o.requires_grad_(True))
    was_training = model.training
    model.eval()
    try:
        for j, t in enumerate(texts):
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            out = model(**ids, use_cache=False)
            logits = out.logits[0, :-1].float()
            target = ids["input_ids"][0, 1:]
            loss = F.cross_entropy(logits, target, reduction="sum") * loss_scale
            loss.backward()
            del out, logits, loss
            if verbose and (j + 1) % 16 == 0:
                print(f"[sides] {j + 1}/{len(texts)} samples", flush=True)
    finally:
        for h in handles:
            h.remove()
        h_emb.remove()
        for prm, r in req:
            prm.requires_grad_(r)
        model.train(was_training)
        torch.cuda.empty_cache()
    for k in keys:
        assert len(gout[k]) == len(xin[k]) == len(texts), (k, len(gout[k]), len(xin[k]))
        for j in range(len(texts)):
            assert gout[k][j].numel() == xin[k][j].numel(), (k, j)
            assert torch.isfinite(gout[k][j]).all(), f"non-finite gradient at {k} sample {j}"
    return xin, gout


def grad_weights(gout, power=1.0, floor=0.0, cap=0.0):
    """Per-module per-token Hessian weights: (g_t^power), optionally capped at
    cap x the median token (W61: the raw weights span 1e5-1e6, a handful of
    sink-receiving tokens then own H), normalised to mean 1 over the whole
    calibration set; optional floor (fraction of the mean)."""
    w = {}
    for k, lst in gout.items():
        ws = [g.clamp(min=0) ** power for g in lst]
        if cap > 0:
            med = torch.cat(ws).median().clamp(min=1e-30)
            ws = [x.clamp(max=cap * med) for x in ws]
        mean = torch.cat(ws).mean().clamp(min=1e-30)
        ws = [x / mean for x in ws]
        if floor > 0:
            ws = [x.clamp(min=floor) for x in ws]
        w[k] = ws
    return w


def _shares(vals, classes):
    tot = sum(float(v.sum()) for v in vals)
    out = {}
    for name, lo, hi in classes:
        s = 0.0
        for v in vals:
            s += float(v[lo:hi].sum()) if hi else float(v[lo:].sum())
        out[name] = s / tot if tot > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calib", default="c4",
                    choices=["c4", "c4win", "pile", "wikitext", "instruct", "c4chat", "ultrachat"])
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="CSV, one row per module")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    texts = load_calib(args.calib, tok, args.n_calib, args.seqlen, seed=args.calib_seed)
    print(f"[sides] {args.model} {args.calib} x{len(texts)}")
    xin, gout = collect_token_sides(model, tok, texts, args.seqlen)
    classes = [("bos", 0, 1), ("p1_7", 1, 8), ("rest", 8, None)]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cols = ["layer", "proj", "n_tokens"] + [f"{s}_{c}" for s in ("A", "G", "AG") for c, _, _ in classes] + \
           ["G_over_A_bos", "G_over_A_p1_7", "g_bos_over_median", "g_p1_7_max_over_median"]
    rows = []
    for (li, p) in xin:
        x, g = xin[(li, p)], gout[(li, p)]
        ag = [a * b for a, b in zip(x, g)]
        A, G, AG = _shares(x, classes), _shares(g, classes), _shares(ag, classes)
        allg = torch.cat(g)
        med = float(allg.median()) if allg.numel() else float("nan")
        gb = torch.stack([v[0] for v in g]).mean()
        gp = torch.cat([v[1:8] for v in g]).max() if all(v.numel() > 8 for v in g) else torch.tensor(float("nan"))
        r = {"layer": li, "proj": p, "n_tokens": int(allg.numel())}
        for c, _, _ in classes:
            r[f"A_{c}"], r[f"G_{c}"], r[f"AG_{c}"] = A[c], G[c], AG[c]
        r["G_over_A_bos"] = G["bos"] / A["bos"] if A["bos"] > 0 else float("nan")
        r["G_over_A_p1_7"] = G["p1_7"] / A["p1_7"] if A["p1_7"] > 0 else float("nan")
        r["g_bos_over_median"] = float(gb) / med if med > 0 else float("nan")
        r["g_p1_7_max_over_median"] = float(gp) / med if med > 0 else float("nan")
        rows.append(r)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[sides] -> {args.out} ({len(rows)} modules)")
    print("[sides] down_proj, layers 0-7: A_bos  G_bos  AG_bos | A_p1_7 G_p1_7 | G/A_bos")
    for r in rows:
        if r["proj"] == "down_proj" and r["layer"] < 8:
            print(f"   L{r['layer']:<2d} {r['A_bos']:.3f} {r['G_bos']:.3f} {r['AG_bos']:.3f} | "
                  f"{r['A_p1_7']:.3f} {r['G_p1_7']:.3f} | {r['G_over_A_bos']:.3g}")
    top = max(rows, key=lambda r: r["A_bos"])
    print(f"[sides] most BOS-dominated module: L{top['layer']}.{top['proj']} A_bos={top['A_bos']:.3f} G_bos={top['G_bos']:.3f}")


if __name__ == "__main__":
    main()
