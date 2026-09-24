"""W81: how much of the second-order loss change caused by the lesion lives in the
cross-position blocks that Lemma 1 drops?

Lemma 1 keeps sum_t (dW x_t)^T G_tt (dW x_t) and drops G_ts, s != t. For the
sink-forming down_proj we measure both on the real network: with the output
perturbation e_t = dW x_t injected after the layer at every position, the full
quadratic form is q_full = 1/2 e^T H e (one Hessian-vector product in the
perturbation), and the block-diagonal part sum_t 1/2 e_t^T H_tt e_t is
estimated with random sign flips across positions, E_sigma[1/2 (sigma o e)^T H
(sigma o e)] (K draws). dW is the real GPTQ lesion of the matrix (W - Q_gptq),
the RTN error (W - Q_rtn) and a Gaussian matrix of the lesion's norm. Also
reported: the first-order term g^T e that OBS drops, the finite-difference
loss change, and the block-diagonal mass on template/newline positions.

  python src/cross_position.py --model meta-llama/Llama-3.1-8B-Instruct --layer 1 \
      --calib c4 --prompts data/ifeval_input_data.jsonl --n 16 --k 8 --out runs/theory/f_l_crosspos.csv
"""
import argparse
import csv
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_jsonl  # noqa: E402
from gptq_core import MaskedGPTQ, rtn_grouped  # noqa: E402
from inject_sink import chat, token_set  # noqa: E402
from quantize_gptq import load_calib  # noqa: E402


def load_fp32(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, device_map="auto",
                                                     attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, device_map="auto",
                                                     attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


@torch.no_grad()
def lesion_matrices(model, tok, texts, layer, seqlen, bits, group_size, percdamp, seed):
    dp = model.model.layers[layer].mlp.down_proj
    g = MaskedGPTQ(dp, name=f"layers.{layer}.down_proj")
    h = dp.register_forward_pre_hook(lambda _m, a: g.add_batch(a[0]))
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=seqlen).to(model.device)
        model(**ids, use_cache=False)
    h.remove()
    W = dp.weight.data.clone()
    Qg = g.quantize(bits=bits, group_size=group_size, sym=True, actorder=True, percdamp=percdamp)
    dp.weight.data = W.clone()                         # restore the full-precision matrix
    g.free()
    Qr, _ = rtn_grouped(W, bits, group_size, True, None, False)
    Eg = (W - Qg).float()
    Er = (W - Qr).float()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    Ern = torch.randn(W.shape, generator=gen).to(W.device)
    Ern = Ern * (Eg.norm() / Ern.norm())
    return {"lesion": Eg, "rtn": Er, "random": Ern}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True, help="sink-forming layer (down_proj perturbed after it)")
    ap.add_argument("--calib", default="c4")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--percdamp", type=float, default=0.05)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=8, help="sign-flip draws for the block-diagonal estimate")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_fp32(args.model)
    texts = load_calib(args.calib, tok, args.n_calib, args.seqlen, seed=args.calib_seed)
    mats = lesion_matrices(model, tok, texts, args.layer, args.seqlen, args.bits, args.group_size,
                           args.percdamp, args.seed)
    print("[crosspos] |E| lesion %.3g  rtn %.3g  random %.3g" % tuple(float(m.norm()) for m in mats.values()), flush=True)
    dp = model.model.layers[args.layer].mlp.down_proj
    tids = token_set(tok, "template,newline")
    prompts = read_jsonl(args.prompts)[: args.n]
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    state = {"eps": None, "x": None}
    h_in = dp.register_forward_pre_hook(lambda _m, a: state.__setitem__("x", a[0].detach()))

    def add_eps(_m, _a, out):
        return out + state["eps"] if state["eps"] is not None else out
    h_out = dp.register_forward_hook(add_eps)

    rows = []
    for pi, ex in enumerate(prompts):
        enc = tok(chat(tok, ex["prompt"]), return_tensors="pt", truncation=True, max_length=args.max_len).to(model.device)
        ids = enc["input_ids"]
        T = ids.shape[1]
        with torch.no_grad():
            state["eps"] = None
            model(**enc, use_cache=False)
            x = state["x"][0].float()                                  # [T, d_ff]
        is_tpl = torch.tensor([int(i) in tids for i in ids[0].tolist()], device=model.device)
        is_tpl[0] = False

        def loss_fn(eps):
            state["eps"] = eps
            logits = model(**enc, use_cache=False).logits[0]
            return F.cross_entropy(logits[:-1].float(), ids[0, 1:], reduction="sum")

        eps0 = torch.zeros(1, T, dp.weight.shape[0], device=model.device, dtype=torch.float32)
        eps0.requires_grad_(True)
        L0 = loss_fn(eps0)
        grad0 = torch.autograd.grad(L0, eps0)[0].detach()
        L0 = float(L0)

        def quad(v):
            _, hv = torch.autograd.functional.hvp(loss_fn, eps0.detach(), v)
            return 0.5 * float((v * hv).sum())

        for name, E in mats.items():
            e = (x @ E.t()).unsqueeze(0)                                 # [1, T, d_model]: dW x_t at every position
            q_full = quad(e)
            first = float((grad0 * e).sum())
            with torch.no_grad():
                fd = float(loss_fn(e)) - L0
            qs, qt, qo = [], [], []
            for _ in range(args.k):
                sgn = (torch.randint(0, 2, (1, T, 1), generator=gen) * 2 - 1).float().to(e.device)
                qs.append(quad(sgn * e))
                if name == "lesion":
                    qt.append(quad(sgn * e * is_tpl.view(1, T, 1).float()))
                    qo.append(quad(sgn * e * (~is_tpl).view(1, T, 1).float()))
            qs_t = torch.tensor(qs)
            row = {"prompt": pi, "dW": name, "T": T, "n_tpl": int(is_tpl.sum()), "L0": L0,
                   "q_full": q_full, "q_diag": float(qs_t.mean()), "q_diag_std": float(qs_t.std()),
                   "ratio_full_over_diag": q_full / max(float(qs_t.mean()), 1e-12),
                   "first_order": first, "finite_diff": fd,
                   "q_diag_tpl": float(torch.tensor(qt).mean()) if qt else float("nan"),
                   "q_diag_ord": float(torch.tensor(qo).mean()) if qo else float("nan")}
            rows.append(row)
            print(f"[crosspos] prompt {pi} {name:7s} T={T} q_full={q_full:.4g} q_diag={row['q_diag']:.4g}"
                  f"±{row['q_diag_std']:.2g} ratio={row['ratio_full_over_diag']:.3f} first={first:.4g} fd={fd:.4g}"
                  + (f" tpl={row['q_diag_tpl']:.4g} ord={row['q_diag_ord']:.4g}" if qt else ""), flush=True)
        state["eps"] = None
    h_in.remove()
    h_out.remove()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    for name in mats:
        rs = [r for r in rows if r["dW"] == name]
        ratios = torch.tensor([r["ratio_full_over_diag"] for r in rs])
        print(f"[crosspos] {name:7s}: ratio full/diag median {ratios.median():.3f} mean {ratios.mean():.3f} "
              f"(n={len(rs)}); mean q_full {sum(r['q_full'] for r in rs)/len(rs):.4g}, "
              f"mean |first-order| {sum(abs(r['first_order']) for r in rs)/len(rs):.4g}")
    les = [r for r in rows if r["dW"] == "lesion"]
    tpl = sum(r["q_diag_tpl"] for r in les) / len(les)
    ordn = sum(r["q_diag_ord"] for r in les) / len(les)
    print(f"[crosspos] lesion block-diagonal mass on template/newline positions: {tpl / max(tpl + ordn, 1e-12):.3f} "
          f"(they are {sum(r['n_tpl'] for r in les) / sum(r['T'] for r in les):.3f} of positions)")
    print(f"[crosspos] -> {args.out}")


if __name__ == "__main__":
    main()
