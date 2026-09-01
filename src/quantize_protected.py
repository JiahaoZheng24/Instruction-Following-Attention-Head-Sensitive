"""v2: in-loop protected GPTQ quantization (protection held out INSIDE the
GPTQ column loop; see gptq_core.py). Saves a fake-quant fp16/bf16 HF
checkpoint (values on the 3-bit grid except protected weights), loadable by
the existing eval pipeline (diagnose_heads.py ablate --model <out_dir>).

Protection modes (--protect):
  none    validation arm: plain masked-GPTQ, no mask — internal reference
  heads   head slices (q rows / o cols / kv-group rows) from a ranking CSV
  tacq    TaCQ-criterion scattered weights: global top weights by saliency
          (src/tacq_salience.py output) up to --budget-params
  randw   random scattered weights at --budget-params (structure control)

Calibration matches W0 protocol: c4, 128 x 2048, g128, sym, desc_act.

Example:
  python src/quantize_protected.py --protect tacq \
      --salience-dir runs/salience_if --budget-params 37624064 \
      --out $STORE/models/qwen2.5-7b-v2gptq3-tacq
"""
import argparse
import csv
import json
import os
import random

import torch

from common import DEFAULT_MODEL, HeadGeom, load_model
from gptq_core import MaskedGPTQ
from quantize_gptq import load_calib

ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("gate_proj", "up_proj", "down_proj")


# ------------------------------------------------------------- masks

def head_masks(geom: HeadGeom, heads, kv: bool, projs: str):
    """-> {proj_name: (row_idx, col_idx)} per layer: dict[layer][proj]."""
    d = geom.head_dim
    out: dict[int, dict[str, tuple]] = {}
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    for l, hs in by_layer.items():
        rows = [i for h in hs for i in range(h * d, (h + 1) * d)]
        m = {}
        if projs in ("all", "qkv"):
            m["q_proj"] = ("rows", rows)
            if kv:
                groups = sorted({geom.kv_group(h) for h in hs})
                g_rows = [i for g in groups for i in range(g * d, (g + 1) * d)]
                m["k_proj"] = ("rows", g_rows)
                m["v_proj"] = ("rows", g_rows)
        if projs in ("all", "o"):
            m["o_proj"] = ("cols", rows)
        out[l] = m
    return out


def build_mask_for(module, layer_idx, proj, args, ctx):
    """Returns bool [rows, cols] mask or None."""
    W = module.weight
    if args.protect == "none":
        return None
    if args.protect == "heads":
        spec = ctx["head_masks"].get(layer_idx, {}).get(proj)
        if spec is None:
            return None
        kind, idx = spec
        m = torch.zeros(W.shape, dtype=torch.bool)
        ix = torch.tensor(idx, dtype=torch.long)
        if kind == "rows":
            m[ix, :] = True
        else:
            m[:, ix] = True
        ctx["selected"] += int(m.sum())
        return m
    if args.protect == "cols":
        cols = ctx["col_masks"].get((layer_idx, proj))
        if not cols:
            return None
        m = torch.zeros(W.shape, dtype=torch.bool)
        m[:, torch.tensor(cols, dtype=torch.long)] = True
        ctx["selected"] += int(m.sum())
        return m
    if args.protect == "coords":
        pts = ctx["coords"].get((layer_idx, proj))
        if not pts:
            return None
        m = torch.zeros(W.shape, dtype=torch.bool)
        for r_, c_ in pts:
            m[r_, c_] = True
        ctx["selected"] += int(m.sum())
        return m
    name = f"model.layers.{layer_idx}.{'self_attn' if proj in ATTN else 'mlp'}.{proj}"
    if args.protect == "tacq":
        sal = torch.load(os.path.join(args.salience_dir,
                                      name.replace("/", "_") + ".pt"))
        m = sal.float() > ctx["threshold"]
        ctx["selected"] += int(m.sum())
        return m
    if args.protect == "randw":
        g = torch.Generator().manual_seed(
            args.seed * 100003 + layer_idx * 101 + ATTN.index(proj) if proj in ATTN
            else args.seed * 100003 + layer_idx * 101 + 50 + MLP.index(proj))
        m = torch.rand(W.shape, generator=g) < ctx["frac"]
        ctx["selected"] += int(m.sum())
        return m
    raise ValueError(args.protect)


# ------------------------------------------------------------- driver

@torch.no_grad()
def capture_layer0_inputs(model, tok, texts, max_len):
    """Run the embedding+prelude once per sample; catch layer-0 inputs."""
    inps, kwargs_list = [], []

    class Catcher(torch.nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, hidden_states, **kw):
            inps.append(hidden_states)
            kwargs_list.append(kw)
            raise RuntimeError("stop")

        def __getattr__(self, name):
            # transformers inspects layer attributes (e.g. attention_type)
            # before calling forward — delegate anything we don't have.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("mod"), name)

    layers = model.model.layers
    layers[0] = Catcher(layers[0])
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True,
                  max_length=max_len).to(model.device)
        try:
            model(**ids)
        except RuntimeError:
            pass
    layers[0] = layers[0].mod
    return inps, kwargs_list


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--protect",
                    choices=["none", "heads", "tacq", "randw", "coords", "cols"],
                    required=True)
    ap.add_argument("--no-actorder", action="store_true",
                    help="GPTQ without desc_act ordering (compensation-order probe)")
    ap.add_argument("--coords-file",
                    help="coords mode: CSV with layer,proj,row,col (e.g. detected "
                         "super weights); protects exactly these weight entries")
    # heads mode
    ap.add_argument("--topk-from")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--kv", action="store_true")
    ap.add_argument("--projs", choices=["all", "qkv", "o"], default="all")
    # tacq / randw modes
    ap.add_argument("--salience-dir")
    ap.add_argument("--budget-params", type=int, default=37_624_064)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-seed", type=int, default=0,
                    help="disjoint c4 calib replicate (for error bars)")
    ap.add_argument("--rtn", action="store_true",
                    help="RTN instead of GPTQ: no calibration, no compensation "
                         "(method-generality arm; same group/sym protocol)")
    ap.add_argument("--rotate", action="store_true",
                    help="QuaRot-style R1: fold norms + fuse a random orthogonal "
                         "rotation into the weights BEFORE quantization. With "
                         "--protect tacq the salience dir must come from a "
                         "matching tacq_salience --rotate run (same seed).")
    ap.add_argument("--rotate-seed", type=int, default=0)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    if args.rotate:
        from rotate_model import fold_and_rotate, logit_probe, probe_report
        ref = logit_probe(model, tok)
        fold_and_rotate(model, seed=args.rotate_seed)
        print(f"[v2] rotation sanity: {probe_report(ref, logit_probe(model, tok))}")
    geom = HeadGeom(model)
    ctx = {"selected": 0}

    if args.protect == "heads":
        assert args.topk_from, "--topk-from required for --protect heads"
        rows = list(csv.DictReader(open(args.topk_from)))
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        heads = [(int(r["layer"]), int(r["head"])) for r in rows[: args.k]]
        ctx["head_masks"] = head_masks(geom, heads, args.kv, args.projs)
    elif args.protect == "tacq":
        sample = torch.load(os.path.join(args.salience_dir, "sample.pt"))
        total = sum(p.numel() for n, p in model.named_parameters()
                    if "layers" in n and p.dim() == 2)
        q = 1.0 - args.budget_params / total
        from common import safe_quantile
        ctx["threshold"] = safe_quantile(sample, q)
        print(f"[v2] tacq threshold={ctx['threshold']:.3e} (target q={q:.5f})")
    elif args.protect == "randw":
        total = sum(p.numel() for n, p in model.named_parameters()
                    if "layers" in n and p.dim() == 2)
        ctx["frac"] = args.budget_params / total
    elif args.protect == "cols":
        # whole input-channel columns, greedily by captured-salience density,
        # until the param budget is filled -> hardware-friendly (OWQ storage)
        assert args.salience_dir, "--salience-dir required for --protect cols"
        import re as _re
        cands = []
        for f in sorted(os.listdir(args.salience_dir)):
            if not f.endswith(".pt") or f == "sample.pt":
                continue
            mm = _re.match(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.pt", f)
            sal = torch.load(os.path.join(args.salience_dir, f), map_location="cpu").float()
            score = sal.sum(dim=0)                     # per input column
            cost = sal.shape[0]
            top = torch.topk(score, min(256, score.numel()))
            for s, c in zip(top.values.tolist(), top.indices.tolist()):
                cands.append((s / cost, int(mm.group(1)), mm.group(2), c, cost))
        cands.sort(reverse=True)
        col_masks: dict[tuple, list] = {}
        spent = 0
        for dens, l, proj, c, cost in cands:
            if spent + cost > args.budget_params:
                continue
            col_masks.setdefault((l, proj), []).append(c)
            spent += cost
        ctx["col_masks"] = col_masks
        print(f"[v2] cols mode: {sum(len(v) for v in col_masks.values())} columns, "
              f"{spent / 1e6:.1f}M params")
    elif args.protect == "coords":
        assert args.coords_file, "--coords-file required for --protect coords"
        coords: dict[tuple, list] = {}
        for r in csv.DictReader(open(args.coords_file)):
            coords.setdefault((int(r["layer"]), r["proj"]), []).append(
                (int(r["row"]), int(r["col"])))
        ctx["coords"] = coords
        print(f"[v2] coords mode: {sum(len(v) for v in coords.values())} weight entries")

    layers = model.model.layers
    if args.rtn:
        with torch.no_grad():
            for li in range(len(layers)):
                layer = layers[li]
                mods = {p: getattr(layer.self_attn, p) for p in ATTN}
                mods.update({p: getattr(layer.mlp, p) for p in MLP})
                for p, m in mods.items():
                    mask = build_mask_for(m, li, p, args, ctx)
                    rtn_quantize_(m.weight.data, args.bits, args.group_size,
                                  mask.to(m.weight.device) if mask is not None else None)
                print(f"[v2-rtn] layer {li + 1}/{len(layers)} "
                      f"(protected so far: {ctx['selected'] / 1e6:.1f}M)", flush=True)
        save_ckpt(model, tok, args, ctx)
        return

    calib = load_calib("c4", tok, args.n_calib, args.seqlen, seed=args.calib_seed)
    print(f"[v2] capturing layer-0 inputs ({len(calib)} samples)")
    inps, kws = capture_layer0_inputs(model, tok, calib, args.seqlen)
    with torch.no_grad():
        for li in range(len(layers)):
            layer = layers[li]
            mods = {p: getattr(layer.self_attn, p) for p in ATTN}
            mods.update({p: getattr(layer.mlp, p) for p in MLP})
            gptq = {p: MaskedGPTQ(m) for p, m in mods.items()}
            handles = [m.register_forward_pre_hook(
                (lambda g: lambda _m, a: g.add_batch(a[0]))(gptq[p]))
                for p, m in mods.items()]
            for j in range(len(inps)):
                layer(inps[j], **kws[j])
            for h in handles:
                h.remove()
            for p, g in gptq.items():
                mask = build_mask_for(mods[p], li, p, args, ctx)
                g.quantize(bits=args.bits, group_size=args.group_size,
                           sym=True, actorder=not args.no_actorder, mask=mask)
                g.free()
            for j in range(len(inps)):
                out = layer(inps[j], **kws[j])
                # transformers >=4.5x returns a plain tensor; older versions a tuple
                inps[j] = out[0] if isinstance(out, tuple) else out
            print(f"[v2] layer {li + 1}/{len(layers)} quantized "
                  f"(protected so far: {ctx['selected'] / 1e6:.1f}M)", flush=True)

    save_ckpt(model, tok, args, ctx)


def rtn_quantize_(W: torch.Tensor, bits: int, group_size: int, mask):
    """In-place grouped sym RTN; masked entries keep fp (protocol-matched)."""
    maxq = 2 ** bits - 1
    zero = (maxq + 1) / 2
    Wf = W.float()
    for c in range(0, W.shape[1], group_size):
        g = Wf[:, c:c + group_size]
        xmax = g.abs().max(dim=1).values.clamp(min=1e-8)
        scale = (2.0 * xmax / maxq).unsqueeze(1)
        q = torch.clamp(torch.round(g / scale) + zero, 0, maxq)
        dq = scale * (q - zero)
        if mask is not None:
            mm = mask[:, c:c + group_size]
            dq[mm] = g[mm]
        Wf[:, c:c + group_size] = dq
    W.copy_(Wf.to(W.dtype))


def save_ckpt(model, tok, args, ctx):
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "PROTECT_PROTOCOL.json"), "w") as f:
        json.dump({"model": args.model, "bits": args.bits,
                   "group_size": args.group_size, "sym": True,
                   "desc_act": not args.rtn and not args.no_actorder,
                   "quantizer": "rtn" if args.rtn else "gptq",
                   "calib": None if args.rtn else "c4", "n_calib": args.n_calib,
                   "protect": args.protect, "budget_params": args.budget_params,
                   "selected_params": ctx["selected"],
                   "topk_from": args.topk_from, "k": args.k,
                   "kv": args.kv, "projs": args.projs, "seed": args.seed,
                   "calib_seed": args.calib_seed, "coords_file": args.coords_file,
                   "rotate": args.rotate,
                   "rotate_seed": args.rotate_seed if args.rotate else None,
                   "format": "fake-quant fp checkpoint (values on b-bit grid)"},
                  f, indent=2)
    print(f"[v2] saved -> {args.out} (protected {ctx['selected'] / 1e6:.1f}M params)")


if __name__ == "__main__":
    main()
