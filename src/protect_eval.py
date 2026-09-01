"""W2: Protect, Don't Repair — slice protection on a quantized checkpoint.

v1 protection = post-hoc slice restoration ("patch after quant"): load a GPTQ
checkpoint, then restore selected weight slices to their original
full-precision values.

Attention-head protection (per query head h, d=head_dim):
  q_proj : output rows  [h*d,(h+1)*d)  (+ bias slice)     -> overwrite rows
  o_proj : input  cols  [h*d,(h+1)*d)                     -> additive delta
  k/v    : per kv-group rows (only with --kv)
  --projs all|qkv|o decomposes the arms: row restoration (q/k/v) fully undoes
  those rows' quantization and is self-consistent; column restoration (o_proj)
  conflicts with GPTQ's cross-column error compensation (the compensation
  assumed the quantized values), so the qkv-vs-o split tests whether post-hoc
  column restoration actively hurts.

MLP-channel protection (budget-matched control for H0-not-attention):
  --mlp-params N --seed S: random intermediate channels (gate/up rows +
  down_proj cols) sampled uniformly across layers until >= N params.

Mechanics: fp16 slices are read directly from the original model's
safetensors shards (no full fp16 model in memory); quantized columns are
recovered kernel-agnostically by probing the module with basis vectors.

Head-set selection: same interface as diagnose_heads.py
  --topk-from FILE --k K | --random K --seed S | --heads "L:H,.." | --layer 0,1
  no selection = noop arm (must reproduce the plain quantized checkpoint).

Example:
  python src/protect_eval.py --quant-model $STORE/models/qwen2.5-7b-gptq3-c4-g128 \
      --topk-from runs/dev_ranking_gptq3.csv --k 32 --kv \
      --prompts data/ifeval_input_data.jsonl --tag prot_top32_dev3
"""
import argparse
import csv
import json
import os
import random

import torch

from common import (DEFAULT_MODEL, HeadGeom, load_model, parse_heads,
                    read_jsonl, write_jsonl)

MAX_NEW_TOKENS = 1280


def chat(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


# ------------------------------------------------------------- fp16 slices

def _weight_map(model_id: str) -> tuple[str, dict]:
    """Resolve safetensors shard paths for the original (unquantized) model."""
    from transformers.utils.hub import cached_file
    try:
        idx_path = cached_file(model_id, "model.safetensors.index.json")
        idx = json.load(open(idx_path))
        return os.path.dirname(idx_path), idx["weight_map"]
    except Exception:
        p = cached_file(model_id, "model.safetensors")
        return os.path.dirname(p), None


class FPSource:
    """Reads individual tensors from the fp model's safetensors shards."""

    def __init__(self, model_id: str):
        self.base, self.wmap = _weight_map(model_id)
        self._open = {}

    def get(self, name: str) -> torch.Tensor:
        from safetensors import safe_open
        shard = self.wmap[name] if self.wmap else "model.safetensors"
        if shard not in self._open:
            self._open[shard] = safe_open(
                os.path.join(self.base, shard), framework="pt", device="cpu")
        return self._open[shard].get_tensor(name)

    def maybe(self, name: str):
        try:
            return self.get(name)
        except Exception:
            return None


# ------------------------------------------------------------- protection

def probe_columns(module, idx: torch.Tensor, in_features: int, device, dtype):
    """Recover W_q[:, idx] of a quantized linear kernel-agnostically.

    module(E) with basis rows E[i, idx[i]] = 1 gives W_q[:, idx_i] + bias;
    subtracting module(0) removes the bias. Returns [out, n] float32.
    """
    n = len(idx)
    E = torch.zeros(n + 1, in_features, device=device, dtype=dtype)
    E[torch.arange(n), idx.to(device)] = 1.0
    with torch.no_grad():
        out = module(E)
    return (out[:n] - out[n]).T.contiguous().float()


class SliceProtector:
    """Restores fp weight slices (rows and/or columns) on quantized modules."""

    def __init__(self, model):
        self.model = model
        self.device = model.model.embed_tokens.weight.device
        self.dtype = next(model.parameters()).dtype
        self.handles = []
        self.n_params = 0

    def protect_rows(self, module, w_fp, b_fp, idx: list[int]):
        """y[..., idx] = x @ W_fp[idx].T + b_fp[idx]  (exact fp restoration)."""
        if not idx:
            return
        ix = torch.tensor(sorted(idx), dtype=torch.long)
        w = w_fp[ix].to(device=self.device, dtype=torch.float32)
        b = b_fp[ix].to(device=self.device, dtype=torch.float32) if b_fp is not None else None
        ix_dev = ix.to(self.device)
        self.n_params += w.numel() + (b.numel() if b is not None else 0)

        def hook(_m, args, out):
            y = args[0].float() @ w.T
            if b is not None:
                y = y + b
            out[..., ix_dev] = y.to(out.dtype)
            return out

        self.handles.append(module.register_forward_hook(hook))

    def protect_cols(self, module, w_fp, idx: list[int]):
        """y += x[..., idx] @ (W_fp[:, idx] - W_q[:, idx]).T."""
        if not idx:
            return
        ix = torch.tensor(sorted(idx), dtype=torch.long)
        w_q = probe_columns(module, ix, w_fp.shape[1], self.device, self.dtype)
        delta = w_fp[:, ix].to(device=self.device, dtype=torch.float32) - w_q
        ix_dev = ix.to(self.device)
        self.n_params += delta.numel()

        def hook(_m, args, out):
            x = args[0]
            return out + (x[..., ix_dev].float() @ delta.T).to(out.dtype)

        self.handles.append(module.register_forward_hook(hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def attn_module(model, layer: int, proj: str):
    return getattr(model.model.layers[layer].self_attn, proj)


def mlp_module(model, layer: int, proj: str):
    return getattr(model.model.layers[layer].mlp, proj)


def protect_heads(prot: SliceProtector, geom: HeadGeom, fp: FPSource,
                  heads, protect_kv: bool, projs: str):
    d = geom.head_dim
    by_layer: dict[int, list[int]] = {}
    for l, h in heads:
        by_layer.setdefault(l, []).append(h)
    for l, hs in sorted(by_layer.items()):
        pfx = f"model.layers.{l}.self_attn."
        rows = [i for h in hs for i in range(h * d, (h + 1) * d)]
        if projs in ("all", "qkv"):
            prot.protect_rows(attn_module(prot.model, l, "q_proj"),
                              fp.get(pfx + "q_proj.weight"),
                              fp.maybe(pfx + "q_proj.bias"), rows)
            if protect_kv:
                groups = sorted({geom.kv_group(h) for h in hs})
                g_rows = [i for g in groups for i in range(g * d, (g + 1) * d)]
                for proj in ("k_proj", "v_proj"):
                    prot.protect_rows(attn_module(prot.model, l, proj),
                                      fp.get(pfx + proj + ".weight"),
                                      fp.maybe(pfx + proj + ".bias"), g_rows)
        if projs in ("all", "o"):
            prot.protect_cols(attn_module(prot.model, l, "o_proj"),
                              fp.get(pfx + "o_proj.weight"), rows)


def protect_mlp(prot: SliceProtector, geom: HeadGeom, fp: FPSource,
                target_params: int, seed: int) -> int:
    """Random intermediate channels (gate/up rows + down cols) up to budget."""
    cfg = prot.model.config
    inter = cfg.intermediate_size
    hidden = cfg.hidden_size
    per_channel = 3 * hidden  # gate row + up row + down col
    n_channels = max(1, target_params // per_channel)
    rng = random.Random(seed)
    all_ch = [(l, c) for l in range(geom.n_layers) for c in range(inter)]
    chosen = rng.sample(all_ch, n_channels)
    by_layer: dict[int, list[int]] = {}
    for l, c in chosen:
        by_layer.setdefault(l, []).append(c)
    for l, cs in sorted(by_layer.items()):
        pfx = f"model.layers.{l}.mlp."
        for proj in ("gate_proj", "up_proj"):
            prot.protect_rows(mlp_module(prot.model, l, proj),
                              fp.get(pfx + proj + ".weight"),
                              fp.maybe(pfx + proj + ".bias"), cs)
        prot.protect_cols(mlp_module(prot.model, l, "down_proj"),
                          fp.get(pfx + "down_proj.weight"), cs)
    return n_channels


# ------------------------------------------------------------- head sets

def pick_heads(args, geom: HeadGeom) -> list[tuple[int, int]]:
    if args.heads:
        return parse_heads(args.heads)
    if args.topk_from:
        rows = list(csv.DictReader(open(args.topk_from)))
        rows.sort(key=lambda r: float(r["score"]), reverse=True)
        return [(int(r["layer"]), int(r["head"])) for r in rows[: args.k]]
    if args.random:
        rng = random.Random(args.seed)
        all_heads = [(l, h) for l in range(geom.n_layers) for h in range(geom.n_heads)]
        return rng.sample(all_heads, args.random)
    if args.layer is not None:
        out = []
        for l in str(args.layer).split(","):
            out += [(int(l), h) for h in range(geom.n_heads)]
        return out
    return []  # noop arm


# ------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="fp weight source")
    ap.add_argument("--quant-model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--kv", action="store_true",
                    help="also protect k/v rows of the touched kv-groups")
    ap.add_argument("--projs", choices=["all", "qkv", "o"], default="all",
                    help="which attention projections to protect (decomposition)")
    ap.add_argument("--heads")
    ap.add_argument("--topk-from")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--random", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", help="protect whole layers, e.g. 0 or 0,1")
    ap.add_argument("--mlp-params", type=int,
                    help="MLP control: random channels up to this param budget")
    ap.add_argument("--selftest", action="store_true",
                    help="numeric check of the protection hooks, then exit")
    args = ap.parse_args()

    model, tok = load_model(args.quant_model)
    geom = HeadGeom(model)
    fp = FPSource(args.model)

    if args.selftest:
        selftest(model, geom, fp)
        return

    prot = SliceProtector(model)
    heads, n_ch = [], 0
    if args.mlp_params:
        n_ch = protect_mlp(prot, geom, fp, args.mlp_params, args.seed)
        print(f"[protect:{args.tag}] MLP control: {n_ch} channels, "
              f"{prot.n_params/1e6:.1f}M params restored")
    else:
        heads = pick_heads(args, geom)
        if heads:
            protect_heads(prot, geom, fp, heads, args.kv, args.projs)
            print(f"[protect:{args.tag}] {len(heads)} heads projs={args.projs} "
                  f"kv={args.kv}: {prot.n_params/1e6:.1f}M params restored")

    prompts = read_jsonl(args.prompts)
    out_rows = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            batch = prompts[i:i + args.batch]
            texts = [chat(tok, ex["prompt"]) for ex in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                 pad_token_id=tok.pad_token_id)
            for ex, seq in zip(batch, gen):
                resp = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                out_rows.append({"prompt": ex["prompt"], "response": resp})
            print(f"[protect:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}",
                  flush=True)
    prot.clear()

    run_dir = os.path.join("runs", os.path.basename(args.quant_model.rstrip("/")), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"quant_model={args.quant_model}\nfp_source={args.model}\n"
                f"heads={heads}\nn={len(heads)}\nkv={args.kv}\nprojs={args.projs}\n"
                f"mlp_channels={n_ch}\nprotected_params={prot.n_params}\n")
    print(f"[protect] {len(out_rows)} responses -> {run_dir}/responses.jsonl")


def selftest(model, geom, fp):
    """Verify hook math on layer 0 against direct fp computation."""
    device = model.model.embed_tokens.weight.device
    dtype = next(model.parameters()).dtype
    d = geom.head_dim
    l, h = 0, 0
    x = torch.randn(3, 5, geom.n_heads * d, device=device, dtype=dtype) * 0.1

    q_mod = attn_module(model, l, "q_proj")
    with torch.no_grad():
        y_before = q_mod(x).clone()
    prot = SliceProtector(model)
    protect_heads(prot, geom, fp, [(l, h)], protect_kv=False, projs="all")
    with torch.no_grad():
        y_after = attn_module(model, l, "q_proj")(x)
        w = fp.get(f"model.layers.{l}.self_attn.q_proj.weight")[:d].to(device).float()
        b = fp.get(f"model.layers.{l}.self_attn.q_proj.bias")[:d].to(device).float()
        ref = (x.float() @ w.T + b).to(dtype)
    err_rows = (y_after[..., :d] - ref).abs().max().item()
    unchanged = (y_after[..., d:] - y_before[..., d:]).abs().max().item()

    o_mod = attn_module(model, l, "o_proj")
    with torch.no_grad():
        z_after = o_mod(x)
        w_o = fp.get(f"model.layers.{l}.self_attn.o_proj.weight").to(device).float()
        rest = torch.arange(d, geom.n_heads * d)
        w_q_rest = probe_columns(o_mod, rest, geom.n_heads * d, device, dtype)
        ref_o = (x[..., :d].float() @ w_o[:, :d].T
                 + x[..., d:].float() @ w_q_rest.T).to(dtype)
    err_cols = (z_after - ref_o).abs().max().item()
    prot.clear()
    print(f"[selftest] q_proj row-restore err={err_rows:.2e} "
          f"untouched-dims drift={unchanged:.2e} o_proj col-delta err={err_cols:.2e}")
    assert err_rows < 5e-2 and err_cols < 5e-2, "protection hooks numerically wrong"
    print("[selftest] OK")


if __name__ == "__main__":
    main()
