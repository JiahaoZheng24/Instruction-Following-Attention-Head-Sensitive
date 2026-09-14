"""W68: matrix transplant — the causal test the derivation asks for.

The lesion is one weight matrix (the sink-forming down_proj quantised by GPTQ
under a BOS-dominated Hessian). Copy exactly that matrix from a DONOR
checkpoint into a TARGET model (fp16, RTN, or another GPTQ checkpoint), leave
everything else untouched, and generate the IFEval responses. No synthesis, no
assumption about the lesion's shape.

  python src/transplant.py --target meta-llama/Llama-3.1-8B-Instruct \
      --donor /store01/.../llama3.1-8b-v2gptq3-none --layer 1 --proj down_proj \
      --prompts data/ifeval_input_data.jsonl --tag tp_l_fp16_les
Responses -> runs/<basename(target)>/<tag>/responses.jsonl (score with src/score_ifeval.py).
"""
import argparse
import gc
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, read_jsonl, write_jsonl  # noqa: E402

MAX_NEW_TOKENS = 1280
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")


def chat(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


def get_linear(model, layer, proj):
    blk = model.model.layers[layer]
    return getattr(blk.self_attn, proj) if proj in ATTN else getattr(blk.mlp, proj)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="model id or checkpoint that receives the matrix")
    ap.add_argument("--donor", required=True, help="checkpoint the matrix is taken from")
    ap.add_argument("--layer", type=int, help="single layer (legacy; same as --layers L)")
    ap.add_argument("--layers", help="W69: layer or range, e.g. '2' or '2-3'")
    ap.add_argument("--proj", default="down_proj",
                    help="'down_proj' (default), 'all' (7 matrices), 'attn' (q,k,v,o), 'mlp' (gate,up,down) or a comma list")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    if args.layers:
        lo, _, hi = args.layers.partition("-")
        layers = list(range(int(lo), int(hi or lo) + 1))
    else:
        assert args.layer is not None, "--layer or --layers required"
        layers = [args.layer]
    projs = {"all": list(ATTN) + ["gate_proj", "up_proj", "down_proj"], "attn": list(ATTN),
             "mlp": ["gate_proj", "up_proj", "down_proj"]}.get(args.proj, args.proj.split(","))

    donor, _ = load_model(args.donor)
    Wd = {(L, p): get_linear(donor, L, p).weight.data.detach().clone().cpu() for L in layers for p in projs}
    del donor
    gc.collect()
    torch.cuda.empty_cache()

    model, tok = load_model(args.target)
    for (L, p), w in Wd.items():
        lin = get_linear(model, L, p)
        Wt = lin.weight.data
        assert Wt.shape == w.shape, (L, p, Wt.shape, w.shape)
        diff = (Wt.float().cpu() - w.float())
        print(f"[transplant] L{L}.{p}: ||W_target - W_donor||_F = {diff.norm():.4g} "
              f"(target ||W||_F {Wt.float().norm():.4g}); max|W| target {Wt.abs().max():.4g} donor {w.abs().max():.4g}; "
              f"rows with largest change: {diff.norm(dim=1).topk(3).indices.tolist()}", flush=True)
        lin.weight.data.copy_(w.to(Wt.device, Wt.dtype))
    args.layer = layers[0]
    args.proj = ",".join(projs) if len(projs) < 7 else "all"
    args.layers_str = args.layers or str(args.layer)

    prompts = read_jsonl(args.prompts)
    out_rows = []
    for i in range(0, len(prompts), args.batch):
        batch = prompts[i:i + args.batch]
        texts = [chat(tok, ex["prompt"]) for ex in batch]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id)
        for ex, seq in zip(batch, gen):
            out_rows.append({"prompt": ex["prompt"],
                             "response": tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)})
        print(f"[transplant:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    run_dir = os.path.join("runs", os.path.basename(args.target.rstrip("/")), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"target={args.target}\ndonor={args.donor}\nlayers={args.layers_str} proj={args.proj}\n")
    print(f"[transplant] {len(out_rows)} responses -> {run_dir}/responses.jsonl")


if __name__ == "__main__":
    main()
