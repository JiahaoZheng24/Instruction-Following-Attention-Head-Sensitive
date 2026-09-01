"""Mechanism experiment: post-hoc SCATTERED restoration (the missing cell).

In-loop scattered protection helps (~+3 on Qwen 3-bit) regardless of
selection criterion; hypothesis: the benefit is not the weights' identity but
their role as free fp parameters that GPTQ's compensation distributes error
into ("error absorbers"). Prediction: the SAME scattered restoration applied
POST-HOC (on the already-quantized fake-quant checkpoint, no compensation
interaction) gives ~nothing.

Works on a v2 fake-quant checkpoint (plain HF format): loads it, overwrites a
random scattered subset of weight entries with the original fp values,
generates IFEval responses in-process (no 15GB checkpoint written).

  python src/posthoc_scatter.py --quant-ckpt $STORE/models/qwen2.5-7b-v2gptq3-none \
      --model Qwen/Qwen2.5-7B-Instruct --budget-params 37624064 --seed 0 \
      --prompts data/ifeval_input_data.jsonl --tag ph_scatter_s0
"""
import argparse
import os

import torch

from common import load_model, read_jsonl, write_jsonl
from protect_eval import FPSource

MAX_NEW_TOKENS = 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quant-ckpt", required=True, help="v2 fake-quant dir")
    ap.add_argument("--model", required=True, help="fp weight source")
    ap.add_argument("--budget-params", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--salience-dir",
                    help="restore the CRITICAL set (global salience threshold) "
                         "instead of a random scatter — the mechanism "
                         "discriminator on a collapse checkpoint: if post-hoc "
                         "restoration rescues, collapse = corrupted critical "
                         "weights; if not, collapse = compensation damage "
                         "propagated into the REST of the network")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model, tok = load_model(args.quant_ckpt)
    fp = FPSource(args.model)
    total = sum(p.numel() for n, p in model.named_parameters()
                if "layers" in n and p.dim() == 2)
    frac = args.budget_params / total
    if args.salience_dir:
        from critical_anatomy import threshold_for
        thr, _ = threshold_for(args.salience_dir, args.budget_params)
    restored = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "layers" not in name or p.dim() != 2:
                continue
            if args.salience_dir:
                f = os.path.join(args.salience_dir,
                                 name.replace(".weight", "") + ".pt")
                if not os.path.exists(f):
                    continue
                m = torch.load(f, map_location="cpu").float() > thr
            else:
                g = torch.Generator().manual_seed(
                    args.seed * 100003 + hash(name) % 65521)
                m = torch.rand(p.shape, generator=g) < frac
            w_fp = fp.get(name).to(p.dtype)
            p.data[m.to(p.device)] = w_fp[m].to(p.device, p.dtype)
            restored += int(m.sum())
    print(f"[ph-scatter] restored {restored / 1e6:.1f}M scattered entries "
          f"({100 * restored / total:.3f}%)")

    prompts = read_jsonl(args.prompts)
    out_rows = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            batch = prompts[i:i + args.batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                             tokenize=False, add_generation_prompt=True)
                     for ex in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                                 pad_token_id=tok.pad_token_id)
            for ex, seq in zip(batch, gen):
                resp = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                out_rows.append({"prompt": ex["prompt"], "response": resp})
            print(f"[ph-scatter:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}",
                  flush=True)

    run_dir = os.path.join("runs", os.path.basename(args.quant_ckpt.rstrip("/")), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"quant_ckpt={args.quant_ckpt}\nbudget={args.budget_params}\n"
                f"seed={args.seed}\nrestored={restored}\nmode=posthoc_scatter\n")
    print(f"[ph-scatter] {len(out_rows)} responses -> {run_dir}/responses.jsonl")


if __name__ == "__main__":
    main()
