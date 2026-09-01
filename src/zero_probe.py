"""Pruning probe: zero the critical set IN THE FP16 MODEL (no quantization).

Distinguishes two readings of the critical set:
  - collapses at fp16 too  => general model criticality (extended super-weight
    structure; quantization merely one way to break it)
  - fp16 tolerates zeroing => quantization-SPECIFIC fragility (the set only
    becomes life-or-death under low-bit noise) — the subtler finding.

  python src/zero_probe.py --model meta-llama/Llama-3.1-8B-Instruct \
      --salience-dir $STORE/salience/llama31-8b-if --budget-params 100000 \
      --prompts data/ifeval_input_data.jsonl --tag zero1e5_llama
"""
import argparse
import os

import torch

from common import load_model, read_jsonl, write_jsonl
from critical_anatomy import threshold_for

MAX_NEW_TOKENS = 1280


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--salience-dir", required=True)
    ap.add_argument("--budget-params", type=int, required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    thr, _ = threshold_for(args.salience_dir, args.budget_params)
    zeroed = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "layers" not in name or p.dim() != 2:
                continue
            f = os.path.join(args.salience_dir, name.replace(".weight", "") + ".pt")
            if not os.path.exists(f):
                continue
            m = (torch.load(f, map_location="cpu").float() > thr).to(p.device)
            p.data[m] = 0.0
            zeroed += int(m.sum())
    print(f"[zero-probe] zeroed {zeroed/1e3:.1f}K critical weights at fp16")

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
            print(f"[zero-probe:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}",
                  flush=True)

    run_dir = os.path.join("runs", os.path.basename(args.model.rstrip("/")), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"model={args.model}\nbudget={args.budget_params}\nzeroed={zeroed}\n"
                f"mode=fp16_zero_probe\nsalience={args.salience_dir}\n")
    print(f"[zero-probe] {len(out_rows)} responses -> {run_dir}/responses.jsonl")


if __name__ == "__main__":
    main()
