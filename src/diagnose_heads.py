"""Causal IF-head diagnosis via mean ablation (RQ1).

Stages:
  calib   Compute per-head mean vectors + activation salience on a calibration
          set (NOT the eval prompts). Saves calib_means.pt, act_salience.csv.
  screen  Per-head output deviation between the FP model and a quantized
          checkpoint on the same prompts (teacher-forced): mean L2 of the
          o_proj-input slice per head -> dev_ranking_<tag>.csv. Quantizer-
          agnostic screening signal (works for GPTQ/AWQ/RTN checkpoints).
  ablate  Generate IFEval responses under a head-ablation configuration.
          Output responses.jsonl is compatible with the official IFEval
          evaluator (google-research/instruction_following_eval).

Head-set selection for `ablate` (exactly one of):
  --heads "L:H,L:H"          explicit set
  --topk-from FILE --k K     top-K heads from a ranking CSV (cols: layer,head,score)
  --random K --seed S        K random heads (control condition)
  --layer L                  all heads of one layer (group ablation)

Protocol constants (fixed 2026-08, do not tune post-hoc):
  greedy decoding, max_new_tokens=1280, chat template, bf16.
  Screening set = 100 stratified IFEval prompts; validation = full 541.

Example:
  python src/diagnose_heads.py calib  --calib-file data/calib_prompts.jsonl
  python src/diagnose_heads.py ablate --prompts data/ifeval_input_data.jsonl \
      --topk-from runs/screen_ranking.csv --k 32 --tag top32
  python src/diagnose_heads.py ablate --prompts data/ifeval_input_data.jsonl \
      --random 32 --seed 0 --tag rand32_s0
  # baseline (no ablation):
  python src/diagnose_heads.py ablate --prompts data/ifeval_input_data.jsonl --tag baseline
Then score every runs/<model>/<tag>/responses.jsonl with the official checker
(see README).
"""
import argparse
import csv
import os
import random

import torch

from common import (DEFAULT_MODEL, HeadAblator, HeadGeom, MeanCapture,
                    load_model, o_proj, parse_heads, read_jsonl, write_jsonl)

MAX_NEW_TOKENS = 1280


def chat(tok, text):
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
    )


def stage_calib(args):
    model, tok = load_model(args.model)
    geom = HeadGeom(model)
    cap = MeanCapture(model, geom)
    prompts = read_jsonl(args.calib_file)
    with torch.no_grad():
        for ex in prompts:
            ids = tok(chat(tok, ex.get("prompt") or ex["text"]), return_tensors="pt",
                      truncation=True, max_length=2048).to(model.device)
            model(**ids)
    means, act_sal = cap.finalize()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save({"means": means, "model": args.model}, os.path.join(args.out_dir, "calib_means.pt"))
    with open(os.path.join(args.out_dir, "act_salience.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "score"])
        for l in range(geom.n_layers):
            for h in range(geom.n_heads):
                w.writerow([l, h, float(act_sal[l, h])])
    print(f"[calib] saved means {tuple(means.shape)} and act_salience.csv -> {args.out_dir}")


def stage_screen(args):
    """Per-head FP-vs-quantized output deviation (screening-grade ranking)."""
    fp_model, tok = load_model(args.model)
    q_model, _ = load_model(args.quant_model)  # needs gptqmodel/auto-gptq or awq in env
    geom = HeadGeom(fp_model)

    def capture(model, store):
        handles = []
        for l in range(geom.n_layers):
            def hook(_m, a, layer=l):
                store[layer] = a[0].detach().float()
            handles.append(o_proj(model, l).register_forward_pre_hook(hook))
        return handles

    dev = torch.zeros(geom.n_layers, geom.n_heads, dtype=torch.float64)
    n_tok = 0
    prompts = read_jsonl(args.prompts)
    with torch.no_grad():
        for ex in prompts:
            ids = tok(chat(tok, ex.get("prompt") or ex["text"]), return_tensors="pt",
                      truncation=True, max_length=2048).to(fp_model.device)
            fp_store, q_store = {}, {}
            h1 = capture(fp_model, fp_store); fp_model(**ids); [h.remove() for h in h1]
            ids_q = {k: v.to(q_model.device) for k, v in ids.items()}
            h2 = capture(q_model, q_store); q_model(**ids_q); [h.remove() for h in h2]
            t = fp_store[0].shape[1]
            n_tok += t
            for l in range(geom.n_layers):
                d = (fp_store[l].cpu() - q_store[l].cpu()).reshape(
                    t, geom.n_heads, geom.head_dim)
                dev[l] += d.norm(dim=-1).to(torch.float64).sum(dim=0)  # L2 per head/token

    dev /= max(n_tok, 1)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"dev_ranking_{args.tag}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "head", "score"])
        for l in range(geom.n_layers):
            for h in range(geom.n_heads):
                w.writerow([l, h, float(dev[l, h])])
    print(f"[screen] {len(prompts)} prompts, {n_tok} tokens -> {out}")


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
        return [(args.layer, h) for h in range(geom.n_heads)]
    return []  # baseline


def stage_ablate(args):
    model, tok = load_model(args.model)
    geom = HeadGeom(model)
    heads = pick_heads(args, geom)

    ablator = None
    if heads:
        means = torch.load(os.path.join(args.out_dir, "calib_means.pt"))["means"]
        ablator = HeadAblator(model, geom, means)
        ablator.set_heads(heads)

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
            print(f"[ablate:{args.tag}] {min(i + args.batch, len(prompts))}/{len(prompts)}", flush=True)
    if ablator:
        ablator.clear()

    run_dir = os.path.join("runs", os.path.basename(args.model), args.tag)
    os.makedirs(run_dir, exist_ok=True)
    write_jsonl(os.path.join(run_dir, "responses.jsonl"), out_rows)
    with open(os.path.join(run_dir, "config.txt"), "w") as f:
        f.write(f"model={args.model}\nheads={heads}\nn={len(heads)}\n")
    print(f"[ablate] {len(out_rows)} responses -> {run_dir}/responses.jsonl "
          f"({len(heads)} heads ablated)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["calib", "screen", "ablate"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--quant-model", help="screen: path/id of quantized checkpoint")
    ap.add_argument("--out-dir", default="runs/diag")
    ap.add_argument("--calib-file")
    ap.add_argument("--prompts")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--heads")
    ap.add_argument("--topk-from")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--random", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int)
    args = ap.parse_args()
    if args.stage == "calib":
        assert args.calib_file, "--calib-file required"
        stage_calib(args)
    elif args.stage == "screen":
        assert args.quant_model and args.prompts, "--quant-model and --prompts required"
        stage_screen(args)
    else:
        assert args.prompts, "--prompts required"
        stage_ablate(args)


if __name__ == "__main__":
    main()
