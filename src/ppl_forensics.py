"""PPL forensics: is a perplexity explosion driven by a few catastrophic
positions (miscalibrated tails; argmax intact -> generation survives) or by
uniform likelihood degradation?

Computes per-token NLL over wikitext-2 windows and reports the distribution:
mean/median NLL, PPL, share of tokens with NLL > {5, 10, 20}, PPL after
excluding the worst 0.1% / 1% of tokens, and top-1 agreement between the
model's argmax and the actual next token.

  python src/ppl_forensics.py --model <ckpt> --tag l32_gptq3 \
      --out runs/ppl_forensics.csv
"""
import argparse
import csv
import os

import torch

from common import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n-windows", type=int, default=40)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    model, tok = load_model(args.model)
    ids = tok(text, return_tensors="pt").input_ids[0]

    nlls = []
    top1 = 0
    total = 0
    with torch.no_grad():
        for w in range(args.n_windows):
            s = w * args.seqlen
            if s + args.seqlen + 1 > ids.numel():
                break
            chunk = ids[s:s + args.seqlen + 1].unsqueeze(0).to(model.device)
            logits = model(chunk[:, :-1]).logits.float()
            targets = chunk[:, 1:]
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nlls.append(nll.flatten().cpu())
            top1 += int((logits.argmax(-1) == targets).sum())
            total += targets.numel()
            print(f"[pplf:{args.tag}] window {w + 1}: "
                  f"mean_nll={float(nll.mean()):.3f}", flush=True)

    nll = torch.cat(nlls)
    srt = nll.sort().values

    def ppl_excl(frac):
        k = int(nll.numel() * (1 - frac))
        return float(srt[:k].mean().exp())

    row = {
        "tag": args.tag,
        "n_tokens": nll.numel(),
        "ppl": round(float(nll.mean().exp()), 2),
        "median_nll": round(float(nll.median()), 3),
        "mean_nll": round(float(nll.mean()), 3),
        "pct_nll_gt5": round(float((nll > 5).float().mean()) * 100, 2),
        "pct_nll_gt10": round(float((nll > 10).float().mean()) * 100, 3),
        "pct_nll_gt20": round(float((nll > 20).float().mean()) * 100, 4),
        "ppl_excl_worst_0.1pct": round(ppl_excl(0.001), 2),
        "ppl_excl_worst_1pct": round(ppl_excl(0.01), 2),
        "top1_agreement": round(top1 / total, 4),
    }
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[pplf] {row}")


if __name__ == "__main__":
    main()
