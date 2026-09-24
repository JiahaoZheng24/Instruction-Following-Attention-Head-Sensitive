"""W82: composition of each calibration draw. For every protocol / seed, the share
of tokens that are newline-only, special, or document-initial (position 0 or 1),
and the number of samples; the candidate calibration-side predictor of collapse
under the co-adaptation hypothesis (RESULTS 9.10bu).

  python src/calib_composition.py --model meta-llama/Llama-3.1-8B-Instruct \
      --arms "c4:0,c4:1,c4:2,c4:3,c4:4,c4win:0,pile:0,wikitext:0,pileshort:0" --out runs/theory/f_l_calib_composition.csv
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inject_sink import token_set  # noqa: E402
from quantize_gptq import load_calib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", required=True, help="comma list of calib:seed")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    nl = token_set(tok, "newline")
    sp = set(int(i) for i in tok.all_special_ids)
    rows = []
    for arm in args.arms.split(","):
        calib, seed = arm.split(":")
        texts = load_calib(calib, tok, args.n_calib, args.seqlen, seed=int(seed))
        n_tok = n_nl = n_sp = n_first = 0
        for t in texts:
            ids = tok(t, truncation=True, max_length=args.seqlen)["input_ids"]
            n_tok += len(ids)
            n_nl += sum(int(i) in nl for i in ids)
            n_sp += sum(int(i) in sp for i in ids)
            n_first += min(2, len(ids))
        rows.append({"calib": calib, "seed": int(seed), "n_samples": len(texts), "n_tokens": n_tok,
                     "mean_len": n_tok / max(len(texts), 1),
                     "newline_share": n_nl / max(n_tok, 1), "special_share": n_sp / max(n_tok, 1),
                     "doc_initial_share": n_first / max(n_tok, 1),
                     "secondary_sink_share": (n_nl + n_sp + n_first) / max(n_tok, 1)})
        print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"[composition] -> {args.out}")


if __name__ == "__main__":
    main()
