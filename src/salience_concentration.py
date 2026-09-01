"""Salience concentration index: the candidate regime predictor.

Activation-outlier statistics failed to predict collapse (Qwen's spikes are
30x Llama's, yet Qwen is graceful). Candidate replacement: how CONCENTRATED
is the salience mass in weight space? Prediction: collapse-prone models
concentrate a large share of total salience |W|*|grad|*|dW| in a tiny set
(Llama >> Qwen/Mistral at the 1e5 scale).

  python src/salience_concentration.py \
      --dirs qwen=$STORE/salience/qwen25-7b-if llama=$STORE/salience/llama31-8b-if \
             mistral=$STORE/salience/mistral7b-if \
      --out runs/salience_concentration.csv
"""
import argparse
import csv
import os

import torch

from critical_anatomy import module_files

BUDGETS = [10_000, 100_000, 1_000_000, 10_000_000]


def analyze(d):
    sample = torch.load(os.path.join(d, "sample.pt"), map_location="cpu").float()
    total_n = 0
    total_mass = 0.0
    for f in module_files(d):
        sal = torch.load(os.path.join(d, f), map_location="cpu").float()
        total_n += sal.numel()
        total_mass += float(sal.sum())
    out = {"total_params_B": round(total_n / 1e9, 2)}
    for b in BUDGETS:
        q = 1.0 - b / total_n
        from common import safe_quantile
        thr = safe_quantile(sample, q)
        mass = 0.0
        for f in module_files(d):
            sal = torch.load(os.path.join(d, f), map_location="cpu").float()
            mass += float(sal[sal > thr].sum())
        out[f"mass_share_top{b:.0e}"] = round(mass / total_mass, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for spec in args.dirs:
        name, path = spec.split("=", 1)
        print(f"[conc] analyzing {name} ({path})")
        rec = {"model": name, **analyze(path)}
        print("  ", rec)
        rows.append(rec)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[conc] -> {args.out}")


if __name__ == "__main__":
    main()
