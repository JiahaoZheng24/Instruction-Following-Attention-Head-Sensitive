"""Compare two HF checkpoints of the same architecture module by module
(CPU, safetensors, no model instantiation). Used for the Mistral damping
forensics: the layer-wise objectives look normal at rho=0.5 yet the model
emits byte garbage, so look at the weights themselves.

Per 2-D weight: relative Frobenius distance, max |diff|, fraction of exact
zeros in each, number of all-zero rows / columns, max |W|. Prints the 15
most-different modules and writes the full table.

  python src/ckpt_diff.py --a $STORE/models/mistral-7b-v2gptq3-none \
      --b $STORE/models/mistral-7b-v2gptq3-damp0p5-keep --out runs/ckpt_diff_mistral.csv
"""
import argparse
import csv
import glob
import os

import torch
from safetensors import safe_open


def load_sd(path):
    sd = {}
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    assert files, f"no safetensors under {path}"
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                sd[k] = h.get_tensor(k)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A, B = load_sd(args.a), load_sd(args.b)
    rows = []
    for k in sorted(A):
        if k not in B or A[k].dim() != 2 or "layers" not in k:
            continue
        a, b = A[k].float(), B[k].float()
        d = a - b
        rows.append({
            "module": k,
            "rel_fro": float(d.norm() / a.norm().clamp(min=1e-12)),
            "max_absdiff": float(d.abs().max()),
            "maxW_a": float(a.abs().max()), "maxW_b": float(b.abs().max()),
            "zero_frac_a": float((a == 0).float().mean()), "zero_frac_b": float((b == 0).float().mean()),
            "zero_rows_a": int((a.abs().sum(1) == 0).sum()), "zero_rows_b": int((b.abs().sum(1) == 0).sum()),
            "zero_cols_a": int((a.abs().sum(0) == 0).sum()), "zero_cols_b": int((b.abs().sum(0) == 0).sum()),
            "nonfinite_b": int((~torch.isfinite(b)).sum()),
        })
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ckpt_diff] {len(rows)} modules -> {args.out}")
    for r in sorted(rows, key=lambda r: -r["rel_fro"])[:15]:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
