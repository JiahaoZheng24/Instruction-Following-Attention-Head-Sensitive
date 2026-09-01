"""Where do the TaCQ-salient weights live? Module/layer anatomy of the mask.

Reads the salience dir, applies the same global-quantile threshold as
quantize_protected.py, and reports the selected-weight distribution:
per module-type totals and a layer x module-type count matrix.

  python src/mask_anatomy.py --salience-dir $STORE/salience/qwen25-7b-if \
      --budget-params 37624064 --out runs/mask_anatomy.csv
"""
import argparse
import csv
import os
import re

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salience-dir", required=True)
    ap.add_argument("--budget-params", type=int, default=37_624_064)
    ap.add_argument("--out", default="runs/mask_anatomy.csv")
    args = ap.parse_args()

    files = [f for f in os.listdir(args.salience_dir)
             if f.endswith(".pt") and f != "sample.pt"]
    total = 0
    sizes = {}
    for f in files:
        sal = torch.load(os.path.join(args.salience_dir, f), map_location="cpu")
        sizes[f] = sal.numel()
        total += sal.numel()
    sample = torch.load(os.path.join(args.salience_dir, "sample.pt"),
                        map_location="cpu").float()
    q = 1.0 - args.budget_params / total
    from common import safe_quantile
    thr = safe_quantile(sample, q)
    print(f"[anatomy] total={total/1e9:.2f}B threshold={thr:.3e} (q={q:.5f})")

    rows = []
    by_proj: dict[str, int] = {}
    for f in sorted(files):
        m = re.match(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.pt", f)
        layer, block, proj = int(m.group(1)), m.group(2), m.group(3)
        sal = torch.load(os.path.join(args.salience_dir, f), map_location="cpu")
        n_sel = int((sal.float() > thr).sum())
        rows.append({"layer": layer, "proj": proj, "numel": sal.numel(),
                     "selected": n_sel, "frac": n_sel / sal.numel()})
        by_proj[proj] = by_proj.get(proj, 0) + n_sel

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["layer", "proj", "numel", "selected", "frac"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["layer"], r["proj"])))

    n_total_sel = sum(by_proj.values())
    print(f"[anatomy] selected {n_total_sel/1e6:.1f}M weights; by module type:")
    for proj, n in sorted(by_proj.items(), key=lambda kv: -kv[1]):
        print(f"  {proj:12s} {n/1e6:7.2f}M  ({100*n/n_total_sel:5.1f}% of mask)")
    print(f"[anatomy] -> {args.out}")


if __name__ == "__main__":
    main()
