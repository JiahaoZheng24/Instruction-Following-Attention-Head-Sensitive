"""Mechanism check: does GPTQ displace the critical weights more than RTN?

The compensation-catastrophe hypothesis predicts: on the critical set, the
quantized-minus-fp displacement |W_q - W_fp| is much larger under GPTQ
(which dumps other columns' error into remaining weights) than under RTN
(pure rounding, bounded by half a grid step) — while on random weights the
two are comparable.

Compares two fake-quant checkpoints (plain HF dirs) against the fp source.

  python src/weight_displacement.py --fp-model meta-llama/Llama-3.1-8B-Instruct \
      --ckpt-gptq $STORE/models/llama3.1-8b-v2gptq3-none \
      --ckpt-rtn  $STORE/models/llama3.1-8b-rtn3-none \
      --salience-dir $STORE/salience/llama31-8b-if --budget-params 100000 \
      --out runs/weight_displacement_llama.csv
"""
import argparse
import csv
import os

import torch

from critical_anatomy import threshold_for, module_files
from protect_eval import FPSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-model", required=True)
    ap.add_argument("--ckpt-gptq", required=True)
    ap.add_argument("--ckpt-rtn", required=True)
    ap.add_argument("--salience-dir", required=True)
    ap.add_argument("--budget-params", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fp = FPSource(args.fp_model)
    gq = FPSource(args.ckpt_gptq)
    rt = FPSource(args.ckpt_rtn)
    thr, _ = threshold_for(args.salience_dir, args.budget_params)

    agg = {"crit_gptq": 0.0, "crit_rtn": 0.0, "rand_gptq": 0.0, "rand_rtn": 0.0,
           "n_crit": 0, "n_rand": 0}
    rows = []
    g = torch.Generator().manual_seed(0)
    for f in module_files(args.salience_dir):
        name = f[:-3] + ".weight"
        sal = torch.load(os.path.join(args.salience_dir, f), map_location="cpu").float()
        m = sal > thr
        n = int(m.sum())
        if n == 0:
            continue
        w_fp = fp.get(name).float()
        d_g = (gq.get(name).float() - w_fp).abs()
        d_r = (rt.get(name).float() - w_fp).abs()
        rand = torch.zeros_like(m)
        idx = torch.randint(0, m.numel(), (n,), generator=g)
        rand.view(-1)[idx] = True
        rec = {"module": f[:-3], "n_crit": n,
               "crit_disp_gptq": float(d_g[m].mean()),
               "crit_disp_rtn": float(d_r[m].mean()),
               "rand_disp_gptq": float(d_g[rand].mean()),
               "rand_disp_rtn": float(d_r[rand].mean())}
        rows.append(rec)
        agg["crit_gptq"] += float(d_g[m].sum())
        agg["crit_rtn"] += float(d_r[m].sum())
        agg["rand_gptq"] += float(d_g[rand].sum())
        agg["rand_rtn"] += float(d_r[rand].sum())
        agg["n_crit"] += n
        agg["n_rand"] += n

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cg = agg["crit_gptq"] / agg["n_crit"]
    cr = agg["crit_rtn"] / agg["n_crit"]
    rg = agg["rand_gptq"] / agg["n_rand"]
    rr = agg["rand_rtn"] / agg["n_rand"]
    print(f"[displacement] mean |W_q - W_fp|:")
    print(f"  critical set : GPTQ={cg:.5f}  RTN={cr:.5f}  ratio={cg / max(cr, 1e-12):.2f}x")
    print(f"  random ctrl  : GPTQ={rg:.5f}  RTN={rr:.5f}  ratio={rg / max(rr, 1e-12):.2f}x")
    print(f"  interaction (crit ratio / rand ratio): "
          f"{(cg / max(cr, 1e-12)) / max(rg / max(rr, 1e-12), 1e-12):.2f}x")
    print(f"[displacement] -> {args.out}")


if __name__ == "__main__":
    main()
