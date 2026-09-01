"""Anatomy of the critical set: is it channel-structured, activation-aligned,
and task-general?

Given a salience dir and a budget, builds the mask and reports:
  1. per-module-type totals (where the mask lives)
  2. CHANNEL CONCENTRATION: fraction of selected weights falling in the top-1%
     of input channels (columns), per module type. High concentration =>
     column-structured protection is deployable (OWQ-style storage).
  3. (--model) overlap between top mask-channels and top ACTIVATION-magnitude
     channels (one forward pass) => is the critical set the massive-activation
     structure the outlier literature protects?
  4. (--compare-dir) Jaccard of the mask vs a second salience's mask at the
     same budget => task-generality (IF-conditioned vs c4-conditioned).

  python src/critical_anatomy.py --salience-dir $STORE/salience/llama31-8b-if \
      --budget-params 100000 --model meta-llama/Llama-3.1-8B-Instruct \
      --out runs/critical_anatomy_llama.csv
"""
import argparse
import csv
import os
import re

import torch

PROMPT = ("Summer is warm. Winter is cold. The capital of France is Paris. "
          "Apple pie is a traditional dessert.")


def module_files(d):
    return sorted(f for f in os.listdir(d) if f.endswith(".pt") and f != "sample.pt")


def threshold_for(d, budget):
    sample = torch.load(os.path.join(d, "sample.pt"), map_location="cpu").float()
    total = sum(torch.load(os.path.join(d, f), map_location="cpu").numel()
                for f in module_files(d))
    q = 1.0 - budget / total
    from common import safe_quantile
    return safe_quantile(sample, q), total


@torch.no_grad()
def activation_channel_scan(model_id):
    """max |input| per channel for every target linear, one forward pass."""
    from common import load_model
    model, tok = load_model(model_id)
    acts = {}

    def make(name):
        def hook(_m, args):
            x = args[0].reshape(-1, args[0].shape[-1]).abs().amax(dim=0)
            prev = acts.get(name)
            acts[name] = x if prev is None else torch.maximum(prev, x)
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers" in name:
            handles.append(mod.register_forward_pre_hook(make(name)))
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    model(**ids)
    for h in handles:
        h.remove()
    out = {k: v.float().cpu() for k, v in acts.items()}
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salience-dir", required=True)
    ap.add_argument("--budget-params", type=int, required=True)
    ap.add_argument("--model", help="optional: activation-channel overlap scan")
    ap.add_argument("--compare-dir", help="optional: second salience dir (task-generality)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    thr, total = threshold_for(args.salience_dir, args.budget_params)
    print(f"[anatomy] threshold={thr:.3e} (budget {args.budget_params} of {total/1e9:.2f}B)")
    acts = activation_channel_scan(args.model) if args.model else {}
    thr2 = None
    if args.compare_dir:
        thr2, _ = threshold_for(args.compare_dir, args.budget_params)

    rows, agg = [], {}
    n_sel_tot = 0
    for f in module_files(args.salience_dir):
        name = f[:-3]
        sal = torch.load(os.path.join(args.salience_dir, f), map_location="cpu").float()
        m = sal > thr
        n_sel = int(m.sum())
        n_sel_tot += n_sel
        if n_sel == 0:
            continue
        proj = re.search(r"\.(\w+_proj)$", name).group(1)
        col_counts = m.sum(dim=0)                       # per input channel
        n_cols = col_counts.numel()
        k1 = max(1, n_cols // 100)                      # top-1% of channels
        top_cols = torch.topk(col_counts, k1).indices
        conc = float(col_counts[top_cols].sum()) / n_sel
        rec = {"module": name, "proj": proj, "selected": n_sel,
               "col_conc_top1pct": round(conc, 4)}
        if name in acts:
            a_top = torch.topk(acts[name], k1).indices
            inter = len(set(top_cols.tolist()) & set(a_top.tolist()))
            rec["act_overlap_top1pct"] = round(inter / k1, 4)
        if thr2 is not None:
            sal2 = torch.load(os.path.join(args.compare_dir, f), map_location="cpu").float()
            m2 = sal2 > thr2
            inter = int((m & m2).sum())
            union = int((m | m2).sum())
            rec["mask_jaccard_vs_compare"] = round(inter / max(union, 1), 4)
        rows.append(rec)
        a = agg.setdefault(proj, {"selected": 0, "conc_w": 0.0, "ovl_w": 0.0, "jac_w": 0.0})
        a["selected"] += n_sel
        a["conc_w"] += conc * n_sel
        a["ovl_w"] += rec.get("act_overlap_top1pct", 0) * n_sel
        a["jac_w"] += rec.get("mask_jaccard_vs_compare", 0) * n_sel

    fields = sorted({k for r in rows for k in r}, key=lambda k: k != "module")
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"[anatomy] total selected {n_sel_tot/1e3:.1f}K; by module type "
          f"(weighted col-concentration | act-overlap | jaccard):")
    for proj, a in sorted(agg.items(), key=lambda kv: -kv[1]["selected"]):
        s = a["selected"]
        print(f"  {proj:12s} {s/1e3:8.1f}K  conc={a['conc_w']/s:.3f}  "
              f"act_ovl={a['ovl_w']/s:.3f}  jac={a['jac_w']/s:.3f}")
    print(f"[anatomy] -> {args.out}")
    print("read-out: conc>~0.5 => channel-structured (column protection deployable); "
          "act_ovl high => critical set = massive-activation structure; "
          "jaccard high => task-general (model-critical, not task-critical).")


if __name__ == "__main__":
    main()
