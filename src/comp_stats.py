"""Aggregate the per-module mechanism logs written by
quantize_protected.py --stats-dir into (a) one summary row per run and
(b) a per-layer table for plotting. No pandas (keeps the pinned stack).

  python src/comp_stats.py --runs llama=runs/stats/llama31-8b \
      q7=runs/stats/qwen25-7b q14=runs/stats/qwen25-14b \
      --summary runs/comp_stats_summary.csv --per-layer runs/comp_stats_layers.csv

Summary columns (per run):
  n_modules, send_total, send_max_module, send_top1pct_mean,
  clip_frac_mean, clip_frac_max, clip_frac_rtn_mean,
  obj_ratio_c4  = sum obj_gptq / sum obj_rtn under the calibration Hessian
                  (<1 means GPTQ wins its own objective, as it should)
  obj_ratio_alt = same under the chat-format Hessian (if logged)
  n_alt_gptq_worse = #modules where GPTQ loses to RTN under the chat Hessian
  comp_disp_rel_mean, hdiag_max_over_mean_max, cond_max
The predictor question is whether any of these separates {llama, q14}
from the graceful set. Optional: --salience-dir + --budget to report the
overlap between top sender columns and the critical set's columns.
"""
import argparse
import csv
import os
from collections import defaultdict


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(path):
    with open(os.path.join(path, "stats.csv")) as f:
        return list(csv.DictReader(f))


def summarize(tag, rows):
    g = lambda k: [v for v in (fnum(r.get(k)) for r in rows) if v is not None]  # noqa: E731
    send = g("send_total")
    obj_g, obj_r = g("obj_gptq"), g("obj_rtn")
    obj_ga, obj_ra = g("obj_gptq_alt"), g("obj_rtn_alt")
    worse_alt = sum(1 for a, b in zip(obj_ga, obj_ra) if a > b)
    worse_c4 = sum(1 for a, b in zip(obj_g, obj_r) if a > b)
    clip = g("clip_frac")
    out = {
        "tag": tag, "n_modules": len(rows),
        "send_total": sum(send), "send_max_module": max(send) if send else "",
        "send_top1pct_mean": _mean(g("send_top1pct")),
        "clip_frac_mean": _mean(clip), "clip_frac_max": max(clip) if clip else "",
        "clip_frac_rtn_mean": _mean(g("clip_frac_rtn")),
        "obj_ratio_c4": (sum(obj_g) / sum(obj_r)) if obj_r and sum(obj_r) else "",
        "n_c4_gptq_worse": worse_c4,
        "obj_ratio_alt": (sum(obj_ga) / sum(obj_ra)) if obj_ra and sum(obj_ra) else "",
        "n_alt_gptq_worse": worse_alt if obj_ga else "",
        "comp_disp_rel_mean": _mean(g("comp_disp_rel")),
        "wdisp_ratio": (sum(g("wdisp_gptq")) / sum(g("wdisp_rtn"))) if g("wdisp_rtn") else "",
        "hdiag_max_over_mean_max": max(g("hdiag_max_over_mean") or [""]),
        "cond_max": max(g("cond_damped") or [""]),
    }
    return out


def _mean(v):
    return sum(v) / len(v) if v else ""


def per_layer(tag, rows):
    by = defaultdict(list)
    for r in rows:
        by[int(r["layer"])].append(r)
    out = []
    for l in sorted(by):
        rs = by[l]
        g = lambda k: [v for v in (fnum(r.get(k)) for r in rs) if v is not None]  # noqa: E731
        og, orr = g("obj_gptq"), g("obj_rtn")
        oga, ora = g("obj_gptq_alt"), g("obj_rtn_alt")
        out.append({
            "tag": tag, "layer": l,
            "send_total": sum(g("send_total")),
            "clip_frac_mean": _mean(g("clip_frac")),
            "obj_ratio_c4": (sum(og) / sum(orr)) if orr and sum(orr) else "",
            "obj_ratio_alt": (sum(oga) / sum(ora)) if ora and sum(ora) else "",
            "comp_disp_rel_mean": _mean(g("comp_disp_rel")),
            "worst_module": max(rs, key=lambda r: fnum(r.get("send_total")) or 0)["proj"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="tag=stats_dir ...")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--per-layer", required=True)
    args = ap.parse_args()

    summ, layers = [], []
    for spec in args.runs:
        tag, path = spec.split("=", 1)
        rows = load(path)
        summ.append(summarize(tag, rows))
        layers.extend(per_layer(tag, rows))

    for path, rows in ((args.summary, summ), (args.per_layer, layers)):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[comp_stats] -> {path} ({len(rows)} rows)")
    for r in summ:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
