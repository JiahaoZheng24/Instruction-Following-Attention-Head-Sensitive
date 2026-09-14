"""Consolidate every per-arm score file into one table so results can be
looked up from a single file.

  python src/build_index.py

Reads  runs/scores_*.csv, runs_archive/scores/scores_*.csv   (IFEval avg4)
       runs/general_*.csv, runs_archive/scores/general_*.csv (MMLU / PPL)
       runs/scores_gsm8k.csv                                  (GSM8K)
Writes runs/INDEX_scores.csv   one row per tag: IFEval metrics, MMLU, PPL, GSM8K, source file
       runs/INDEX_stats.csv    one row per stats dir: n modules, detector max (module, value),
                               BOS/sink dominance at the early down_proj
Newer files win when a tag appears twice.
"""
import csv
import glob
import os


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    rows = {}
    files = sorted(glob.glob("runs/scores_*.csv") + glob.glob("runs_archive/scores/scores_*.csv"), key=os.path.getmtime)
    for f in files:
        if os.path.basename(f) == "scores_gsm8k.csv":
            continue
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if "avg4" in r and r.get("tag"):
                rows[r["tag"]] = {"tag": r["tag"], "prompt_strict": r.get("prompt_strict", ""), "inst_strict": r.get("inst_strict", ""),
                                  "prompt_loose": r.get("prompt_loose", ""), "inst_loose": r.get("inst_loose", ""), "avg4": r["avg4"],
                                  "mmlu": "", "ppl": "", "gsm8k": "", "source": f}
    for f in sorted(glob.glob("runs/general_*.csv") + glob.glob("runs_archive/scores/general_*.csv"), key=os.path.getmtime):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            t = r.get("tag")
            if not t:
                continue
            rows.setdefault(t, {"tag": t, "prompt_strict": "", "inst_strict": "", "prompt_loose": "", "inst_loose": "", "avg4": "", "mmlu": "", "ppl": "", "gsm8k": "", "source": f})
            rows[t]["mmlu"] = r.get("mmlu_acc", ""); rows[t]["ppl"] = r.get("wikitext2_ppl", "")
    for f in sorted(glob.glob("runs_archive/scores/scores_gsm8k*.csv") + glob.glob("runs/scores_gsm8k.csv"), key=os.path.getmtime):
        if os.path.exists(f):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                t = r.get("tag")
                if t:
                    rows.setdefault(t, {"tag": t, "prompt_strict": "", "inst_strict": "", "prompt_loose": "", "inst_loose": "", "avg4": "", "mmlu": "", "ppl": "", "gsm8k": "", "source": f})
                    rows[t]["gsm8k"] = r.get("accuracy", "")
    out = "runs/INDEX_scores.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "avg4", "prompt_strict", "inst_strict", "prompt_loose", "inst_loose", "mmlu", "ppl", "gsm8k", "source"])
        w.writeheader()
        for t in sorted(rows):
            w.writerow(rows[t])
    print(f"[index] {len(rows)} tags -> {out}")

    srows = []
    for sd in sorted(glob.glob("runs/stats/*/")):
        f = os.path.join(sd, "stats.csv")
        if not os.path.exists(f):
            continue
        R = list(csv.DictReader(open(f, encoding="utf-8")))
        best = ("", float("nan")); dom = ("", float("nan")); alt = ("", float("nan"))
        for r in R:
            if r.get("tplpos_ratio"):
                pr = [fl(v) for v in r["tplpos_ratio"].split(",")]
                m = max(pr[2:]) if len(pr) >= 8 else float("nan")
                if m == m and (best[1] != best[1] or m > best[1]):
                    best = (f"L{r['layer']}.{r['proj']}", m)
            if r.get("xnormpos") and r["proj"] == "down_proj" and int(r["layer"]) <= 6:
                xn = [fl(v) for v in r["xnormpos"].split(",")]; xo = fl(r.get("xnorm_ord", ""))
                d = max(xn) / xo if xo and xo == xo and xo > 0 else float("nan")
                if d == d and (dom[1] != dom[1] or d > dom[1]):
                    dom = (f"L{r['layer']}", d)
            if r.get("obj_gptq_alt") and r.get("obj_rtn_alt"):
                a = fl(r["obj_gptq_alt"]) / max(fl(r["obj_rtn_alt"]), 1e-12)
                if a == a and (alt[1] != alt[1] or a > alt[1]):
                    alt = (f"L{r['layer']}.{r['proj']}", a)
        srows.append({"stats_dir": sd.rstrip("/\\"), "n_modules": len(R), "detector_max_module": best[0], "detector_max": f"{best[1]:.1f}" if best[1] == best[1] else "",
                      "sink_dominance_layer": dom[0], "sink_dominance": f"{dom[1]:.0f}" if dom[1] == dom[1] else "",
                      "tokenavg_max_module": alt[0], "tokenavg_max": f"{alt[1]:.1f}" if alt[1] == alt[1] else "",
                      "has_abs_err_cols": int(any(r.get("tplpos_err_gptq") for r in R))})
    out2 = "runs/INDEX_stats.csv"
    with open(out2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()) if srows else ["stats_dir"])
        w.writeheader(); w.writerows(srows)
    print(f"[index] {len(srows)} stats dirs -> {out2}")


if __name__ == "__main__":
    main()
