
import argparse, os, json, glob
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    M = z["mean_layer_head"]     # [L,H]
    per_prompt = z["per_prompt"] # [P,L,H]
    meta = json.loads(str(z["meta"].tolist()))
    return M, per_prompt, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_glob", required=True, help='e.g., "artifacts/attn_*.npz"')
    ap.add_argument("--ifeval_csv", required=True, help="CSV combining FP16/GPTQ/AWQ avg4 per tag")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = glob.glob(args.attn_glob)
    runs = {}
    for f in files:
        M, P, meta = load_npz(f)
        runs[meta["run_tag"]] = {"M": M, "P": P, "meta": meta}

    assert "fp16" in runs, "Need a run with tag 'fp16' for baseline."
    base = runs["fp16"]["M"]   # [L,H]

    # Compute per-run ΔA = A_fp16 - A_run (positive means sensitivity reduced by quantization)
    rows = []
    for tag, obj in runs.items():
        M = obj["M"]
        delta = base - M
        # ISI = mean of positive deltas
        ISI = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0.0
        rows.append({"tag": tag, "ISI": ISI})

        # save per-layer/head csv
        L,H = M.shape
        grid = []
        for l in range(L):
            for h in range(H):
                grid.append({"tag": tag, "layer": l, "head": h, "A": M[l,h], "delta": delta[l,h]})
        pd.DataFrame(grid).to_csv(os.path.join(args.out_dir, f"per_head_{tag}.csv"), index=False)

    df_isi = pd.DataFrame(rows).sort_values("tag")
    df_isi.to_csv(os.path.join(args.out_dir, "isi_by_tag.csv"), index=False)

    # correlate ISI with IFEval avg4 drop
    df_scores = pd.read_csv(args.ifeval_csv)
    # normalize drop w.r.t fp16
    base_avg4 = float(df_scores.loc[df_scores["tag"]=="fp16", "avg4"].iloc[0])
    df_scores["drop"] = (base_avg4 - df_scores["avg4"]) / base_avg4

    df = pd.merge(df_isi, df_scores[["tag","avg4","drop"]], on="tag", how="left")

    # Exclude fp16 from correlation (drop=0 by def)
    sub = df[df["tag"]!="fp16"].dropna(subset=["ISI","drop"])
    if len(sub) >= 2:
        r, p = pearsonr(sub["ISI"], sub["drop"])
    else:
        r, p = (np.nan, np.nan)

    df["pearson_r_vs_drop"] = r
    df["pearson_p_vs_drop"] = p
    df.to_csv(os.path.join(args.out_dir, "summary_isi_vs_ifeval.csv"), index=False)
    print("Summary:\n", df)

if __name__ == "__main__":
    main()
