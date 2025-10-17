
import argparse, os, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    M = z["mean_layer_head"]     # [L,H]
    meta = json.loads(str(z["meta"].tolist()))
    return M, meta

def plot_heatmap(M, title, out_png):
    plt.figure()
    plt.imshow(M, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = glob.glob(args.attn_glob)

    mats = {}
    for f in files:
        M, meta = load_npz(f)
        mats[meta["run_tag"]] = M
        plot_heatmap(M, f"Instruction Attention – {meta['run_tag']}", os.path.join(args.out_dir, f"heat_{meta['run_tag']}.png"))

    if "fp16" in mats:
        base = mats["fp16"]
        for tag, M in mats.items():
            if tag=="fp16": continue
            delta = base - M
            plot_heatmap(delta, f"ΔA (fp16 - {tag})", os.path.join(args.out_dir, f"heat_delta_{tag}.png"))

if __name__ == "__main__":
    main()
