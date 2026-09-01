"""Dissociation analysis (the paper's key insight experiment, run in W1).

Question: are IF-critical heads the same heads that capability-agnostic
salience signals would protect? If overlap is LOW, existing salience-based
mixed-precision methods (AWQ-style activation magnitude, Hessian/gradient
saliency a la TaCQ/SpQR) cannot find IF-heads in principle -> our targeted
protection is necessary. If overlap is HIGH, the story collapses; know early.

Rankings compared (each a CSV with cols layer,head,score):
  causal   from ablation results (build with your Delta-IFEval per head)
  isi      old attention-deviation signal (backup/artifacts/per_head_*.csv,
           convert first; screening-grade only)
  act      activation magnitude per head (produced by diagnose_heads.py calib)
  grad     |W * dW| gradient saliency aggregated per q-head slice over
           q_proj rows + o_proj cols (+ k/v per kv-group), computed here

Outputs: overlap_matrix.csv (Jaccard@k for k in {16,32,64} + Spearman),
         overlap_heatmap.png

Example:
  python src/dissociation.py --grad-calib data/calib_prompts.jsonl \
      --rankings causal=runs/causal_ranking.csv act=runs/diag/act_salience.csv \
      --out runs/dissociation
"""
import argparse
import csv
import itertools
import os

import numpy as np
import torch

from common import DEFAULT_MODEL, HeadGeom, load_model, read_jsonl


def load_ranking(path, geom) -> np.ndarray:
    """CSV(layer,head,score) -> flat score array [L*H]."""
    arr = np.full(geom.n_layers * geom.n_heads, np.nan)
    for r in csv.DictReader(open(path)):
        arr[int(r["layer"]) * geom.n_heads + int(r["head"])] = float(r["score"])
    assert not np.isnan(arr).any(), f"incomplete ranking: {path}"
    return arr


def grad_saliency(model_id, calib_file) -> tuple[np.ndarray, "HeadGeom"]:
    """Per-q-head |W * dL/dW| on attention projections, teacher-forced LM loss."""
    model, tok = load_model(model_id, dtype=torch.bfloat16)
    geom = HeadGeom(model)
    for p in model.parameters():
        p.requires_grad_(False)
    projs = []
    for l in range(geom.n_layers):
        attn = model.model.layers[l].self_attn
        for m in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
            m.weight.requires_grad_(True)
        projs.append(attn)

    for ex in read_jsonl(calib_file):
        text = tok.apply_chat_template(
            [{"role": "user", "content": ex.get("prompt") or ex["text"]}],
            tokenize=False, add_generation_prompt=False)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        out.loss.backward()

    d, g = geom.head_dim, geom.group
    scores = np.zeros((geom.n_layers, geom.n_heads))
    for l, attn in enumerate(projs):
        def sal(m):  # |W * grad|
            return (m.weight.detach() * m.weight.grad).abs() if m.weight.grad is not None \
                else torch.zeros_like(m.weight)
        sq, sk, sv, so = sal(attn.q_proj), sal(attn.k_proj), sal(attn.v_proj), sal(attn.o_proj)
        for h in range(geom.n_heads):
            kv = h // g
            s = (sq[h * d:(h + 1) * d, :].sum()          # q rows per head
                 + so[:, h * d:(h + 1) * d].sum()        # o cols per head
                 + (sk[kv * d:(kv + 1) * d, :].sum()     # k/v rows per kv-group,
                    + sv[kv * d:(kv + 1) * d, :].sum()) / g)  # shared across group
            scores[l, h] = float(s)
    return scores.reshape(-1), geom


def jaccard_at_k(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ta, tb = set(np.argsort(-a)[:k]), set(np.argsort(-b)[:k])
    return len(ta & tb) / len(ta | tb)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(-a)), np.argsort(np.argsort(-b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--rankings", nargs="+", default=[],
                    help="name=path.csv pairs (cols: layer,head,score)")
    ap.add_argument("--grad-calib", help="calib jsonl; if set, computes grad saliency")
    ap.add_argument("--out", default="runs/dissociation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    named: dict[str, np.ndarray] = {}
    geom = None
    if args.grad_calib:
        g, geom = grad_saliency(args.model, args.grad_calib)
        named["grad"] = g
        np.save(os.path.join(args.out, "grad_salience.npy"), g)
    if geom is None:
        model, _ = load_model(args.model)  # geometry only; TODO: read config w/o weights
        geom = HeadGeom(model)
        del model
    for spec in args.rankings:
        name, path = spec.split("=", 1)
        named[name] = load_ranking(path, geom)

    names = list(named)
    ks = (16, 32, 64)
    with open(os.path.join(args.out, "overlap_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "spearman"] + [f"jaccard@{k}" for k in ks])
        for a, b in itertools.combinations(names, 2):
            row = [a, b, round(spearman(named[a], named[b]), 4)]
            row += [round(jaccard_at_k(named[a], named[b], k), 4) for k in ks]
            w.writerow(row)
            print("overlap", row)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(names)
        m = np.eye(n)
        for i, j in itertools.combinations(range(n), 2):
            m[i, j] = m[j, i] = jaccard_at_k(named[names[i]], named[names[j]], 32)
        fig, ax = plt.subplots(figsize=(1.2 * n + 2, 1.2 * n + 1.5))
        im = ax.imshow(m, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(n), names)
        ax.set_yticks(range(n), names)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                        color="w" if m[i, j] < 0.6 else "k")
        ax.set_title("Head-ranking overlap (Jaccard@32)")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "overlap_heatmap.png"), dpi=200)
        print("saved", os.path.join(args.out, "overlap_heatmap.png"))
    except ImportError:
        print("matplotlib not available; skipped heatmap")


if __name__ == "__main__":
    main()
