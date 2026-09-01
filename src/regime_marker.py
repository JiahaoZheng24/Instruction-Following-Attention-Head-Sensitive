"""Regime marker: can a one-forward-pass outlier statistic predict which
quantization regime a model falls into (graceful / rescuable-collapse /
terminal)? Measures per-model activation-outlier structure:

  - max |hidden activation| across layers (massive-activation magnitude)
  - number of channels whose max |activation| exceeds 50 / 100
  - kurtosis of hidden activations (heavy-tail index)
  - max |down_proj input| (intermediate spikes)

Appends one row per model to the CSV. The paper's claim: collapse propensity
under compensation-based quantizers tracks these statistics.

  python src/regime_marker.py --model meta-llama/Llama-3.1-8B-Instruct \
      --out runs/regime_markers.csv
"""
import argparse
import csv
import os

import torch

from common import load_model

PROMPT = ("Summer is warm. Winter is cold. The capital of France is Paris. "
          "Apple pie is a traditional dessert. The quick brown fox jumps "
          "over the lazy dog while the orchestra plays a quiet nocturne.")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    hidden_max = {}     # per layer: max |h| and its channel
    down_in_max = {}
    samples = []

    def h_hook(layer):
        def hook(_m, args_, out):
            h = (out[0] if isinstance(out, tuple) else out).detach().float()
            hidden_max[layer] = float(h.abs().max())
            if layer % 4 == 0:
                samples.append(h.flatten()[:: max(1, h.numel() // 20000)].cpu())
        return hook

    def d_hook(layer):
        def hook(_m, args_):
            down_in_max[layer] = float(args_[0].detach().abs().max())
        return hook

    handles = []
    ch_max = None
    for l, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(h_hook(l)))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(d_hook(l)))

    def emb_hook(_m, a, out):
        return out
    ids = tok(PROMPT, return_tensors="pt").to(model.device)

    # channel-wise max over the final-layer hidden states via a norm hook
    ch_store = {}

    def norm_hook(_m, a):
        x = a[0].detach().float().reshape(-1, a[0].shape[-1])
        prev = ch_store.get("x")
        cur = x.abs().amax(dim=0).cpu()
        ch_store["x"] = cur if prev is None else torch.maximum(prev, cur)
    handles.append(model.model.norm.register_forward_pre_hook(norm_hook))

    model(**ids)
    for h in handles:
        h.remove()

    ch = ch_store["x"]
    flat = torch.cat(samples)
    k = float(((flat - flat.mean()) ** 4).mean() / (flat.var() ** 2))
    row = {
        "model": args.model,
        "max_hidden_act": round(max(hidden_max.values()), 1),
        "argmax_layer": max(hidden_max, key=hidden_max.get),
        "n_channels_gt50": int((ch > 50).sum()),
        "n_channels_gt100": int((ch > 100).sum()),
        "act_kurtosis": round(k, 1),
        "max_downproj_in": round(max(down_in_max.values()), 1),
    }
    print("[marker]", row)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
