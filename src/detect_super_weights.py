"""Detect super weights (Yu et al., 2411.07191) via activation spikes:
a super weight W_down[i, j] produces a massive spike at output channel i of
mlp.down_proj fed by a massive input at intermediate channel j. Detection =
one forward pass; iterate by zeroing the found weight and re-running until
no spike above threshold remains (their protocol).

Also (optionally) reports where each super weight ranks in a TaCQ salience
tensor — the overlap check for the rescue-curve attribution.

  python src/detect_super_weights.py --model meta-llama/Llama-3.1-8B-Instruct \
      --out runs/super_weights_llama.csv \
      [--salience-dir $STORE/salience/llama31-8b-if]
"""
import argparse
import csv
import os

import torch

from common import load_model

PROMPT = ("Summer is warm. Winter is cold. The capital of France is Paris. "
          "Apple pie is a traditional dessert.")


@torch.no_grad()
def spike_scan(model, tok):
    """One forward pass; per layer, the biggest |down_proj output| spike and
    the co-located |input| spike. Returns list of candidate records."""
    stats = {}

    def make(layer):
        def hook(mod, args, out):
            x = args[0][0]          # [T, inter]
            y = out[0]              # [T, hidden]
            my, flat = y.abs().max(), y.abs().argmax()
            t, i = divmod(int(flat), y.shape[-1])
            j = int(x[t].abs().argmax())
            stats[layer] = {"out_spike": float(my), "row": i, "col": j,
                            "in_spike": float(x[t, j].abs()), "token": t}
        return hook

    handles = [model.model.layers[l].mlp.down_proj.register_forward_hook(make(l))
               for l in range(model.config.num_hidden_layers)]
    ids = tok(PROMPT, return_tensors="pt").to(model.device)
    model(**ids)
    for h in handles:
        h.remove()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=70.0,
                    help="abs output-spike threshold to count as super weight")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--salience-dir", help="optional: rank SW in TaCQ salience")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    found = []
    zeroed = []
    for it in range(args.max_iter):
        stats = spike_scan(model, tok)
        layer, rec = max(stats.items(), key=lambda kv: kv[1]["out_spike"])
        print(f"[detect] iter {it}: layer {layer} out_spike={rec['out_spike']:.1f} "
              f"row={rec['row']} col={rec['col']} in_spike={rec['in_spike']:.1f}")
        if rec["out_spike"] < args.threshold:
            break
        W = model.model.layers[layer].mlp.down_proj.weight
        w_val = float(W.data[rec["row"], rec["col"]])
        found.append({"layer": layer, "proj": "down_proj",
                      "row": rec["row"], "col": rec["col"], "weight": w_val,
                      "in_spike": rec["in_spike"], "out_spike": rec["out_spike"]})
        zeroed.append((W, rec["row"], rec["col"], W.data[rec["row"], rec["col"]].clone()))
        W.data[rec["row"], rec["col"]] = 0.0
    for W, r, c, v in zeroed:   # restore
        W.data[r, c] = v

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "proj", "row", "col",
                                          "weight", "in_spike", "out_spike"])
        w.writeheader()
        w.writerows(found)
    print(f"[detect] {len(found)} super weights -> {args.out}")

    if args.salience_dir and found:
        print("[detect] TaCQ-salience rank of each super weight:")
        for r in found:
            name = f"model.layers.{r['layer']}.mlp.down_proj.pt"
            sal = torch.load(os.path.join(args.salience_dir, name),
                             map_location="cpu").float()
            v = sal[r["row"], r["col"]]
            pct = float((sal > v).sum()) / sal.numel()
            print(f"  L{r['layer']} ({r['row']},{r['col']}): salience={v:.3e}, "
                  f"top-{100 * pct:.5f}% of module")


if __name__ == "__main__":
    main()
