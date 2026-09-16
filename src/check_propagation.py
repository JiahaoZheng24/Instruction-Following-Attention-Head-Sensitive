"""W72z: propagation check for the layer-wise quantisation loop.

Emulates quantize_protected.py's loop exactly -- capture_layer0_inputs, then for
every layer: one 'Hessian' call per sample (hooks discarded here), RTN-quantise
the layer's seven linears, one 'propagation' call per sample -- and records the
propagated activations at every depth. Afterwards the model is fully quantised,
so a plain forward with output_hidden_states=True must reproduce those
activations layer by layer. The W71 defect (a KV cache carried into the loop,
so the propagation call attended the layer's pre-quantisation K/V) fails this
check at every layer after the first; bf16 recomputation noise passes it.

  python src/check_propagation.py --model meta-llama/Llama-3.1-8B-Instruct --n 4
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model  # noqa: E402
from quantize_gptq import load_calib  # noqa: E402
from quantize_protected import capture_layer0_inputs, rtn_quantize_, ATTN, MLP  # noqa: E402


def rel(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return ((a - b).norm() / b.norm().clamp(min=1e-6)).item()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4, help="calibration samples")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max relative error tolerated at any layer (bf16 noise is ~1e-2)")
    args = ap.parse_args()

    model, tok = load_model(args.model)
    model.eval()
    texts = load_calib("c4", tok, args.n, args.seqlen, seed=0)
    enc = [tok(t, return_tensors="pt", truncation=True, max_length=args.seqlen).to(model.device) for t in texts]
    print(f"[check] {args.model}: {len(texts)} samples, lengths {[e['input_ids'].shape[1] for e in enc]}", flush=True)

    # --- the loop, exactly as quantize_protected.py does it ---
    inps, kws = capture_layer0_inputs(model, tok, texts, args.seqlen)
    for j, kw in enumerate(kws):
        bad = [k for k in ("past_key_values", "past_key_value", "use_cache")
               if kw.get(k, None) not in (None, False)]
        assert not bad, f"sample {j}: cache kwargs still active: {bad}"
    layers = model.model.layers
    stage = {}   # depth k -> list of propagated inputs to layer k (after layers < k are quantised)
    for li, layer in enumerate(layers):
        for j in range(len(inps)):                   # 'Hessian' pass (hooks would fire here)
            layer(inps[j], **kws[j])
        mods = {p: getattr(layer.self_attn, p) for p in ATTN}
        mods.update({p: getattr(layer.mlp, p) for p in MLP})
        for m in mods.values():                      # quantise the layer in place
            rtn_quantize_(m.weight.data, args.bits, args.group_size, None, sym=True)
        for j in range(len(inps)):                   # propagation pass
            out = layer(inps[j], **kws[j])
            inps[j] = out[0] if isinstance(out, tuple) else out
        stage[li + 1] = [x.detach().clone().cpu() for x in inps]
        if (li + 1) % 8 == 0:
            print(f"[check] propagated through layer {li + 1}/{len(layers)}", flush=True)

    # --- reference: plain forward of the now fully quantised model ---
    worst = -1.0
    worst_at = (0, 0)
    per_layer = []
    for j, e in enumerate(enc):
        out = model(**e, use_cache=False, output_hidden_states=True)
        hs = out.hidden_states           # hs[k] = input to layer k for k < L (hs[L] is after the final norm)
        for k in range(1, len(layers)):
            r = rel(stage[k][j].to(hs[k].device), hs[k])
            per_layer.append((k, j, r))
            if r > worst:
                worst, worst_at = r, (k, j)
    by_layer = {}
    for k, j, r in per_layer:
        by_layer[k] = max(by_layer.get(k, 0.0), r)
    print("[check] max relative error by layer (input to layer k):")
    print("        " + " ".join(f"{k}:{v:.3g}" for k, v in sorted(by_layer.items())))
    verdict = "PASS" if worst <= args.tol else "FAIL"
    print(f"[check] {verdict}: worst relative error {worst:.4g} at layer {worst_at[0]}, sample {worst_at[1]} "
          f"(tolerance {args.tol})", flush=True)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
