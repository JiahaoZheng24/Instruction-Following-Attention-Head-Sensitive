"""TaCQ-style per-weight saliency for IF: |W| * |dL/dW| * |W - RTN_b(W)|.

Faithful to TaCQ (2504.07389): Magnitude-Sharpened Gradient x Quantization-
Aware Localization. Gradient = single accumulated backward over calibration
data (grads summed across batches, then |.|). Task-conditioning for IF: the
calibration set is chat-formatted instruction data (data/calib_prompts.jsonl),
mirroring TaCQ's task-specific arm (their strongest setting).

Output: one fp16 tensor per linear module -> <out_dir>/<module_name>.pt
plus sample.pt (random subsample of all saliency values, for global
quantile thresholding in quantize_protected.py).

  python src/tacq_salience.py --calib-file data/calib_prompts.jsonl \
      --bits 3 --out-dir runs/salience_if
"""
import argparse
import os
import random

import torch

from common import DEFAULT_MODEL, load_model, read_jsonl


def rtn_err(W: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    """|W - RTN_b(W)| with per-row grouped sym quantization (protocol-matched)."""
    maxq = 2 ** bits - 1
    zero = (maxq + 1) / 2
    err = torch.empty_like(W)
    for c in range(0, W.shape[1], group_size):
        g = W[:, c:c + group_size]
        xmax = g.abs().max(dim=1).values.clamp(min=1e-8)
        scale = (2.0 * xmax / maxq).unsqueeze(1)
        q = torch.clamp(torch.round(g / scale) + zero, 0, maxq)
        err[:, c:c + group_size] = (g - scale * (q - zero)).abs()
    return err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calib-file")
    ap.add_argument("--calib-kind", choices=["chat", "c4"], default="chat",
                    help="chat = IF-conditioned (calib-file, chat template); "
                         "c4 = general-text-conditioned (task-generality arm)")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rotate", action="store_true",
                    help="compute salience in the R1-rotated basis (must match "
                         "the quantize_protected --rotate run: same seed)")
    ap.add_argument("--rotate-seed", type=int, default=0)
    args = ap.parse_args()

    model, tok = load_model(args.model)
    if args.rotate:
        from rotate_model import fold_and_rotate
        fold_and_rotate(model, seed=args.rotate_seed)
    model.train(False)
    targets = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "layers" in name:
            mod.weight.requires_grad_(True)
            targets[name] = mod

    if args.calib_kind == "c4":
        from quantize_gptq import load_calib
        texts = load_calib("c4", tok, args.n_calib, args.max_len)
    else:
        assert args.calib_file, "--calib-file required for chat kind"
        rows = read_jsonl(args.calib_file)[: args.n_calib]
        texts = [tok.apply_chat_template(
            [{"role": "user", "content": ex.get("prompt") or ex["text"]}],
            tokenize=False, add_generation_prompt=True) for ex in rows]
    print(f"[salience] accumulating gradients over {len(texts)} {args.calib_kind} samples")
    for i, text in enumerate(texts):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.max_len).to(model.device)
        out = model(**ids, labels=ids["input_ids"])
        out.loss.backward()   # grads ACCUMULATE across batches (no zero_grad)
        if (i + 1) % 16 == 0:
            print(f"[salience] {i + 1}/{len(texts)} loss={out.loss.item():.3f}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(0)
    samples = []
    with torch.no_grad():
        for name, mod in targets.items():
            W = mod.weight.data.float()
            sal = (W.abs() * mod.weight.grad.float().abs()
                   * rtn_err(W, args.bits)).to(torch.float16)
            torch.save(sal.cpu(), os.path.join(args.out_dir, name.replace("/", "_") + ".pt"))
            flat = sal.flatten()
            idx = torch.randint(0, flat.numel(), (max(1, flat.numel() // 512),))
            samples.append(flat[idx].float().cpu())
            mod.weight.grad = None
    torch.save(torch.cat(samples), os.path.join(args.out_dir, "sample.pt"))
    print(f"[salience] saved {len(targets)} module tensors -> {args.out_dir}")


if __name__ == "__main__":
    main()
