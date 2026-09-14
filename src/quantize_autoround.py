"""W56: AutoRound (Intel, Cheng et al. 2023) with OUR c4 calibration set,
written out as a fake-quant HF checkpoint.

Why this method: it optimises the same token-averaged block reconstruction
objective as GPTQ, but by signed-gradient descent on a rounding offset that
is bounded to half a quantisation step (plus learned min/max clipping).
It therefore cannot produce the range-level displacement of P3'.

Pre-registered prediction (RESULTS 9.10af): NO collapse on Llama-3.1-8B or
Qwen2.5-14B at 3-bit (bounded displacement -> no range-bound trade), i.e. the
objective's blind spot alone is not sufficient; unbounded OBS compensation
is necessary. If AutoRound collapses too, P3' is only GPTQ's expression of
a blind spot that suffices by itself.

    pip install auto-round   (in a cloned env, see jobs/w56_setup.sh)
    python src/quantize_autoround.py --model meta-llama/Llama-3.1-8B-Instruct --bits 3 --group-size 128 --out $STORE/models/llama3.1-8b-ar3
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import load_model  # noqa: E402
from quantize_gptq import load_calib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=200, help="AutoRound default 200")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--calib", choices=["c4", "c4win", "pileours", "pile"], default="c4",
                    help="c4/c4win/pileours = OUR loaders fed as a DataLoader (see load_calib); pile = AutoRound own NeelNanda/pile-10k loader")
    ap.add_argument("--asym", action="store_true", help="default symmetric to match our protocol")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from auto_round import AutoRound

    model, tok = load_model(args.model, dtype=torch.bfloat16)

    calib_used = "NeelNanda/pile-10k"
    dataset = "NeelNanda/pile-10k"
    ar_seqlen = args.seqlen
    OURS = {"c4": "c4", "c4win": "c4win", "pileours": "pile"}
    if args.calib in OURS:
        # W56 lesson: AutoRound's text loader DROPS every sample shorter than
        # seqlen tokens; our c4 docs are mostly short, so the first run
        # calibrated on 3 samples. Feed the exact variable-length samples our
        # GPTQ arms see (same docs, same truncation, BOS added by the tokenizer)
        # as a DataLoader of {input_ids, attention_mask} batches of size 1, and
        # set AutoRound's seqlen to the shortest sample so nothing is filtered.
        docs = load_calib(OURS[args.calib], tok, args.n_calib, args.seqlen, seed=0)
        enc = [tok(d, return_tensors="pt", truncation=True, max_length=args.seqlen).input_ids[0] for d in docs]
        lens = [int(e.numel()) for e in enc]
        ar_seqlen = max(1, min(lens))

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(enc)

            def __getitem__(self, i):
                ids = enc[i].unsqueeze(0)
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        dataset = torch.utils.data.DataLoader(_DS(), batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
        calib_used = (f"{OURS[args.calib]} x{len(enc)} (our loader/truncation as GPTQ arms; variable length, "
                      f"min {min(lens)} / mean {sum(lens) / len(lens):.0f} / max {max(lens)} tokens; batch 1)")
        print(f"[ar] {OURS[args.calib]} calibration: {len(enc)} samples, token lengths min {min(lens)} mean {sum(lens) / len(lens):.0f} max {max(lens)}")

    def build(ds, seqlen, bs):
        return AutoRound(model, tok, bits=args.bits, group_size=args.group_size, sym=not args.asym,
                         iters=args.iters, nsamples=args.n_calib, seqlen=seqlen,
                         batch_size=bs, dataset=ds, device="cuda",
                         low_gpu_mem_usage=False)

    t0 = time.time()
    try:
        ar = build(dataset, ar_seqlen, 1 if args.calib in OURS else args.batch_size)
        ar.quantize()
    except Exception as e:  # noqa: BLE001
        if args.calib not in OURS:
            raise
        print(f"[ar] WARNING: our-loader DataLoader path failed ({type(e).__name__}: {e}); "
              f"falling back to AutoRound's default pile-10k. Record this in RESULTS.", flush=True)
        calib_used = "NeelNanda/pile-10k (FALLBACK, c4 loader rejected)"
        model, tok = load_model(args.model, dtype=torch.bfloat16)
        ar = build("NeelNanda/pile-10k", args.seqlen, args.batch_size)
        ar.quantize()
    print(f"[ar] quantisation done in {(time.time() - t0) / 60:.1f} min")

    os.makedirs(args.out, exist_ok=True)
    saved = "save_quantized(format=fake)"
    try:
        ar.save_quantized(args.out, format="fake", inplace=True)
    except Exception as e:  # noqa: BLE001
        print(f"[ar] save_quantized(fake) failed ({type(e).__name__}: {e}); saving the in-memory "
              f"fake-quantised model with save_pretrained", flush=True)
        ar.model.save_pretrained(args.out)
        saved = "save_pretrained of in-memory q-dq model"
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "PROTECT_PROTOCOL.json"), "w") as f:
        json.dump({"model": args.model, "bits": args.bits, "group_size": args.group_size,
                   "quantizer": "autoround", "sym": not args.asym, "iters": args.iters,
                   "calib": calib_used, "n_calib": args.n_calib, "seqlen": args.seqlen, "ar_seqlen": ar_seqlen,
                   "protect": "none", "selected_params": 0, "saved_via": saved,
                   "format": "fake-quant fp checkpoint"}, f, indent=2)
    print(f"[ar] saved -> {args.out}")


if __name__ == "__main__":
    main()
