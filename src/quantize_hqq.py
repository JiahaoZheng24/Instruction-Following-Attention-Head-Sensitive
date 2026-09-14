"""W56: HQQ (Half-Quadratic Quantization, Badri & Shaji 2023) as a
CALIBRATION-FREE control. No Hessian, no compensation: per-group scale and
zero-point are fitted by a half-quadratic solver on the weights alone.
Written out as a fake-quant HF checkpoint (dequantized weights), same
format the IFEval / GSM8K pipeline consumes.

Pre-registered prediction (RESULTS 9.10af): no collapse on Llama-3.1-8B or
Qwen2.5-14B at 3-bit (no calibration Hessian -> no BOS-dominated objective ->
no collinear trade); IFEval between RTN and the cured GPTQ arms.

Protocol deviation vs our GPTQ/RTN arms: HQQ is ASYMMETRIC (zero-point per
group); our frozen protocol is symmetric. Stated in PROTECT_PROTOCOL.json.

    pip install hqq
    python src/quantize_hqq.py --model meta-llama/Llama-3.1-8B-Instruct --bits 3 --group-size 128 --out $STORE/models/llama3.1-8b-hqq3
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import load_model  # noqa: E402

LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def hqq_dequant(lin: torch.nn.Linear, bits: int, group_size: int) -> torch.Tensor:
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    kw = dict(nbits=bits, group_size=group_size, axis=1)
    # older/newer hqq differ in which flags exist; keep the fake-quant path plain
    for extra in (dict(quant_zero=False, quant_scale=False, offload_meta=False), {}):
        try:
            cfg = BaseQuantizeConfig(**kw, **extra)
            break
        except TypeError:
            continue
    try:
        hl = HQQLinear(lin, quant_config=cfg, compute_dtype=torch.float16,
                       device=lin.weight.device, initialize=True, del_orig=False)
    except TypeError:
        hl = HQQLinear(lin, quant_config=cfg, compute_dtype=torch.float16, device=lin.weight.device)
    Wq = hl.dequantize()
    del hl
    return Wq.reshape(lin.weight.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model, tok = load_model(args.model, dtype=torch.float16)
    n = 0
    t0 = time.time()
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            for name, mod in layer.named_modules():
                if isinstance(mod, torch.nn.Linear) and name.split(".")[-1] in LINEARS:
                    Wq = hqq_dequant(mod, args.bits, args.group_size)
                    err = float((Wq.float() - mod.weight.float()).norm() / mod.weight.float().norm())
                    mod.weight.data.copy_(Wq.to(mod.weight.dtype))
                    n += 1
                    if name.endswith("down_proj"):
                        print(f"[hqq] L{li} {name}: rel err {err:.4f}", flush=True)
    print(f"[hqq] quantised {n} linears in {time.time() - t0:.0f}s")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    with open(os.path.join(args.out, "PROTECT_PROTOCOL.json"), "w") as f:
        json.dump({"model": args.model, "bits": args.bits, "group_size": args.group_size,
                   "quantizer": "hqq", "sym": False, "calib": None, "n_calib": 0,
                   "protect": "none", "selected_params": 0,
                   "note": "HQQ half-quadratic, asymmetric per-group zero-point, axis=1 (groups along input dim); "
                           "calibration-free; fake-quant dequantized weights",
                   "format": "fake-quant fp checkpoint"}, f, indent=2)
    print(f"[hqq] saved -> {args.out}")


if __name__ == "__main__":
    main()
