"""W90: quantize with GPTQModel end to end, touching nothing but bits and group size.

Reviewer question (pre-submission review, 2026-09-23): the paper's main arms reproduce GPTQModel's
calibration *sampling shape* inside our own loop (act-order on, no true-sequential pass, fake-quant).
Does the collapse also appear under the library's own default code path?  This script leaves every
QuantizeConfig field at its library default (desc_act unset, true_sequential on, damp_percent 0.05,
sym), feeds the calibration rows exactly as the GPTQModel README does (raw C4 rows, tokenised by the
library one row at a time, no concatenation) and saves the packed checkpoint.  IFEval is then run by
the job through the usual harness, which loads the packed weights with the library's TORCH backend.

  python src/quantize_gptqmodel_default.py --model meta-llama/Llama-3.1-8B-Instruct --bits 3 \
      --n-rows 1024 --out /store01/yshi4/jzheng7/models/w90_llama31_gptqmodel_default_1024

--n-rows 1024 is the README example; --n-rows 128 matches the paper's sample count.
"""
import argparse
import json
import os
import time


def readme_calibration(n_rows: int) -> list[str]:
    """The GPTQModel README recipe: one C4 shard, the first n_rows texts, nothing else."""
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
    return ds.select(range(n_rows))["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-rows", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    import gptqmodel
    from gptqmodel import GPTQModel, QuantizeConfig

    calib = readme_calibration(args.n_rows)
    # nothing but bits and group size: every other field is whatever this gptqmodel version defaults to.
    # offload_to_disk_path is a QuantizeConfig field in gptqmodel 5.x (a scratch directory, not a
    # quantization setting); it is pointed at node-local $TMPDIR so array tasks do not race on NFS.
    extra = {}
    if os.environ.get("IFH_OFFLOAD_DIR"):
        extra["offload_to_disk_path"] = os.environ["IFH_OFFLOAD_DIR"]
    qcfg = QuantizeConfig(bits=args.bits, group_size=args.group_size, **extra)
    print(f"[w90] gptqmodel {getattr(gptqmodel, '__version__', 'unknown')}")
    print(f"[w90] QuantizeConfig as constructed: {qcfg}")
    print(f"[w90] {args.model}: {len(calib)} raw C4 rows (README shard), bits={args.bits} g={args.group_size}")

    t0 = time.time()
    model = GPTQModel.load(args.model, qcfg)
    model.quantize(calib, batch_size=args.batch_size)
    dt = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)

    # record exactly what the library used (its own saved config is the authority)
    saved = {}
    for name in ("quantize_config.json", "quant_config.json"):
        p = os.path.join(args.out, name)
        if os.path.exists(p):
            saved = json.load(open(p))
            break
    with open(os.path.join(args.out, "W90_PROTOCOL.json"), "w") as f:
        json.dump({"model": args.model, "bits": args.bits, "group_size": args.group_size,
                   "calibration": f"GPTQModel README recipe: first {args.n_rows} rows of en/c4-train.00001-of-01024",
                   "library": f"gptqmodel=={getattr(gptqmodel, '__version__', 'unknown')}",
                   "quantize_seconds": round(dt), "library_saved_config": saved}, f, indent=2)
    print(f"[w90] saved {args.out} in {dt:.0f}s; library config: {saved}")


if __name__ == "__main__":
    main()
