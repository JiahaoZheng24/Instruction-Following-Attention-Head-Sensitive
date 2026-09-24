"""W91: which setting separates GPTQModel's default path (no collapse, W90) from our loop (collapse)?

W90 (2026-09-23): under gptqmodel 5.6.12 defaults, Llama-3.1-8B-Instruct scores IFEval 0.646 (128 rows) /
0.665 (1,024 rows) and Qwen2.5-14B 0.759 / 0.745, all ABOVE RTN, where our loop on the same sampling shape
gives 0.155 / 0.426.  The library's defaults differ from our loop in: desc_act=False (ours: act-order on),
act_group_aware=True (group-aware reordering), true_sequential=True (ours: block-input Hessians),
damp_auto_increment=0.01 (ours: fixed rho), and in how the rows are fed (README shard vs our C4 stream).
This script quantizes with GPTQModel and lets each of these be set on the command line, one at a time or
together, so the job can bracket the cause.  Everything not named stays at the library default.

  python src/quantize_gptqmodel_ablate.py --model meta-llama/Llama-3.1-8B-Instruct --n-rows 128 \
      --set desc_act=True --out /store01/yshi4/jzheng7/models/w91_l_descact
  --set may be repeated; values are parsed as Python literals (True/False/0/0.05).
  --calib readme  : first n rows of en/c4-train.00001-of-01024 (GPTQModel README recipe; W90)
  --calib paper   : the paper's short-document stream (quantize_gptq.load_calib('c4'), 128 docs)
  --pretokenize   : tokenize the rows ourselves with the chat model's tokenizer (BOS added) and pass
                    input_ids/attention_mask dicts, so the library's own text handling is bypassed
"""
import argparse
import ast
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def rows_readme(n):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
    return list(ds.select(range(n))["text"])


def rows_paper(n, tokenizer):
    from quantize_gptq import load_calib
    return load_calib("c4", tokenizer, n, 2048)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--n-rows", type=int, default=128)
    ap.add_argument("--calib", choices=["readme", "paper"], default="readme")
    ap.add_argument("--pretokenize", action="store_true")
    ap.add_argument("--set", action="append", default=[], help="QuantizeConfig field=value, repeatable")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gptqmodel
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import AutoTokenizer

    overrides = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        overrides[k] = ast.literal_eval(v)
    if os.environ.get("IFH_OFFLOAD_DIR"):
        overrides["offload_to_disk_path"] = os.environ["IFH_OFFLOAD_DIR"]
    qcfg = QuantizeConfig(bits=args.bits, group_size=args.group_size, **overrides)
    print(f"[w91] gptqmodel {getattr(gptqmodel, '__version__', 'unknown')}  overrides={overrides}")
    print(f"[w91] QuantizeConfig: {qcfg}")

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = rows_readme(args.n_rows) if args.calib == "readme" else rows_paper(args.n_rows, tok)
    if args.pretokenize:
        enc = [tok(r, truncation=True, max_length=2048, return_tensors=None) for r in rows]
        calib = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in enc]
        n_bos = sum(1 for e in enc if e["input_ids"][0] == tok.bos_token_id) if tok.bos_token_id is not None else -1
        print(f"[w91] pretokenized {len(calib)} rows, mean {sum(len(e['input_ids']) for e in enc)/len(enc):.0f} tokens, "
              f"rows starting with BOS: {n_bos}")
    else:
        calib = rows
        print(f"[w91] {len(calib)} raw rows ({args.calib}); the library tokenizes them")

    t0 = time.time()
    model = GPTQModel.load(args.model, qcfg)
    model.quantize(calib, batch_size=1)
    dt = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    model.save(args.out)
    saved = {}
    for name in ("quantize_config.json", "quant_config.json"):
        p = os.path.join(args.out, name)
        if os.path.exists(p):
            saved = json.load(open(p))
            break
    with open(os.path.join(args.out, "W91_PROTOCOL.json"), "w") as f:
        json.dump({"model": args.model, "bits": args.bits, "group_size": args.group_size, "n_rows": args.n_rows,
                   "calib": args.calib, "pretokenize": args.pretokenize, "overrides": {k: str(v) for k, v in overrides.items()},
                   "library": f"gptqmodel=={getattr(gptqmodel, '__version__', 'unknown')}",
                   "quantize_seconds": round(dt), "library_saved_config": saved}, f, indent=2)
    print(f"[w91] saved {args.out} in {dt:.0f}s")


if __name__ == "__main__":
    main()
