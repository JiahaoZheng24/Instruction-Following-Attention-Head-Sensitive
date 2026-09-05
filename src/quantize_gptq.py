"""Quantize an instruct model with GPTQModel (maintained AutoGPTQ fork) and
smoke-test the result for garbled generation.

Why re-quantize from scratch: old checkpoints are gone/possibly corrupted, and
the paper needs controlled quantization (documented calibration set, library
version, config) anyway. Old 3-bit runs produced garbled output — could be
kernel flakiness (AutoGPTQ 3-bit kernels are notoriously unreliable; Marlin has
no 3-bit support) rather than true 3-bit damage; the smoke test below is the
first check.

Config protocol (fixed; log any deviation):
  group_size=128, sym=True, desc_act=True (act-order matters at low bits)
  calib c4      = 128 C4 documents x 2048 tokens (literature standard; DEFAULT)
  calib instruct= data/calib_prompts.jsonl chat-formatted (separate baseline arm
                  for the instruction-calibration comparison — do NOT make it
                  the default or the baseline contrast disappears)

Example:
  python src/quantize_gptq.py --model Qwen/Qwen2.5-7B-Instruct --bits 3 \
      --calib c4 --out /store01/yshi4/jzheng7/models/qwen2.5-7b-gptq3-c4-g128
"""
import argparse
import json
import os

SMOKE_PROMPTS = [
    "Write exactly two sentences about the ocean, in English.",
    "List three fruits as a bulleted list. Do not add any explanation.",
    "Respond with a valid JSON object containing keys 'name' and 'age'.",
]


def load_calib(kind: str, tokenizer, n: int, seqlen: int, seed: int = 0) -> list[str]:
    """seed k>0 -> skip the first k*n eligible docs (disjoint calib replicates)."""
    if kind == "instruct":
        rows = [json.loads(l) for l in open("data/calib_prompts.jsonl", encoding="utf-8")]
        rows = rows[seed * n: seed * n + n]
        return [tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False, add_generation_prompt=True) for r in rows]
    if kind == "wikitext":
        # W21 calibration-corpus arm: wikitext-2-raw train, non-overlapping
        # seqlen-token windows (GPTQ paper's alternative calibration set).
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        start = seed * n * seqlen
        texts = []
        for i in range(n):
            a = start + i * seqlen
            if a + seqlen > ids.numel():
                break
            texts.append(tokenizer.decode(ids[a:a + seqlen]))
        return texts
    # c4 (default, literature-standard)
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    texts, skip = [], seed * n
    for ex in ds:
        if len(ex["text"]) > 200:
            if skip > 0:
                skip -= 1
                continue
            texts.append(ex["text"][: seqlen * 8])  # rough char cap, tokenizer trims
        if len(texts) >= n:
            break
    return texts


def smoke_test(out_dir: str, backend: str = "TORCH") -> None:
    """Generate a few responses; flag repetition/garbage heuristically.

    backend=TORCH by default: the Triton 3-bit dequant kernel is broken in the
    current stack (identical garbage output for different prompts + triton
    kernel-wrap warnings; W0 smoke test 2026-08). AUTO reproduces that bug.
    """
    from gptqmodel import BACKEND, GPTQModel
    model = GPTQModel.load(out_dir, backend=BACKEND[backend])
    print(f"[smoke] backend={backend}")
    tok = model.tokenizer
    print("\n===== SMOKE TEST =====")
    responses = []
    for p in SMOKE_PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        out = model.generate(**ids, do_sample=False, max_new_tokens=128)
        resp = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        responses.append(resp)
        uniq = len(set(resp.split())) / max(len(resp.split()), 1)
        ascii_ratio = sum(c.isascii() for c in resp) / max(len(resp), 1)
        flag = " [!GARBLED?]" if (uniq < 0.4 or ascii_ratio < 0.8) else ""
        print(f"\n--- {p}\n{resp[:400]}{flag}")
    # The signature that caught the 3-bit packing bug (2026-08-25): identical
    # output for DIFFERENT prompts = mechanically corrupt weights, regardless
    # of how the text looks. Genuine low-bit damage is input-dependent.
    if len(set(responses)) < len(responses):
        print("\n[!!] MECHANICAL CORRUPTION: identical output for different "
              "prompts. Do NOT use this checkpoint — suspect packing/kernel "
              "bug (check gptqmodel version, see GPTQModel#1278).")
    print("\n===== SMOKE TEST DONE (inspect above; [!GARBLED?] = repetition/"
          "non-ascii heuristic tripped) =====")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bits", type=int, required=True, choices=[2, 3, 4, 8])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib", choices=["c4", "instruct"], default="c4")
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-only", action="store_true",
                    help="skip quantization, just smoke-test an existing checkpoint")
    ap.add_argument("--v2", action="store_true",
                    help="GPTAQ / GPTQv2 asymmetric calibration (gptqmodel QuantizeConfig(v2=True)): "
                         "W30 'OBS family' arm — does the 2025 descendant collapse too?")
    ap.add_argument("--damp-percent", type=float, default=None,
                    help="gptqmodel damp_percent (library default if omitted; W30 packed damping arm)")
    ap.add_argument("--backend", default="TORCH",
                    help="gptqmodel BACKEND for the smoke test: TORCH (safe, default), "
                         "AUTO/TRITON (reproduces the broken 3-bit kernel), MARLIN, EXLLAMA_V2")
    args = ap.parse_args()

    if not args.smoke_only:
        from gptqmodel import GPTQModel, QuantizeConfig
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        calib = load_calib(args.calib, tok, args.n_calib, args.seqlen)
        print(f"[quantize] {args.model} -> {args.bits}bit g{args.group_size} "
              f"calib={args.calib}({len(calib)})")
        # Offload dir MUST be per-process and ideally node-local: with several
        # array tasks sharing one cwd, the default ./gptqmodel_offload/ on NFS
        # races between tasks (rmtree FileNotFoundError / truncated
        # safetensors, job 1385695). Job scripts set IFH_OFFLOAD_DIR=$TMPDIR/...
        extra = {}
        if os.environ.get("IFH_OFFLOAD_DIR"):
            extra["offload_to_disk_path"] = os.environ["IFH_OFFLOAD_DIR"]
        if args.v2:
            extra["v2"] = True
        if args.damp_percent is not None:
            extra["damp_percent"] = args.damp_percent
        qcfg = QuantizeConfig(bits=args.bits, group_size=args.group_size,
                              sym=True, desc_act=True, **extra)
        print(f"[quantize] QuantizeConfig extras: {extra}  (damp_percent in effect: "
              f"{getattr(qcfg, 'damp_percent', 'n/a')})")
        model = GPTQModel.load(args.model, qcfg)
        model.quantize(calib, batch_size=2)
        os.makedirs(args.out, exist_ok=True)
        model.save(args.out)
        import gptqmodel as _gq
        with open(os.path.join(args.out, "QUANT_PROTOCOL.json"), "w") as f:
            json.dump({"model": args.model, "bits": args.bits,
                       "group_size": args.group_size, "sym": True, "desc_act": True,
                       "calib": args.calib, "n_calib": args.n_calib,
                       "v2_gptaq": args.v2, "damp_percent": getattr(qcfg, 'damp_percent', None),
                       "library": f"gptqmodel=={getattr(_gq, '__version__', 'unknown')}"},
                      f, indent=2)
        del model
        import torch
        torch.cuda.empty_cache()

    smoke_test(args.out, backend=args.backend)


if __name__ == "__main__":
    main()
