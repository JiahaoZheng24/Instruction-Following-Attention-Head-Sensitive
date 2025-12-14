#!/usr/bin/env python3
"""
dump_attn.py - Extract attention weights for instruction tokens

FIXED VERSION - Resolves UnboundLocalError with 'text' variable
"""

import argparse
import os
import json
import re
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def log(msg):
    """Simple logging with flush"""
    print(msg, flush=True)


def find_spans(text, pats):
    """Find and merge overlapping spans matching patterns"""
    spans = []
    for pat in pats:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            spans.append((m.start(), m.end()))
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def char_to_token_idxs(text, tokenizer, spans):
    """Convert character spans to token indices"""
    if not spans:
        return []
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    idxs = []
    for tid, (s, e) in enumerate(enc["offset_mapping"]):
        for cs, ce in spans:
            if not (e <= cs or s >= ce):
                idxs.append(tid)
                break
    return sorted(set(idxs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=4096)
    args = ap.parse_args()

    # Instruction patterns
    regex_list = [
        r"two words", r"start with", r"json", r"format", r"only output.*",
        r"no explanation", r"do not explain", r"output only",
        r"title", r"bullet point", r"all lowercase", r"markdown",
        r"three sections", r"highlight", r"exactly \d+ bullet",
        r"use exactly \d+ words", r"all capital", r"comma",
        r"postscript", r"first word", r"end your response with",
        r"wrap.*in json", r"include keywords", r"letter frequency",
        r"repeat the prompt", r"in your entire response"
    ]

    log("=" * 70)
    log(f"ATTENTION EXTRACTION: {args.run_tag}")
    log("=" * 70)
    log(f"Instruction patterns: {len(regex_list)} patterns")

    # Load tokenizer
    log(f"\n[1/5] Loading tokenizer from: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        log("✓ Tokenizer loaded successfully")
    except Exception as e:
        log(f"✗ FAILED to load tokenizer: {e}")
        sys.exit(1)

    # Disable Flash Attention
    log("\n[2/5] Disabling Flash Attention backends...")
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        log("✓ Flash/SDPA disabled, using math backend")
    except Exception:
        log("⚠️  Could not configure attention backends")

    # Load model
    log(f"\n[3/5] Loading model: {args.model_id}")
    log("This may take several minutes...")

    if torch.cuda.is_available():
        log(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        log("⚠️  CUDA not available, using CPU")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        log("✓ FP16 / non-GPTQ model loaded via AutoModelForCausalLM.from_pretrained(...)")
        log(f"  Model type: {model.__class__.__name__}")
        log(f"  Device: {model.device}")
    except Exception as e:
        log(f"✗ FAILED to load model: {e}")
        sys.exit(1)

    # Configure attention
    log("\n[4/5] Configuring attention mechanism...")
    try:
        model.set_attn_implementation("eager")
        log("✓ Set attention implementation to 'eager'")
    except Exception as e:
        log(f"⚠️  Could not set attention implementation: {e}")

    model.eval()
    log("✓ Model set to eval mode")

    # Load prompts
    log(f"\n[5/5] Processing prompts from: {args.prompts_jsonl}")
    if not os.path.exists(args.prompts_jsonl) or os.path.getsize(args.prompts_jsonl) == 0:
        log(f"✗ ERROR: Prompts file missing or empty: {args.prompts_jsonl}")
        sys.exit(1)

    try:
        with open(args.prompts_jsonl, "r", encoding="utf-8") as f:
            prompts = [json.loads(line) for line in f if line.strip()]
        log(f"✓ Loaded {len(prompts)} prompts")
    except Exception as e:
        log(f"✗ FAILED to load prompts: {e}")
        sys.exit(1)

    if len(prompts) == 0:
        log("✗ ERROR: No valid prompts in file")
        sys.exit(1)

    # Process each prompt
    log(f"\nProcessing {len(prompts)} prompts...")
    attn_per_prompt = []
    failed_count = 0

    for idx, ex in enumerate(prompts):
        if (idx + 1) % 5 == 0 or idx == 0:
            log(f"  [{idx + 1}/{len(prompts)}] Processing...")

        # Extract text (user prompt)
        user_text = ex.get("prompt") or ex.get("text")
        if not user_text:
            log(f"  ⚠️  Skipping sample {idx}: no prompt text")
            failed_count += 1
            continue

        # IMPORTANT: keep tokenization consistent with eval (chat template)
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": user_text},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Find instruction spans in *user_text*, then map to char positions in formatted_prompt
        spans_rel = find_spans(user_text, regex_list)
        start_char = formatted_prompt.find(user_text)
        if start_char == -1:
            # Fallback: operate on formatted prompt directly (less precise)
            spans = find_spans(formatted_prompt, regex_list)
        else:
            spans = [(s + start_char, e + start_char) for (s, e) in spans_rel]

        idxs = char_to_token_idxs(formatted_prompt, tokenizer, spans)

        # Debug output
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  Found {len(spans)} instruction spans in formatted prompt")
            print(f"  Instruction token indices: {len(idxs)} tokens")

        # Warning if no instruction tokens found
        if len(idxs) == 0:
            print(f"  ⚠️  WARNING: No instruction tokens found for sample {idx}!")
            print(f"  User text preview: {user_text[:200]}...")
            # Continue anyway - will compute zero attention

        # Tokenize formatted prompt
        try:
            toks = tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            )
            input_ids = toks["input_ids"].to(model.device)
            attn_mask = toks["attention_mask"].to(model.device)
        except Exception as e:
            log(f"  ✗ Tokenization failed for sample {idx}: {e}")
            failed_count += 1
            continue

        # Run model with attention extraction
        try:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

            attentions = out.attentions
            if attentions is None:
                raise RuntimeError(
                    "Got attentions=None. Your backend is still not in 'eager' mode.\n"
                    "Fixes to try:\n"
                    "  1) Ensure transformers>=4.41, and call model.set_attn_implementation('eager').\n"
                    "  2) Disable Flash/SDPA kernels via torch.backends.cuda.enable_flash_sdp(False), etc.\n"
                    "  3) If using quantized backends that swallow attentions, switch to a HF model that supports attentions.\n"
                )

            # Extract per-head attention to instruction tokens
            mats = []
            for A in attentions:  # [bs, heads, tgt, src]
                H = A[0]  # [heads, tgt, src]
                if len(idxs) == 0:
                    # No instruction tokens - use zero attention
                    v = torch.zeros(H.size(0), dtype=torch.float16, device=H.device)
                else:
                    # Average attention to instruction tokens across positions
                    v = H[..., idxs].mean(dim=(1, 2))
                mats.append(v.detach().float().cpu().numpy())

            LH = np.stack(mats, axis=0)  # [L, H]
            attn_per_prompt.append(LH)

        except Exception as e:
            log(f"  ✗ Model forward failed for sample {idx}: {e}")
            failed_count += 1
            continue

    # Check results
    if len(attn_per_prompt) == 0:
        log("\n✗ ERROR: No valid prompts produced attention matrices")
        log("Check regex/dataset or model configuration")
        sys.exit(1)

    if failed_count > 0:
        log(f"\n⚠️  {failed_count}/{len(prompts)} prompts failed")

    # Save results
    arr = np.stack(attn_per_prompt, axis=0)  # [P, L, H]
    mean_over_prompts = arr.mean(axis=0)  # [L, H]

    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, f"attn_{args.run_tag}.npz")
    meta = {"model_id": args.model_id, "run_tag": args.run_tag}

    np.savez(outp, mean_layer_head=mean_over_prompts, per_prompt=arr, meta=json.dumps(meta))

    log(f"\n✓ Saved: {outp}")
    log(f"  Shape: {mean_over_prompts.shape}")
    log(f"  Processed: {len(attn_per_prompt)}/{len(prompts)} samples")
    log("\n" + "=" * 70)
    log("✅ EXTRACTION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()