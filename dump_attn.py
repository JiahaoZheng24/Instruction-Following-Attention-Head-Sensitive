#!/usr/bin/env python3
"""
Extract attention weights focusing on instruction tokens
Robust version with extensive error handling and debugging
"""
import argparse
import os
import json
import re
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:
    AutoGPTQForCausalLM = None


def log(msg):
    """Print with flush for real-time logging in qsub"""
    print(msg, flush=True)


def find_spans(text, pats):
    """Find character spans matching regex patterns"""
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
    parser = argparse.ArgumentParser(description="Extract attention on instruction tokens")
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID or path")
    parser.add_argument("--run_tag", required=True, help="Tag for this run (e.g., fp16, gptq3)")
    parser.add_argument("--prompts_jsonl", required=True, help="Path to prompts JSONL file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    # 新增：可选的 tokenizer_id，默认用 model_id
    parser.add_argument(
        "--tokenizer_id",
        default=None,
        help="Optional separate tokenizer id/path (default: use model_id)",
    )
    args = parser.parse_args()

    log("=" * 70)
    log(f"ATTENTION EXTRACTION: {args.run_tag}")
    log("=" * 70)

    # Instruction regex patterns
    regex_list = [
        # 原有的patterns
        r"two words", r"start with", r"json", r"format",
        r"only output.*", r"no explanation", r"do not explain", r"output only",

        # 新增patterns - 匹配更多instruction类型
        r"title", r"wrapped in", r"double angular brackets",
        r"repeat", r"word for word",
        r"bullet point", r"numbered list",
        r"all lowercase", r"all capital",
        r"less than \d+ words", r"at least \d+ words",
        r"postscript", r"P\.S\.",
        r"first word", r"end with",
        r"placeholder", r"\[.*?\]",
    ]
    log(f"Instruction patterns: {len(regex_list)} patterns")

    # Step 1: Load tokenizer
    tok_id = args.tokenizer_id or args.model_id
    log(f"\n[1/5] Loading tokenizer from: {tok_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_id,
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        log("✓ Tokenizer loaded successfully")
    except Exception as e:
        log(f"✗ FAILED to load tokenizer: {e}")
        sys.exit(1)

    # Step 2: Disable Flash Attention
    log("\n[2/5] Disabling Flash Attention backends...")
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        log("✓ Flash/SDPA disabled, using math backend")
    except Exception as e:
        log(f"⚠ Warning: Could not disable Flash/SDPA: {e}")

    # Step 3: Load model with extensive error handling
    log(f"\n[3/5] Loading model: {args.model_id}")
    log("This may take several minutes...")

    try:
        # Check CUDA availability
        if not torch.cuda.is_available():
            log("⚠ WARNING: CUDA not available, using CPU (will be slow)")
            device_map = "cpu"
        else:
            log(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            device_map = "auto"

        # Decide whether this is a GPTQ quantized model or a normal HF model
        quant_config_path = os.path.join(args.model_id, "quantize_config.json")
        is_gptq = os.path.exists(quant_config_path)

        if is_gptq:
            log(f"   Detected GPTQ checkpoint (found {quant_config_path})")
            if AutoGPTQForCausalLM is None:
                raise RuntimeError(
                    "This looks like a GPTQ quantized model, but auto_gptq is not installed "
                    "in this environment. Please install auto-gptq or run in the lm-eval env."
                )
            # 用 AutoGPTQ 正确加载 3bit 模型
            model = AutoGPTQForCausalLM.from_quantized(
                args.model_id,
                device_map=device_map,
                trust_remote_code=True,
                use_triton=False,
            )
            log("✓ GPTQ model loaded via AutoGPTQForCausalLM.from_quantized(...)")
        else:
            # 正常 FP16 / 非量化模型走原来的 HF 路径
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            log("✓ FP16 / non-GPTQ model loaded via AutoModelForCausalLM.from_pretrained(...)")

        # Print model info
        log(f"  Model type: {type(model).__name__}")
        log(f"  Device: {next(model.parameters()).device}")

    except Exception as e:
        log(f"✗ FAILED to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Set eager attention
    log("\n[4/5] Configuring attention mechanism...")
    try:
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
            log("✓ Set attention implementation to 'eager'")
        elif hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"
            log("✓ Set config.attn_implementation = 'eager'")
        else:
            log("⚠ Could not set eager attention explicitly")
            log("  Will try output_attentions=True and see if it works")
    except Exception as e:
        log(f"⚠ Warning setting eager attention: {e}")

    model.eval()
    log("✓ Model set to eval mode")

    # Step 5: Load and process prompts
    log(f"\n[5/5] Processing prompts from: {args.prompts_jsonl}")

    if not os.path.exists(args.prompts_jsonl):
        log(f"✗ ERROR: File not found: {args.prompts_jsonl}")
        sys.exit(1)

    if os.path.getsize(args.prompts_jsonl) == 0:
        log(f"✗ ERROR: File is empty: {args.prompts_jsonl}")
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


        # 添加debug输出
        # 添加debug输出
        print(f"  Found {len(spans)} instruction spans (in formatted prompt): {spans[:3]}...")
        print(f"  Instruction token indices (first 10): {idxs[:10]}...")
        print(f"  Total instruction tokens: {len(idxs)}")

        # 如果没有找到instruction tokens，警告
        if len(idxs) == 0:
            print(f"  ⚠️  WARNING: No instruction tokens found!")
            print(f"  Prompt preview: {text[:200]}...")

        # Truncate very long texts
        if len(text) > 10000:
            log(f"  [WARNING] Prompt {idx} is very long ({len(text)} chars), truncating")
            text = text[:10000]

        try:
            # Find instruction spans
            spans = find_spans(text, regex_list)
            idxs = char_to_token_idxs(text, tokenizer, spans)

            # Tokenize
            toks = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            )
            input_ids = toks["input_ids"].to(model.device)
            attn_mask = toks["attention_mask"].to(model.device)

            # Run model with attention extraction
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_attentions=True,
                )

            # Check if attentions were returned
            if outputs.attentions is None:
                log(f"  [ERROR] Prompt {idx}: attentions is None!")
                log(f"  This means eager attention is not working properly")
                log(f"  Skipping this prompt...")
                failed_count += 1
                continue

            # Extract attention weights
            mats = []
            for A in outputs.attentions:  # [batch, heads, tgt, src]
                H = A[0]  # [heads, tgt, src]
                if len(idxs) == 0:
                    v = torch.zeros(H.size(0), dtype=torch.float16, device=H.device)
                else:
                    v = H[..., idxs].mean(dim=(1, 2))
                mats.append(v.detach().float().cpu().numpy())

            LH = np.stack(mats, axis=0)  # [L, H]
            attn_per_prompt.append(LH)

            # Clear cache periodically
            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            log(f"  [ERROR] Failed to process prompt {idx}: {str(e)[:100]}")
            failed_count += 1
            continue

    log(f"\n✓ Successfully processed {len(attn_per_prompt)}/{len(prompts)} prompts")
    if failed_count > 0:
        log(f"⚠ Failed to process {failed_count} prompts")

    if len(attn_per_prompt) == 0:
        log("✗ ERROR: No prompts were successfully processed")
        log("Check the error messages above for details")
        sys.exit(1)

    # Save results
    log("\nSaving results...")
    arr = np.stack(attn_per_prompt, axis=0)  # [P, L, H]
    mean_over_prompts = arr.mean(axis=0)  # [L, H]

    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"attn_{args.run_tag}.npz")

    meta = {
        "model_id": args.model_id,
        "run_tag": args.run_tag,
        "num_prompts": len(attn_per_prompt),
        "num_failed": failed_count,
    }

    np.savez(
        output_path,
        mean_layer_head=mean_over_prompts,
        per_prompt=arr,
        meta=json.dumps(meta),
    )

    log(f"✓ Saved to: {output_path}")
    log(f"  Shape: {mean_over_prompts.shape}")
    log(f"  Mean attention: {mean_over_prompts.mean():.6f}")
    log(f"  Std attention: {mean_over_prompts.std():.6f}")

    log("\n" + "=" * 70)
    log("✅ ATTENTION EXTRACTION COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()