import argparse, os, json, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def find_spans(text, pats):
    spans = []
    for pat in pats:
        for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
            spans.append((m.start(), m.end()))
    if not spans: return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def char_to_token_idxs(text, tokenizer, spans):
    if not spans: return []
    # Include special tokens to get correct offset mapping
    enc = tokenizer(text, return_offsets_mapping=True)
    idxs = []
    for tid, (s, e) in enumerate(enc["offset_mapping"]):
        # Skip special tokens (they have offset (0, 0))
        if s == 0 and e == 0:
            continue
        for cs, ce in spans:
            if not (e <= cs or s >= ce):
                idxs.append(tid);
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

    # 1) regex for instruction spans (è‹±æ–‡ IFEval)
    regex_list = [r"two words", r"start with", r"json", r"format", r"only output.*", r"no explanation",
                  r"do not explain", r"output only"]

    # 2) tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.padding_side = "left";
    tokenizer.truncation_side = "left"

    # 3) modelï¼ˆå¼ºåˆ¶ä½¿ç”¨ eager attention å®žçŽ°ï¼Œå…³é—­ sdpa/flashï¼‰
    #    è¿™æ · output_attentions=True æ‰èƒ½æ‹¿åˆ°æ³¨æ„åŠ›çŸ©é˜µ
    #    æ³¨æ„ï¼šä»¥ä¸‹ä¸‰è¡Œç¡®ä¿ PyTorch ä¸èµ° flash/sdpa å¿«é€Ÿè·¯å¾„
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # å°è¯•åˆ‡æ¢åˆ° eager å®žçŽ°ï¼ˆTransformers 4.41+ å¸¸è§æ–¹æ³•ï¼‰
    try:
        model.set_attn_implementation("eager")
        # æŸäº›æ¨¡åž‹è¿˜å¯ä»¥ï¼šmodel.config.attn_implementation = "eager"
    except Exception:
        pass

    model.eval()

    # 4) load prompts
    if not os.path.exists(args.prompts_jsonl) or os.path.getsize(args.prompts_jsonl) == 0:
        raise ValueError(f"No prompts found at {args.prompts_jsonl}")
    prompts = [json.loads(x) for x in open(args.prompts_jsonl, "r", encoding="utf-8") if x.strip()]
    if len(prompts) == 0:
        raise ValueError(f"No prompts loaded from {args.prompts_jsonl}. Check file content.")

    attn_per_prompt = []
    for ex in prompts:
        text = ex.get("prompt") or ex.get("text")
        if not text:
            continue

        # Apply chat template (same as lm-eval does)
        messages = [{"role": "user", "content": text}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Find instruction spans in the ORIGINAL text
        spans_in_original = find_spans(text, regex_list)

        # Map spans to the formatted prompt
        # Find where the original text appears in the formatted prompt
        start_pos = formatted_prompt.find(text)
        if start_pos == -1:
            # If exact match fails, try to find spans in formatted_prompt directly
            spans = find_spans(formatted_prompt, regex_list)
        else:
            # Adjust span positions to account for chat template prefix
            spans = [(s + start_pos, e + start_pos) for (s, e) in spans_in_original]

        # Find token indices for instruction spans in the FORMATTED prompt
        idxs = char_to_token_idxs(formatted_prompt, tokenizer, spans)

        # Tokenize the FORMATTED prompt (with chat template)
        toks = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids = toks["input_ids"].to(model.device)
        attn_mask = toks["attention_mask"].to(model.device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

        attentions = out.attentions  # list of [bs, heads, tgt, src] or None

        # å¦‚æžœè¿˜æ˜¯ Noneï¼Œç›´æŽ¥ç»™å‡ºæ˜Žç¡®æŒ‡å¼•
        if attentions is None:
            raise RuntimeError(
                "Got attentions=None. Your backend is still not in 'eager' mode.\n"
                "Fixes to try:\n"
                "  1) Ensure transformers>=4.41, and call model.set_attn_implementation('eager').\n"
                "  2) Disable Flash/SDPA kernels via torch.backends.cuda.enable_flash_sdp(False), etc.\n"
                "  3) If using quantized backends that swallow attentions, switch to a HF model that supports attentions.\n"
            )

        mats = []
        for A in attentions:  # [bs, heads, tgt, src]
            H = A[0]  # [heads, tgt, src]
            if len(idxs) == 0:
                v = torch.zeros(H.size(0), dtype=torch.float16, device=H.device)
            else:
                v = H[..., idxs].mean(dim=(1, 2))
            mats.append(v.detach().float().cpu().numpy())
        LH = np.stack(mats, axis=0)  # [L,H]
        attn_per_prompt.append(LH)

    if len(attn_per_prompt) == 0:
        raise ValueError("No valid prompts produced attention matrices. Check regex/dataset.")

    arr = np.stack(attn_per_prompt, axis=0)  # [P,L,H]
    mean_over_prompts = arr.mean(axis=0)  # [L,H]
    os.makedirs(args.out_dir, exist_ok=True)
    outp = os.path.join(args.out_dir, f"attn_{args.run_tag}.npz")
    meta = {"model_id": args.model_id, "run_tag": args.run_tag}
    np.savez(outp, mean_layer_head=mean_over_prompts, per_prompt=arr, meta=json.dumps(meta))
    print("Saved:", outp, "shape:", mean_over_prompts.shape)


if __name__ == "__main__":
    main()