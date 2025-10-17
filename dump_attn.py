import argparse, os, json, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def find_spans(text, pats):
    spans=[]
    for pat in pats:
        for m in re.finditer(pat, text, flags=re.IGNORECASE|re.DOTALL):
            spans.append((m.start(), m.end()))
    if not spans: return []
    spans.sort()
    merged=[list(spans[0])]
    for s,e in spans[1:]:
        if s<=merged[-1][1]:
            merged[-1][1]=max(merged[-1][1],e)
        else: merged.append([s,e])
    return [tuple(x) for x in merged]

def char_to_token_idxs(text, tokenizer, spans):
    if not spans: return []
    enc=tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    idxs=[]
    for tid,(s,e) in enumerate(enc["offset_mapping"]):
        for cs,ce in spans:
            if not (e<=cs or s>=ce):
                idxs.append(tid); break
    return sorted(set(idxs))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--run_tag", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_length", type=int, default=4096)
    args=ap.parse_args()

    # 1) regex for instruction spans (英文 IFEval)
    regex_list=[r"two words", r"start with", r"json", r"format", r"only output.*", r"no explanation", r"do not explain", r"output only"]

    # 2) tokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.padding_side="left"; tokenizer.truncation_side="left"

    # 3) model（强制使用 eager attention 实现，关闭 sdpa/flash）
    #    这样 output_attentions=True 才能拿到注意力矩阵
    #    注意：以下三行确保 PyTorch 不走 flash/sdpa 快速路径
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    model=AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 尝试切换到 eager 实现（Transformers 4.41+ 常见方法）
    try:
        model.set_attn_implementation("eager")
        # 某些模型还可以：model.config.attn_implementation = "eager"
    except Exception:
        pass

    model.eval()

    # 4) load prompts
    if not os.path.exists(args.prompts_jsonl) or os.path.getsize(args.prompts_jsonl)==0:
        raise ValueError(f"No prompts found at {args.prompts_jsonl}")
    prompts=[json.loads(x) for x in open(args.prompts_jsonl,"r",encoding="utf-8") if x.strip()]
    if len(prompts)==0:
        raise ValueError(f"No prompts loaded from {args.prompts_jsonl}. Check file content.")

    attn_per_prompt=[]
    for ex in prompts:
        text=ex.get("prompt") or ex.get("text")
        if not text:
            continue
        spans=find_spans(text, regex_list)
        idxs=char_to_token_idxs(text, tokenizer, spans)

        toks=tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
        input_ids=toks["input_ids"].to(model.device)
        attn_mask=toks["attention_mask"].to(model.device)

        with torch.no_grad():
            out=model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)

        attentions = out.attentions  # list of [bs, heads, tgt, src] or None

        # 如果还是 None，直接给出明确指引
        if attentions is None:
            raise RuntimeError(
                "Got attentions=None. Your backend is still not in 'eager' mode.\n"
                "Fixes to try:\n"
                "  1) Ensure transformers>=4.41, and call model.set_attn_implementation('eager').\n"
                "  2) Disable Flash/SDPA kernels via torch.backends.cuda.enable_flash_sdp(False), etc.\n"
                "  3) If using quantized backends that swallow attentions, switch to a HF model that supports attentions.\n"
            )

        mats=[]
        for A in attentions:    # [bs, heads, tgt, src]
            H=A[0]             # [heads, tgt, src]
            if len(idxs)==0:
                v=torch.zeros(H.size(0), dtype=torch.float16, device=H.device)
            else:
                v=H[..., idxs].mean(dim=(1,2))
            mats.append(v.detach().float().cpu().numpy())
        LH=np.stack(mats,axis=0)   # [L,H]
        attn_per_prompt.append(LH)

    if len(attn_per_prompt)==0:
        raise ValueError("No valid prompts produced attention matrices. Check regex/dataset.")

    arr=np.stack(attn_per_prompt,axis=0)      # [P,L,H]
    mean_over_prompts=arr.mean(axis=0)        # [L,H]
    os.makedirs(args.out_dir, exist_ok=True)
    outp=os.path.join(args.out_dir, f"attn_{args.run_tag}.npz")
    meta={"model_id":args.model_id,"run_tag":args.run_tag}
    np.savez(outp, mean_layer_head=mean_over_prompts, per_prompt=arr, meta=json.dumps(meta))
    print("Saved:", outp, "shape:", mean_over_prompts.shape)

if __name__=="__main__":
    main()
