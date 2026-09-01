#!/usr/bin/env python3
"""
IFEval evaluation with attention-head compensation.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


# -----------------------------
# 映射量化路径 -> 原始 tokenizer id
# -----------------------------
def get_fallback_tokenizer_path(model_path: str):
    base = os.path.basename(model_path.rstrip("/"))
    candidates = [base]
    if base.startswith("quantized_"):
        candidates.append(base[len("quantized_"):])

    fallback_map = [
        (r"Qwen_Qwen2\.5-0\.5B-Instruct_\d+bit$", "Qwen/Qwen2.5-0.5B-Instruct"),
        (r"Qwen_Qwen2\.5-1\.5B-Instruct_\d+bit$", "Qwen/Qwen2.5-1.5B-Instruct"),
        (r"Qwen_Qwen2\.5-3B-Instruct_\d+bit$", "Qwen/Qwen2.5-3B-Instruct"),
        (r"Qwen_Qwen2\.5-7B-Instruct_\d+bit$", "Qwen/Qwen2.5-7B-Instruct"),
        (r"Qwen_Qwen2\.5-14B-Instruct_\d+bit$", "Qwen/Qwen2.5-14B-Instruct"),
    ]

    for cand in candidates:
        for pattern, fallback_name in fallback_map:
            if re.search(pattern, cand):
                print(f"[INFO] Using fallback tokenizer '{fallback_name}' for model path '{model_path}'")
                return fallback_name

    print(f"[INFO] No fallback tokenizer mapping for {model_path}, will use model_path as tokenizer")
    return None


# -----------------------------
# IFEval sample
# -----------------------------
@dataclass
class IFEvalSample:
    key: str
    prompt: str
    instruction_id_list: List[int]
    kwargs: Dict


def load_ifeval_data(jsonl_path: str, max_samples: int = None) -> List[IFEvalSample]:
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and len(data) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(
                IFEvalSample(
                    key=obj.get("key", f"sample_{i}"),
                    prompt=obj.get("prompt", ""),
                    instruction_id_list=obj.get("instruction_id_list", []),
                    kwargs=obj.get("kwargs", {}),
                )
            )
    return data


# -----------------------------
# 简单 scoring：先占位
# -----------------------------
def check_instruction_following(sample: IFEvalSample, answer: str) -> float:
    """Placeholder scoring function"""
    answer = (answer or "").strip()
    if not answer:
        return 0.0
    return 1.0


# -----------------------------
# ✅ 修复1: 去掉 self，修正参数顺序
# -----------------------------
def evaluate_sample(sample: IFEvalSample, model, tokenizer, max_new_tokens=256):
    """Evaluate a single sample with chat template"""

    prompt = sample.prompt
    if not prompt:
        return None

    # Apply chat template (like lm-eval does)
    messages = [
        {
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize formatted prompt
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    ).to(model.device)

    # Generate with correct parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return answer


# -----------------------------
# ✅ 修复2: AttentionCompensator 完整版
# -----------------------------
class AttentionCompensator:
    """Attention compensation using 2D mean_layer_head differences"""

    def __init__(self, attn_fp16, attn_quant, top_heads, alpha):
        """
        Args:
            attn_fp16: [L, H] FP16 attention values
            attn_quant: [L, H] Quantized attention values
            top_heads: List of (layer, head) tuples
            alpha: Compensation strength
        """
        self.attn_fp16 = attn_fp16
        self.attn_quant = attn_quant
        self.attn_diff = attn_fp16 - attn_quant
        self.top_heads = top_heads
        self.alpha = alpha
        self.n_layers, self.n_heads = attn_fp16.shape
        self.handles = []

    def register_hooks(self, model):
        """Register hooks to compensate attention output"""

        # Get config for num_heads
        if hasattr(model, 'config'):
            config = model.config
        elif hasattr(model, 'model') and hasattr(model.model, 'config'):
            config = model.model.config
        else:
            print("[ERROR] Cannot find model config!")
            return

        num_heads = getattr(config, 'num_attention_heads', None)
        hidden_size = getattr(config, 'hidden_size', None)

        if num_heads is None or hidden_size is None:
            print(f"[ERROR] Cannot get head config")
            return

        print(f"[INFO] Model config: {num_heads} heads, {hidden_size} hidden_size")

        def make_hook(layer_idx, head_idx, num_heads, hidden_size):
            """Create hook with num_heads from config"""

            def hook(module, input, output):
                # Get output
                if isinstance(output, tuple):
                    attn_output = output[0]
                else:
                    attn_output = output

                batch_size, seq_len, hidden_dim = attn_output.shape

                # Get head_dim from module or calculate
                if hasattr(module, 'head_dim'):
                    head_dim = module.head_dim
                else:
                    head_dim = hidden_size // num_heads

                # Bounds check
                if layer_idx >= self.n_layers or head_idx >= num_heads:
                    return output

                # Get compensation
                diff_scalar = float(self.attn_diff[layer_idx, head_idx])
                if abs(diff_scalar) < 1e-10 or abs(self.alpha) < 1e-10:
                    return output

                # Compute scale factor
                quant_attn = self.attn_quant[layer_idx, head_idx]
                if quant_attn > 1e-8:
                    target_attn = quant_attn + self.alpha * diff_scalar
                    scale_factor = target_attn / quant_attn
                else:
                    scale_factor = 1.0

                # Reshape and scale
                attn_output = attn_output.view(batch_size, seq_len, num_heads, head_dim)
                with torch.no_grad():
                    attn_output[:, :, head_idx, :] *= scale_factor
                attn_output = attn_output.view(batch_size, seq_len, hidden_dim)

                # Return
                if isinstance(output, tuple):
                    return (attn_output,) + output[1:]
                else:
                    return attn_output

            return hook

        # Find layers
        self.handles = []
        if hasattr(model, 'model'):
            base = model.model
        else:
            base = model

        if hasattr(base, 'model') and hasattr(base.model, 'layers'):
            layers = base.model.layers
        elif hasattr(base, 'layers'):
            layers = base.layers
        else:
            print("[ERROR] Cannot find layers!")
            return

        print(f"[INFO] Found {len(layers)} layers")

        # Register hooks
        for (layer_idx, head_idx) in self.top_heads:
            try:
                if layer_idx >= len(layers):
                    continue

                attn_module = layers[layer_idx].self_attn
                h = attn_module.register_forward_hook(
                    make_hook(layer_idx, head_idx, num_heads, hidden_size)
                )
                self.handles.append(h)
                print(f"✓ Registered hook on layer {layer_idx}, head {head_idx}")
            except Exception as e:
                print(f"✗ Failed on layer {layer_idx}, head {head_idx}: {e}")

    def remove_hooks(self):
        """Remove all registered hooks"""
        for h in self.handles:
            h.remove()
        self.handles = []


# -----------------------------
# ✅ 修复3: evaluate_ifeval 正确调用
# -----------------------------
def evaluate_ifeval(
        model,
        tokenizer,
        data: List[IFEvalSample],
        attn_fp16: np.ndarray = None,
        attn_quant: np.ndarray = None,
        top_heads: List[Tuple[int, int]] = None,
        alpha: float = 0.0,
        max_new_tokens: int = 512,
) -> Dict:
    """Evaluate IFEval with optional attention compensation"""

    # Setup compensation if needed
    if attn_fp16 is not None and attn_quant is not None and top_heads is not None and abs(alpha) > 1e-8:
        compensator = AttentionCompensator(attn_fp16, attn_quant, top_heads, alpha)
        compensator.register_hooks(model)
        print(f"[INFO] Compensation enabled with alpha={alpha}")
    else:
        compensator = None
        print("[INFO] No compensation (baseline)")

    scores = []
    per_sample = []

    for i, sample in enumerate(data):
        print(f"\n----- Sample {i + 1}/{len(data)}: {sample.key} -----")
        print("Prompt:", sample.prompt[:200].replace("\n", " "))

        # ✅ 正确的调用顺序
        answer = evaluate_sample(sample, model, tokenizer, max_new_tokens=max_new_tokens)
        print("Answer:", (answer or "")[:200].replace("\n", " "))

        score = check_instruction_following(sample, answer)
        scores.append(score)
        per_sample.append({
            "key": sample.key,
            "prompt": sample.prompt,
            "answer": answer,
            "score": score,
            "instruction_id_list": sample.instruction_id_list,
            "kwargs": sample.kwargs,
        })

    # Clean up hooks
    if compensator is not None:
        compensator.remove_hooks()

    scores = np.array(scores, dtype=float)
    results = {
        "mean_score": float(scores.mean()) if len(scores) > 0 else 0.0,
        "num_samples": int(len(scores)),
        "per_sample": per_sample,
        "alpha": alpha,
    }
    return results


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="IFEval with attention compensation")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", default=None)
    ap.add_argument("--ifeval_data", required=True)
    ap.add_argument("--attn_fp16", required=True)
    ap.add_argument("--attn_quant", required=True)
    ap.add_argument("--top_heads", required=True)
    ap.add_argument("--alpha_list", type=str, default="0.0,0.5,1.0")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("=" * 70)
    print("IFEval with Attention Compensation")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    print(f"Tokenizer path: {args.tokenizer_path}")
    print(f"IFEval data: {args.ifeval_data}")
    print(f"FP16 attention: {args.attn_fp16}")
    print(f"Quant attention: {args.attn_quant}")
    print(f"Top heads: {args.top_heads}")
    print(f"Alpha list: {args.alpha_list}")
    print(f"Max samples: {args.max_samples}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Output: {args.output}")

    # 1) Load data
    print("\n1. Loading IFEval data...")
    data = load_ifeval_data(args.ifeval_data, max_samples=args.max_samples)
    print(f"   Loaded {len(data)} samples")

    # 2) Load model + tokenizer
    print("\n2. Loading model & tokenizer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Using device: {device}")

    # Determine tokenizer source
    if args.tokenizer_path is not None:
        tokenizer_src = args.tokenizer_path
    else:
        tokenizer_src = get_fallback_tokenizer_path(args.model_path) or args.model_path
    print(f"   Tokenizer source: {tokenizer_src}")

    # Check if GPTQ model
    quant_config_path = os.path.join(args.model_path, "quantize_config.json")
    is_gptq = os.path.exists(quant_config_path)

    if is_gptq:
        print(f"   Detected GPTQ model (found {quant_config_path})")
        try:
            from auto_gptq import AutoGPTQForCausalLM
        except ImportError as e:
            raise RuntimeError(
                "GPTQ model detected but auto_gptq not installed"
            ) from e

        model = AutoGPTQForCausalLM.from_quantized(
            args.model_path,
            device_map={"": 0} if device == "cuda" else None,
            trust_remote_code=True,
            use_triton=False,
        )
        print("   Loaded GPTQ model")
    else:
        print("   Loading as normal HF model")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map={"": 0} if device == "cuda" else None,
            trust_remote_code=True,
        )
        print("   Loaded HF model")

    print(f"   Loading tokenizer from: {tokenizer_src}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3) Load attention tensors
    print("\n3. Loading attention tensors...")
    fp16_npz = np.load(args.attn_fp16)
    quant_npz = np.load(args.attn_quant)

    if "mean_layer_head" not in fp16_npz or "mean_layer_head" not in quant_npz:
        raise KeyError("mean_layer_head not found in npz files")

    fp16_mean = fp16_npz["mean_layer_head"]  # [L, H]
    quant_mean = quant_npz["mean_layer_head"]

    if fp16_mean.shape != quant_mean.shape:
        raise ValueError(f"Shape mismatch: fp16 {fp16_mean.shape}, quant {quant_mean.shape}")

    print(f"   Attention shape: {fp16_mean.shape}")

    # 4) Load top heads
    print("\n4. Loading top-K heads...")
    with open(args.top_heads, "r", encoding="utf-8") as f:
        top_heads_json = json.load(f)

    if "critical_heads" in top_heads_json:
        crit_list = top_heads_json["critical_heads"]
    else:
        crit_list = top_heads_json

    top_heads = [(int(h["layer"]), int(h["head"])) for h in crit_list]
    print(f"   Top heads: {top_heads}")

    # 5) Run evaluation for each alpha
    print("\n5. Running evaluation...")
    alpha_values = [float(x) for x in args.alpha_list.split(",") if x.strip()]
    all_results = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_src,
        "ifeval_data": args.ifeval_data,
        "attn_fp16": args.attn_fp16,
        "attn_quant": args.attn_quant,
        "top_heads_json": top_heads_json,
        "alpha_list": alpha_values,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "results": {},
    }

    for alpha in alpha_values:
        print(f"\n[RUN] alpha = {alpha}")
        if abs(alpha) < 1e-8:
            # Baseline: no compensation
            res = evaluate_ifeval(
                model, tokenizer, data,
                attn_fp16=None,
                attn_quant=None,
                top_heads=None,
                alpha=0.0,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            # ✅ 传入正确的参数
            res = evaluate_ifeval(
                model, tokenizer, data,
                attn_fp16=fp16_mean,
                attn_quant=quant_mean,
                top_heads=top_heads,
                alpha=alpha,
                max_new_tokens=args.max_new_tokens,
            )
        all_results["results"][f"alpha_{alpha}"] = res
        print(f"   alpha={alpha} mean score: {res['mean_score']:.4f}")

    # 6) Save results
    print("\n6. Saving results...")
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"   Saved to {args.output}")
    print("\nAll done.")


if __name__ == "__main__":
    main()