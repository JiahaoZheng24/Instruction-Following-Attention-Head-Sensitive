#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N GPTQ4_COMPENSATION

set -e

echo "========================================"
echo "GPTQ4 Compensation Experiment"
echo "========================================"
date

#############################
# 0) Paths & Variables
#############################
BASE_DIR="/users/jzheng7/ifattn"
ARTI_DIR="$BASE_DIR/artifacts/gptq4_compensation"
mkdir -p "$ARTI_DIR"

# Conda environment
CONDA_ENV="attention"

# Models
FP16_MODEL="Qwen/Qwen2.5-7B-Instruct"
QUANT_MODEL="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_4bit"

# IFEval samples
FP16_SAMPLES="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct/samples_ifeval_2025-10-09T13-58-12.528717.jsonl"
QUANT_SAMPLES="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_4bit/samples_ifeval_2025-10-14T16-06-35.354326.jsonl"

#############################
# 1) Environment
#############################
echo "[1] Activating conda environment..."
conda activate "$CONDA_ENV"

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#############################
# 2) Select samples
#############################
echo "[2] Selecting FP16✓ GPTQ4✗ samples..."
python "$BASE_DIR/select_samples_for_compensation.py" \
  --fp16_samples "$FP16_SAMPLES" \
  --quant_samples "$QUANT_SAMPLES" \
  --strategy failure_only \
  --max_samples 5 \
  --output "$ARTI_DIR/selected_prompts.jsonl"

#############################
# 3) Dump attentions (if not exists)
#############################
if [[ ! -f "$ARTI_DIR/attn_fp16.npz" ]]; then
  echo "[3a] Dumping FP16 attention..."
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$FP16_MODEL" \
    --run_tag "fp16" \
    --prompts_jsonl "$ARTI_DIR/selected_prompts.jsonl" \
    --out_dir "$ARTI_DIR"
else
  echo "[3a] FP16 attention already exists, skipping..."
fi

if [[ ! -f "$ARTI_DIR/attn_gptq4.npz" ]]; then
  echo "[3b] Dumping GPTQ4 attention..."
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$QUANT_MODEL" \
    --run_tag "gptq4" \
    --prompts_jsonl "$ARTI_DIR/selected_prompts.jsonl" \
    --out_dir "$ARTI_DIR"
else
  echo "[3b] GPTQ4 attention already exists, skipping..."
fi

#############################
# 4) Identify critical heads
#############################
echo "[4] Identifying critical heads..."
python "$BASE_DIR/identify_critical_heads.py" \
  --fp16_attn "$ARTI_DIR/attn_fp16.npz" \
  --quant_attn "$ARTI_DIR/attn_gptq4.npz" \
  --top_k 10 \
  --out_dir "$ARTI_DIR"

#############################
# 5) Run compensation experiments
#############################
echo "[5] Running compensation experiments..."
for ALPHA in 0.0 5.0 10.0 20.0; do
  echo "  Testing alpha=$ALPHA"
  python "$BASE_DIR/eval_ifeval_with_compensation.py" \
    --model_path "$QUANT_MODEL" \
    --ifeval_data "$ARTI_DIR/selected_prompts.jsonl" \
    --attn_fp16 "$ARTI_DIR/attn_fp16.npz" \
    --attn_quant "$ARTI_DIR/attn_gptq4.npz" \
    --top_heads "$ARTI_DIR/critical_heads_gptq4.json" \
    --alpha_list "$ALPHA" \
    --max_samples 5 \
    --max_new_tokens 1280 \
    --output "$ARTI_DIR/compensation_alpha${ALPHA}.jsonl"
done

#############################
# 6) Analyze results
#############################
echo "[6] Analyzing compensation results..."
python "$BASE_DIR/analyze_compensation_results.py" \
  --baseline "$ARTI_DIR/compensation_alpha0.0.jsonl" \
  --compensated "$ARTI_DIR/compensation_alpha0.5.jsonl" \
               "$ARTI_DIR/compensation_alpha1.0.jsonl" \
               "$ARTI_DIR/compensation_alpha2.0.jsonl" \
  --output "$ARTI_DIR/compensation_summary.csv"

echo "========================================"
echo "✅ GPTQ4 Compensation Experiment Complete!"
echo "========================================"
echo "Results saved to: $ARTI_DIR"
date

#cd /users/jzheng7/ifattn
#conda activate attention
## 1. 查看选中的样本key
#jq '.key' /users/jzheng7/ifattn/artifacts/gptq4_compensation/selected_prompts.jsonl
#
## 2. 查看baseline结果的前10行
#head -20 /users/jzheng7/ifattn/artifacts/gptq4_compensation/compensation_alpha0.0.jsonl
#
## 3. 检查这些key在原始GPTQ4结果中的follow_instruction_list
#grep -A 5 '"key": 12' /users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_4bit/samples_ifeval_2025-10-14T16-06-35.354326.jsonl


#python - <<'PY'
#from transformers import AutoModelForCausalLM, AutoTokenizer
#
#model_path = "/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_3bit"
#tokenizer_path = "Qwen/Qwen2.5-7B-Instruct"
#
#print("Loading model...")
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
#model = AutoModelForCausalLM.from_pretrained(
#    model_path,
#    device_map="auto",
#    torch_dtype="auto",
#    trust_remote_code=True,
#)
#
## Use the EXACT same prompt from IFEval key=1012
#prompt = "Write an email to my boss telling him that I am quitting. The email must contain a title wrapped in double angular brackets, i.e. <<title>>.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)"
#
#messages = [
#    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#    {"role": "user", "content": prompt}
#]
#
#formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
#
#print("Generating (like lm-eval: max_gen_toks=1280, temperature=0.0)...")
#outputs = model.generate(
#    **inputs,
#    max_new_tokens=1280,  # ← 改成和lm-eval一样大
#    do_sample=False,
#    temperature=0.0,
#    pad_token_id=tokenizer.pad_token_id,
#    eos_token_id=tokenizer.eos_token_id,
#)
#
#input_len = inputs.input_ids.shape[1]
#generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
#
#print("="*70)
#print("Generated:")
#print(generated[:500])  # 显示前500字符
#print("="*70)
#
## Check for degenerate patterns
#if "!" * 10 in generated:
#    print("❌ Found exclamation marks")
#elif "ospel" in generated.lower() or len(set(generated.split()[:10])) < 3:
#    print("❌ Found repetitive pattern")
#elif "<<" in generated and ">>" in generated:
#    print("✅ Contains title markers - looks good!")
#else:
#    print("⚠️ Generated something, but check manually")
#PY



