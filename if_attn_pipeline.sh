#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N IFATTN_Qwen2.5_7B_full

set -e

#############################
# 0) Paths & Variables
#############################
BASE_DIR="/users/jzheng7/ifattn"
ARTI_DIR="$BASE_DIR/artifacts"
mkdir -p "$ARTI_DIR"

# Conda environment name
CONDA_ENV="IFEval"

# ---- your actual files ----
SAMPLES_JSONL="/users/jzheng7/ifattn/selected_prompts.jsonl"   # 已经生成好的样本文件
FP16_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"                     # HuggingFace 模型ID

# GPTQ模型路径 (不同bit)
GPTQ2_MODEL_PATH="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_2bit"
GPTQ3_MODEL_PATH="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_3bit"
GPTQ4_MODEL_PATH="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_4bit"
GPTQ8_MODEL_PATH="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_8bit"

# lm-eval results.json 路径
FP16_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct/results_2025-10-09T13-58-12.528717.json"
GPTQ2_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_2bit/results_2025-10-22T15-55-18.912810.json"
GPTQ3_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_3bit/results_2025-10-18T02-53-19.455189.json"
GPTQ4_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_4bit/results_2025-10-14T16-06-35.354326.json"
GPTQ8_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_8bit/results_2025-10-22T15-46-23.674011.json"

TOPK=10   # 样本数（你生成的就是10条）

#############################
# 1) Environment
#############################
echo "[IFATTN] Activate conda: $CONDA_ENV"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"

#pip install -U --no-input torch transformers accelerate einops matplotlib numpy scipy pandas tqdm pyyaml
#pip install -U --no-input auto-gptq optimum autoawq || true

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#############################
# 2) Verify selected_prompts.jsonl
#############################
if [[ ! -s "$SAMPLES_JSONL" ]]; then
  echo "[ERROR] No samples found at $SAMPLES_JSONL"
  exit 1
fi
echo "[OK] Using selected prompts: $SAMPLES_JSONL"

#############################
# 3) Dump attentions (FP16 / GPTQ 2/3/4/8 bit)
#############################
echo "[IFATTN] Dump FP16 attentions"
python "$BASE_DIR/dump_attn.py" \
  --model_id "$FP16_MODEL_ID" \
  --run_tag "fp16" \
  --prompts_jsonl "$SAMPLES_JSONL" \
  --out_dir "$ARTI_DIR"

if [[ -n "$GPTQ2_MODEL_PATH" && -d "$GPTQ2_MODEL_PATH" ]]; then
  echo "[IFATTN] Dump GPTQ-2bit attentions"
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$GPTQ2_MODEL_PATH" \
    --run_tag "gptq2" \
    --prompts_jsonl "$SAMPLES_JSONL" \
    --out_dir "$ARTI_DIR"
fi

if [[ -n "$GPTQ3_MODEL_PATH" && -d "$GPTQ3_MODEL_PATH" ]]; then
  echo "[IFATTN] Dump GPTQ-3bit attentions"
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$GPTQ3_MODEL_PATH" \
    --run_tag "gptq3" \
    --prompts_jsonl "$SAMPLES_JSONL" \
    --out_dir "$ARTI_DIR"
fi

if [[ -n "$GPTQ4_MODEL_PATH" && -d "$GPTQ4_MODEL_PATH" ]]; then
  echo "[IFATTN] Dump GPTQ-4bit attentions"
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$GPTQ4_MODEL_PATH" \
    --run_tag "gptq4" \
    --prompts_jsonl "$SAMPLES_JSONL" \
    --out_dir "$ARTI_DIR"
fi

if [[ -n "$GPTQ8_MODEL_PATH" && -d "$GPTQ8_MODEL_PATH" ]]; then
  echo "[IFATTN] Dump GPTQ-8bit attentions"
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$GPTQ8_MODEL_PATH" \
    --run_tag "gptq8" \
    --prompts_jsonl "$SAMPLES_JSONL" \
    --out_dir "$ARTI_DIR"
fi

#############################
# 4) Parse IFEval results → CSV
#############################
IFCSV="$ARTI_DIR/ifeval_scores_all_models.csv"
rm -f "$IFCSV"

if [[ -f "$FP16_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$FP16_RESULTS_JSON" --out_csv "$IFCSV" --tag "fp16"
fi

if [[ -f "$GPTQ2_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$GPTQ2_RESULTS_JSON" --out_csv "$IFCSV" --tag "gptq2"
fi

if [[ -f "$GPTQ3_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$GPTQ3_RESULTS_JSON" --out_csv "$IFCSV" --tag "gptq3"
fi

if [[ -f "$GPTQ4_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$GPTQ4_RESULTS_JSON" --out_csv "$IFCSV" --tag "gptq4"
fi

if [[ -f "$GPTQ8_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$GPTQ8_RESULTS_JSON" --out_csv "$IFCSV" --tag "gptq8"
fi

#############################
# 5) Compute ISI & Correlation
#############################
python "$BASE_DIR/compute_metrics.py" \
  --attn_glob "$ARTI_DIR/attn_*.npz" \
  --ifeval_csv "$IFCSV" \
  --out_dir "$ARTI_DIR"

#############################
# 6) Visualization
#############################
python "$BASE_DIR/viz.py" \
  --attn_glob "$ARTI_DIR/attn_*.npz" \
  --out_dir "$ARTI_DIR"

#############################
# 7) Analyze by Instruction Type
#############################
echo "[IFATTN] Analyzing by instruction type"
python "$BASE_DIR/analyze_by_instruction_type.py" \
  --prompts_jsonl "$SAMPLES_JSONL" \
  --attn_glob "$ARTI_DIR/attn_*.npz" \
  --out_dir "$ARTI_DIR/by_instruction_type"

echo "[IFATTN] ✅ All done. Check outputs in: $ARTI_DIR"
