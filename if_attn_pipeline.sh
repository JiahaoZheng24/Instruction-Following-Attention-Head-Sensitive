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
GPTQ_MODEL_PATH="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_4bit"     # 如果没有可留空 ""
AWQ_MODEL_PATH=""       # 如果没有可留空 ""

# lm-eval results.json 路径
FP16_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct/results_2025-10-09T13-58-12.528717.json"
GPTQ_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_4bit/results_2025-10-09T13-57-56.919923.json"
AWQ_RESULTS_JSON="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct-AWQ/results_2025-10-09T14-34-11.060643.json"

TOPK=10   # 样本数（你生成的就是10条）

#############################
# 1) Environment
#############################
echo "[IFATTN] Activate conda: $CONDA_ENV"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"

pip install -U --no-input torch transformers accelerate einops matplotlib numpy scipy pandas tqdm pyyaml
pip install -U --no-input auto-gptq optimum autoawq || true

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
# 3) Dump attentions (FP16 / GPTQ / AWQ)
#############################
echo "[IFATTN] Dump FP16 attentions"
python "$BASE_DIR/dump_attn.py" \
  --model_id "$FP16_MODEL_ID" \
  --run_tag "fp16" \
  --prompts_jsonl "$SAMPLES_JSONL" \
  --out_dir "$ARTI_DIR"

if [[ -n "$GPTQ_MODEL_PATH" ]]; then
  echo "[IFATTN] Dump GPTQ attentions"
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$GPTQ_MODEL_PATH" \
    --run_tag "gptq4" \
    --prompts_jsonl "$SAMPLES_JSONL" \
    --out_dir "$ARTI_DIR"
fi

if [[ -n "$AWQ_MODEL_PATH" ]]; then
  echo "[IFATTN] Dump AWQ attentions"
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$AWQ_MODEL_PATH" \
    --run_tag "awq4" \
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
if [[ -f "$GPTQ_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$GPTQ_RESULTS_JSON" --out_csv "$IFCSV" --tag "gptq4"
fi
if [[ -f "$AWQ_RESULTS_JSON" ]]; then
  python "$BASE_DIR/parse_ifeval.py" --results_json "$AWQ_RESULTS_JSON" --out_csv "$IFCSV" --tag "awq4"
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

echo "[IFATTN] ✅ All done. Check outputs in: $ARTI_DIR"
