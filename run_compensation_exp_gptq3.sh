#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@qa-a10-018.crc.nd.edu
#$ -l gpu_card=2
#$ -N COMPENSATION_GPTQ3

set -e

#############################
# Attention Compensation Experiment - GPTQ3
# Strategy A: FP16 vs GPTQ3
#############################

echo "================================"
echo "GPTQ3 COMPENSATION EXPERIMENT"
echo "================================"
echo "Start time: $(date)"
echo ""

#############################
# 0) Configuration
#############################

BASE_DIR="/users/jzheng7/ifattn"
ARTI_DIR="$BASE_DIR/artifacts"
COMP_DIR="$ARTI_DIR/compensation_3bit"
mkdir -p "$COMP_DIR"

# Conda environment
CONDA_ENV="attention"

# Models
FP16_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
GPTQ3_MODEL_PATH="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_3bit"

# Sample files (lm-eval outputs)
FP16_SAMPLES_JSONL="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct/samples_ifeval_2025-10-09T13-58-12.528717.jsonl"
GPTQ3_SAMPLES_JSONL="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_3bit/samples_ifeval_2025-10-18T02-53-19.455189.jsonl"

# Experiment parameters
TOP_K=10
ALPHA_VALUES="0.0 0.5 1.0"   # alpha 少一点

#############################
# 1) Environment Setup
#############################

echo "[STEP 1] Setting up environment..."
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#############################
# 2) Select Samples (FP16✓ & GPTQ3✗)
#############################

echo ""
echo "[STEP 2] Selecting failure samples (FP16✓ & GPTQ3✗)..."
echo "  FP16 samples: $FP16_SAMPLES_JSONL"
echo "  GPTQ3 samples: $GPTQ3_SAMPLES_JSONL"
echo ""

if [[ ! -f "$FP16_SAMPLES_JSONL" ]]; then
    echo "[ERROR] FP16 samples not found: $FP16_SAMPLES_JSONL"
    exit 1
fi

if [[ ! -f "$GPTQ3_SAMPLES_JSONL" ]]; then
    echo "[ERROR] GPTQ3 samples not found: $GPTQ3_SAMPLES_JSONL"
    exit 1
fi

SELECTED_PROMPTS="$COMP_DIR/selected_prompts.jsonl"

python "$BASE_DIR/select_samples_for_compensation.py" \
    --fp16_samples "$FP16_SAMPLES_JSONL" \
    --quant_samples "$GPTQ3_SAMPLES_JSONL" \
    --output "$SELECTED_PROMPTS" \
    --strategy failure_only \
    --max_samples 1

if [[ ! -f "$SELECTED_PROMPTS" ]]; then
    echo "[ERROR] Failed to select samples"
    exit 1
fi

NUM_SELECTED=$(wc -l < "$SELECTED_PROMPTS")
echo ""
echo "✓ Selected $NUM_SELECTED samples"

#############################
# 3) Dump Attentions
#############################

echo ""
echo "[STEP 3] Dumping attention data..."

# Dump FP16 attentions
FP16_ATTN="$COMP_DIR/attn_fp16.npz"

if [[ ! -f "$FP16_ATTN" ]]; then
    echo "  [3.1] Dumping FP16 attentions..."
    python "$BASE_DIR/dump_attn.py" \
        --model_id "$FP16_MODEL_ID" \
        --run_tag "fp16" \
        --prompts_jsonl "$SELECTED_PROMPTS" \
        --out_dir "$COMP_DIR"
else
    echo "  [3.1] FP16 attention already exists: $FP16_ATTN"
fi

# Dump GPTQ3 attentions
GPTQ3_ATTN="$COMP_DIR/attn_gptq3.npz"

if [[ ! -f "$GPTQ3_ATTN" ]]; then
    echo "  [3.2] Dumping GPTQ3 attentions..."
    python "$BASE_DIR/dump_attn.py" \
        --model_id "$GPTQ3_MODEL_PATH" \
        --run_tag "gptq3" \
        --prompts_jsonl "$SELECTED_PROMPTS" \
        --out_dir "$COMP_DIR"
else
    echo "  [3.2] GPTQ3 attention already exists: $GPTQ3_ATTN"
fi

echo ""
echo "✓ Attention data ready"

#############################
# 4) Identify Critical Heads
#############################

echo ""
echo "[STEP 4] Identifying critical attention heads..."
echo "  FP16 attention: $FP16_ATTN"
echo "  GPTQ3 attention: $GPTQ3_ATTN"
echo "  Top K: $TOP_K"
echo ""

CRITICAL_HEADS_JSON="$COMP_DIR/critical_heads_gptq3.json"

python "$BASE_DIR/identify_critical_heads.py" \
    --fp16_attn "$FP16_ATTN" \
    --quant_attn "$GPTQ3_ATTN" \
    --out_dir "$COMP_DIR" \
    --top_k $TOP_K \
    --method positive

if [[ ! -f "$CRITICAL_HEADS_JSON" ]]; then
    echo "[ERROR] Failed to identify critical heads"
    exit 1
fi

echo ""
echo "✓ Critical heads identified: $CRITICAL_HEADS_JSON"

#############################
# 5) Run Compensation Experiments
#############################

echo ""
echo "[STEP 5] Running compensation experiments..."
echo "  Testing alpha values: 0.0 0.5 1.0"
echo ""

RESULTS_DIR="$COMP_DIR/results"
mkdir -p "$RESULTS_DIR"

python "$BASE_DIR/eval_ifeval_with_compensation.py" \
    --model_path "$GPTQ3_MODEL_PATH" \
    --tokenizer_path "$FP16_MODEL_ID" \
    --ifeval_data "$SELECTED_PROMPTS" \
    --attn_fp16 "$FP16_ATTN" \
    --attn_quant "$GPTQ3_ATTN" \
    --top_heads "$CRITICAL_HEADS_JSON" \
    --alpha_list "0.0,0.5,1.0" \
    --max_samples 1 \
    --max_new_tokens 256 \
    --output "$COMP_DIR/comp_results_gptq3.json"

echo ""
echo "✓ All compensation experiments complete"

#############################
# 6) Aggregate and Visualize Results
#############################

echo ""
echo "[STEP 6] Aggregating results..."

# 下面这一整段 aggregate_results.py 和你原来的一样，就不重复贴了
# 保持不变即可
