#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu
#$ -l gpu_card=1
#$ -N ATTN_COMPENSATION

set -e

################################################################################
# Attention Compensation Experiment Pipeline
#
# Tests whether attention compensation can repair instruction-following
# degradation in quantized models by boosting critical attention heads.
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: qsub run_compensation_experiment.sh
#   3. Results will be saved to OUTPUT_DIR
################################################################################

echo "========================================"
echo "Attention Compensation Experiment"
echo "========================================"
date

################################################################################
# CONFIGURATION - Edit this section for your experiments
################################################################################

# Project directories
BASE_DIR="/users/jzheng7/ifattn"
EXPERIMENT_NAME="gptq4_compensation"  # Change this for different experiments
OUTPUT_DIR="$BASE_DIR/artifacts/$EXPERIMENT_NAME"

# Conda environment (must be pre-configured with all dependencies)
CONDA_ENV="attention"

# Models
FP16_MODEL="Qwen/Qwen2.5-7B-Instruct"
QUANT_MODEL="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_4bit"
QUANT_METHOD="gptq4"  # Used for tagging outputs (gptq4, awq4, etc.)

# IFEval benchmark data
FP16_SAMPLES="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct/samples_ifeval_2025-10-09T13-58-12.528717.jsonl"
QUANT_SAMPLES="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_4bit/samples_ifeval_2025-10-14T16-06-35.354326.jsonl"

# Experiment parameters
SAMPLE_STRATEGY="failure_only"  # Options: failure_only, both_wrong, all
MAX_SAMPLES=5                    # Number of samples to test
TOP_K_HEADS=10                   # Number of critical heads to compensate
ALPHA_VALUES="0.0 5.0 10.0 20.0" # Compensation strengths to test
MAX_GEN_TOKENS=1280              # Maximum generation length

################################################################################
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
################################################################################

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

################################################################################
# Step 1: Environment Setup
################################################################################
echo "[1/6] Activating conda environment: $CONDA_ENV"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"

################################################################################
# Step 2: Sample Selection
################################################################################
echo "[2/6] Selecting samples (strategy: $SAMPLE_STRATEGY)..."
python "$BASE_DIR/select_samples_for_compensation.py" \
  --fp16_samples "$FP16_SAMPLES" \
  --quant_samples "$QUANT_SAMPLES" \
  --strategy "$SAMPLE_STRATEGY" \
  --max_samples "$MAX_SAMPLES" \
  --output "$OUTPUT_DIR/selected_prompts.jsonl"

################################################################################
# Step 3: Extract Attention Weights
################################################################################
echo "[3/6] Extracting attention weights..."

# FP16 baseline attention
if [[ ! -f "$OUTPUT_DIR/attn_fp16.npz" ]]; then
  echo "  → Extracting FP16 baseline attention..."
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$FP16_MODEL" \
    --run_tag "fp16" \
    --prompts_jsonl "$OUTPUT_DIR/selected_prompts.jsonl" \
    --out_dir "$OUTPUT_DIR"
else
  echo "  ✓ FP16 attention already exists, skipping"
fi

# Quantized model attention
if [[ ! -f "$OUTPUT_DIR/attn_${QUANT_METHOD}.npz" ]]; then
  echo "  → Extracting quantized model attention..."
  python "$BASE_DIR/dump_attn.py" \
    --model_id "$QUANT_MODEL" \
    --run_tag "$QUANT_METHOD" \
    --prompts_jsonl "$OUTPUT_DIR/selected_prompts.jsonl" \
    --out_dir "$OUTPUT_DIR"
else
  echo "  ✓ Quantized attention already exists, skipping"
fi

################################################################################
# Step 4: Identify Critical Attention Heads
################################################################################
echo "[4/6] Identifying top-$TOP_K_HEADS critical attention heads..."
python "$BASE_DIR/identify_critical_heads.py" \
  --fp16_attn "$OUTPUT_DIR/attn_fp16.npz" \
  --quant_attn "$OUTPUT_DIR/attn_${QUANT_METHOD}.npz" \
  --top_k "$TOP_K_HEADS" \
  --out_dir "$OUTPUT_DIR"

################################################################################
# Step 5: Run Compensation Experiments
################################################################################
echo "[5/6] Running compensation experiments..."
echo "  Alpha values: $ALPHA_VALUES"

for ALPHA in $ALPHA_VALUES; do
  OUTPUT_FILE="$OUTPUT_DIR/compensation_alpha${ALPHA}.jsonl"

  if [[ -f "$OUTPUT_FILE" ]]; then
    echo "  ✓ Alpha=$ALPHA already completed, skipping"
    continue
  fi

  echo "  → Testing alpha=$ALPHA..."
  python "$BASE_DIR/eval_ifeval_with_compensation.py" \
    --model_path "$QUANT_MODEL" \
    --ifeval_data "$OUTPUT_DIR/selected_prompts.jsonl" \
    --attn_fp16 "$OUTPUT_DIR/attn_fp16.npz" \
    --attn_quant "$OUTPUT_DIR/attn_${QUANT_METHOD}.npz" \
    --top_heads "$OUTPUT_DIR/critical_heads_${QUANT_METHOD}.json" \
    --alpha_list "$ALPHA" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_GEN_TOKENS" \
    --output "$OUTPUT_FILE"
done

################################################################################
# Step 6: Analyze Results
################################################################################
echo "[6/6] Analyzing compensation results..."

# Build list of result files
RESULT_FILES=""
for ALPHA in $ALPHA_VALUES; do
  RESULT_FILES="$RESULT_FILES $OUTPUT_DIR/compensation_alpha${ALPHA}.jsonl"
done

python "$BASE_DIR/analyze_compensation_results.py" \
  --results $RESULT_FILES \
  --output "$OUTPUT_DIR/summary.csv" \
  --detailed_output "$OUTPUT_DIR/detailed_analysis.csv"

################################################################################
# Done!
################################################################################
echo ""
echo "========================================"
echo "✅ Experiment Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key output files:"
echo "  - summary.csv              : High-level statistics"
echo "  - detailed_analysis.csv    : Per-sample comparison"
echo "  - critical_heads_*.json    : Top-K degraded heads"
echo "  - compensation_alpha*.jsonl: Raw outputs for each alpha"
echo ""
date