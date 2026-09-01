#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W6b_2bit
#$ -t 1-4
# W6b: complete the 2x2 (model x regime) matrix — Qwen at 2-bit is Qwen's
# collapse regime (qbase_gptq2 = 0.12). Does tacq rescue it like Llama@3bit
# (criterion matters in collapse) while randw/sw-only calibrate the size?
# Overlaps TaCQ's own 2-bit battleground, with the IF eval they never ran.
# NOTE: reuses the 3-bit-computed salience (|dW| term differs at 2-bit;
# recorded in protocol as a limitation).
#   qsub -hold_jid IFH_W6_detect jobs/w6b_qwen2bit.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
FULL="data/ifeval_input_data.jsonl"
BUDGET=37624064
SAL_DIR="$STORE/salience/qwen25-7b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) ARM="b2_none";  ARGS="--protect none" ;;
  2) ARM="b2_tacq";  ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $BUDGET" ;;
  3) ARM="b2_randw"; ARGS="--protect randw --budget-params $BUDGET --seed 0" ;;
  4) ARM="b2_sw";    ARGS="--protect coords --coords-file runs/super_weights_qwen.csv" ;;
  *) echo "bad task id"; exit 1 ;;
esac

CKPT="$STORE/models/qwen2.5-7b-v2gptq2-$ARM"
TAG="v2_$ARM"
echo "[W6b] task=$SGE_TASK_ID arm=$ARM"

python src/quantize_protected.py \
  --model "$MODEL_ID" --bits 2 --group-size 128 --out "$CKPT" $ARGS

python src/diagnose_heads.py ablate \
  --model "$CKPT" --prompts "$FULL" --tag "$TAG" --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename $CKPT)/$TAG/responses.jsonl" \
  --input-data "$FULL" --tag "$TAG" --scores-csv "runs/scores_${TAG}.csv"

echo "[W6b] ✅ $TAG scored"
