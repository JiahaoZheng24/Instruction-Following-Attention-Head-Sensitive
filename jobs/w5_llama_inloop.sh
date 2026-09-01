#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W5_inloop
#$ -t 1-4
# W5 stage 3: the four decisive in-loop arms on Llama-3.1-8B-Instruct.
# Replicates the dissociation on a second model family.
#   qsub -hold_jid IFH_W5_screen jobs/w5_llama_inloop.sh
# Budget: keep the SAME FRACTION as Qwen (0.58% of layer params).
# Llama-3.1-8B layer params ~= 6.9B -> 40.2M.

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
FULL="data/ifeval_input_data.jsonl"
BUDGET=40200000
SAL_DIR="$STORE/salience/llama31-8b-if"
RANK="runs/dev_ranking_llama3.csv"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) ARM="none";    ARGS="--protect none" ;;
  2) ARM="heads32"; ARGS="--protect heads --topk-from $RANK --k 32 --kv --projs all" ;;
  3) ARM="tacq";    ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $BUDGET" ;;
  4) ARM="randw";   ARGS="--protect randw --budget-params $BUDGET --seed 0" ;;
  *) echo "bad task id"; exit 1 ;;
esac

CKPT="$STORE/models/llama3.1-8b-v2gptq3-$ARM"
TAG="v2l_$ARM"
echo "[W5-inloop] task=$SGE_TASK_ID arm=$ARM"

python src/quantize_protected.py \
  --model "$MODEL_ID" \
  --bits 3 --group-size 128 \
  --out "$CKPT" \
  $ARGS

python src/diagnose_heads.py ablate \
  --model "$CKPT" --prompts "$FULL" --tag "$TAG" --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename $CKPT)/$TAG/responses.jsonl" \
  --input-data "$FULL" --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"

echo "[W5-inloop] ✅ $TAG scored"
