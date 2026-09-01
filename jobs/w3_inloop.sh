#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W3_inloop
#$ -t 1-5
# W3/v2: IN-LOOP protected GPTQ (protection held out inside the column loop,
# TaCQ-style surgery) + full-541 IFEval. Decides the paper's outcome matrix:
#   arm2 works                  -> back to head-protection method paper
#   arm2 fails, arm3 works      -> dissociation paper (function != damage)
#   arm2 & arm3 fail            -> boundary paper (IF immune to weight protection)
# All arms compare against arm1 (v2_none, internal reference; also sanity vs
# qbase_gptq3 0.6832 — small deviation acceptable, documented pipeline diff).
#   qsub -hold_jid IFH_W3_salience jobs/w3_inloop.sh
# NOTE: each arm writes a ~15GB fake-quant checkpoint under $STORE/models.

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
FULL="data/ifeval_input_data.jsonl"
BUDGET=37624064        # = W2 prot_top32_dev3 budget (0.58% of layer params)
TACQ_NATIVE=22900000   # TaCQ's own 0.35% of ~6.53B layer params
SAL_DIR="$STORE/salience/qwen25-7b-if"   # ~13GB, lives outside runs/

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) ARM="none";     ARGS="--protect none" ;;
  2) ARM="heads32";  ARGS="--protect heads --topk-from runs/dev_ranking_gptq3.csv --k 32 --kv --projs all" ;;
  3) ARM="tacq";     ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $BUDGET" ;;
  4) ARM="randw";    ARGS="--protect randw --budget-params $BUDGET --seed 0" ;;
  5) ARM="tacq035";  ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $TACQ_NATIVE" ;;
  *) echo "bad task id"; exit 1 ;;
esac

CKPT="$STORE/models/qwen2.5-7b-v2gptq3-$ARM"
TAG="v2_$ARM"
echo "[W3-inloop] task=$SGE_TASK_ID arm=$ARM ckpt=$CKPT"

python src/quantize_protected.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --bits 3 --group-size 128 \
  --out "$CKPT" \
  $ARGS

python src/diagnose_heads.py ablate \
  --model "$CKPT" \
  --prompts "$FULL" \
  --tag "$TAG" \
  --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename $CKPT)/$TAG/responses.jsonl" \
  --input-data "$FULL" \
  --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"

echo "[W3-inloop] ✅ $TAG scored -> runs/scores_${TAG}.csv"
