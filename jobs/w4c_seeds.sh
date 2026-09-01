#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W4c_seeds
#$ -t 1-9
# W4c: statistical hardening + the subsumption arm.
#  1-8: {none, heads32, randw, tacq} x calib seeds {1,2} (seed 0 = existing
#       W3 runs) -> cross-seed error bars for the tacq>random claim.
#  9:   g64 + tacq protection combo: if g64+tacq ~ g64 alone (0.7193),
#       protection is fully subsumed by group granularity — final nail.
# Each task: in-loop quantize (~1-2h) + full-541 IFEval + score.
# Disk: each writes a ~15GB fake-quant ckpt; safe to delete after scoring
# EXCEPT keep seed-1 of none/tacq for reruns.
#   qsub jobs/w4c_seeds.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
FULL="data/ifeval_input_data.jsonl"
BUDGET=37624064
SAL_DIR="$STORE/salience/qwen25-7b-if"
RANK="runs/dev_ranking_gptq3.csv"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

GS=128
case "$SGE_TASK_ID" in
  1) ARM="none_cs1";    CS=1; ARGS="--protect none" ;;
  2) ARM="heads32_cs1"; CS=1; ARGS="--protect heads --topk-from $RANK --k 32 --kv --projs all" ;;
  3) ARM="randw_cs1";   CS=1; ARGS="--protect randw --budget-params $BUDGET --seed 1" ;;
  4) ARM="tacq_cs1";    CS=1; ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $BUDGET" ;;
  5) ARM="none_cs2";    CS=2; ARGS="--protect none" ;;
  6) ARM="heads32_cs2"; CS=2; ARGS="--protect heads --topk-from $RANK --k 32 --kv --projs all" ;;
  7) ARM="randw_cs2";   CS=2; ARGS="--protect randw --budget-params $BUDGET --seed 2" ;;
  8) ARM="tacq_cs2";    CS=2; ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $BUDGET" ;;
  9) ARM="g64tacq";     CS=0; GS=64
     ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params $BUDGET" ;;
  *) echo "bad task id"; exit 1 ;;
esac

CKPT="$STORE/models/qwen2.5-7b-v2gptq3-$ARM"
TAG="v2_$ARM"
echo "[W4c] task=$SGE_TASK_ID arm=$ARM calib_seed=$CS group=$GS"

python src/quantize_protected.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --bits 3 --group-size $GS --calib-seed $CS \
  --out "$CKPT" \
  $ARGS

python src/diagnose_heads.py ablate \
  --model "$CKPT" --prompts "$FULL" --tag "$TAG" --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename $CKPT)/$TAG/responses.jsonl" \
  --input-data "$FULL" --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"

echo "[W4c] ✅ $TAG scored"
