#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_W1_ablate
#$ -t 1-11
# Task array: one ablation config per task, runs in parallel across the
# zzheng3_Lab H200 nodes. Submit AFTER w1_calib.sh finishes:
#   qsub -hold_jid IFH_W1_calib jobs/w1_ablate.sh

set -e

STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
MODEL_TAG="Qwen2.5-7B-Instruct"          # basename used in runs/<model>/<tag>/

FULL="data/ifeval_input_data.jsonl"      # 541 prompts (baseline / validation)
SCREEN="data/ifeval_screen100.jsonl"     # 100-prompt screening subset

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"

#############################
# Config table: TAG | PROMPTS | extra diagnose_heads.py args
# 1-2: baselines (full set + screening subset reference)
# 3-6: top-k from screening rankings (ISI = quant-deviation, ACT = magnitude)
# 7-11: random controls (selectivity!) — 3 seeds at k=32, plus k=16/64
#############################
case "$SGE_TASK_ID" in
  1)  TAG="baseline_full"; PROMPTS="$FULL";   ARGS="" ;;
  2)  TAG="screen_base";   PROMPTS="$SCREEN"; ARGS="" ;;
  3)  TAG="top16_dev3";    PROMPTS="$SCREEN"; ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 16" ;;
  4)  TAG="top32_dev3";    PROMPTS="$SCREEN"; ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 32" ;;
  5)  TAG="top64_dev3";    PROMPTS="$SCREEN"; ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 64" ;;
  6)  TAG="top32_act";     PROMPTS="$SCREEN"; ARGS="--topk-from runs/diag/act_salience.csv --k 32" ;;
  7)  TAG="rand16_s0";     PROMPTS="$SCREEN"; ARGS="--random 16 --seed 0" ;;
  8)  TAG="rand32_s0";     PROMPTS="$SCREEN"; ARGS="--random 32 --seed 0" ;;
  9)  TAG="rand32_s1";     PROMPTS="$SCREEN"; ARGS="--random 32 --seed 1" ;;
  10) TAG="rand32_s2";     PROMPTS="$SCREEN"; ARGS="--random 32 --seed 2" ;;
  11) TAG="rand64_s0";     PROMPTS="$SCREEN"; ARGS="--random 64 --seed 0" ;;
  *)  echo "bad task id $SGE_TASK_ID"; exit 1 ;;
esac

echo "[W1-ablate] task=$SGE_TASK_ID tag=$TAG prompts=$PROMPTS args=$ARGS"

python src/diagnose_heads.py ablate \
  --model "$MODEL_ID" \
  --out-dir runs/diag \
  --prompts "$PROMPTS" \
  --tag "$TAG" \
  --batch 16 \
  $ARGS

python src/score_ifeval.py \
  --responses "runs/$MODEL_TAG/$TAG/responses.jsonl" \
  --input-data "$PROMPTS" \
  --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"   # per-task CSV: parallel tasks must not share one file

echo "[W1-ablate] ✅ $TAG scored -> runs/scores_${TAG}.csv"
# Merge after ALL tasks finish:
#   awk 'FNR==1 && NR!=1 {next} 1' runs/scores_*.csv > runs/scores.csv
# After all 11 tasks finish, the selectivity read-out is:
#   Δ(top32_dev3 vs screen_base)  vs  Δ(rand32_s* vs screen_base)
# and the causal ranking for dissociation comes from per-head follow-up runs.
