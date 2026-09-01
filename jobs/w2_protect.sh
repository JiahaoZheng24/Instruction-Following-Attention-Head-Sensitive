#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_W2_protect
#$ -t 1-9
# W2 protection experiment (v1: post-hoc head-slice restoration on gptq3).
# All arms run on the FULL 541 IFEval set — this is the paper's money table.
#   qsub jobs/w2_protect.sh
# Read-out vs qbase_gptq3 (0.6832 avg4) and qbase_fp16 (0.7652):
#   recovery  = protect_top32_dev3 − qbase_gptq3   (how much comes back)
#   selectivity = that recovery vs protect_rand32_s* (budget-matched)
#   principle   = vs protect_top32_act (salience) and protect_layer01 (CASIA-style)
# Per-constraint types to watch: language / combination / punctuation
# (the three biggest 3-bit victims).

set -e

STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
QUANT="$STORE/models/qwen2.5-7b-gptq3-c4-g128"
QUANT_TAG="qwen2.5-7b-gptq3-c4-g128"     # basename used in runs/<model>/<tag>/
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"

#############################
# 1: noop sanity (0 heads — must reproduce qbase_gptq3 scores exactly-ish)
# 2-4: targeted protection, k = 16/32/64 from the quant-fragility ranking
# 5-7: random controls, k=32, 3 seeds (budget-matched selectivity)
# 8: activation-salience protection (AWQ-style criterion control)
# 9: CASIA-style layer protection (layers 0,1 = 56 heads; larger budget,
#    reported honestly — the structural comparison, not budget-matched)
#############################
case "$SGE_TASK_ID" in
  1) TAG="prot_noop";       ARGS="" ;;
  2) TAG="prot_top16_dev3"; ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 16 --kv" ;;
  3) TAG="prot_top32_dev3"; ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 32 --kv" ;;
  4) TAG="prot_top64_dev3"; ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 64 --kv" ;;
  5) TAG="prot_rand32_s0";  ARGS="--random 32 --seed 0 --kv" ;;
  6) TAG="prot_rand32_s1";  ARGS="--random 32 --seed 1 --kv" ;;
  7) TAG="prot_rand32_s2";  ARGS="--random 32 --seed 2 --kv" ;;
  8) TAG="prot_top32_act";  ARGS="--topk-from runs/diag/act_salience.csv --k 32 --kv" ;;
  9) TAG="prot_layer01";    ARGS="--layer 0,1 --kv" ;;
  *) echo "bad task id $SGE_TASK_ID"; exit 1 ;;
esac

echo "[W2-protect] task=$SGE_TASK_ID tag=$TAG args=$ARGS"

# numeric selftest of the protection hooks once, in the sanity task
if [ "$SGE_TASK_ID" = "1" ]; then
  python src/protect_eval.py --model "$MODEL_ID" --quant-model "$QUANT" \
    --prompts "$FULL" --tag selftest --selftest
fi

python src/protect_eval.py \
  --model "$MODEL_ID" \
  --quant-model "$QUANT" \
  --prompts "$FULL" \
  --tag "$TAG" \
  --batch 16 \
  $ARGS

python src/score_ifeval.py \
  --responses "runs/$QUANT_TAG/$TAG/responses.jsonl" \
  --input-data "$FULL" \
  --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"   # per-task CSV: parallel tasks must not share one file

echo "[W2-protect] ✅ $TAG scored -> runs/scores_${TAG}.csv"
