#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W2c_budget
#$ -t 1-6
# W2c: concentration curve — how much must be restored (post-hoc, v1
# machinery) before IF comes back? Fractions: 0.5% (done in W2b) -> 2% -> 5%
# -> 11% (ALL attention) -> 25%. attn_all is the decisive H1-attention arm.
# If recovery needs >=25%, average bits are worse than plain 4-bit ->
# quantitative epitaph of small-budget protection (paper main figure).
#   qsub jobs/w2c_budget.sh    (avoid node qa-h200-007 if it still hangs:
#   qsub -l h='!qa-h200-007' jobs/w2c_budget.sh)

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
QUANT="$STORE/models/qwen2.5-7b-gptq3-c4-g128"
QUANT_TAG="qwen2.5-7b-gptq3-c4-g128"
FULL="data/ifeval_input_data.jsonl"
ALL_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) TAG="attn_all";        ARGS="--layer $ALL_LAYERS --kv" ;;             # ~823M = 11%
  2) TAG="mlp_match_attn";  ARGS="--mlp-params 823000000 --seed 0" ;;      # 11% budget-matched
  3) TAG="mlp_2pct";        ARGS="--mlp-params 152000000 --seed 0" ;;
  4) TAG="mlp_5pct";        ARGS="--mlp-params 380000000 --seed 0" ;;
  5) TAG="mlp_25pct";       ARGS="--mlp-params 1900000000 --seed 0" ;;
  6) TAG="mlp_match_attn_s1"; ARGS="--mlp-params 823000000 --seed 1" ;;
  *) echo "bad task id"; exit 1 ;;
esac

echo "[W2c] task=$SGE_TASK_ID tag=$TAG args=$ARGS"

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
  --scores-csv "runs/scores_${TAG}.csv"

echo "[W2c] ✅ $TAG scored -> runs/scores_${TAG}.csv"
