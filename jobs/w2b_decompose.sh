#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_W2b_decomp
#$ -t 1-6
# W2b: decompose the v1 protection null result (full 541, gptq3).
# Two hypotheses from W2:
#  (A) o_proj post-hoc column restoration fights GPTQ's cross-column error
#      compensation and actively hurts -> compare qkv-only vs o-only vs all.
#  (B) 3-bit IF damage lives in the MLPs, not attention (GPTQ log: MLP losses
#      ~100x attention) -> budget-matched random-MLP-channel protection.
# Read-out vs qbase_gptq3 (0.6832) and prot_top32_dev3 (0.7010, 37.6M params):
#   qkv > all  => hypothesis A confirmed (drop o_proj restoration in v1)
#   mlp >> head arms at same budget => hypothesis B confirmed (retarget W3)
#   qsub jobs/w2b_decompose.sh

set -e

STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
QUANT="$STORE/models/qwen2.5-7b-gptq3-c4-g128"
QUANT_TAG="qwen2.5-7b-gptq3-c4-g128"
FULL="data/ifeval_input_data.jsonl"
BUDGET=37624064   # = prot_top32_dev3's protected params (budget-matched MLP)

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) TAG="prot32_qkv";  ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 32 --kv --projs qkv" ;;
  2) TAG="prot32_o";    ARGS="--topk-from runs/dev_ranking_gptq3.csv --k 32 --projs o" ;;
  3) TAG="mlp_b_s0";    ARGS="--mlp-params $BUDGET --seed 0" ;;
  4) TAG="mlp_b_s1";    ARGS="--mlp-params $BUDGET --seed 1" ;;
  5) TAG="mlp_b_s2";    ARGS="--mlp-params $BUDGET --seed 2" ;;
  6) TAG="rand32_qkv_s0"; ARGS="--random 32 --seed 0 --kv --projs qkv" ;;
  *) echo "bad task id $SGE_TASK_ID"; exit 1 ;;
esac

echo "[W2b] task=$SGE_TASK_ID tag=$TAG args=$ARGS"

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

echo "[W2b] ✅ $TAG scored -> runs/scores_${TAG}.csv"
