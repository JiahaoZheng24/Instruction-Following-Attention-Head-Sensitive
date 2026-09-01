#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W6_rescue
#$ -t 1-5
# W6 stage 2: THE RESCUE CURVE — how many protected weights does it take to
# un-collapse Llama-3.1 at 3-bit? Budget spans 7 orders of magnitude:
#   ~10 (detected super weights only) -> 1e4 -> 1e5 -> 1e6 -> 1e7 (tacq-ranked)
# (4.02e7 = existing v2l_tacq 0.6426; 0 = v2l_none 0.1496.)
# If the super-weights-only arm already recovers most of the rescue, the
# collapse regime is attributed: tacq's benefit = super-weight coverage.
#   qsub -hold_jid IFH_W6_detect jobs/w6_rescue.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
FULL="data/ifeval_input_data.jsonl"
SAL_DIR="$STORE/salience/llama31-8b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) ARM="sw_only"; ARGS="--protect coords --coords-file runs/super_weights_llama.csv" ;;
  2) ARM="tacq1e4"; ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params 10000" ;;
  3) ARM="tacq1e5"; ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params 100000" ;;
  4) ARM="tacq1e6"; ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params 1000000" ;;
  5) ARM="tacq1e7"; ARGS="--protect tacq --salience-dir $SAL_DIR --budget-params 10000000" ;;
  *) echo "bad task id"; exit 1 ;;
esac

CKPT="$STORE/models/llama3.1-8b-v2gptq3-$ARM"
TAG="v2l_$ARM"
echo "[W6-rescue] task=$SGE_TASK_ID arm=$ARM"

python src/quantize_protected.py \
  --model "$MODEL_ID" --bits 3 --group-size 128 --out "$CKPT" $ARGS

python src/diagnose_heads.py ablate \
  --model "$CKPT" --prompts "$FULL" --tag "$TAG" --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename $CKPT)/$TAG/responses.jsonl" \
  --input-data "$FULL" --tag "$TAG" --scores-csv "runs/scores_${TAG}.csv"

echo "[W6-rescue] ✅ $TAG scored"
