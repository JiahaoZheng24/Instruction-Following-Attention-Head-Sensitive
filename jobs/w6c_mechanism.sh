#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W6c_mech
#$ -t 1-2
# W6c: the error-absorber mechanism test. In-loop scattered protection helps
# (~+3 on Qwen 3-bit) with criterion irrelevant; if the SAME scattered
# restoration applied POST-HOC (no compensation interaction) gives ~nothing,
# the benefit is mechanistically located in GPTQ's compensation ("scattered
# fp entries absorb quantization error"), not in the weights themselves.
# Requires $STORE/models/qwen2.5-7b-v2gptq3-none (kept checkpoint).
#   qsub jobs/w6c_mechanism.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

TAG="ph_scatter_s$((SGE_TASK_ID - 1))"
python src/posthoc_scatter.py \
  --quant-ckpt "$STORE/models/qwen2.5-7b-v2gptq3-none" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --budget-params 37624064 --seed $((SGE_TASK_ID - 1)) \
  --prompts "$FULL" --tag "$TAG"

python src/score_ifeval.py \
  --responses "runs/qwen2.5-7b-v2gptq3-none/$TAG/responses.jsonl" \
  --input-data "$FULL" --tag "$TAG" --scores-csv "runs/scores_${TAG}.csv"

echo "[W6c] ✅ $TAG scored"
