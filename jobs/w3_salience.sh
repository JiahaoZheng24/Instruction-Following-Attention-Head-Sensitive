#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W3_salience
# TaCQ-style saliency |W|*|dL/dW|*|W-RTN3(W)|, task-conditioned on IF data.
# Output feeds the tacq arm of jobs/w3_inloop.sh.
#   qsub jobs/w3_salience.sh
#   qsub -hold_jid IFH_W3_salience jobs/w3_inloop.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

# NOT under runs/: ~13GB of per-module tensors, cluster-side input only —
# keeps runs/ light enough to copy wholesale to the laptop.
SAL_DIR="$STORE/salience/qwen25-7b-if"
mkdir -p "$SAL_DIR"

python src/tacq_salience.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --calib-file data/calib_prompts.jsonl \
  --bits 3 \
  --out-dir "$SAL_DIR"

echo "[W3-salience] ✅ -> $SAL_DIR"
