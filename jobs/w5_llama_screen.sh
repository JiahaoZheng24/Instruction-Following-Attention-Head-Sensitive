#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W5_screen
# W5 stage 2: per-head FP-vs-gptq3 deviation ranking for Llama (needed by the
# heads32 arm). Loads both models on one H200 (~24GB total, fine).
#   qsub -hold_jid IFH_W5_prep jobs/w5_llama_screen.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

python src/diagnose_heads.py screen \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --quant-model "$STORE/models/llama3.1-8b-gptq3-c4-g128" \
  --prompts data/ifeval_screen100.jsonl \
  --tag llama3 \
  --out-dir runs

echo "[W5-screen] ✅ -> runs/dev_ranking_llama3.csv"
