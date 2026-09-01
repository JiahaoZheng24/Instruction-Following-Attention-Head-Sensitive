#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W8b_close
#$ -t 1-2
# W8b closeout: the last two gaps before the experiment freeze.
#  1: Mistral fp16 IFEval baseline (needed for its relative-drop numbers)
#  2: salience concentration index across the three models (regime predictor)
#   qsub jobs/w8b_closeout.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1)
    python src/diagnose_heads.py ablate --model mistralai/Mistral-7B-Instruct-v0.3 \
      --prompts "$FULL" --tag qbase_mistral_fp16 --batch 16
    python src/score_ifeval.py \
      --responses "runs/Mistral-7B-Instruct-v0.3/qbase_mistral_fp16/responses.jsonl" \
      --input-data "$FULL" --tag qbase_mistral_fp16 \
      --scores-csv runs/scores_qbase_mistral_fp16.csv ;;
  2)
    python src/salience_concentration.py \
      --dirs qwen="$STORE/salience/qwen25-7b-if" \
             llama="$STORE/salience/llama31-8b-if" \
             mistral="$STORE/salience/mistral7b-if" \
      --out runs/salience_concentration.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W8b] ✅ task $SGE_TASK_ID done"
