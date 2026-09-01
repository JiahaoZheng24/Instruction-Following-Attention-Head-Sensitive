#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W9_mif
#$ -t 1-7
# W9 (freeze batch): second IF benchmark + the last invariant arm.
#  1-6: Multi-IF (multi-turn, English) on the six key checkpoints —
#       does the regime picture replicate on a second IF benchmark?
#  7:   Mistral tacq arm (IFEval): completes the "protection is null in the
#       graceful regime" invariant on the third model.
#   qsub jobs/w9_multiif.sh
# EXPERIMENT FREEZE after this batch.

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

mif () {  # $1 model, $2 tag
  python src/multi_if.py --model "$1" --tag "$2" --batch 8 \
    --scores-csv "runs/scores_mif_$2.csv"
}

case "$SGE_TASK_ID" in
  1) mif Qwen/Qwen2.5-7B-Instruct                      mif_qwen_fp16 ;;
  2) mif "$STORE/models/qwen2.5-7b-gptq3-c4-g128"      mif_qwen_gptq3 ;;
  3) mif "$STORE/models/qwen2.5-7b-v2gptq3-tacq"       mif_qwen_tacq ;;
  4) mif meta-llama/Llama-3.1-8B-Instruct              mif_llama_fp16 ;;
  5) mif "$STORE/models/llama3.1-8b-v2gptq3-none"      mif_llama_none ;;
  6) mif "$STORE/models/llama3.1-8b-v2gptq3-tacq"      mif_llama_tacq ;;
  7) CKPT="$STORE/models/mistral-7b-v2gptq3-tacq"
     python src/quantize_protected.py --model mistralai/Mistral-7B-Instruct-v0.3 \
       --bits 3 --group-size 128 --protect tacq \
       --salience-dir "$STORE/salience/mistral7b-if" --budget-params 40000000 \
       --out "$CKPT"
     python src/diagnose_heads.py ablate --model "$CKPT" --prompts "$FULL" \
       --tag v2m_tacq --batch 16
     python src/score_ifeval.py --responses "runs/$(basename $CKPT)/v2m_tacq/responses.jsonl" \
       --input-data "$FULL" --tag v2m_tacq --scores-csv runs/scores_v2m_tacq.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W9] ✅ task $SGE_TASK_ID done"
