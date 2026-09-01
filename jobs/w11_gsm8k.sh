#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W11_gsm
#$ -t 1-6
# W11 (ICLR framing): GSM8K reasoning on the six regime-defining checkpoints —
# same six as Multi-IF (W9). No new quantization; capability-generality
# evidence (IF + knowledge + reasoning + PPL all show the same regime law?).
#   qsub jobs/w11_gsm8k.sh   (independent of W10, submit together)

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

gsm () {  # $1 model, $2 tag
  python src/gsm8k_eval.py --model "$1" --tag "$2" --batch 16 \
    --scores-csv "runs/scores_gsm8k.csv"
}

case "$SGE_TASK_ID" in
  1) gsm Qwen/Qwen2.5-7B-Instruct                      gsm_qwen_fp16 ;;
  2) gsm "$STORE/models/qwen2.5-7b-gptq3-c4-g128"      gsm_qwen_gptq3 ;;
  3) gsm "$STORE/models/qwen2.5-7b-v2gptq3-tacq"       gsm_qwen_tacq ;;
  4) gsm meta-llama/Llama-3.1-8B-Instruct              gsm_llama_fp16 ;;
  5) gsm "$STORE/models/llama3.1-8b-v2gptq3-none"      gsm_llama_none ;;
  6) gsm "$STORE/models/llama3.1-8b-v2gptq3-tacq"      gsm_llama_tacq ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W11] ✅ task $SGE_TASK_ID done"
