#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W4b_cap
#$ -t 1-5
# W4b: capability-contrast experiment — the paper's centerpiece contrast.
# Question: does protection rescue KNOWLEDGE (MMLU) while failing to rescue
# IF (established)? Plus wikitext-2 PPL: the proxy the outlier literature
# optimizes ("PPL fine, capability broken").
# Read-out:
#   v2_tacq MMLU >> v2_none MMLU  while  v2_tacq IFEval ~ v2_none IFEval
#   => damage localizability DIFFERS BY CAPABILITY (positive scientific claim)
#   qsub jobs/w4b_capability.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) TAG="gen_fp16";      MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  2) TAG="gen_gptq3";     MODEL="$STORE/models/qwen2.5-7b-gptq3-c4-g128" ;;
  3) TAG="gen_v2none";    MODEL="$STORE/models/qwen2.5-7b-v2gptq3-none" ;;
  4) TAG="gen_v2tacq";    MODEL="$STORE/models/qwen2.5-7b-v2gptq3-tacq" ;;
  5) TAG="gen_v2heads32"; MODEL="$STORE/models/qwen2.5-7b-v2gptq3-heads32" ;;
  *) echo "bad task id"; exit 1 ;;
esac

if [ ! -d "$MODEL" ] && [[ "$MODEL" == /* ]]; then
  echo "[W4b] checkpoint $MODEL missing (deleted?) — skipping"; exit 0
fi

echo "[W4b] task=$SGE_TASK_ID tag=$TAG model=$MODEL"

python src/eval_general.py \
  --model "$MODEL" \
  --tag "$TAG" \
  --batch 8 \
  --scores-csv "runs/general_${TAG}.csv"

echo "[W4b] ✅ $TAG -> runs/general_${TAG}.csv"
