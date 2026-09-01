#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_W1_qbase
#$ -t 1-5
# Full-541 IFEval baselines on the FRESH quantized checkpoints + FP16 reference.
# This re-measures the bit-width narrative (old "gptq3 drops 15 pts / garbled"
# numbers came from the deleted, possibly-broken checkpoints).
#   qsub jobs/w1_quant_baselines.sh          # (after w0_quantize finished)
# Read-out: runs/scores_qbase_*.csv -> how much does clean 3-bit really drop?

set -e

STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"
mkdir -p runs

case "$SGE_TASK_ID" in
  1) TAG="qbase_fp16";  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  2) TAG="qbase_gptq2"; MODEL="$STORE/models/qwen2.5-7b-gptq2-c4-g128" ;;
  3) TAG="qbase_gptq3"; MODEL="$STORE/models/qwen2.5-7b-gptq3-c4-g128" ;;
  4) TAG="qbase_gptq4"; MODEL="$STORE/models/qwen2.5-7b-gptq4-c4-g128" ;;
  5) TAG="qbase_gptq8"; MODEL="$STORE/models/qwen2.5-7b-gptq8-c4-g128" ;;
  *) echo "bad task id"; exit 1 ;;
esac

echo "[qbase] $TAG <- $MODEL"

python src/diagnose_heads.py ablate \
  --model "$MODEL" \
  --prompts data/ifeval_input_data.jsonl \
  --tag "$TAG" \
  --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename "$MODEL")/$TAG/responses.jsonl" \
  --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"

echo "[qbase] ✅ $TAG -> runs/scores_${TAG}.csv"
# Merge when all 5 done:
#   awk 'FNR==1 && NR!=1 {next} 1' runs/scores_qbase_*.csv > runs/scores_qbase.csv
