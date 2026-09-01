#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W4a_fork
#$ -t 1-3
# W4a: the calibration-space fork — decides the paper's direction.
#  1. instruct-calibrated GPTQ 3-bit: if this recovers IF, damage is
#     "diffuse in weight space but concentrated in calibration space"
#     -> paper flips to a constructive method (IF-aware calibration).
#  2. g64 3-bit (bpw 3.53): honest "same-bits alternative" ruler — does a
#     finer group size beat every protection arm?
#  3. mask anatomy: where do the TaCQ-salient weights live (free figure).
#   qsub jobs/w4a_calib_fork.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
FULL="data/ifeval_input_data.jsonl"
SAL_DIR="$STORE/salience/qwen25-7b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/qwen2.5-7b-gptq3-instruct-g128"; TAG="qcal_instruct3"
     python src/quantize_gptq.py --model Qwen/Qwen2.5-7B-Instruct \
       --bits 3 --group-size 128 --calib instruct --out "$CKPT" ;;
  2) CKPT="$STORE/models/qwen2.5-7b-gptq3-c4-g64"; TAG="qcal_g64_3"
     python src/quantize_gptq.py --model Qwen/Qwen2.5-7B-Instruct \
       --bits 3 --group-size 64 --calib c4 --out "$CKPT" ;;
  3) python src/mask_anatomy.py --salience-dir "$SAL_DIR" \
       --budget-params 37624064 --out runs/mask_anatomy.csv
     echo "[W4a] ✅ anatomy -> runs/mask_anatomy.csv"; exit 0 ;;
  *) echo "bad task id"; exit 1 ;;
esac

python src/diagnose_heads.py ablate \
  --model "$CKPT" \
  --prompts "$FULL" \
  --tag "$TAG" \
  --batch 16

python src/score_ifeval.py \
  --responses "runs/$(basename $CKPT)/$TAG/responses.jsonl" \
  --input-data "$FULL" \
  --tag "$TAG" \
  --scores-csv "runs/scores_${TAG}.csv"

echo "[W4a] ✅ $TAG scored -> runs/scores_${TAG}.csv"
