#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=10:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W41
#$ -t 1-4
# W41 (optional, last): is the detector's one false positive real?
# W40: Mistral-7B-v0.2 L1 down_proj loses to RTN 31x at template positions,
# yet RTN on blocks 0-1 down_proj gave IFEval -2.4 (single seed; graceful-model
# seed noise is +-1..3). Qwen2.5-7B: detector 3.1 (weak), rule -0.9.
# Seed-1 replicates of none / +rule for both models.
#   qsub jobs/w41_fp_check.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7V02="mistralai/Mistral-7B-Instruct-v0.2"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (3-bit, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$M7V02" mistral-7b-v02-v2gptq3-none_cs1      v2m702_none_cs1      --calib-seed 1 ;;
  2) fq "$M7V02" mistral-7b-v02-v2gptq3-rtn01down_cs1 v2m702_rtn01down_cs1 --rtn-modules "0-1:down_proj" --calib-seed 1 ;;
  3) fq "$Q7"    qwen2.5-7b-v2gptq3-none_cs1          v2q7_none_cs1        --calib-seed 1 ;;
  4) fq "$Q7"    qwen2.5-7b-v2gptq3-rtn01down_cs1     v2q7_rtn01down_cs1   --rtn-modules "0-1:down_proj" --calib-seed 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W41] done task $SGE_TASK_ID"
