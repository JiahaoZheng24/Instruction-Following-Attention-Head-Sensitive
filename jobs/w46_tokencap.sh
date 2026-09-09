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
#$ -N IFH_W46
#$ -t 1-6
# W46 (optional): soft token weighting. W45 showed the full token-normalised
# Hessian costs the Mistral family 3.7-4.5 points on both seeds (their BOS
# massive activation apparently must be reproduced accurately), while it
# cures the collapses. --hess-token-cap 10 only scales tokens whose norm
# exceeds 10x rms down to that ceiling (BOS at 480-800x rms becomes 10x),
# leaving every ordinary token untouched. Prediction: still cures Llama/Q14,
# no longer hurts Mistral.
#   qsub jobs/w46_tokencap.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (3-bit g128, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --hess-token-cap 10 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$LLAMA"                              llama3.1-8b-v2gptq3-tcap10    v2l_tcap10    ;;
  2) fq "$Q14"                                qwen2.5-14b-v2gptq3-tcap10    v2q14_tcap10  ;;
  3) fq mistralai/Mistral-7B-Instruct-v0.3    mistral-7b-v2gptq3-tcap10     v2m_tcap10    ;;
  4) fq mistralai/Mistral-7B-Instruct-v0.2    mistral-7b-v02-v2gptq3-tcap10 v2m702_tcap10 ;;
  5) fq "$NEMO"                               mistral-nemo-12b-v2gptq3-tcap10 v2nemo_tcap10 ;;
  6) fq meta-llama/Meta-Llama-3-8B-Instruct   llama3-8b-v2gptq3-tcap10      v2l3_tcap10   ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W46] done task $SGE_TASK_ID"
