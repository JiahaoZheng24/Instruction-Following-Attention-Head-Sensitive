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
#$ -N IFH_W47
#$ -t 1-5
# W47: the soft token cap, correctly dosed. W46's --hess-token-cap 10 was
# relative to the rms token norm, which the BOS token itself dominates
# (Llama layer-1 down_proj: rms ~10.7, BOS 481, typical ~1), so BOS was only
# capped to ~107x typical and Llama still collapsed (.154). The cap is now
# relative to the MEDIAN token norm: BOS becomes 10x typical (from 480-800x).
# Prediction: cures Llama/Q14 like the full normalisation, without the
# Mistral-family loss (-3.7 / -4.5).
#   qsub jobs/w47_tokencap_median.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"

fq () {  # $1 model $2 ckpt $3 tag   (3-bit g128, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --hess-token-cap 10 --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$LLAMA"                              llama3.1-8b-v2gptq3-tcapmed10    v2l_tcapmed10    ;;
  2) fq "$Q14"                                qwen2.5-14b-v2gptq3-tcapmed10    v2q14_tcapmed10  ;;
  3) fq mistralai/Mistral-7B-Instruct-v0.3    mistral-7b-v2gptq3-tcapmed10     v2m_tcapmed10    ;;
  4) fq mistralai/Mistral-7B-Instruct-v0.2    mistral-7b-v02-v2gptq3-tcapmed10 v2m702_tcapmed10 ;;
  5) fq "$NEMO"                               mistral-nemo-12b-v2gptq3-tcapmed10 v2nemo_tcapmed10 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W47] done task $SGE_TASK_ID"
