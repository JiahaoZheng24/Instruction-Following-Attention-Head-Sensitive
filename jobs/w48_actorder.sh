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
#$ -N IFH_W48
#$ -t 1-5
# W48 (optional): is the Mistral-family loss under token re-weighting an
# act-order effect? Re-weighting the Hessian changes its diagonal, hence the
# act-order column permutation; Mistral is known to be act-order-fragile
# (rho=0.5 + act-order collapses, without act-order it is fine). W47: the
# 10x median cap keeps BOS at 10x yet still costs Mistral -4.9 / -4.2, so the
# loss is not the BOS weight itself.
#  1-3  Mistral-7B-v0.3, act-order OFF: none / token-norm / cap-10
#  4    Llama-3.1-8B, act-order OFF, token-norm  (is Llama's cure the
#       objective, not the permutation? none without act-order is .156)
#  5    Nemo, cap-10, seed 1  (replicate of +6.3)
#   qsub jobs/w48_actorder.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (3-bit g128, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$M7"    mistral-7b-v2gptq3-noao           v2m_noao           --no-actorder ;;
  2) fq "$M7"    mistral-7b-v2gptq3-tnorm-noao     v2m_tnorm_noao     --no-actorder --hess-token-norm ;;
  3) fq "$M7"    mistral-7b-v2gptq3-tcapmed10-noao v2m_tcapmed10_noao --no-actorder --hess-token-cap 10 ;;
  4) fq "$LLAMA" llama3.1-8b-v2gptq3-tnorm-noao    v2l_tnorm_noao     --no-actorder --hess-token-norm ;;
  5) fq "$NEMO"  mistral-nemo-12b-v2gptq3-tcapmed10_cs1 v2nemo_tcapmed10_cs1 --hess-token-cap 10 --calib-seed 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W48] done task $SGE_TASK_ID"
