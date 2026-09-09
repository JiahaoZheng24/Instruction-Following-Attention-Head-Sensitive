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
#$ -N IFH_W45
#$ -t 1-8
# W45 (last): are the token-norm losses on graceful models real or seed noise?
# W43: Mistral-7B-v0.2 -3.4, Llama-3-8B -2.6, gemma-2-2b -2.4, Mistral-7B-v0.3
# -2.2 (single seed each; graceful-model seed noise is +-1..3).
# Seed-1 replicates of none and token-norm for the four models.
#   qsub jobs/w45_tnorm_seeds.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (3-bit g128, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq mistralai/Mistral-7B-Instruct-v0.2      mistral-7b-v02-v2gptq3-tnorm_cs1 v2m702_tnorm_cs1 --hess-token-norm --calib-seed 1 ;;
  2) fq mistralai/Mistral-7B-Instruct-v0.3      mistral-7b-v2gptq3-none_cs1      v2m_none_cs1     --calib-seed 1 ;;
  3) fq mistralai/Mistral-7B-Instruct-v0.3      mistral-7b-v2gptq3-tnorm_cs1     v2m_tnorm_cs1    --hess-token-norm --calib-seed 1 ;;
  4) fq meta-llama/Meta-Llama-3-8B-Instruct     llama3-8b-v2gptq3-none_cs1       v2l3_none_cs1    --calib-seed 1 ;;
  5) fq meta-llama/Meta-Llama-3-8B-Instruct     llama3-8b-v2gptq3-tnorm_cs1      v2l3_tnorm_cs1   --hess-token-norm --calib-seed 1 ;;
  6) fq google/gemma-2-2b-it                    gemma2-2b-v2gptq3-none_cs1       v2g22_none_cs1   --calib-seed 1 ;;
  7) fq google/gemma-2-2b-it                    gemma2-2b-v2gptq3-tnorm_cs1      v2g22_tnorm_cs1  --hess-token-norm --calib-seed 1 ;;
  8) fq "$Q14"                                  qwen2.5-14b-v2gptq3-tnorm_cs1    v2q14_tnorm_cs1  --hess-token-norm --calib-seed 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W45] done task $SGE_TASK_ID"
