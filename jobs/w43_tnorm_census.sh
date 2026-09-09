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
#$ -N IFH_W43
#$ -t 1-18
# W43: the mechanism-derived repair (token-normalised calibration Hessian,
# --hess-token-norm, c4 otherwise unchanged) across the whole census.
# W42: it cures Llama (.150 -> .648 / .668) and Q14 (.412 -> .720) with a
# one-line change. Question: what does it do on the 14 graceful models
# (baselines = the census GPTQ none arms), on the Mistral act-order x damping
# collapse (trigger 2), and at 4-bit?
#  1-14  3-bit g128 token-norm on every remaining census model
#  15    Mistral-7B-v0.3 rho=0.5 (trigger 2) + token-norm
#  16-18 4-bit token-norm: Llama, Q14, Nemo
#   qsub jobs/w43_tnorm_census.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 8h"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"

fq () {  # $1 model $2 ckpt $3 tag $4 bits $5.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3" bits="$4"; shift 4
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits "$bits" --group-size 128 --protect none --hess-token-norm "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1)  fq "$Q7"                                    qwen2.5-7b-v2gptq3-tnorm      v2q7_tnorm     3 ;;
  2)  fq "$NEMO"                                  mistral-nemo-12b-v2gptq3-tnorm v2nemo_tnorm  3 ;;
  3)  fq meta-llama/Llama-3.2-3B-Instruct         llama3.2-3b-v2gptq3-tnorm     v2l32_tnorm    3 ;;
  4)  fq meta-llama/Llama-3.2-1B-Instruct         llama3.2-1b-v2gptq3-tnorm     v2l321b_tnorm  3 ;;
  5)  fq google/gemma-2-9b-it                     gemma2-9b-v2gptq3-tnorm       v2g29_tnorm    3 ;;
  6)  fq google/gemma-2-2b-it                     gemma2-2b-v2gptq3-tnorm       v2g22_tnorm    3 ;;
  7)  fq tiiuae/Falcon3-7B-Instruct               falcon3-7b-v2gptq3-tnorm      v2f3_tnorm     3 ;;
  8)  fq HuggingFaceTB/SmolLM2-1.7B-Instruct      smollm2-1.7b-v2gptq3-tnorm    v2sm_tnorm     3 ;;
  9)  fq Qwen/Qwen2.5-3B-Instruct                 qwen2.5-3b-v2gptq3-tnorm      v2q3_tnorm     3 ;;
  10) fq Qwen/Qwen2.5-32B-Instruct                qwen2.5-32b-v2gptq3-tnorm     v2q32_tnorm    3 ;;
  11) fq mistralai/Mistral-Small-24B-Instruct-2501 mistral-24b-v2gptq3-tnorm    v2m24_tnorm    3 ;;
  12) fq meta-llama/Meta-Llama-3-8B-Instruct      llama3-8b-v2gptq3-tnorm       v2l3_tnorm     3 ;;
  13) fq mistralai/Mistral-7B-Instruct-v0.2       mistral-7b-v02-v2gptq3-tnorm  v2m702_tnorm   3 ;;
  14) fq mistralai/Mistral-7B-Instruct-v0.3       mistral-7b-v2gptq3-tnorm      v2m_tnorm      3 ;;
  15) fq mistralai/Mistral-7B-Instruct-v0.3       mistral-7b-v2gptq3-damp0p5-tnorm v2m_damp0p5_tnorm 3 --percdamp 0.5 ;;
  16) fq "$LLAMA"                                 llama3.1-8b-v2gptq4-tnorm     v2l4_tnorm     4 ;;
  17) fq "$Q14"                                   qwen2.5-14b-v2gptq4-tnorm     v2q14_4_tnorm  4 ;;
  18) fq "$NEMO"                                  mistral-nemo-12b-v2gptq4-tnorm v2nemo4_tnorm 4 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W43] done task $SGE_TASK_ID"
