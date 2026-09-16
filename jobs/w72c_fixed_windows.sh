#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=12:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W72c
#$ -t 1-38
# W72c (2026-09-15). Pre-registered in RESULTS 9.10bm. FIXED-CODE RE-RUN, part c:
# the window-protocol census (128 x 2048 windows). Tags f_<model>_<arm>.
#   1-16   16 models, GPTQ3 c4win
#   17-32  16 models, GPTQ3 pile
#   33-38  6 models, GPTQ3 pile gw+tn
# Llama / Q14 window arms are in W72a. Checkpoints deleted after scoring.
#   qsub jobs/w72c_fixed_windows.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 11h"

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|q7"
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "Qwen/Qwen2.5-3B-Instruct|q3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "mistralai/Mistral-7B-Instruct-v0.3|m7"
  "google/gemma-2-9b-it|g29"
  "meta-llama/Meta-Llama-3-8B-Instruct|l3"
  "meta-llama/Llama-3.2-3B-Instruct|l32"
  "tiiuae/Falcon3-7B-Instruct|f3"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct|sm"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b"
  "allenai/Llama-3.1-Tulu-3-8B|tulu3"
  "Qwen/Qwen3-8B|qwen3_8b"
  "ibm-granite/granite-3.1-8b-instruct|granite"
)
PILE=(
  "Qwen/Qwen2.5-7B-Instruct|q7"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "tiiuae/Falcon3-7B-Instruct|f3"
  "allenai/Llama-3.1-Tulu-3-8B|tulu3"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct|sm"
)

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
qz () {   # $1 model $2 ckpt-name $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --out "$CK" "$@"
  finish "$CK" "$tag"
}

id=$SGE_TASK_ID
if [ "$id" -le 32 ]; then
  if [ "$id" -le 16 ]; then calib=c4win; idx=$((id - 1)); else calib=pile; idx=$((id - 17)); fi
  entry="${MODELS[$idx]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  qz "$model" "f-${s}-gptq3-${calib}" "f_${s}_${calib}" --calib "$calib"
elif [ "$id" -le 38 ]; then
  entry="${PILE[$((id - 33))]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  qz "$model" "f-${s}-gptq3-pile-gwtn" "f_${s}_pile_gwtn" --calib pile --hess-grad-weight --hess-token-norm
else
  echo "bad task id"; exit 1
fi
echo "[W72c] done task $SGE_TASK_ID"
