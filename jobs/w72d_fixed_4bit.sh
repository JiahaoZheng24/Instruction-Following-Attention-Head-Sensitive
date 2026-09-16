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
#$ -N IFH_W72d
#$ -t 1-20
# W72d (2026-09-15). Pre-registered in RESULTS 9.10bm. FIXED-CODE RE-RUN, part d:
# the 4-bit census. Tags f_<model>_<arm>. RTN4 arms are unaffected and not re-run.
#   1-16   16 models, GPTQ4 short-c4 (Llama / Q14 GPTQ4 are in W72a)
#   17-20  4-bit gw+tn on the four W65 models (m7, qwen3_14b, q32, hermes3)
# Checkpoints deleted after scoring.
#   qsub jobs/w72d_fixed_4bit.sh
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
GW4=(
  "mistralai/Mistral-7B-Instruct-v0.3|m7"
  "Qwen/Qwen3-14B|qwen3_14b"
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
)

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
qz () {   # $1 model $2 ckpt-name $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 4 --group-size 128 --protect none --calib c4 --out "$CK" "$@"
  finish "$CK" "$tag"
}

id=$SGE_TASK_ID
if [ "$id" -le 16 ]; then
  entry="${MODELS[$((id - 1))]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  qz "$model" "f-${s}-gptq4-none" "f_${s}_gptq4"
elif [ "$id" -le 20 ]; then
  entry="${GW4[$((id - 17))]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  qz "$model" "f-${s}-gptq4-gwtn" "f_${s}_gptq4_gwtn" --hess-grad-weight --hess-token-norm
else
  echo "bad task id"; exit 1
fi
echo "[W72d] done task $SGE_TASK_ID"
