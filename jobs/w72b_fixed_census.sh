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
#$ -N IFH_W72b
#$ -t 1-64
# W72b (2026-09-15). Pre-registered in RESULTS 9.10bm. FIXED-CODE RE-RUN, part b:
# the short-c4 3-bit census. Tags are f_<model>_<arm> ("f" = after the KV-cache
# fix of W71); the old tags stay untouched for the honesty ledger.
#   1-52   26 models x {GPTQ3 none, GPTQ3 gw+tn}   (Llama / Q14 are in W72a)
#   53-64  harsher recipes on the 6 W55 models: group -1 | percdamp 0.01
# RTN / fp16 / AutoRound / HQQ arms are not affected by the fix and are not re-run.
# Checkpoints deleted after scoring.
#   qsub jobs/w72b_fixed_census.sh
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
  "tiiuae/Falcon3-10B-Instruct|falcon3_10b"
  "NousResearch/Hermes-2-Pro-Llama-3-8B|hermes2pro_l3_8b"
  "mistralai/Mistral-7B-Instruct-v0.1|mistral_7b_v01"
  "Qwen/Qwen2.5-0.5B-Instruct|qwen25_0p5b"
  "Qwen/Qwen2.5-1.5B-Instruct|qwen25_1p5b"
  "Qwen/Qwen2-7B-Instruct|qwen2_7b"
  "Qwen/Qwen3-4B|qwen3_4b"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|r1_llama_8b"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|r1_qwen_14b"
  "HuggingFaceTB/SmolLM3-3B|smollm3_3b"
)
HARSH=(
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b"
  "mistralai/Mistral-7B-Instruct-v0.3|m7"
)

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
qz () {   # $1 model $2 ckpt-name $3 tag $4.. quantiser flags (later flags override earlier ones)
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@"
  finish "$CK" "$tag"
}

id=$SGE_TASK_ID
if [ "$id" -le 52 ]; then
  idx=$(( (id - 1) / 2 )); entry="${MODELS[$idx]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  if [ $(( id % 2 )) -eq 1 ]; then
    qz "$model" "f-${s}-gptq3-none" "f_${s}_none"
  else
    qz "$model" "f-${s}-gptq3-gwtn" "f_${s}_gwtn" --hess-grad-weight --hess-token-norm
  fi
elif [ "$id" -le 64 ]; then
  idx=$(( (id - 53) / 2 )); entry="${HARSH[$idx]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  if [ $(( (id - 53) % 2 )) -eq 0 ]; then
    qz "$model" "f-${s}-gptq3-gm1"      "f_${s}_gm1"      --group-size -1
  else
    qz "$model" "f-${s}-gptq3-damp0p01" "f_${s}_damp0p01" --percdamp 0.01
  fi
else
  echo "bad task id"; exit 1
fi
echo "[W72b] done task $SGE_TASK_ID"
