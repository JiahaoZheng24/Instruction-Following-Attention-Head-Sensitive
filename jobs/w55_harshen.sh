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
#$ -N IFH_W55
#$ -t 1-20
# W55: PREDICTED NEW COLLAPSES under equally standard recipes. The collapse is
# a threshold on (down_proj template error) x (block background error). Two
# standard knobs raise the product: the ORIGINAL GPTQ default damping 0.01
# (AutoGPTQ default; most public GPTQ checkpoints) and per-channel grouping
# (g=-1, raises the background x1.3). Pre-registered (RESULTS 9.10ad):
#   detector >= 60 -> Q32B (69), Hermes-3 (63), Nemo (60): COLLAPSE under g=-1
#                     (GPTQ below RTN by > 20); rho=0.01 not predicted either way
#   detector 30-33 -> Ministral (33), Qwen3-14B (31), Mistral-v0.3 (30): drop >= 10
#                     vs the g128 arm but may stay above the line
# Each model: GPTQ rho=0.01 g128 | GPTQ g=-1 rho=0.05 | RTN g=-1 (baseline);
# plus RTN g128 for Q32B and Nemo (missing). Checkpoints deleted after eval.
#   qsub jobs/w55_harshen.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 8h"

MODELS=(
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b"
  "mistralai/Mistral-7B-Instruct-v0.3|m7"
)
fq () {  # $1 model $2 ckpt $3 tag $4 group $5.. flags
  local model="$1" ckpt="$2" tag="$3" grp="$4"; shift 4
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size "$grp" --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

if [ "$SGE_TASK_ID" -le 18 ]; then
  idx=$(( (SGE_TASK_ID - 1) / 3 )); kind=$(( (SGE_TASK_ID - 1) % 3 ))
  entry="${MODELS[$idx]}"; model="${entry%%|*}"; s="${entry##*|}"
  case $kind in
    0) fq "$model" "harsh-${s}-gptq3-damp0p01" "h_${s}_damp0p01" 128 --percdamp 0.01 ;;
    1) fq "$model" "harsh-${s}-gptq3-gm1"      "h_${s}_gm1"      -1 ;;
    2) fq "$model" "harsh-${s}-rtn3-gm1"       "h_${s}_rtn_gm1"  -1 --quantizer rtn ;;
  esac
elif [ "$SGE_TASK_ID" -eq 19 ]; then
  fq Qwen/Qwen2.5-32B-Instruct harsh-q32-rtn3 h_q32_rtn 128 --quantizer rtn
elif [ "$SGE_TASK_ID" -eq 20 ]; then
  fq mistralai/Mistral-Nemo-Instruct-2407 harsh-nemo-rtn3 h_nemo_rtn 128 --quantizer rtn
else
  echo "bad task id"; exit 1
fi
echo "[W55] done task $SGE_TASK_ID"
