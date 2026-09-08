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
#$ -N IFH_W40
#$ -t 1-9
# W40: PROSPECTIVE test of the position-resolved detector (last batch).
# W38 stats: max over template positions 2-7 of the per-position GPTQ/RTN
# output-error ratio ranks the culprit first on Llama (L1 down_proj, 102x)
# and Q14 (L4 down_proj, 456x). W38 census: the zero-bit rule (RTN on the two
# early down_proj) helped Nemo (+4.9) and Llama-3.2-3B (+2.3) and slightly
# hurt gemma-2-9b (-1.3), Mistral-7B-v0.2 (-2.4), Qwen2.5-7B (-0.9).
# Prediction: the detector fires (ratio > 2 on an early down_proj) on Nemo
# and Llama-3.2-3B and stays silent on the other three.
#  1-5  position statistics (no checkpoint) on the five census models
#  6-9  4-bit seed replicates: Q14 none / +rule, Nemo none / +rule (seed 1)
#   qsub jobs/w40_detector.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"
L32_3B="meta-llama/Llama-3.2-3B-Instruct"
G9="google/gemma-2-9b-it"
M7V02="mistralai/Mistral-7B-Instruct-v0.2"
mkdir -p runs/stats

posstats () {  # $1 model $2 stats-dir $3.. flags   (no checkpoint)
  local model="$1" sd="$2"; shift 2
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
     --stats-dir "$sd" --stats-chat-n 64 --no-save "$@" --out "$STORE/models/_unused_posstats"
}
fq4 () {  # $1 model $2 ckpt $3 tag $4.. flags   (4-bit, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 4 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) posstats "$NEMO"   runs/stats/nemo-12b-pos ;;
  2) posstats "$L32_3B" runs/stats/llama32-3b-pos ;;
  3) posstats "$G9"     runs/stats/gemma2-9b-pos ;;
  4) posstats "$M7V02"  runs/stats/mistral-7b-v02-pos ;;
  5) posstats "$Q7"     runs/stats/qwen25-7b-pos ;;
  6) fq4 "$Q14"  qwen2.5-14b-v2gptq4-none_cs1        v2q14_4_none_cs1        --calib-seed 1 ;;
  7) fq4 "$Q14"  qwen2.5-14b-v2gptq4-l4down-rtn_cs1  v2q14_4_l4down_rtn_cs1  --rtn-modules "4:down_proj" --calib-seed 1 ;;
  8) fq4 "$NEMO" mistral-nemo-12b-v2gptq4-none_cs1   v2nemo4_none_cs1        --calib-seed 1 ;;
  9) fq4 "$NEMO" mistral-nemo-12b-v2gptq4-rtn01down_cs1 v2nemo4_rtn01down_cs1 --rtn-modules "0-1:down_proj" --calib-seed 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W40] done task $SGE_TASK_ID"
