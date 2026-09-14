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
#$ -N IFH_W66
#$ -t 1-56
# W66: (a) second calibration seed for the four cured arms; (b) the THIRD GATE
# measured without quantization: inject the lesion (BOS-direction leakage into
# the residual stream at chat-template positions after the sink-forming layer)
# into the fp16 model with a dose beta and read IFEval -> per-model tolerance
# beta*; (c) fp16 attention mass onto BOS / template positions per layer;
# (d) is the real GPTQ artifact BOS-direction leakage? (cosine, Llama & Q14).
# Pre-registered in RESULTS 9.10bb.
#    1-4   gw+tn seed 1: Llama c4, Q14 c4, Mistral-v0.3 c4win, Llama pile
#    5-14  measure (fp16, eager attention, 32 IFEval prompts) -> runs/sink/<s>_measure.csv
#   15-16  artifact: GPTQ3 c4 checkpoint vs fp16 at L1 (Llama) / L4 (Q14) -> runs/sink/<s>_artifact.csv
#   17-56  inject beta in {1,2,4,8} x 10 models -> IFEval tag inj_<s>_b<beta>
#   qsub jobs/w66_sink_tolerance.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs/sink

# model | short | sink-forming layer (max A_bos down_proj, W59/W63; gemma has none -> L1)
MODELS=(
  "$LLAMA|l|1"
  "$Q14|q14|4"
  "$Q7|q7|3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo|2"
  "$M7|m7|1"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3|1"
  "allenai/Llama-3.1-Tulu-3-8B|tulu3|1"
  "Qwen/Qwen3-14B|qwen3_14b|6"
  "tiiuae/Falcon3-7B-Instruct|f3|3"
  "google/gemma-2-9b-it|g29|1"
)
BETAS=(1 2 4 8)

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
gt_seed () {   # $1 model $2 ckpt $3 tag $4 calib
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none \
     --calib "$4" --calib-seed 1 --hess-grad-weight --hess-token-norm --out "$CK"
  finish "$CK" "$3"
}
score () {   # $1 model-id  $2 tag   (responses already written by inject_sink.py)
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

id=$SGE_TASK_ID
if [ "$id" -le 4 ]; then
  case $id in
    1) gt_seed "$LLAMA" llama3.1-8b-v2gptq3-gwtn-cs1        v2l_gwtn_cs1        c4 ;;
    2) gt_seed "$Q14"   qwen2.5-14b-v2gptq3-gwtn-cs1        v2q14_gwtn_cs1      c4 ;;
    3) gt_seed "$M7"    mistral-7b-v03-v2gptq3-c4win-gwtn-cs1 v2m7_c4win_gwtn_cs1 c4win ;;
    4) gt_seed "$LLAMA" llama3.1-8b-v2gptq3-pile-gwtn-cs1   v2l_pile_gwtn_cs1   pile ;;
  esac
elif [ "$id" -le 14 ]; then
  entry="${MODELS[$((id - 5))]}"; IFS='|' read -r model s L <<< "$entry"
  $T python src/inject_sink.py measure --model "$model" --prompts "$FULL" --n 32 --out "runs/sink/${s}_measure.csv"
elif [ "$id" -le 16 ]; then
  if [ "$id" -eq 15 ]; then model="$LLAMA"; s=l; L=1; ck=llama3.1-8b-v2gptq3-art; else model="$Q14"; s=q14; L=4; ck=qwen2.5-14b-v2gptq3-art; fi
  CK="$STORE/models/$ck"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK"
  $T python src/inject_sink.py artifact --model "$model" --quant "$CK" --prompts "$FULL" --n 32 --layer "$L" --out "runs/sink/${s}_artifact.csv"
  rm -rf "$CK"
else
  k=$((id - 17)); mi=$((k / 4)); bi=$((k % 4))
  entry="${MODELS[$mi]}"; IFS='|' read -r model s L <<< "$entry"; beta="${BETAS[$bi]}"
  tag="inj_${s}_b${beta}"
  $T python src/inject_sink.py inject --model "$model" --prompts "$FULL" --layer "$L" --beta "$beta" --tag "$tag" --batch 16
  score "$model" "$tag"
fi
echo "[W66] done task $SGE_TASK_ID"
