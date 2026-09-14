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
#$ -N IFH_W64
#$ -t 1-50
# W64 (universality, part 2). Pre-registered in RESULTS 9.10ax.
#   1-34  4-bit census: GPTQ g128 vs RTN g128 on the 16 W58 models (+ RTN4 for
#         Llama, Q14). Claim U2: at 4-bit GPTQ - RTN >= -2 everywhere (the blind
#         spot never bites); the 4 -> 3 cliff is the objective's, not the grid's.
#         odd id = GPTQ4 (u4_<s>_gptq4), even id = RTN4 (u4_<s>_rtn4)
#   35-44 corrected objective (gw+tn) on 10 HELD-OUT blind-test models with
#         GPTQ3/RTN3 already measured. Claim U3 (prospective): gw+tn >= RTN - 3
#         on every model; >= GPTQ wherever GPTQ < RTN.
#   45-50 gw+tn under pile 128x2048 windows on 6 models whose plain-GPTQ pile
#         numbers exist (W58). Claim U4: |gw+tn(pile) - gw+tn(c4)| <= 3.
# Checkpoints deleted after IFEval.
#   qsub jobs/w64_universal_census.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

M4=(
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
  "meta-llama/Llama-3.1-8B-Instruct|l"
  "Qwen/Qwen2.5-14B-Instruct|q14"
)
HELD=(
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

id=$SGE_TASK_ID
if [ "$id" -le 34 ]; then
  idx=$(( (id - 1) / 2 )); entry="${M4[$idx]}"; model="${entry%%|*}"; s="${entry##*|}"
  [ -n "$model" ] || { echo "bad task id"; exit 1; }
  if [ $(( id % 2 )) -eq 1 ]; then
    if [ "$s" = "l" ] || [ "$s" = "q14" ]; then echo "[W64] GPTQ4 for $s exists (v2l4_none / Q14 4-bit); skipping"; exit 0; fi
    CK="$STORE/models/u4-${s}-v2gptq4"
    $T python src/quantize_protected.py --model "$model" --bits 4 --group-size 128 --protect none --out "$CK"
    finish "$CK" "u4_${s}_gptq4"
  else
    CK="$STORE/models/u4-${s}-rtn4"
    $T python src/quantize_protected.py --model "$model" --bits 4 --group-size 128 --protect none --rtn --out "$CK"
    finish "$CK" "u4_${s}_rtn4"
  fi
elif [ "$id" -le 44 ]; then
  entry="${HELD[$((id - 35))]}"; model="${entry%%|*}"; s="${entry##*|}"
  CK="$STORE/models/held-${s}-v2gptq3-gwtn"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
     --calib c4 --hess-grad-weight --hess-token-norm --out "$CK"
  finish "$CK" "held_${s}_gwtn"
else
  entry="${PILE[$((id - 45))]}"; model="${entry%%|*}"; s="${entry##*|}"
  CK="$STORE/models/gt-${s}-v2gptq3-pile-gwtn"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
     --calib pile --hess-grad-weight --hess-token-norm --out "$CK"
  finish "$CK" "gt_${s}_gwtn_pile"
fi
echo "[W64] done task $SGE_TASK_ID"
