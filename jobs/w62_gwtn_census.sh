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
#$ -N IFH_W62
#$ -t 1-21
# W62: the corrected objective (w_t = g_t / ||x_t||^2 : input whitening x
# output sensitivity; --hess-grad-weight --hess-token-norm) on the rest of the
# census, plus the missing controls. W61: it cured Llama c4 .651, Llama pile
# .649, Q14 .728, Mistral-v0.3 c4win .449, Nemo .545 (all >= RTN, >= tnorm
# except Mistral by 1 point). Pre-registered in RESULTS 9.10av.
#   1-14  gw+tn 3-bit, our short c4, 14 census models (Q7/Llama/Q14/Nemo done)
#   15    Q14 pile gw+tn (pile collapse .291)
#   16    Llama c4win gw+tn (.643 plain)
#   17    Llama 4-bit gw+tn (no harm at 4-bit; tnorm 4-bit .757)
#   18-20 GSM8K for Llama c4 / Q14 c4 / Llama pile gw+tn (re-quantised; GSM8K only)
#   21    Mistral-v0.3 c4win tnorm ALONE (attribution control for the W60/W61 cure)
# Checkpoints deleted after evaluation.
#   qsub jobs/w62_gwtn_census.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"
M7="mistralai/Mistral-7B-Instruct-v0.3"

MODELS=(
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "Qwen/Qwen2.5-3B-Instruct|q3"
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
run_gsm () {   # $1 ckpt  $2 tag
  $T python src/gsm8k_eval.py --model "$1" --tag "$2" --batch 16 --scores-csv runs/scores_gsm8k.csv
}
finish () {    # $1 ckpt  $2 tag  [$3 = gsm -> GSM8K instead of IFEval]
  if [ "${3:-}" = "gsm" ]; then run_gsm "$1" "gsm_$2"; else run_ifeval "$1" "$2"; fi
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
gt () {   # $1 model $2 ckpt $3 tag $4 calib $5 bits [$6 gsm]
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits "$5" --group-size 128 --protect none \
     --calib "$4" --hess-grad-weight --hess-token-norm --out "$CK"
  finish "$CK" "$3" "${6:-}"
}

id=$SGE_TASK_ID
if [ "$id" -le 14 ]; then
  entry="${MODELS[$((id - 1))]}"; model="${entry%%|*}"; s="${entry##*|}"
  gt "$model" "gt-${s}-v2gptq3-gwtn" "gt_${s}_gwtn" c4 3
else
  case $id in
    15) gt "$Q14"   qwen2.5-14b-v2gptq3-pile-gwtn   v2q14_pile_gwtn  pile  3 ;;
    16) gt "$LLAMA" llama3.1-8b-v2gptq3-c4win-gwtn  v2l_c4win_gwtn   c4win 3 ;;
    17) gt "$LLAMA" llama3.1-8b-v2gptq4-gwtn        v2l4_gwtn        c4    4 ;;
    18) gt "$LLAMA" llama3.1-8b-v2gptq3-gwtn-gsm    v2l_gwtn         c4    3 gsm ;;
    19) gt "$Q14"   qwen2.5-14b-v2gptq3-gwtn-gsm    v2q14_gwtn       c4    3 gsm ;;
    20) gt "$LLAMA" llama3.1-8b-v2gptq3-pile-gwtn-gsm v2l_pile_gwtn  pile  3 gsm ;;
    21) CK="$STORE/models/mistral-7b-v03-v2gptq3-c4win-tnorm"
        $T python src/quantize_protected.py --model "$M7" --bits 3 --group-size 128 --protect none \
           --calib c4win --hess-token-norm --out "$CK"
        finish "$CK" v2m7_c4win_tnorm ;;
    *) echo "bad task id"; exit 1 ;;
  esac
fi
echo "[W62] done task $SGE_TASK_ID"
