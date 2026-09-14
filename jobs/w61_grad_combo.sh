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
#$ -N IFH_W61
#$ -t 1-8
# W61: G-weighting x input normalisation, and a cap on the G weights.
# W60: scalar output-side weights (H_G) cured Q14 (.412 -> .699 = RTN) and
# Mistral-v0.3 c4win (.172 -> .458 > RTN), lifted Nemo / Hermes-3 above RTN,
# but left Llama c4 at .430 (< RTN .565) and made Llama pile WORSE (.402 -> .139).
# H_G is still 97-99% BOS at the lesion (||x_bos||^2 too large) and its
# weights span 1e5-1e6, so a handful of sink-receiving tokens own H.
# Two principled repairs, pre-registered in RESULTS 9.10at:
#   gw+tnorm : w_t = g_t / ||x_t||^2  (per-token whitened input x sensitivity;
#              BOS dominance gone, G kept)          -> tasks 1-4 (+ Nemo control 8)
#   gw cap100: w_t = min(g_t, 100 x median)        -> tasks 5-7
#   qsub jobs/w61_grad_combo.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
q () {   # $1 model $2 ckpt $3 tag $4 calib $5... extra flags
  CK="$STORE/models/$2"; m="$1"; ck="$2"; tag="$3"; cal="$4"; shift 4
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none \
     --calib "$cal" --hess-grad-weight "$@" --out "$CK"
  finish "$CK" "$tag"
}

case "$SGE_TASK_ID" in
  1) q "$LLAMA" llama3.1-8b-v2gptq3-gwtn        v2l_gwtn        c4    --hess-token-norm ;;
  2) q "$LLAMA" llama3.1-8b-v2gptq3-pile-gwtn   v2l_pile_gwtn   pile  --hess-token-norm ;;
  3) q "$Q14"   qwen2.5-14b-v2gptq3-gwtn        v2q14_gwtn      c4    --hess-token-norm ;;
  4) q "$M7"    mistral-7b-v03-v2gptq3-c4win-gwtn v2m7_c4win_gwtn c4win --hess-token-norm ;;
  5) q "$LLAMA" llama3.1-8b-v2gptq3-gwcap       v2l_gwcap       c4    --hess-grad-cap 100 ;;
  6) q "$LLAMA" llama3.1-8b-v2gptq3-pile-gwcap  v2l_pile_gwcap  pile  --hess-grad-cap 100 ;;
  7) q "$Q14"   qwen2.5-14b-v2gptq3-gwcap       v2q14_gwcap     c4    --hess-grad-cap 100 ;;
  8) q "$NEMO"  mistral-nemo-12b-v2gptq3-gwtn   v2nemo_gwtn     c4    --hess-token-norm ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W61] done task $SGE_TASK_ID"
