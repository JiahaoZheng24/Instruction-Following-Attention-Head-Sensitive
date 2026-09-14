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
#$ -N IFH_W60
#$ -t 1-10
# W60: OUTPUT-SIDE-WEIGHTED GPTQ (H_G = sum_t g_t x_t x_t^T, g_t = ||dL/dy_t||^2,
# one backward per calibration sample; --hess-grad-weight). Same frozen recipe
# otherwise (3-bit g128 sym act-order rho=0.05). Pre-registered in RESULTS
# 9.10aq: all three collapse arms recover to >= RTN, the four harmless models
# stay within +-3 of plain GPTQ. Checkpoints deleted after IFEval.
# Requires the W59 smoke output (task 1 of jobs/w59_hessian_sides.sh).
#   qsub jobs/w59_hessian_sides.sh   # first; check logs/IFH_W59.*.1
#   qsub jobs/w60_grad_hessian.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
[ -s runs/sides/smoke_l_c4.csv ] || { echo "W59 smoke output runs/sides/smoke_l_c4.csv missing: run W59 task 1 first"; exit 5; }

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
gw () {   # $1 model $2 ckpt $3 tag $4 calib $5 power
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none \
     --calib "$4" --hess-grad-weight --hess-grad-power "$5" --out "$CK"
  finish "$CK" "$3"
}

case "$SGE_TASK_ID" in
   1) gw "$LLAMA" llama3.1-8b-v2gptq3-gw        v2l_gw          c4    1.0 ;;
   2) gw "$Q14"   qwen2.5-14b-v2gptq3-gw        v2q14_gw        c4    1.0 ;;
   3) gw "$M7"    mistral-7b-v03-v2gptq3-c4win-gw v2m7_c4win_gw c4win 1.0 ;;
   4) gw "$LLAMA" llama3.1-8b-v2gptq3-pile-gw   v2l_pile_gw     pile  1.0 ;;
   5) gw "$LLAMA" llama3.1-8b-v2gptq3-gw05      v2l_gw05        c4    0.5 ;;
   6) gw "$Q14"   qwen2.5-14b-v2gptq3-gw05      v2q14_gw05      c4    0.5 ;;
   7) gw "$Q7"    qwen2.5-7b-v2gptq3-gw         v2q7_gw         c4    1.0 ;;
   8) gw mistralai/Mistral-Nemo-Instruct-2407   mistral-nemo-12b-v2gptq3-gw v2nemo_gw c4 1.0 ;;
   9) gw NousResearch/Hermes-3-Llama-3.1-8B     hermes3-8b-v2gptq3-gw       v2hermes3_gw c4 1.0 ;;
  10) gw tiiuae/Falcon3-7B-Instruct             falcon3-7b-v2gptq3-gw       v2f3_gw   c4 1.0 ;;
   *) echo "bad task id"; exit 1 ;;
esac
echo "[W60] done task $SGE_TASK_ID"
