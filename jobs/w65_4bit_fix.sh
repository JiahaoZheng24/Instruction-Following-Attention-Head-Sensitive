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
#$ -N IFH_W65
#$ -t 1-6
# W65: the 4-bit census (W64) refuted "4-bit is immune": GPTQ4 loses to RTN4 on
# Mistral-7B-v0.3 (-9.5; GPTQ4 .450 is below its own GPTQ3 .468), Qwen3-14B
# (-5.7; .470 < GPTQ3 .487), Qwen2.5-32B (-3.1), Hermes-3 (-2.2); mean over 17
# models -0.6. Does the corrected objective (w_t = g_t/||x_t||^2) close the
# 4-bit gap the same way it closes the 3-bit one? Pre-registered in RESULTS
# 9.10ay. Also fills two holes: Q14 RTN4 (U2 table) and Q7 gw+tn at 3-bit
# (U4 compared pile gw+tn against G-only). Checkpoints deleted after IFEval.
#   qsub jobs/w65_4bit_fix.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
Q32="Qwen/Qwen2.5-32B-Instruct"

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
gt () {   # $1 model $2 ckpt $3 tag $4 bits
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits "$4" --group-size 128 --protect none \
     --calib c4 --hess-grad-weight --hess-token-norm --out "$CK"
  finish "$CK" "$3"
}

case "$SGE_TASK_ID" in
  1) gt "$M7"                              u4-m7-v2gptq4-gwtn        u4_m7_gwtn        4 ;;
  2) gt Qwen/Qwen3-14B                     u4-qwen3_14b-v2gptq4-gwtn u4_qwen3_14b_gwtn 4 ;;
  3) gt "$Q32"                             u4-q32-v2gptq4-gwtn       u4_q32_gwtn       4 ;;
  4) gt NousResearch/Hermes-3-Llama-3.1-8B u4-hermes3-v2gptq4-gwtn   u4_hermes3_gwtn   4 ;;
  5) CK="$STORE/models/u4-q14-rtn4"
     $T python src/quantize_protected.py --model "$Q14" --bits 4 --group-size 128 --protect none --rtn --out "$CK"
     finish "$CK" u4_q14_rtn4 ;;
  6) gt "$Q7"                              gt-q7-v2gptq3-gwtn        gt_q7_gwtn        3 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W65] done task $SGE_TASK_ID"
