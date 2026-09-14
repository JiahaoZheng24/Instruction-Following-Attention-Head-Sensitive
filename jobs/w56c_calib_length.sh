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
#$ -N IFH_W56C
#$ -t 1-7
# W56c: W56b showed AutoRound is BROKEN by our c4 calibration set (Llama .460,
# Q14 .181) and EXCELLENT with its default pile-10k (Llama .716). Two things
# differ between those sets: corpus (c4 vs pile) and sample length (our c4 =
# 128 short docs, mean 420 tokens, BOS share ~1/420; pile = 128 x 2048, BOS
# share 1/2048). This batch separates them, for AutoRound AND for GPTQ (the
# latter is also the missing protocol check: does the literature-standard
# 128 x 2048 c4 budget collapse Llama/Q14 under GPTQ?). Pre-registered in
# RESULTS 9.10aj. Checkpoints deleted after evaluation.
#   qsub jobs/w56c_calib_length.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

finish () {    # $1 ckpt  $2 tag
  run_ifeval "$1" "$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
in_ar_env () {
  conda activate IFEval_ar || { echo "env IFEval_ar missing: run bash jobs/w56_setup.sh"; exit 4; }
  "$@"; local rc=$?
  conda activate "$CONDA_ENV"
  return $rc
}
gq () {   # $1 model $2 ckpt $3 tag $4 calib   (GPTQ, frozen protocol, only the calibration set changes)
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none --calib "$4" --out "$CK"
  finish "$CK" "$3"
}
ar () {   # $1 model $2 ckpt $3 tag $4 calib
  CK="$STORE/models/$2"
  in_ar_env $T python src/quantize_autoround.py --model "$1" --bits 3 --group-size 128 --calib "$4" --out "$CK"
  finish "$CK" "$3"
}

case "$SGE_TASK_ID" in
  1) gq "$LLAMA" llama3.1-8b-v2gptq3-c4win  v2l_c4win    c4win ;;
  2) gq "$Q14"   qwen2.5-14b-v2gptq3-c4win  v2q14_c4win  c4win ;;
  3) gq "$LLAMA" llama3.1-8b-v2gptq3-pile   v2l_pile     pile  ;;
  4) gq "$Q14"   qwen2.5-14b-v2gptq3-pile   v2q14_pile   pile  ;;
  5) ar "$LLAMA" llama3.1-8b-ar3c4win       m_l_ar3c4win   c4win ;;
  6) ar "$Q14"   qwen2.5-14b-ar3c4win       m_q14_ar3c4win c4win ;;
  7) ar "$Q14"   qwen2.5-14b-ar3pile        m_q14_ar3pile  pile  ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W56C] done task $SGE_TASK_ID"
