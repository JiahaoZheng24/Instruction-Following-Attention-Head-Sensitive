#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=08:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W81
#$ -t 1-5
# W81 (2026-09-19). Two reviewer gaps of the theory section, pre-registered in RESULTS 9.10bt.
#   1    cross-position curvature at the sink-forming layer (Lemma 1 drops G_ts, s != t):
#        full quadratic form vs block-diagonal estimate for the real lesion, the RTN error and
#        a random matrix, Llama layer 1, 16 chat prompts, fp32 -> runs/theory/f_l_crosspos.csv
#   2    the same on Qwen2.5-14B, layer 4 -> runs/theory/f_q14_crosspos.csv
#   3-5  cost of the corrected objective at 14B and 32B (quantise only, no checkpoint):
#        Q14 gw+tn; Q32 GPTQ; Q32 gw+tn  -> wall-clock and peak memory in the log
#   qsub jobs/w81_theory_gaps.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"
Q32="Qwen/Qwen2.5-32B-Instruct"
GWTN="--hess-grad-weight --hess-token-norm"
mkdir -p runs/theory

timed () {   # $1 model $2 tag $3.. flags   (quantise only; --no-save needs --stats-dir, which we discard)
  m="$1"; tag="$2"; shift 2
  t0=$SECONDS
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 \
     --stats-dir "runs/stats/_w81_$tag" --no-save --out "$STORE/models/_unused_w81_$tag" "$@"
  echo "[W81-timing] tag=$tag quantise_seconds=$((SECONDS - t0))"
  rm -rf "runs/stats/_w81_$tag"
}

case "$SGE_TASK_ID" in
  1) $T python src/cross_position.py --model "$LLAMA" --layer 1 --calib c4 --prompts "$FULL" --n 16 --k 8 --out runs/theory/f_l_crosspos.csv ;;
  2) $T python src/cross_position.py --model "$Q14"   --layer 4 --calib c4 --prompts "$FULL" --n 16 --k 8 --out runs/theory/f_q14_crosspos.csv ;;
  3) timed "$Q14" q14_gwtn $GWTN ;;
  4) timed "$Q32" q32_none ;;
  5) timed "$Q32" q32_gwtn $GWTN ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W81] done task $SGE_TASK_ID"
