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
#$ -N IFH_W56
#$ -t 1-10
# W56: METHOD CENSUS on the two collapsing models. Does the finding depend on
# GPTQ, or on the three ingredients (token-averaged layer objective, unbounded
# OBS compensation, original basis)? Pre-registered in RESULTS 9.10af:
#   AutoRound  (same objective, displacement bounded to half a step) -> NO collapse
#   HQQ        (calibration-free, no Hessian)                        -> NO collapse
#   SpQR-style (GPTQ + 1% outliers by OBS saliency err^2/[H^-1]_jj)  -> STILL collapses
#              (the collinear BOS block has large [H^-1]_jj, so its weights look
#              cheap and are not isolated; contrast hmag = |W|sqrt(H_jj) which cures)
# IFEval on every arm; GSM8K on the 3-bit arms (+ the missing RTN GSM8K baselines).
# Requires: bash jobs/w56_setup.sh once (conda env IFEval_ar with auto-round, hqq).
# Checkpoints deleted after evaluation.
#   qsub jobs/w56_methods.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

run_gsm () {   # $1 ckpt  $2 tag
  $T python src/gsm8k_eval.py --model "$1" --tag "$2" --batch 16 --scores-csv runs/scores_gsm8k.csv
}
finish () {    # $1 ckpt  $2 tag  $3 gsm(0/1)
  run_ifeval "$1" "$2"
  [ "$3" = "1" ] && run_gsm "$1" "gsm_$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
in_ar_env () {   # run a command inside the AutoRound/HQQ env, then return to IFEval
  conda activate IFEval_ar || { echo "env IFEval_ar missing: run bash jobs/w56_setup.sh"; exit 4; }
  "$@"; local rc=$?
  conda activate "$CONDA_ENV"
  return $rc
}

case "$SGE_TASK_ID" in
  1) CK="$STORE/models/llama3.1-8b-ar3"
     in_ar_env $T python src/quantize_autoround.py --model "$LLAMA" --bits 3 --group-size 128 --out "$CK"
     finish "$CK" m_l_ar3 1 ;;
  2) CK="$STORE/models/qwen2.5-14b-ar3"
     in_ar_env $T python src/quantize_autoround.py --model "$Q14" --bits 3 --group-size 128 --out "$CK"
     finish "$CK" m_q14_ar3 1 ;;
  3) CK="$STORE/models/llama3.1-8b-ar4"
     in_ar_env $T python src/quantize_autoround.py --model "$LLAMA" --bits 4 --group-size 128 --out "$CK"
     finish "$CK" m_l4_ar 0 ;;
  4) CK="$STORE/models/llama3.1-8b-hqq3"
     in_ar_env $T python src/quantize_hqq.py --model "$LLAMA" --bits 3 --group-size 128 --out "$CK"
     finish "$CK" m_l_hqq3 1 ;;
  5) CK="$STORE/models/qwen2.5-14b-hqq3"
     in_ar_env $T python src/quantize_hqq.py --model "$Q14" --bits 3 --group-size 128 --out "$CK"
     finish "$CK" m_q14_hqq3 1 ;;
  6) CK="$STORE/models/llama3.1-8b-hqq4"
     in_ar_env $T python src/quantize_hqq.py --model "$LLAMA" --bits 4 --group-size 128 --out "$CK"
     finish "$CK" m_l4_hqq 0 ;;
  7) CK="$STORE/models/llama3.1-8b-v2gptq3-spqr1"
     $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect spqr --frac 0.01 --scale-excl-mask --out "$CK"
     finish "$CK" m_l_spqr1 1 ;;
  8) CK="$STORE/models/qwen2.5-14b-v2gptq3-spqr1"
     $T python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 --protect spqr --frac 0.01 --scale-excl-mask --out "$CK"
     finish "$CK" m_q14_spqr1 1 ;;
  9) CK="$STORE/models/llama3.1-8b-rtn3-gsm"     # GSM8K for the plain 3-bit RTN baseline (IFEval .565 exists)
     $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --quantizer rtn --out "$CK"
     run_gsm "$CK" gsm_l_rtn3; rm -rf "$CK" ;;
  10) CK="$STORE/models/qwen2.5-14b-rtn3-gsm"    # GSM8K for the plain 3-bit RTN baseline (IFEval .697 exists)
     $T python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 --protect none --quantizer rtn --out "$CK"
     run_gsm "$CK" gsm_q14_rtn3; rm -rf "$CK" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W56] done task $SGE_TASK_ID"
