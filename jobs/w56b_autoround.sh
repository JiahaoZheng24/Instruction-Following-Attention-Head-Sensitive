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
#$ -N IFH_W56B
#$ -t 1-4
# W56b: AutoRound RERUN with a valid calibration set. W56 tasks 1-3 are void:
# AutoRound's text loader dropped every c4 doc shorter than 2048 tokens and
# quantised on 3 samples (log: "valid samples count is 3"). Now the exact
# 128 variable-length c4 samples of the GPTQ arms are fed as a DataLoader.
# Task 4 = AutoRound with its own default calibration (pile-10k, 128 x 2048)
# to separate "method" from "calibration set" if Llama collapses again.
# Pre-registered (RESULTS 9.10ag). Checkpoints deleted after evaluation.
#   qsub jobs/w56b_autoround.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

run_gsm () { $T python src/gsm8k_eval.py --model "$1" --tag "$2" --batch 16 --scores-csv runs/scores_gsm8k.csv; }
finish () {    # $1 ckpt  $2 tag  $3 gsm(0/1)
  run_ifeval "$1" "$2"
  [ "$3" = "1" ] && run_gsm "$1" "gsm_$2"
  cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$1"
}
in_ar_env () {
  conda activate IFEval_ar || { echo "env IFEval_ar missing: run bash jobs/w56_setup.sh"; exit 4; }
  "$@"; local rc=$?
  conda activate "$CONDA_ENV"
  return $rc
}

case "$SGE_TASK_ID" in
  1) CK="$STORE/models/llama3.1-8b-ar3b"
     in_ar_env $T python src/quantize_autoround.py --model "$LLAMA" --bits 3 --group-size 128 --calib c4 --out "$CK"
     finish "$CK" m_l_ar3b 1 ;;
  2) CK="$STORE/models/qwen2.5-14b-ar3b"
     in_ar_env $T python src/quantize_autoround.py --model "$Q14" --bits 3 --group-size 128 --calib c4 --out "$CK"
     finish "$CK" m_q14_ar3b 1 ;;
  3) CK="$STORE/models/llama3.1-8b-ar4b"
     in_ar_env $T python src/quantize_autoround.py --model "$LLAMA" --bits 4 --group-size 128 --calib c4 --out "$CK"
     finish "$CK" m_l4_arb 0 ;;
  4) CK="$STORE/models/llama3.1-8b-ar3pile"
     in_ar_env $T python src/quantize_autoround.py --model "$LLAMA" --bits 3 --group-size 128 --calib pile --out "$CK"
     finish "$CK" m_l_ar3pile 1 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W56B] done task $SGE_TASK_ID"
