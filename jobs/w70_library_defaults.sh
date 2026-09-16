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
#$ -N IFH_W70
#$ -t 1-4
# W70 (2026-09-15). Pre-registered in RESULTS 9.10bk. Closes the abstract claim
# "GPTQ under the calibration protocol a major library uses by default":
# GPTQModel's README example = 1024 c4 documents, per-row tokenisation, no
# concatenation, damp_percent 0.1. Our census used 128 documents and rho 0.05.
#   1  Llama  c4 x1024  percdamp 0.10   (GPTQModel README defaults)
#   2  Q14    c4 x1024  percdamp 0.10
#   3  Llama  c4 x1024  percdamp 0.05   (control: document count alone)
#   4  Q14    c4 x1024  percdamp 0.05
# Checkpoints deleted after scoring.
#   qsub jobs/w70_library_defaults.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 11h"

q1024 () {   # $1 model $2 ckpt-name $3 percdamp $4 tag
  CK="$STORE/models/$2"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none \
     --calib c4 --n-calib 1024 --percdamp "$3" --out "$CK"
  run_ifeval "$CK" "$4"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}

case "$SGE_TASK_ID" in
  1) q1024 "$LLAMA" llama3.1-8b-v2gptq3-c4x1024-d10 0.10 v2l_c4x1024_d10 ;;
  2) q1024 "$Q14"   qwen2.5-14b-v2gptq3-c4x1024-d10 0.10 v2q14_c4x1024_d10 ;;
  3) q1024 "$LLAMA" llama3.1-8b-v2gptq3-c4x1024-d05 0.05 v2l_c4x1024_d05 ;;
  4) q1024 "$Q14"   qwen2.5-14b-v2gptq3-c4x1024-d05 0.05 v2q14_c4x1024_d05 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W70] done task $SGE_TASK_ID"
