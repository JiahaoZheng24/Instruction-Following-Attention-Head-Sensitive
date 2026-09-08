#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=10:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W39
#$ -t 1-8
# W39: harden the 4-bit headline (W38: Llama 4-bit GPTQ .745 -> .782 with RTN
# on the two early down_proj; packed gptqmodel 4-bit is already .776, so the
# claim must be seed-replicated inside the v2 protocol) and extend it to the
# two models where the 3-bit rule helped most.
#  1  Llama 4-bit none, calib seed 1           (noise floor of the .745)
#  2  Llama 4-bit RTN blocks 0-1 down_proj, seed 1   (replicate of .782)
#  3  Llama 4-bit RTN layer-1 down_proj only   (single module at 4-bit)
#  4  Llama 4-bit damp 0.1 none                (library-default damping: is the
#     packed .776 explained by damping alone?)
#  5  Q14 4-bit none
#  6  Q14 4-bit RTN layer-4 down_proj
#  7  Nemo 4-bit none
#  8  Nemo 4-bit RTN blocks 0-1 down_proj
# Also re-submit the fixed position-statistics tasks:  qsub -t 1-3 jobs/w38_impact.sh
#   qsub jobs/w39_4bit.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (4-bit, deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 4 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$LLAMA" llama3.1-8b-v2gptq4-none_cs1      v2l4_none_cs1      --calib-seed 1 ;;
  2) fq "$LLAMA" llama3.1-8b-v2gptq4-rtn01down_cs1 v2l4_rtn01down_cs1 --rtn-modules "0-1:down_proj" --calib-seed 1 ;;
  3) fq "$LLAMA" llama3.1-8b-v2gptq4-rtn1down      v2l4_rtn1down      --rtn-modules "1:down_proj" ;;
  4) fq "$LLAMA" llama3.1-8b-v2gptq4-damp0p1       v2l4_damp0p1       --percdamp 0.1 ;;
  5) fq "$Q14"   qwen2.5-14b-v2gptq4-none          v2q14_4_none ;;
  6) fq "$Q14"   qwen2.5-14b-v2gptq4-l4down-rtn    v2q14_4_l4down_rtn --rtn-modules "4:down_proj" ;;
  7) fq "$NEMO"  mistral-nemo-12b-v2gptq4-none     v2nemo4_none ;;
  8) fq "$NEMO"  mistral-nemo-12b-v2gptq4-rtn01down v2nemo4_rtn01down --rtn-modules "0-1:down_proj" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W39] done task $SGE_TASK_ID"
