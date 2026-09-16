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
#$ -N IFH_W73
#$ -t 1-4
# W73 (2026-09-15): the cells of the paper's Table 1 that no earlier batch produced.
# Fixed-code path; tags f_<model>_<arm>.
#   1  Llama RTN3  : MMLU + PPL            (RTN is unaffected by the loop fix; never measured on these)
#   2  Q14   RTN3  : MMLU + PPL
#   3  Llama GPTQ4 : GSM8K + MMLU + PPL + Multi-IF
#   4  Q14   GPTQ4 : GSM8K + MMLU + PPL + Multi-IF
#   qsub jobs/w73_table_gaps.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"

quant () {   # $1 model $2 ckpt-name $3.. flags -> echoes ckpt path
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --group-size 128 --protect none --calib c4 --out "$CK" "$@" >&2
  echo "$CK"
}
gen () {  $T python src/eval_general.py --model "$1" --tag "$2" --scores-csv "runs/general_$2.csv"; }
gsm () {  $T python src/gsm8k_eval.py --model "$1" --tag "gsm_$2" --batch 16 --scores-csv runs/scores_gsm8k.csv; }
mif () {  $T python src/multi_if.py --model "$1" --tag "mif_$2" --batch 8 --scores-csv "runs/scores_mif_$2.csv"; }

case "$SGE_TASK_ID" in
  1) CK=$(quant "$LLAMA" f-l-rtn3-gen  --bits 3 --rtn); gen "$CK" f_l_rtn3;   rm -rf "$CK" ;;
  2) CK=$(quant "$Q14"   f-q14-rtn3-gen --bits 3 --rtn); gen "$CK" f_q14_rtn3; rm -rf "$CK" ;;
  3) CK=$(quant "$LLAMA" f-l-gptq4-ds  --bits 4); gsm "$CK" f_l_gptq4;   gen "$CK" f_l_gptq4;   mif "$CK" f_l_gptq4;   rm -rf "$CK" ;;
  4) CK=$(quant "$Q14"   f-q14-gptq4-ds --bits 4); gsm "$CK" f_q14_gptq4; gen "$CK" f_q14_gptq4; mif "$CK" f_q14_gptq4; rm -rf "$CK" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W73] done task $SGE_TASK_ID"
