#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=06:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W84
#$ -t 1-9
# W84 (2026-09-20). THE COLLAPSE NEEDS TWO SINK MATRICES. Pre-registered in RESULTS 9.10ca.
#   W83: (0,down_proj)+(1,down_proj) of the collapsed short-c4 GPTQ3 Llama, transplanted into the RTN3
#   Llama, collapse it (.139); layer 1 whole (7 matrices, lesion included) is healthy (.572); layer-0
#   attention or gate/up plus the lesion are healthy (.577/.536). Layer 0's down_proj is itself a weaker
#   sink matrix (BOS share of tr H .27 against .998, overshoot 3.7x against 8.7x).
#   1  RTN layer-0 down_proj only, everything else GPTQ (fixed code; pre-fix value was .629)
#   2  transplant (0,down_proj) ALONE into RTN3 Llama            -> is one sink matrix enough?
#   3  into the COLLAPSED short-c4 net: c4win's (0,down_proj)    -> does the window version disarm it?
#   4  into the COLLAPSED short-c4 net: c4win's (0-1,down_proj)  -> control for task 3
#   5  Qwen2.5-14B: transplant (4,down_proj)+(5,down_proj) into the RTN3 Qwen  (sink L4, second L5)
#   6  Qwen2.5-14B: transplant (4,down_proj) alone into the RTN3 Qwen
#   7  Qwen2.5-14B: RTN layers 4-5 down_proj only, everything else GPTQ
#   8  Mistral-7B under c4win (its collapsing protocol): transplant (0-1,down_proj) into the RTN3 Mistral
#   9  Mistral-7B under c4win: transplant (1,down_proj) alone into the RTN3 Mistral
#   qsub jobs/w84_two_sinks.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 5h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs runs/protocols

quant () {   # $1 model $2 ckpt-name $3.. flags  -> echoes ckpt path
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@" >&2
  echo "$CK"
}
score () {   # $1 model-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
tps () {     # $1 target $2 donor $3 spec $4 tag
  $T python src/transplant.py --target "$1" --donor "$2" --spec "$3" --prompts "$FULL" --tag "$4" --batch 16
  score "$1" "$4"
}
qz () {      # $1 model $2 ckpt $3 tag $4.. flags : quantise + IFEval + protocol, then delete
  CK=$(quant "$1" "$2" "${@:4}"); run_ifeval "$CK" "$3"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}

case "$SGE_TASK_ID" in
  1) qz "$LLAMA" f-l-rtn0down f_l_rtn0down --rtn-modules "0:down_proj" ;;
  2) D=$(quant "$LLAMA" f-l-donor84-2); TG=$(quant "$LLAMA" f-l-rtn3-tp84-2 --rtn)
     tps "$TG" "$D" "0:down_proj" f_tp_l_rtn_L0down; rm -rf "$D" "$TG" ;;
  3) D=$(quant "$LLAMA" f-l-c4win-donor84-3 --calib c4win); TG=$(quant "$LLAMA" f-l-short-tgt84-3)
     tps "$TG" "$D" "0:down_proj" f_tp_l_short_cwinL0; rm -rf "$D" "$TG" ;;
  4) D=$(quant "$LLAMA" f-l-c4win-donor84-4 --calib c4win); TG=$(quant "$LLAMA" f-l-short-tgt84-4)
     tps "$TG" "$D" "0-1:down_proj" f_tp_l_short_cwinL01; rm -rf "$D" "$TG" ;;
  5) D=$(quant "$Q14" f-q14-donor84-5); TG=$(quant "$Q14" f-q14-rtn3-tp84-5 --rtn)
     tps "$TG" "$D" "4-5:down_proj" f_tp_q14_rtn_L45down; rm -rf "$D" "$TG" ;;
  6) D=$(quant "$Q14" f-q14-donor84-6); TG=$(quant "$Q14" f-q14-rtn3-tp84-6 --rtn)
     tps "$TG" "$D" "4:down_proj" f_tp_q14_rtn_L4down; rm -rf "$D" "$TG" ;;
  7) qz "$Q14" f-q14-rtn45down f_q14_rtn45down --rtn-modules "4-5:down_proj" ;;
  8) D=$(quant "$M7" f-m7-donor84-8 --calib c4win); TG=$(quant "$M7" f-m7-rtn3-tp84-8 --rtn --calib c4win)
     tps "$TG" "$D" "0-1:down_proj" f_tp_m7_rtn_L01down; rm -rf "$D" "$TG" ;;
  9) D=$(quant "$M7" f-m7-donor84-9 --calib c4win); TG=$(quant "$M7" f-m7-rtn3-tp84-9 --rtn --calib c4win)
     tps "$TG" "$D" "1:down_proj" f_tp_m7_rtn_L1down; rm -rf "$D" "$TG" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W84] done task $SGE_TASK_ID"
