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
#$ -N IFH_W85
#$ -t 1-9
# W85 (2026-09-21). IS THE PAIR SUFFICIENT, AND WHAT CARRIES MISTRAL? Pre-registered in RESULTS 9.10cd.
#   W84 left three holes:
#   (a) Llama seed 1 does not collapse (.531) although its two sink matrices look like seed 0's
#       (BOS share .247 against .266, same lesion size) -- is its PAIR benign, or is its background?
#   (b) Tulu-3, the same base model under the same protocol, has the same two sinks (.266) and does
#       not collapse (.726) -- same question across models.
#   (c) Qwen is healthy under c4win although its sink matrix alone collapses an RTN network: is the
#       c4win version of that matrix a weaker lesion, or is the protocol dependence in the background?
#   (d) Mistral's window collapse is carried by neither down-projection -- where is it?
#   1  Llama seed 1: (0-1,down_proj) into the RTN3 Llama
#   2  Llama seed 1: all 14 matrices of layers 0-1 into the RTN3 Llama
#   3  Tulu-3: its own (0-1,down_proj) into the RTN3 Tulu-3
#   4  Qwen2.5-14B: the c4win (4,down_proj) into the RTN3 Qwen
#   5  Mistral c4win: all 14 matrices of layers 0-1 into the RTN3 Mistral
#   6  Mistral c4win: layer 1 whole (7 matrices)
#   7  Mistral c4win: layers 0-3 down_proj (the phenotype is spread over ordinary tokens)
#   8  Mistral c4win: all 28 matrices of layers 0-3
#   9  Tulu-3 RTN3 baseline (missing from the index; task 3 cannot be read without it)
#   qsub jobs/w85_sufficiency.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 5h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
TULU="allenai/Llama-3.1-Tulu-3-8B"
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

id=$SGE_TASK_ID
case "$id" in
  1|2) D=$(quant "$LLAMA" "f-l-donor85-$id" --calib-seed 1); TG=$(quant "$LLAMA" "f-l-rtn3-tp85-$id" --rtn)
       [ "$id" = 1 ] && { SPEC="0-1:down_proj"; TAG=f_tp_l_rtn_cs1_L01down; } || { SPEC="0-1:all"; TAG=f_tp_l_rtn_cs1_L01all; }
       tps "$TG" "$D" "$SPEC" "$TAG"; rm -rf "$D" "$TG" ;;
  3)   D=$(quant "$TULU" f-tulu3-donor85-3); TG=$(quant "$TULU" f-tulu3-rtn3-tp85-3 --rtn)
       tps "$TG" "$D" "0-1:down_proj" f_tp_tulu3_rtn_L01down; rm -rf "$D" "$TG" ;;
  4)   D=$(quant "$Q14" f-q14-c4win-donor85-4 --calib c4win); TG=$(quant "$Q14" f-q14-rtn3-tp85-4 --rtn)
       tps "$TG" "$D" "4:down_proj" f_tp_q14_rtn_cwinL4down; rm -rf "$D" "$TG" ;;
  5|6|7|8)
       D=$(quant "$M7" "f-m7-donor85-$id" --calib c4win); TG=$(quant "$M7" "f-m7-rtn3-tp85-$id" --rtn --calib c4win)
       case "$id" in
         5) SPEC="0-1:all";        TAG=f_tp_m7_rtn_L01all ;;
         6) SPEC="1:all";          TAG=f_tp_m7_rtn_L1all ;;
         7) SPEC="0-3:down_proj";  TAG=f_tp_m7_rtn_L03down ;;
         8) SPEC="0-3:all";        TAG=f_tp_m7_rtn_L03all ;;
       esac
       tps "$TG" "$D" "$SPEC" "$TAG"; rm -rf "$D" "$TG" ;;
  9)   CK=$(quant "$TULU" f-tulu3-rtn3 --rtn); run_ifeval "$CK" f_tulu3_rtn3
       cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
       rm -rf "$CK" ;;
  *)   echo "bad task id"; exit 1 ;;
esac
echo "[W85] done task $SGE_TASK_ID"
