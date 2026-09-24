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
#$ -N IFH_W83
#$ -t 1-8
# W83 (2026-09-19). WHICH OF THE THIRTEEN MATRICES ARMS THE LESION? Pre-registered in RESULTS 9.10bx.
#   W82 showed: lesion + all-RTN network .528 (f_tp_l_rtn_les); lesion + GPTQ layers 0-1 + RTN layers 2-31 .152
#   (f_l_rtntail_2). Fixed-matrix transplant from the collapsed short-c4 GPTQ3 Llama (donor) into the RTN3 Llama
#   (target); every arm includes the lesion (layer 1 down_proj).
#   1  0-1:all                                   14 matrices  (must reproduce .15: consistency check)
#   2  1:all                                     layer 1 whole
#   3  0:all;1:down_proj                         layer 0 whole + lesion
#   4  1:gate_proj,up_proj,down_proj             layer-1 MLP
#   5  1:q_proj,k_proj,v_proj,o_proj,down_proj   layer-1 attention + lesion
#   6  0-1:down_proj                             the two down-projections
#   7  0:q_proj,k_proj,v_proj,o_proj;1:down_proj layer-0 attention + lesion
#   8  0:gate_proj,up_proj;1:down_proj           layer-0 gate/up + lesion
#   qsub jobs/w83_background_split.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 5h"
mkdir -p runs

quant () {   # $1 model $2 ckpt-name $3.. flags  -> echoes ckpt path
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@" >&2
  echo "$CK"
}
score () {   # $1 model-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

SPECS=("0-1:all" "1:all" "0:all;1:down_proj" "1:gate_proj,up_proj,down_proj" \
       "1:q_proj,k_proj,v_proj,o_proj,down_proj" "0-1:down_proj" \
       "0:q_proj,k_proj,v_proj,o_proj;1:down_proj" "0:gate_proj,up_proj;1:down_proj")
TAGS=(f_tp_l_rtn_L01all f_tp_l_rtn_L1all f_tp_l_rtn_L0all_les f_tp_l_rtn_L1mlp \
      f_tp_l_rtn_L1attn_les f_tp_l_rtn_L01down f_tp_l_rtn_L0attn_les f_tp_l_rtn_L0mlp_les)
id=$SGE_TASK_ID
SPEC="${SPECS[$((id - 1))]}"; TAG="${TAGS[$((id - 1))]}"
echo "[W83] task $id spec=$SPEC tag=$TAG"

D=$(quant "$LLAMA" "f-l-donor83-$id")
TG=$(quant "$LLAMA" "f-l-rtn3-tp83-$id" --rtn)
$T python src/transplant.py --target "$TG" --donor "$D" --spec "$SPEC" --prompts "$FULL" --tag "$TAG" --batch 16
score "$TG" "$TAG"
rm -rf "$D" "$TG"
echo "[W83] done task $SGE_TASK_ID"
