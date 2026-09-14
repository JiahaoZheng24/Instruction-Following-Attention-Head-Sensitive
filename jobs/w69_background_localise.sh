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
#$ -N IFH_W69
#$ -t 1-17
# W69 (2026-09-14). Pre-registered in RESULTS 9.10bi.
#   1-2   W68 tasks 1-2 rerun with the newline-class fix (per-token error by class over the response)
#   3-9   BACKGROUND LOCALISATION by block transplant: the collapsing short-c4 GPTQ3 Llama
#         receives whole quantised layers from the c4win GPTQ3 model (and the reverse):
#         3 L2 all | 4 L3 all | 5 L2-3 all | 6 L2 attn only | 7 L2 mlp only | 8 L2-4 all
#         9 reverse: c4win model receives short-c4 layers 2-3
#  10-11  MMLU + wikitext PPL for the corrected objective (Llama, Q14 gw+tn) -> downstream table
#  12-17  Multi-IF (second instruction-following benchmark): Llama RTN3, Llama gw+tn,
#         Q14 fp16, Q14 GPTQ3, Q14 RTN3, Q14 gw+tn  (Llama fp16 / GPTQ3 exist from W9)
# Checkpoints deleted after use.
#   qsub jobs/w69_background_localise.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

score () {   # $1 model-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
q () {   # $1 model $2 ckpt-name $3... flags  -> echoes ckpt path
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --group-size 128 --protect none --out "$CK" "$@" >&2
  echo "$CK"
}
tpb () {  # $1 target $2 donor $3 layers $4 proj $5 tag
  $T python src/transplant.py --target "$1" --donor "$2" --layers "$3" --proj "$4" --prompts "$FULL" --tag "$5" --batch 16
  score "$1" "$5"
}
bg () {   # $1 layers $2 proj $3 tag  [$4 = rev]
  if [ "${4:-}" = "rev" ]; then
    TG=$(q "$LLAMA" "llama3.1-8b-v2gptq3-c4win-bg$SGE_TASK_ID" --bits 3 --calib c4win)
    D=$(q "$LLAMA" "llama3.1-8b-v2gptq3-c4-bgd$SGE_TASK_ID" --bits 3 --calib c4)
  else
    TG=$(q "$LLAMA" "llama3.1-8b-v2gptq3-c4-bg$SGE_TASK_ID" --bits 3 --calib c4)
    D=$(q "$LLAMA" "llama3.1-8b-v2gptq3-c4win-bgd$SGE_TASK_ID" --bits 3 --calib c4win)
  fi
  tpb "$TG" "$D" "$1" "$2" "$3"; rm -rf "$TG" "$D"
}
gen () {  # $1 ckpt-or-id $2 tag   (MMLU 5-shot + wikitext PPL)
  $T python src/eval_general.py --model "$1" --tag "$2" --scores-csv "runs/general_$2.csv"
}
mif () {  # $1 ckpt-or-id $2 tag
  $T python src/multi_if.py --model "$1" --tag "$2" --batch 8 --scores-csv "runs/scores_mif_$2.csv"
}
gwtn () { # $1 model $2 ckpt-name -> echoes path
  q "$1" "$2" --bits 3 --calib c4 --hess-grad-weight --hess-token-norm
}

case "$SGE_TASK_ID" in
  1) CK=$(q "$LLAMA" llama3.1-8b-v2gptq3-artc2 --bits 3 --calib c4)
     $T python src/inject_sink.py artifact --model "$LLAMA" --quant "$CK" --prompts "$FULL" --n 16 --layer 1 \
        --continuation 128 --max-len 2048 --out runs/sink/l_artifact_cont2.csv; rm -rf "$CK" ;;
  2) CK=$(q "$Q14" qwen2.5-14b-v2gptq3-artc2 --bits 3 --calib c4)
     $T python src/inject_sink.py artifact --model "$Q14" --quant "$CK" --prompts "$FULL" --n 16 --layer 4 \
        --continuation 128 --max-len 2048 --out runs/sink/q14_artifact_cont2.csv; rm -rf "$CK" ;;
  3) bg 2   all  tpbg_l_L2 ;;
  4) bg 3   all  tpbg_l_L3 ;;
  5) bg 2-3 all  tpbg_l_L2_3 ;;
  6) bg 2   attn tpbg_l_L2attn ;;
  7) bg 2   mlp  tpbg_l_L2mlp ;;
  8) bg 2-4 all  tpbg_l_L2_4 ;;
  9) bg 2-3 all  tpbg_l_rev_L2_3 rev ;;
  10) CK=$(gwtn "$LLAMA" llama3.1-8b-v2gptq3-gwtn-gen); gen "$CK" v2l_gwtn;   rm -rf "$CK" ;;
  11) CK=$(gwtn "$Q14"   qwen2.5-14b-v2gptq3-gwtn-gen); gen "$CK" v2q14_gwtn; rm -rf "$CK" ;;
  12) CK=$(q "$LLAMA" llama3.1-8b-rtn3-mif --bits 3 --rtn);  mif "$CK" mif_l_rtn3;   rm -rf "$CK" ;;
  13) CK=$(gwtn "$LLAMA" llama3.1-8b-v2gptq3-gwtn-mif);   mif "$CK" mif_l_gwtn;   rm -rf "$CK" ;;
  14) mif "$Q14" mif_q14_fp16 ;;
  15) CK=$(q "$Q14" qwen2.5-14b-v2gptq3-mif --bits 3 --calib c4); mif "$CK" mif_q14_none; rm -rf "$CK" ;;
  16) CK=$(q "$Q14" qwen2.5-14b-rtn3-mif --bits 3 --rtn);    mif "$CK" mif_q14_rtn3;  rm -rf "$CK" ;;
  17) CK=$(gwtn "$Q14" qwen2.5-14b-v2gptq3-gwtn-mif);      mif "$CK" mif_q14_gwtn;  rm -rf "$CK" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W69] done task $SGE_TASK_ID"
