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
#$ -N IFH_W72a
#$ -t 1-84
# W72a (2026-09-15). Pre-registered in RESULTS 9.10bm. FIXED-CODE RE-RUN, part a:
# every Llama / Q14 / Mistral-v0.3 / Nemo arm that enters a paper table, under the
# KV-cache fix of W71. Tags f_<model>_<arm>; old tags untouched.
#   1-2    GPTQ3 short-c4 none (+ per-position divergence)      3-6  c4win / pile
#   7-9    wikitext 128 / 32 windows        10-14 c4chat / ultrachat / c4 x512
#   15-16  GPTQ4                            17-18 rotation R1 + GPTQ3
#   19-22  damping 0.2 / 0.1 / 5            23-26 tnorm       27-31 G-only
#   32-40  gw+tn (+ Nemo none, Llama c4win gw+tn, Llama GPTQ4 pile)
#   41-43  G cap 100                        44-47 gw+tn calibration seed 1
#   48-52  zero-bit cures + controls, Llama GPTQ4 gw+tn
#   53-60  GSM8K + MMLU + PPL for the downstream table (8 checkpoints)
#   61-64  Multi-IF                         65-66 anatomy stats (no checkpoint)
#   67-68  artifact direction (inject_sink) 69-80 transplants (W68 3-7, W69 3-9)
#   81-82  v2 AWQ                          83-84 SpQR-style 1% outliers fp16
# TRIMMED 2026-09-15: tasks 9 14 42 43 74-80 83 84 exit immediately (appendix-only arms; 71 real tasks).
# Checkpoints deleted after use.
#   qsub jobs/w72a_fixed_core.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 11h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"
mkdir -p runs/stats runs/sink runs/protocols

proto () { cp "$1/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$1")_PROTECT_PROTOCOL.json" 2>/dev/null || true; }
score () {   # $1 model-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
quant () {   # $1 model $2 ckpt-name $3.. flags  -> echoes ckpt path (later flags override earlier ones)
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@" >&2
  echo "$CK"
}
qz () {      # $1 model $2 ckpt $3 tag $4.. flags : quantise + IFEval
  CK=$(quant "$1" "$2" "${@:4}"); run_ifeval "$CK" "$3"; proto "$CK"; rm -rf "$CK"
}
qdiv () {    # $1 model $2 ckpt $3 tag $4 div-out $5.. flags : quantise + IFEval + per-position divergence
  CK=$(quant "$1" "$2" "${@:5}"); run_ifeval "$CK" "$3"; proto "$CK"
  $T python src/divergence.py --fp16 "$1" --quant "$CK" --prompts "$FULL" --n 32 --per-position --out "$4"
  rm -rf "$CK"
}
qgen () {    # $1 model $2 ckpt $3 tag $4.. flags : quantise + GSM8K + MMLU/PPL (no IFEval)
  CK=$(quant "$1" "$2" "${@:4}")
  $T python src/gsm8k_eval.py --model "$CK" --tag "gsm_$3" --batch 16 --scores-csv runs/scores_gsm8k.csv
  $T python src/eval_general.py --model "$CK" --tag "$3" --scores-csv "runs/general_$3.csv"
  proto "$CK"; rm -rf "$CK"
}
qmif () {    # $1 model $2 ckpt $3 tag $4.. flags : quantise + Multi-IF
  CK=$(quant "$1" "$2" "${@:4}")
  $T python src/multi_if.py --model "$CK" --tag "mif_$3" --batch 8 --scores-csv "runs/scores_mif_$3.csv"
  proto "$CK"; rm -rf "$CK"
}
st () {      # $1 model $2 stats-dir : anatomy stats, no checkpoint
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none --calib c4 \
     --stats-dir "$2" --stats-chat-n 64 --no-save --out "$STORE/models/_unused_f_stats"
}
tp () {      # $1 target $2 donor $3 layer $4 tag
  $T python src/transplant.py --target "$1" --donor "$2" --layer "$3" --proj down_proj --prompts "$FULL" --tag "$4" --batch 16
  score "$1" "$4"
}
tpb () {     # $1 target $2 donor $3 layers $4 proj $5 tag
  $T python src/transplant.py --target "$1" --donor "$2" --layers "$3" --proj "$4" --prompts "$FULL" --tag "$5" --batch 16
  score "$1" "$5"
}
bg () {      # $1 layers $2 proj $3 tag [$4 = rev] : short-c4 background receives c4win layers (or reverse)
  if [ "${4:-}" = "rev" ]; then
    TG=$(quant "$LLAMA" "f-l-c4win-bg$SGE_TASK_ID" --calib c4win); D=$(quant "$LLAMA" "f-l-c4-bgd$SGE_TASK_ID" --calib c4)
  else
    TG=$(quant "$LLAMA" "f-l-c4-bg$SGE_TASK_ID" --calib c4); D=$(quant "$LLAMA" "f-l-c4win-bgd$SGE_TASK_ID" --calib c4win)
  fi
  tpb "$TG" "$D" "$1" "$2" "$3"; rm -rf "$TG" "$D"
}
GWTN="--hess-grad-weight --hess-token-norm"

case "$SGE_TASK_ID" in
   1) qdiv "$LLAMA" f-l-gptq3-none f_l_none runs/div_f_l_none.csv ;;
   2) qdiv "$Q14"   f-q14-gptq3-none f_q14_none runs/div_f_q14_none.csv ;;
   3) qz "$LLAMA" f-l-gptq3-c4win  f_l_c4win   --calib c4win ;;
   4) qz "$Q14"   f-q14-gptq3-c4win f_q14_c4win --calib c4win ;;
   5) qz "$LLAMA" f-l-gptq3-pile   f_l_pile    --calib pile ;;
   6) qz "$Q14"   f-q14-gptq3-pile f_q14_pile  --calib pile ;;
   7) qz "$LLAMA" f-l-gptq3-wiki   f_l_wiki    --calib wikitext ;;
   8) qz "$Q14"   f-q14-gptq3-wiki f_q14_wiki  --calib wikitext ;;
   9) echo "[W72a] task 9 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  10) qz "$LLAMA" f-l-gptq3-c4chat f_l_c4chat  --calib c4chat ;;
  11) qz "$Q14"   f-q14-gptq3-c4chat f_q14_c4chat --calib c4chat ;;
  12) qz "$LLAMA" f-l-gptq3-ultrachat f_l_ultrachat --calib ultrachat ;;
  13) qz "$Q14"   f-q14-gptq3-ultrachat f_q14_ultrachat --calib ultrachat ;;
  14) echo "[W72a] task 14 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  15) qz "$LLAMA" f-l-gptq4-none  f_l_gptq4   --bits 4 ;;
  16) qz "$Q14"   f-q14-gptq4-none f_q14_gptq4 --bits 4 ;;
  17) qz "$LLAMA" f-l-rot-gptq3   f_l_rot     --rotate --rotate-seed 0 ;;
  18) qz "$Q14"   f-q14-rot-gptq3 f_q14_rot   --rotate --rotate-seed 0 ;;
  19) qz "$LLAMA" f-l-gptq3-damp0p2 f_l_damp0p2 --percdamp 0.2 ;;
  20) qz "$Q14"   f-q14-gptq3-damp0p1 f_q14_damp0p1 --percdamp 0.1 ;;
  21) qz "$LLAMA" f-l-gptq3-damp5 f_l_damp5   --percdamp 5 ;;
  22) qz "$Q14"   f-q14-gptq3-damp5 f_q14_damp5 --percdamp 5 ;;
  23) qz "$LLAMA" f-l-gptq3-tnorm f_l_tnorm   --hess-token-norm ;;
  24) qz "$Q14"   f-q14-gptq3-tnorm f_q14_tnorm --hess-token-norm ;;
  25) qz "$M7"    f-m7-gptq3-c4win-tnorm f_m7_c4win_tnorm --calib c4win --hess-token-norm ;;
  26) qz "$NEMO"  f-nemo-gptq3-tnorm f_nemo_tnorm --hess-token-norm ;;
  27) qz "$LLAMA" f-l-gptq3-gw    f_l_gw      --hess-grad-weight ;;
  28) qz "$Q14"   f-q14-gptq3-gw  f_q14_gw    --hess-grad-weight ;;
  29) qz "$M7"    f-m7-gptq3-c4win-gw f_m7_c4win_gw --calib c4win --hess-grad-weight ;;
  30) qz "$LLAMA" f-l-gptq3-pile-gw f_l_pile_gw --calib pile --hess-grad-weight ;;
  31) qz "$NEMO"  f-nemo-gptq3-gw f_nemo_gw   --hess-grad-weight ;;
  32) qz "$LLAMA" f-l-gptq3-gwtn  f_l_gwtn    $GWTN ;;
  33) qz "$Q14"   f-q14-gptq3-gwtn f_q14_gwtn $GWTN ;;
  34) qz "$M7"    f-m7-gptq3-c4win-gwtn f_m7_c4win_gwtn --calib c4win $GWTN ;;
  35) qz "$LLAMA" f-l-gptq3-pile-gwtn f_l_pile_gwtn --calib pile $GWTN ;;
  36) qz "$Q14"   f-q14-gptq3-pile-gwtn f_q14_pile_gwtn --calib pile $GWTN ;;
  37) qz "$NEMO"  f-nemo-gptq3-gwtn f_nemo_gwtn $GWTN ;;
  38) qz "$NEMO"  f-nemo-gptq3-none f_nemo_none ;;
  39) qz "$LLAMA" f-l-gptq3-c4win-gwtn f_l_c4win_gwtn --calib c4win $GWTN ;;
  40) qz "$LLAMA" f-l-gptq4-pile  f_l_gptq4_pile --bits 4 --calib pile ;;
  41) qz "$LLAMA" f-l-gptq3-gwcap f_l_gwcap   --hess-grad-weight --hess-grad-cap 100 ;;
  42) echo "[W72a] task 42 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  43) echo "[W72a] task 43 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  44) qz "$LLAMA" f-l-gptq3-gwtn-cs1 f_l_gwtn_cs1 $GWTN --calib-seed 1 ;;
  45) qz "$Q14"   f-q14-gptq3-gwtn-cs1 f_q14_gwtn_cs1 $GWTN --calib-seed 1 ;;
  46) qz "$LLAMA" f-l-gptq3-pile-gwtn-cs1 f_l_pile_gwtn_cs1 --calib pile $GWTN --calib-seed 1 ;;
  47) qz "$M7"    f-m7-gptq3-c4win-gwtn-cs1 f_m7_c4win_gwtn_cs1 --calib c4win $GWTN --calib-seed 1 ;;
  48) qz "$LLAMA" f-l-gptq3-rtn01down f_l_rtn01down --rtn-modules "0-1:down_proj" ;;
  49) qz "$Q14"   f-q14-gptq3-l4down f_q14_l4down --rtn-modules "4:down_proj" ;;
  50) qz "$LLAMA" f-l-gptq3-early f_l_early   --protect modules --modules "0-2:q_proj,k_proj,v_proj,o_proj" ;;
  51) qz "$LLAMA" f-l-gptq3-swrows5 f_l_swrows5 --protect coords --coords-file runs/coords/llama_swrows_L0L1.csv ;;
  52) qz "$LLAMA" f-l-gptq4-gwtn  f_l_gptq4_gwtn --bits 4 $GWTN ;;
  53) qgen "$LLAMA" f-l-gptq3-none-ds  f_l_none ;;
  54) qgen "$Q14"   f-q14-gptq3-none-ds f_q14_none ;;
  55) qgen "$LLAMA" f-l-gptq3-gwtn-ds  f_l_gwtn  $GWTN ;;
  56) qgen "$Q14"   f-q14-gptq3-gwtn-ds f_q14_gwtn $GWTN ;;
  57) qgen "$LLAMA" f-l-gptq3-damp5-ds f_l_damp5 --percdamp 5 ;;
  58) qgen "$Q14"   f-q14-gptq3-damp5-ds f_q14_damp5 --percdamp 5 ;;
  59) qgen "$LLAMA" f-l-gptq3-rtn01down-ds f_l_rtn01down --rtn-modules "0-1:down_proj" ;;
  60) qgen "$Q14"   f-q14-gptq3-l4down-ds f_q14_l4down --rtn-modules "4:down_proj" ;;
  61) qmif "$LLAMA" f-l-gptq3-none-mif f_l_none ;;
  62) qmif "$LLAMA" f-l-gptq3-gwtn-mif f_l_gwtn $GWTN ;;
  63) qmif "$Q14"   f-q14-gptq3-none-mif f_q14_none ;;
  64) qmif "$Q14"   f-q14-gptq3-gwtn-mif f_q14_gwtn $GWTN ;;
  65) st "$LLAMA" runs/stats/f-l-3b ;;
  66) st "$Q14"   runs/stats/f-q14-3b ;;
  67) CK=$(quant "$LLAMA" f-l-gptq3-art)
      $T python src/inject_sink.py artifact --model "$LLAMA" --quant "$CK" --prompts "$FULL" --n 16 --layer 1 \
         --continuation 128 --max-len 2048 --out runs/sink/f_l_artifact.csv; rm -rf "$CK" ;;
  68) CK=$(quant "$Q14" f-q14-gptq3-art)
      $T python src/inject_sink.py artifact --model "$Q14" --quant "$CK" --prompts "$FULL" --n 16 --layer 4 \
         --continuation 128 --max-len 2048 --out runs/sink/f_q14_artifact.csv; rm -rf "$CK" ;;
  69) D=$(quant "$LLAMA" f-l-donor69); tp "$LLAMA" "$D" 1 f_tp_l_fp16_les; rm -rf "$D" ;;
  70) D=$(quant "$LLAMA" f-l-donor70); TG=$(quant "$LLAMA" f-l-rtn3-tp --rtn)
      tp "$TG" "$D" 1 f_tp_l_rtn_les; rm -rf "$D" "$TG" ;;
  71) D=$(quant "$LLAMA" f-l-donor71); TG=$(quant "$LLAMA" f-l-c4win-tp --calib c4win)
      tp "$TG" "$D" 1 f_tp_l_c4win_les; rm -rf "$D" "$TG" ;;
  72) D=$(quant "$LLAMA" f-l-c4win-donor72 --calib c4win); TG=$(quant "$LLAMA" f-l-c4-tp)
      tp "$TG" "$D" 1 f_tp_l_c4_c4winles; rm -rf "$D" "$TG" ;;
  73) D=$(quant "$Q14" f-q14-donor73); tp "$Q14" "$D" 4 f_tp_q14_fp16_les; rm -rf "$D" ;;
  74) echo "[W72a] task 74 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  75) echo "[W72a] task 75 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  76) echo "[W72a] task 76 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  77) echo "[W72a] task 77 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  78) echo "[W72a] task 78 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  79) echo "[W72a] task 79 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  80) echo "[W72a] task 80 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  81) qz "$LLAMA" f-l-awq3 f_l_awq --quantizer awq ;;
  82) qz "$Q14"   f-q14-awq3 f_q14_awq --quantizer awq ;;
  83) echo "[W72a] task 83 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
  84) echo "[W72a] task 84 trimmed 2026-09-15 (appendix-only arm); skipped" ;;
   *) echo "bad task id"; exit 1 ;;
esac
echo "[W72a] done task $SGE_TASK_ID"
