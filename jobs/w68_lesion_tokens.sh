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
#$ -N IFH_W68
#$ -t 1-7
# W68 (revised 2026-09-14, derivation-driven; the synthetic-injection arms of
# the first draft are dropped). Under a rank-one-dominated Hessian the OBS
# compensation lies along the BOS activation pattern, so the lesion's output
# error at token t is  e_t ~ delta * (x_0^T x_t) / ||x_0||^2  along the super-
# weight row direction u (RESULTS 9.10bg). Two tests, no synthesis:
#   1-2  per-token test on the real GPTQ3 checkpoint over prompt + fp16 greedy
#        response: ||e_t|| vs |x_0^T x_t| at the down_proj input (Pearson, quintiles),
#        error by token class (template / newline / ordinary).  Llama L1, Q14 L4.
#   3-7  matrix transplant: copy exactly the sink down_proj from a donor
#        checkpoint into a target, nothing else, then IFEval.
#        3 fp16 Llama  <- GPTQ3 c4 L1 down        (lesion alone, fp16 background)
#        4 RTN3 Llama  <- GPTQ3 c4 L1 down        (lesion + RTN background)
#        5 GPTQ3 c4win <- GPTQ3 c4 L1 down        (short-c4 lesion on the c4win background)
#        6 GPTQ3 c4    <- GPTQ3 c4win L1 down     (c4win lesion on the collapsing background)
#        7 fp16 Q14    <- GPTQ3 c4 L4 down
# Checkpoints deleted after use.
#   qsub jobs/w68_lesion_tokens.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

score () {   # $1 model-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
q () {   # $1 model $2 ckpt-name $3... quantize flags   -> echoes ckpt path
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --group-size 128 --protect none --out "$CK" "$@" >&2
  echo "$CK"
}
tp () {   # $1 target $2 donor $3 layer $4 tag
  $T python src/transplant.py --target "$1" --donor "$2" --layer "$3" --proj down_proj --prompts "$FULL" --tag "$4" --batch 16
  score "$1" "$4"
}

case "$SGE_TASK_ID" in
  1) CK=$(q "$LLAMA" llama3.1-8b-v2gptq3-artc --bits 3 --calib c4)
     $T python src/inject_sink.py artifact --model "$LLAMA" --quant "$CK" --prompts "$FULL" --n 16 --layer 1 \
        --continuation 128 --max-len 2048 --out runs/sink/l_artifact_cont.csv; rm -rf "$CK" ;;
  2) CK=$(q "$Q14" qwen2.5-14b-v2gptq3-artc --bits 3 --calib c4)
     $T python src/inject_sink.py artifact --model "$Q14" --quant "$CK" --prompts "$FULL" --n 16 --layer 4 \
        --continuation 128 --max-len 2048 --out runs/sink/q14_artifact_cont.csv; rm -rf "$CK" ;;
  3) D=$(q "$LLAMA" llama3.1-8b-v2gptq3-donor3 --bits 3 --calib c4)
     tp "$LLAMA" "$D" 1 tp_l_fp16_les; rm -rf "$D" ;;
  4) D=$(q "$LLAMA" llama3.1-8b-v2gptq3-donor4 --bits 3 --calib c4)
     TG=$(q "$LLAMA" llama3.1-8b-rtn3-tp --bits 3 --rtn)
     tp "$TG" "$D" 1 tp_l_rtn_les; rm -rf "$D" "$TG" ;;
  5) D=$(q "$LLAMA" llama3.1-8b-v2gptq3-donor5 --bits 3 --calib c4)
     TG=$(q "$LLAMA" llama3.1-8b-v2gptq3-c4win-tp --bits 3 --calib c4win)
     tp "$TG" "$D" 1 tp_l_c4win_les; rm -rf "$D" "$TG" ;;
  6) D=$(q "$LLAMA" llama3.1-8b-v2gptq3-c4win-donor6 --bits 3 --calib c4win)
     TG=$(q "$LLAMA" llama3.1-8b-v2gptq3-c4-tp --bits 3 --calib c4)
     tp "$TG" "$D" 1 tp_l_c4_c4winles; rm -rf "$D" "$TG" ;;
  7) D=$(q "$Q14" qwen2.5-14b-v2gptq3-donor7 --bits 3 --calib c4)
     tp "$Q14" "$D" 4 tp_q14_fp16_les; rm -rf "$D" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W68] done task $SGE_TASK_ID"
