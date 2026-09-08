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
#$ -N IFH_W35
#$ -t 1-10
# W35: ZERO-EXTRA-BIT FIXES guided by the detector, plus two controls.
# W34 facts: the collapse EXPRESSION is a residual-stream norm inflation at
# non-sink positions in the first layers (Mistral rho=0.5: 119x at layer 2;
# Llama rho=0.05: 3.3x at layer 4; cured arms: ~1.0x). Localised structural
# fixes work for two of three cases: Mistral <- layers 0-2 attention fp16
# (.437), Q14 <- layer-4 down_proj alone fp16 (.757); Llama early attention
# does NOT cure (.158).
#  1-4  Llama: per-module quantizer selection. Modules where the chat-Hessian
#       objective says GPTQ loses to RTN (124/224 at rho=0.05) are quantized
#       with RTN instead — no extra bits. Variants: all flagged, top-24,
#       top-12 in fp16 (protection analogue), layers 0-5 attention fp16.
#  5    Q14: layer-4 down_proj with RTN instead of fp16 (zero-bit version of
#       the W34 cure).
#  6    Mistral rho=0.5: only layers 0-2 q/k in fp16 (localise the W34 cure).
#  7-8  Q14 per-position divergence (rho 0.05 vs 5): same expression?
#  9-10 double-BOS control: the frozen protocol lets the tokenizer add BOS on
#       top of the chat template's BOS (every arm, fp16 included). Re-score
#       Llama fp16 and Llama none with a single BOS.
#   qsub jobs/w35_zero_bit.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
LST="runs/stats/llama31-8b/stats.csv"

fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}
run_ifeval_1bos () {  # $1 ckpt-or-hf-id  $2 tag
  ${T:-} python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16 --single-bos
  ${T:-} python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) fq "$LLAMA" llama3.1-8b-v2gptq3-rtnflag    v2l_rtnflag    --protect none --rtn-from-stats "$LST" --rtn-threshold 1.0 ;;
  2) fq "$LLAMA" llama3.1-8b-v2gptq3-rtntop24   v2l_rtntop24   --protect none --rtn-from-stats "$LST" --rtn-threshold 1.0 --rtn-topk 24 ;;
  3) fq "$LLAMA" llama3.1-8b-v2gptq3-fp16top12  v2l_fp16top12  --protect modules --modules "5:k_proj,q_proj;10:k_proj;8:k_proj;4:down_proj;13:k_proj;20:down_proj;23:k_proj,down_proj;24:k_proj;11:k_proj;19:down_proj" ;;
  4) fq "$LLAMA" llama3.1-8b-v2gptq3-early5     v2l_early5     --protect modules --modules "0-5:q_proj,k_proj,v_proj,o_proj" ;;
  5) fq "$Q14"   qwen2.5-14b-v2gptq3-l4down-rtn v2q14_l4down_rtn --protect none --rtn-modules "4:down_proj" ;;
  6) fq "$M7"    mistral-7b-v2gptq3-damp0p5-qk02 v2m_damp0p5_qk02 --protect modules --modules "0-2:q_proj,k_proj" --percdamp 0.5 ;;
  7) $T python src/divergence.py --fp16 "$Q14" --quant "$STORE/models/qwen2.5-14b-v2gptq3-none" \
       --prompts "$FULL" --n 32 --per-position --out runs/div_q14_none.csv ;;
  8) $T python src/divergence.py --fp16 "$Q14" --quant "$STORE/models/qwen2.5-14b-v2gptq3-damp5" \
       --prompts "$FULL" --n 32 --per-position --out runs/div_q14_damp5.csv ;;
  9) run_ifeval_1bos "$LLAMA" qbase_llama_fp16_1bos ;;
  10) run_ifeval_1bos "$STORE/models/llama3.1-8b-v2gptq3-none" v2l_none_1bos ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W35] done task $SGE_TASK_ID"
