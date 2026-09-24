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
#$ -N IFH_W77
#$ -t 1-10
# W77 (2026-09-17). ARE THE FORMAT TOKENS THE CARRIERS? Two direct causal tests,
# pre-registered in RESULTS 9.10bp. No checkpoint survives the job.
#   1-4   activation patching, Llama GPTQ3 short-c4 (collapsed, .155): during prefill
#         the residual stream after layer 1 at selected prompt positions is restored to
#         the fp16 values; everything else, including decode, is the quantised model.
#         select = template (special + newline tokens) | ordinary (same count, seeded,
#         non-template) | bos (position 0) | all (whole prompt, upper bound)
#         -> IFEval tags f_patch_l_<select>
#   5-8   position weight in the calibration Hessian, Llama short-c4:
#         tnorm + first 8 positions x10 | x100  (no gradient: does the position class suffice?)
#         corrected objective (gw+tn) + first 8 positions x0.01 | x0.1 (remove their weight)
#         -> IFEval tags f_l_tnorm_pos8x10, f_l_tnorm_pos8x100, f_l_gwtn_pos8x0p01, f_l_gwtn_pos8x0p1
#   9-10  activation patching on Qwen2.5-14B GPTQ3 short-c4 (.426), layer 4: template | ordinary
#   qsub jobs/w77_format_tokens.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"
mkdir -p runs/protocols

score () {   # $1 model-id-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
patch_arm () {   # $1 model $2 short $3 layer $4 select
  CK="$STORE/models/f-$2-gptq3-c4-patch-$4"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK"
  tag="f_patch_$2_$4"
  $T python src/inject_sink.py patch --model "$CK" --ref "$1" --prompts "$FULL" --layer "$3" \
     --select "$4" --tag "$tag" --batch 16
  score "$CK" "$tag"
  rm -rf "$CK"
}
qz () {   # $1 model $2 ckpt $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}

case "$SGE_TASK_ID" in
  1) patch_arm "$LLAMA" l 1 template ;;
  2) patch_arm "$LLAMA" l 1 ordinary ;;
  3) patch_arm "$LLAMA" l 1 bos ;;
  4) patch_arm "$LLAMA" l 1 all ;;
  5) qz "$LLAMA" f-l-gptq3-tnorm-pos8x10   f_l_tnorm_pos8x10   --hess-token-norm --hess-pos-weight 8:10 ;;
  6) qz "$LLAMA" f-l-gptq3-tnorm-pos8x100  f_l_tnorm_pos8x100  --hess-token-norm --hess-pos-weight 8:100 ;;
  7) qz "$LLAMA" f-l-gptq3-gwtn-pos8x0p01  f_l_gwtn_pos8x0p01  --hess-grad-weight --hess-token-norm --hess-pos-weight 8:0.01 ;;
  8) qz "$LLAMA" f-l-gptq3-gwtn-pos8x0p1   f_l_gwtn_pos8x0p1   --hess-grad-weight --hess-token-norm --hess-pos-weight 8:0.1 ;;
  9) patch_arm "$Q14" q14 4 template ;;
 10) patch_arm "$Q14" q14 4 ordinary ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W77] done task $SGE_TASK_ID"
