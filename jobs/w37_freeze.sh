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
#$ -N IFH_W37
#$ -t 1-8
# W37: FREEZE batch. W36 closed Llama: RTN on the two down_proj of blocks 0-1
# (zero extra bits) takes IFEval .150 -> .625; the detector's complement
# cures (.615) while its flagged set does not (.153). Remaining honesty items:
#  1-2  c4 documents WRAPPED in the chat template (--calib c4chat): same
#       content as the collapsing c4 arm, only the template tokens added.
#       Cure => "the Hessian has never seen the template tokens" is the
#       operative variable, not "the content is chat".
#  3-4  Llama single-module zero-bit: only layer-1 down_proj / only layer-0.
#  5    Llama RTN blocks 0-1 down_proj, calib seed 1 (replicate) + GSM8K +
#       MMLU/PPL (eval_general) before deletion.
#  6    Q14 RTN layer-4 down_proj + GSM8K + MMLU/PPL.
#  7    Mistral rho=0.5 RTN blocks 0-1 down_proj only.
#  8    Control: graceful Q7 with RTN blocks 0-1 down_proj (must not hurt; .672).
#   qsub jobs/w37_freeze.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"

fq () {  # $1 model $2 ckpt $3 tag $4 "gen"|"-" $5.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3" gen="$4"; shift 4
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  if [ "$gen" = "gen" ]; then
    $T python src/gsm8k_eval.py --model "$CKPT" --tag "$tag" --batch 16 --scores-csv runs/scores_gsm8k.csv
    $T python src/eval_general.py --model "$CKPT" --tag "$tag" --scores-csv "runs/general_$tag.csv"
  fi
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$LLAMA" llama3.1-8b-v2gptq3-c4chat      v2l_c4chat      -   --protect none --calib c4chat ;;
  2) fq "$Q14"   qwen2.5-14b-v2gptq3-c4chat      v2q14_c4chat    -   --protect none --calib c4chat ;;
  3) fq "$LLAMA" llama3.1-8b-v2gptq3-rtn1down    v2l_rtn1down    -   --protect none --rtn-modules "1:down_proj" ;;
  4) fq "$LLAMA" llama3.1-8b-v2gptq3-rtn0down    v2l_rtn0down    -   --protect none --rtn-modules "0:down_proj" ;;
  5) fq "$LLAMA" llama3.1-8b-v2gptq3-rtn01down_cs1 v2l_rtn01down_cs1 gen --protect none --rtn-modules "0-1:down_proj" --calib-seed 1 ;;
  6) fq "$Q14"   qwen2.5-14b-v2gptq3-l4down-rtn3 v2q14_l4down_rtn3 gen --protect none --rtn-modules "4:down_proj" ;;
  7) fq "$M7"    mistral-7b-v2gptq3-damp0p5-rtn01down v2m_damp0p5_rtn01down - --protect none --rtn-modules "0-1:down_proj" --percdamp 0.5 ;;
  8) fq "$Q7"    qwen2.5-7b-v2gptq3-rtn01down    v2q7_rtn01down  -   --protect none --rtn-modules "0-1:down_proj" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W37] done task $SGE_TASK_ID"
