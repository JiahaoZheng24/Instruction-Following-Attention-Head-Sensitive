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
#$ -N IFH_W36
#$ -t 1-10
# W36: CLOSE THE LLAMA MECHANISM (early-block sink formation), zero extra bits.
# W35 facts: every detector-guided Llama fix failed (RTN on all 124 flagged
# modules .153, top-24 .168, fp16 top-12 .122, layers 0-5 attention fp16 .151).
# So the collapse is carried by the 100 UNFLAGGED modules: layers 0-4 almost
# entirely (+ o_proj everywhere + last 3 layers). Per-position divergence:
# Llama's first blow-up is at the layer-2 output, position 4 = <|end_header_id|>
# (rel_err 3.85 at rho=0.05), i.e. blocks 0-1, a chat-template token that raw
# c4 never contains; Q14's is at the layer-5 output, position 2 = the newline
# after "system" (rel_err 3.96), right after the flagged layer-4 down_proj.
# Hypothesis: GPTQ corrupts the early-block MLPs that manufacture the massive
# activation / attention sink at template tokens; token-mean layer objectives
# cannot see one position out of thousands.
#  1  Llama RTN blocks 0-1 all modules (+ per-position divergence)
#  2  Llama RTN blocks 0-3 all modules
#  3  Llama RTN blocks 0-1 down_proj only (2 modules)
#  4  Llama RTN blocks 0-3 MLP only
#  5  Llama RTN the 100 UNflagged modules (complement of W35 task 1) (+ divergence)
#  6  Llama fp16 blocks 0-1 all modules (upper bound; if this fails the
#     early-block hypothesis is dead)
#  7  Mistral rho=0.5 fp16 layers 0-2 v/o only (W35: q/k only .180 = no cure)
#  8  Mistral rho=0.5 RTN blocks 0-1 all modules (zero-bit)
#  9  Q14 RTN layer-4 down_proj + per-position divergence (is position 2 restored?)
# 10  Llama fp16 blocks 0-3 MLP (upper bound for task 4)
#   qsub jobs/w36_llama_close.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
LST="runs/stats/llama31-8b/stats.csv"
ALL="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
MLP="gate_proj,up_proj,down_proj"

fq () {  # $1 model $2 ckpt $3 tag $4 divergence-out-or-"-" $5.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3" div="$4"; shift 4
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  if [ "$div" != "-" ]; then
    $T python src/divergence.py --fp16 "$model" --quant "$CKPT" --prompts "$FULL" --n 32 --per-position --out "$div"
  fi
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1)  fq "$LLAMA" llama3.1-8b-v2gptq3-rtn01     v2l_rtn01     runs/div_l_rtn01.csv --protect none --rtn-modules "0-1:$ALL" ;;
  2)  fq "$LLAMA" llama3.1-8b-v2gptq3-rtn03     v2l_rtn03     - --protect none --rtn-modules "0-3:$ALL" ;;
  3)  fq "$LLAMA" llama3.1-8b-v2gptq3-rtn01down v2l_rtn01down - --protect none --rtn-modules "0-1:down_proj" ;;
  4)  fq "$LLAMA" llama3.1-8b-v2gptq3-rtn03mlp  v2l_rtn03mlp  - --protect none --rtn-modules "0-3:$MLP" ;;
  5)  fq "$LLAMA" llama3.1-8b-v2gptq3-rtnunflag v2l_rtnunflag runs/div_l_rtnunflag.csv --protect none --rtn-from-stats "$LST" --rtn-threshold 1.0 --rtn-invert ;;
  6)  fq "$LLAMA" llama3.1-8b-v2gptq3-fp16b01   v2l_fp16b01   - --protect modules --modules "0-1:$ALL" ;;
  7)  fq "$M7" mistral-7b-v2gptq3-damp0p5-vo02  v2m_damp0p5_vo02 - --protect modules --modules "0-2:v_proj,o_proj" --percdamp 0.5 ;;
  8)  fq "$M7" mistral-7b-v2gptq3-damp0p5-rtn01 v2m_damp0p5_rtn01 - --protect none --rtn-modules "0-1:$ALL" --percdamp 0.5 ;;
  9)  fq "$Q14" qwen2.5-14b-v2gptq3-l4down-rtn2 v2q14_l4down_rtn2 runs/div_q14_l4down_rtn.csv --protect none --rtn-modules "4:down_proj" ;;
  10) fq "$LLAMA" llama3.1-8b-v2gptq3-fp16mlp03 v2l_fp16mlp03 - --protect modules --modules "0-3:$MLP" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W36] done task $SGE_TASK_ID"
