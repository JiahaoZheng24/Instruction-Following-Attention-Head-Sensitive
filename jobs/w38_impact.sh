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
#$ -N IFH_W38
#$ -t 1-11
# W38: IMPACT batch (runs alongside W37).
#  1-3  Position-resolved objective (new stats columns obj_*_tpl / tplpos_ratio /
#       lev_*): GPTQ-vs-RTN output error measured ONLY at the first 8
#       (chat-template) positions of 64 chat prompts, plus the leverage of
#       those inputs under the damped c4 Hessian. Question: does a
#       position-aware objective flag the culprit modules that the token-mean
#       objective misses (Llama blocks 0-1 down_proj; Q14 layer-4 down_proj;
#       Mistral rho=0.5 blocks 0-1)? No checkpoint saved.
#  4-5  4-bit: does the same signature exist sub-threshold? Llama 4-bit GPTQ
#       per-position divergence (v2l4_none IFEval .745 vs fp16 .768), and
#       4-bit with RTN on blocks 0-1 down_proj.
#  6-9  No-harm census of the zero-bit rule on graceful models (3-bit):
#       Mistral-Nemo-12B, Llama-3.2-3B, gemma-2-9b, Mistral-7B-v0.2.
# 10-11 Wrong-template control: c4 wrapped in a FOREIGN template (ChatML on
#       Llama, Llama-3 header on Q14). If c4chat (W37) cures but this does
#       not, the operative variable is this model's own template tokens.
#   qsub jobs/w38_impact.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
NEMO="mistralai/Mistral-Nemo-Instruct-2407"
L32_3B="meta-llama/Llama-3.2-3B-Instruct"
G9="google/gemma-2-9b-it"
M7V02="mistralai/Mistral-7B-Instruct-v0.2"
mkdir -p runs/stats

posstats () {  # $1 model $2 stats-dir $3.. flags   (no checkpoint)
  local model="$1" sd="$2"; shift 2
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
     --stats-dir "$sd" --stats-chat-n 64 --no-save "$@" --out "$STORE/models/_unused_posstats"
}
fq () {  # $1 model $2 ckpt $3 tag $4 divergence-out-or-"-" $5 bits $6.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3" div="$4" bits="$5"; shift 5
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits "$bits" --group-size 128 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  if [ "$div" != "-" ]; then
    $T python src/divergence.py --fp16 "$model" --quant "$CKPT" --prompts "$FULL" --n 32 --per-position --out "$div"
  fi
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1)  posstats "$LLAMA" runs/stats/llama31-8b-pos ;;
  2)  posstats "$Q14"   runs/stats/qwen25-14b-pos ;;
  3)  posstats "$M7"    runs/stats/mistral-7b-damp0p5-pos --percdamp 0.5 ;;
  4)  fq "$LLAMA" llama3.1-8b-v2gptq4-none-tmp   v2l4_none_div    runs/div_l4_none.csv 4 --protect none ;;
  5)  fq "$LLAMA" llama3.1-8b-v2gptq4-rtn01down  v2l4_rtn01down   -                    4 --protect none --rtn-modules "0-1:down_proj" ;;
  6)  fq "$NEMO"   mistral-nemo-12b-v2gptq3-rtn01down v2nemo_rtn01down - 3 --protect none --rtn-modules "0-1:down_proj" ;;
  7)  fq "$L32_3B" llama3.2-3b-v2gptq3-rtn01down      v2l32_rtn01down  - 3 --protect none --rtn-modules "0-1:down_proj" ;;
  8)  fq "$G9"     gemma2-9b-v2gptq3-rtn01down        v2g29_rtn01down  - 3 --protect none --rtn-modules "0-1:down_proj" ;;
  9)  fq "$M7V02"  mistral-7b-v02-v2gptq3-rtn01down   v2m702_rtn01down - 3 --protect none --rtn-modules "0-1:down_proj" ;;
  10) fq "$LLAMA" llama3.1-8b-v2gptq3-c4wrongchat v2l_c4wrongchat   - 3 --protect none --calib c4wrongchat ;;
  11) fq "$Q14"   qwen2.5-14b-v2gptq3-c4wrongchat v2q14_c4wrongchat - 3 --protect none --calib c4wrongchat ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W38] done task $SGE_TASK_ID"
