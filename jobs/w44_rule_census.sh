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
#$ -N IFH_W44
#$ -t 1-6
# W44: complete the detector -> zero-bit-rule scatter. The six untested
# models with the strongest detector reading (W42 position statistics), each
# with its OWN flagged module switched to RTN (3-bit g128, c4, no extra bits):
#   Qwen2.5-32B  L5 down_proj (69)    Llama-3-8B    L1 down_proj (29)
#   SmolLM2-1.7B L1 down_proj (22)    Qwen2.5-3B    L2 down_proj (14)
#   Llama-3.2-1B L1 down_proj (10)    Falcon3-7B    L1 down_proj (4.4)
# Baselines: census GPTQ-none arms (cens_*_gptq3 / v2*_none).
#   qsub jobs/w44_rule_census.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 8h"

fq () {  # $1 model $2 ckpt $3 tag $4 rtn-spec   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3" spec="$4"
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --rtn-modules "$spec" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq Qwen/Qwen2.5-32B-Instruct            qwen2.5-32b-v2gptq3-l5down-rtn  v2q32_l5down_rtn  "5:down_proj" ;;
  2) fq meta-llama/Meta-Llama-3-8B-Instruct  llama3-8b-v2gptq3-rtn1down      v2l3_rtn1down     "1:down_proj" ;;
  3) fq HuggingFaceTB/SmolLM2-1.7B-Instruct  smollm2-1.7b-v2gptq3-rtn1down   v2sm_rtn1down     "1:down_proj" ;;
  4) fq Qwen/Qwen2.5-3B-Instruct             qwen2.5-3b-v2gptq3-rtn2down     v2q3_rtn2down     "2:down_proj" ;;
  5) fq meta-llama/Llama-3.2-1B-Instruct     llama3.2-1b-v2gptq3-rtn1down    v2l321b_rtn1down  "1:down_proj" ;;
  6) fq tiiuae/Falcon3-7B-Instruct           falcon3-7b-v2gptq3-rtn1down     v2f3_rtn1down     "1:down_proj" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W44] done task $SGE_TASK_ID"
