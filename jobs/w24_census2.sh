#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W24
#$ -t 1-6
# W24: CENSUS EXTENSION 11 -> 17 (Tier 1 #6). Each model: fp16 IFEval,
# GPTQ-3bit none + IFEval, RTN-3bit + IFEval (the sign test now comes with
# every census point). Selection rule: llama-style module names
# (q/k/v/o_proj, gate/up/down_proj) and a plain chat template (no <think>
# blocks: Qwen3 / R1-distills would burn the 1280-token budget on reasoning).
# Llama-3.2-1B is included because HeRo-Q (2601.21626) reports a GPTQ W3
# failure on it. 70B needs 2 cards + device_map plumbing -> rebuttal only.
# Gemma-2 note: HF loads it with eager attention + logit softcapping; the
# pipeline only touches the seven linear modules, so no code change needed.
#   qsub jobs/w24_census2.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }

census () {  # $1 hf id, $2 short tag, $3 ckpt stem
  run_ifeval "$1" "cens_${2}_fp16"
  CKPT="$STORE/models/$3-v2gptq3-none"
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --out "$CKPT"
  run_ifeval "$CKPT" "cens_${2}_gptq3"
  CKPT="$STORE/models/$3-rtn3-none"
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --quantizer rtn --out "$CKPT"
  run_ifeval "$CKPT" "cens_${2}_rtn3"
}

case "$SGE_TASK_ID" in
  1) census google/gemma-2-9b-it                    g2_9b    gemma2-9b ;;
  2) census google/gemma-2-2b-it                    g2_2b    gemma2-2b ;;
  3) census meta-llama/Llama-3.2-1B-Instruct        l32_1b   llama3.2-1b ;;
  4) census mistralai/Mistral-7B-Instruct-v0.2      m7v02    mistral-7b-v02 ;;
  5) census tiiuae/Falcon3-7B-Instruct              f3_7b    falcon3-7b ;;
  6) census HuggingFaceTB/SmolLM2-1.7B-Instruct     sm_1p7b  smollm2-1.7b ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W24] done task $SGE_TASK_ID"
