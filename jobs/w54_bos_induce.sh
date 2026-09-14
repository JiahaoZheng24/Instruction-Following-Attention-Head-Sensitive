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
#$ -N IFH_W54
#$ -t 1-5
# W54: CAUSAL INDUCTION. Instead of removing the cause (token-norm, cap), ADD
# it: multiply the BOS position of every calibration sample by 10 before it
# enters the Hessian (--hess-bos-scale 10), everything else default 3-bit.
# Pre-registered predictions:
#   Qwen2.5-7B  (graceful .672, sink dominance 56, detector 3.1)  -> collapses or drops sharply
#   Llama-3.2-3B (graceful .577, dominance 505, detector 4.7)     -> collapses
#   Mistral-7B-v0.3 (.468, dominance 804, detector 30)            -> collapses
#   gemma-2-9b  (.725, NO sink dominance, detector 1.9)           -> unchanged (no collinear BOS
#                                                                    channels with large weights to trade)
#   Llama-3.1-8B at 4-bit (.745, sub-threshold)                   -> crosses the threshold, collapses
# Checkpoints deleted after evaluation.
#   qsub jobs/w54_bos_induce.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"

fq () {  # $1 model $2 ckpt $3 tag $4 bits
  local model="$1" ckpt="$2" tag="$3" bits="$4"
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits "$bits" --group-size 128 --protect none --hess-bos-scale 10 --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) fq "$Q7"                                    qwen2.5-7b-v2gptq3-bos10     v2q7_bos10    3 ;;
  2) fq meta-llama/Llama-3.2-3B-Instruct         llama3.2-3b-v2gptq3-bos10    v2l32_bos10   3 ;;
  3) fq mistralai/Mistral-7B-Instruct-v0.3       mistral-7b-v2gptq3-bos10     v2m_bos10     3 ;;
  4) fq google/gemma-2-9b-it                     gemma2-9b-v2gptq3-bos10      v2g29_bos10   3 ;;
  5) fq "$LLAMA"                                 llama3.1-8b-v2gptq4-bos10    v2l4_bos10    4 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W54] done task $SGE_TASK_ID"
