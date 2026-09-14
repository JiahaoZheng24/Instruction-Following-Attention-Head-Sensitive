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
#$ -N IFH_W53B
#$ -t 1-38
# W53b: BLIND TEST, stage B (ground truth). Submit ONLY after the stage-A
# predictions are written in RESULTS.md. Odd task = 3-bit GPTQ none + IFEval,
# even task = 3-bit RTN + IFEval, for the 19 stage-A models in the same order.
# Collapse := GPTQ below RTN by more than 20 points. Every checkpoint is
# deleted right after its evaluation.
#   qsub jobs/w53b_blind_eval.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"

MODELS=(
  "allenai/Llama-3.1-Tulu-3-8B|tulu3_8b"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3_8b"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|r1_llama_8b"
  "NousResearch/Hermes-2-Pro-Llama-3-8B|hermes2pro_l3_8b"
  "Qwen/Qwen3-4B|qwen3_4b"
  "Qwen/Qwen3-8B|qwen3_8b"
  "Qwen/Qwen3-14B|qwen3_14b"
  "Qwen/Qwen2.5-1.5B-Instruct|qwen25_1p5b"
  "Qwen/Qwen2.5-0.5B-Instruct|qwen25_0p5b"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|r1_qwen_14b"
  "Qwen/Qwen2-7B-Instruct|qwen2_7b"
  "ibm-granite/granite-3.1-8b-instruct|granite31_8b"
  "HuggingFaceTB/SmolLM3-3B|smollm3_3b"
  "google/gemma-3-1b-it|gemma3_1b"
  "mistralai/Ministral-8B-Instruct-2410|ministral_8b"
  "CohereLabs/aya-expanse-8b|aya_8b"
  "tiiuae/Falcon3-10B-Instruct|falcon3_10b"
  "meta-llama/Llama-2-7b-chat-hf|llama2_7b"
  "mistralai/Mistral-7B-Instruct-v0.1|mistral_7b_v01"
)
idx=$(( (SGE_TASK_ID - 1) / 2 ))
entry="${MODELS[$idx]}"
model="${entry%%|*}"; short="${entry##*|}"
[ -n "$model" ] || { echo "bad task id"; exit 1; }

if [ $(( SGE_TASK_ID % 2 )) -eq 1 ]; then
  CKPT="$STORE/models/blind-${short}-v2gptq3-none"; TAG="blind_${short}_gptq3"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --out "$CKPT"
else
  CKPT="$STORE/models/blind-${short}-rtn3"; TAG="blind_${short}_rtn3"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --quantizer rtn --out "$CKPT"
fi
run_ifeval "$CKPT" "$TAG"
cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
rm -rf "$CKPT"
echo "[W53B] done task $SGE_TASK_ID ($TAG)"
