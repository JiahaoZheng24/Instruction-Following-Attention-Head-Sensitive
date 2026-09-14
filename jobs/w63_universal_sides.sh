#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=3:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W63
#$ -t 1-22
# W63 (universality, part 1): the A/G blind spot on EVERY remaining census and
# blind-test model (W59 measured 6). One forward+backward per sample, our short
# c4 x128, output runs/sides/<s>_c4.csv. Pre-registered in RESULTS 9.10ax:
# every model with a BOS/sink-dominated down_proj (input-norm ratio >= 20, W42)
# has A_bos >= 0.5 and G_bos <= 0.01 at that module; models without such a
# layer (gemma-2 / gemma-3) show no module with A_bos >= 0.5.
# gemma-2/3 use soft-capped attention; if sdpa backward fails they are reported
# as not measurable.
#   qsub jobs/w63_universal_sides.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 2h"
mkdir -p runs/sides

MODELS=(
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "Qwen/Qwen2.5-3B-Instruct|q3"
  "google/gemma-2-9b-it|g29"
  "google/gemma-2-2b-it|g22"
  "meta-llama/Meta-Llama-3-8B-Instruct|l3"
  "meta-llama/Llama-3.2-3B-Instruct|l32"
  "meta-llama/Llama-3.2-1B-Instruct|l32_1b"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct|sm"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b"
  "allenai/Llama-3.1-Tulu-3-8B|tulu3"
  "Qwen/Qwen3-8B|qwen3_8b"
  "ibm-granite/granite-3.1-8b-instruct|granite"
  "mistralai/Mistral-7B-Instruct-v0.2|m7v02"
  "tiiuae/Falcon3-10B-Instruct|falcon3_10b"
  "NousResearch/Hermes-2-Pro-Llama-3-8B|hermes2pro"
  "Qwen/Qwen2-7B-Instruct|qwen2_7b"
  "Qwen/Qwen3-4B|qwen3_4b"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|r1_llama_8b"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|r1_qwen_14b"
  "HuggingFaceTB/SmolLM3-3B|smollm3"
)
entry="${MODELS[$((SGE_TASK_ID - 1))]}"; model="${entry%%|*}"; s="${entry##*|}"
[ -n "$model" ] || { echo "bad task id"; exit 1; }
$T python src/grad_weights.py --model "$model" --calib c4 --n-calib 128 --seqlen 2048 --out "runs/sides/${s}_c4.csv"
echo "[W63] done task $SGE_TASK_ID ($s)"
