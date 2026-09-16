#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=06:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W74
#$ -t 1-16
# W74 (2026-09-15). LESION SIGNATURE ACROSS THE CENSUS, fixed code, stats only
# (no checkpoint, no IFEval). For each of the 16 census models the stats mode
# records, per module: template-position error of GPTQ vs RTN (tplpos_err_*),
# top-row share of the error (p4_toprow_share_*), clipping (p4_toprow_nclip,
# clip_frac), the position-resolved objective (tplpos_ratio) and sink dominance.
# This lets Corollary 1(i) (row concentration, clipping) and the lesion size be
# reported for every sink model, not only for the two that collapse.
# Llama / Q14 stats are W72a tasks 65-66 (runs/stats/f-l-3b, f-q14-3b).
#   qsub jobs/w74_lesion_census.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 5h"
mkdir -p runs/stats

MODELS=(
  "Qwen/Qwen2.5-7B-Instruct|q7"
  "Qwen/Qwen2.5-32B-Instruct|q32"
  "Qwen/Qwen2.5-3B-Instruct|q3"
  "mistralai/Mistral-Nemo-Instruct-2407|nemo"
  "mistralai/Mistral-7B-Instruct-v0.3|m7"
  "google/gemma-2-9b-it|g29"
  "meta-llama/Meta-Llama-3-8B-Instruct|l3"
  "meta-llama/Llama-3.2-3B-Instruct|l32"
  "tiiuae/Falcon3-7B-Instruct|f3"
  "HuggingFaceTB/SmolLM2-1.7B-Instruct|sm"
  "NousResearch/Hermes-3-Llama-3.1-8B|hermes3"
  "mistralai/Ministral-8B-Instruct-2410|ministral"
  "Qwen/Qwen3-14B|qwen3_14b"
  "allenai/Llama-3.1-Tulu-3-8B|tulu3"
  "Qwen/Qwen3-8B|qwen3_8b"
  "ibm-granite/granite-3.1-8b-instruct|granite"
)

entry="${MODELS[$((SGE_TASK_ID - 1))]}"; model="${entry%%|*}"; s="${entry##*|}"
[ -n "$model" ] || { echo "bad task id"; exit 1; }
$T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none --calib c4 \
   --stats-dir "runs/stats/f-${s}-3b" --stats-chat-n 64 --no-save --out "$STORE/models/_unused_f_stats_${s}"
echo "[W74] done task $SGE_TASK_ID ($s)"
