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
#$ -N IFH_W53A
#$ -t 1-19
# W53a: BLIND TEST, stage A (detector only, no checkpoints, no IFEval).
# 19 models we have never evaluated. For each: 3-bit g128 GPTQ statistics
# with the per-position columns -> detector reading (max GPTQ/RTN error at
# template positions 2-7), sink dominance (BOS norm / typical at the early
# down_proj), and the down_proj template error. Predictions (collapse / no
# collapse) are written into RESULTS.md from these numbers BEFORE stage B
# (jobs/w53b_blind_eval.sh) runs IFEval on GPTQ and RTN.
# Structural priors registered before running:
#   - Llama-3.1-8B fine-tunes (Tulu-3, Hermes-3, R1-Distill-Llama-8B): early
#     layers essentially unchanged -> predicted COLLAPSE.
#   - Hermes-2-Pro (Llama-3-8B base, graceful .611): predicted graceful, detector ~30.
#   - Qwen3 / small Qwen2.5 / R1-Distill-Qwen-14B / Qwen2-7B: genuinely open.
#   - gemma-3-1b: predicted silent (gemma-2 has no sink dominance).
# Model weights land in $HF_HOME; delete them with jobs/hf_cache_rm.sh after stage B.
#   qsub jobs/w53a_blind_stats.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
mkdir -p runs/stats

posstats () {  # $1 model $2 stats-dir
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none \
     --stats-dir "$2" --stats-chat-n 64 --no-save --out "$STORE/models/_unused_blind"
}

case "$SGE_TASK_ID" in
  1)  posstats allenai/Llama-3.1-Tulu-3-8B                 runs/stats/blind-tulu3-8b-pos ;;
  2)  posstats NousResearch/Hermes-3-Llama-3.1-8B          runs/stats/blind-hermes3-8b-pos ;;
  3)  posstats deepseek-ai/DeepSeek-R1-Distill-Llama-8B    runs/stats/blind-r1-llama-8b-pos ;;
  4)  posstats NousResearch/Hermes-2-Pro-Llama-3-8B        runs/stats/blind-hermes2pro-l3-8b-pos ;;
  5)  posstats Qwen/Qwen3-4B                               runs/stats/blind-qwen3-4b-pos ;;
  6)  posstats Qwen/Qwen3-8B                               runs/stats/blind-qwen3-8b-pos ;;
  7)  posstats Qwen/Qwen3-14B                              runs/stats/blind-qwen3-14b-pos ;;
  8)  posstats Qwen/Qwen2.5-1.5B-Instruct                  runs/stats/blind-qwen25-1p5b-pos ;;
  9)  posstats Qwen/Qwen2.5-0.5B-Instruct                  runs/stats/blind-qwen25-0p5b-pos ;;
  10) posstats deepseek-ai/DeepSeek-R1-Distill-Qwen-14B    runs/stats/blind-r1-qwen-14b-pos ;;
  11) posstats Qwen/Qwen2-7B-Instruct                      runs/stats/blind-qwen2-7b-pos ;;
  12) posstats ibm-granite/granite-3.1-8b-instruct         runs/stats/blind-granite31-8b-pos ;;
  13) posstats HuggingFaceTB/SmolLM3-3B                    runs/stats/blind-smollm3-3b-pos ;;
  14) posstats google/gemma-3-1b-it                        runs/stats/blind-gemma3-1b-pos ;;
  15) posstats mistralai/Ministral-8B-Instruct-2410        runs/stats/blind-ministral-8b-pos ;;
  16) posstats CohereLabs/aya-expanse-8b                   runs/stats/blind-aya-8b-pos ;;
  17) posstats tiiuae/Falcon3-10B-Instruct                 runs/stats/blind-falcon3-10b-pos ;;
  18) posstats meta-llama/Llama-2-7b-chat-hf               runs/stats/blind-llama2-7b-pos ;;
  19) posstats mistralai/Mistral-7B-Instruct-v0.1          runs/stats/blind-mistral-7b-v01-pos ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W53A] done task $SGE_TASK_ID"
