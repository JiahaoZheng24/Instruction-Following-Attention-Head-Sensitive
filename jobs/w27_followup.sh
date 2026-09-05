#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W27
#$ -t 1-9
#$ -hold_jid IFH_W20,IFH_W21,IFH_W23
# W27: FOLLOW-UP EVALS on checkpoints produced by W20/W21 (held until they
# finish). Extends the detection map and the capability-generality table to
# the new arms without any new quantization:
#   1-4 PPL+MMLU on the Llama damping ladder (does likelihood track the
#       IFEval recovery, or stay dissociated?)
#   5-6 PPL+MMLU on AWQ-sym Llama / Q14 (is the AWQ arm graceful on every axis?)
#   7-8 GSM8K on AWQ-sym Llama / Q14 (reasoning side of the AWQ prediction)
#   9   aggregate the W23 mechanism logs -> runs/comp_stats_*.csv
# If a held-for checkpoint is missing (that W20/W21 task failed) the task
# exits non-zero and the rest of the array is unaffected.
#   qsub jobs/w27_followup.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }

gen () {  # $1 ckpt $2 tag
  [ -d "$1" ] || { echo "missing $1"; exit 2; }
  python src/eval_general.py --model "$1" --tag "$2" --scores-csv "runs/general_$2.csv"
}
gsm () {  # $1 ckpt $2 tag
  [ -d "$1" ] || { echo "missing $1"; exit 2; }
  python src/gsm8k_eval.py --model "$1" --tag "$2" --batch 16 --scores-csv runs/scores_gsm8k.csv
}

case "$SGE_TASK_ID" in
  1) gen "$STORE/models/llama3.1-8b-v2gptq3-damp0p01" gen_l_damp0p01 ;;
  2) gen "$STORE/models/llama3.1-8b-v2gptq3-damp0p2"  gen_l_damp0p2 ;;
  3) gen "$STORE/models/llama3.1-8b-v2gptq3-damp1"    gen_l_damp1 ;;
  4) gen "$STORE/models/llama3.1-8b-v2gptq3-damp5"    gen_l_damp5 ;;
  5) gen "$STORE/models/llama3.1-8b-awq3-none"        gen_l_awq ;;
  6) gen "$STORE/models/qwen2.5-14b-awq3-none"        gen_q14_awq ;;
  7) gsm "$STORE/models/llama3.1-8b-awq3-none"        gsm_llama_awq ;;
  8) gsm "$STORE/models/qwen2.5-14b-awq3-none"        gsm_q14_awq ;;
  9) python src/comp_stats.py \
       --runs llama=runs/stats/llama31-8b q7=runs/stats/qwen25-7b q14=runs/stats/qwen25-14b \
       --summary runs/comp_stats_summary.csv --per-layer runs/comp_stats_layers.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W27] done task $SGE_TASK_ID"
