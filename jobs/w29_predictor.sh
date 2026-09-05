#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W29
#$ -t 1-20
# W29: QUANTIZATION-TIME PREDICTOR across the whole census + loose ends.
# W23/W28 showed the chat-Hessian objective ratio (GPTQ vs RTN evaluated on
# chat-format inputs) is the one statistic that orders llama (1.02) > q14
# (0.91) > q7 (0.86) and drops to 0.82 when the collapse is cured. Tasks
# 1-14 compute it for every remaining census model (stats-only GPTQ pass,
# no checkpoint, no eval; ~1-3 h each). If the two collapses are the only
# models with median ratio ~1 / many "GPTQ worse than RTN" modules, the
# paper gets a positive prediction result at zero deployment cost.
# 15-17: capability-wide cure check (GSM8K on damp5 Llama/Q14; PPL+MMLU Q14 damp5).
# 18-20: does dampening rescue 2-bit? (Qwen-7B 2-bit none was 0.136 "terminal";
#        Llama 2-bit none + damp5). Decides whether "terminal" stays.
#   qsub jobs/w29_predictor.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
mkdir -p runs/stats

stats_arm () {  # $1 model  $2 stats subdir
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --stats-dir "runs/stats/$2" --stats-chat-n 64 \
    --no-save --out "$STORE/models/_unused_$2"
}
damp_arm () {  # $1 model $2 ckpt $3 tag $4 bits $5 percdamp
  CKPT="$STORE/models/$2"
  python src/quantize_protected.py --model "$1" --bits "$4" --group-size 128 \
    --protect none --percdamp "$5" --out "$CKPT"
  run_ifeval "$CKPT" "$3"
}

case "$SGE_TASK_ID" in
  1)  stats_arm mistralai/Mistral-7B-Instruct-v0.3          mistral-7b ;;
  2)  stats_arm mistralai/Mistral-Small-24B-Instruct-2501   mistral-24b ;;
  3)  stats_arm Qwen/Qwen2.5-32B-Instruct                   qwen25-32b ;;
  4)  stats_arm allenai/OLMo-2-1124-7B-Instruct             olmo2-7b ;;
  5)  stats_arm meta-llama/Meta-Llama-3-8B-Instruct         llama3-8b ;;
  6)  stats_arm meta-llama/Llama-3.2-3B-Instruct            llama32-3b ;;
  7)  stats_arm mistralai/Mistral-Nemo-Instruct-2407        mistral-nemo-12b ;;
  8)  stats_arm Qwen/Qwen2.5-3B-Instruct                    qwen25-3b ;;
  9)  stats_arm meta-llama/Llama-3.2-1B-Instruct            llama32-1b ;;
  10) stats_arm mistralai/Mistral-7B-Instruct-v0.2          mistral-7b-v02 ;;
  11) stats_arm tiiuae/Falcon3-7B-Instruct                  falcon3-7b ;;
  12) stats_arm HuggingFaceTB/SmolLM2-1.7B-Instruct         smollm2-1.7b ;;
  13) stats_arm google/gemma-2-9b-it                        gemma2-9b ;;
  14) stats_arm google/gemma-2-2b-it                        gemma2-2b ;;
  15) python src/gsm8k_eval.py --model "$STORE/models/llama3.1-8b-v2gptq3-damp5" \
        --tag gsm_llama_damp5 --batch 16 --scores-csv runs/scores_gsm8k.csv ;;
  16) python src/gsm8k_eval.py --model "$STORE/models/qwen2.5-14b-v2gptq3-damp5" \
        --tag gsm_q14_damp5 --batch 16 --scores-csv runs/scores_gsm8k.csv ;;
  17) python src/eval_general.py --model "$STORE/models/qwen2.5-14b-v2gptq3-damp5" \
        --tag gen_q14_damp5 --scores-csv runs/general_gen_q14_damp5.csv ;;
  18) damp_arm "$Q7"    qwen2.5-7b-v2gptq2-damp5   v2q_b2_damp5  2 5 ;;
  19) damp_arm "$LLAMA" llama3.1-8b-v2gptq2-none   v2l_b2_none   2 0.05 ;;
  20) damp_arm "$LLAMA" llama3.1-8b-v2gptq2-damp5  v2l_b2_damp5  2 5 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W29] done task $SGE_TASK_ID"
