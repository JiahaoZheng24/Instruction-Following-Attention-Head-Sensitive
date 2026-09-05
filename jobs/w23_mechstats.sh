#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W23
#$ -t 1-6
# W23: MECHANISM LOG + DIVERGENCE ONSET (Tier 0 #5). No new IFEval runs.
# Tasks 1-3 re-run the frozen GPTQ pass (same checkpoint as *-none, not saved)
# with logging: per module compensation displacement, per-column sent mass,
# clipping fraction, layer objective tr(dHd^T) for GPTQ vs RTN under the c4
# Hessian AND a chat-format Hessian (64 instruct prompts). Three questions:
#   Q1 does GPTQ win its own objective on c4 yet LOSE under chat inputs in
#      the collapsing models?  (distribution-shift account)
#   Q2 is compensation mass / clipping concentrated in a few modules, and are
#      those the critical-set modules (k/q_proj Llama, MLP Q14)?
#   Q3 does any of these statistics separate {llama, q14} from q7 - a
#      quantization-time predictor where all fp16 statistics failed?
# Tasks 4-6: layer of divergence (fp16 vs none ckpt) on 32 chat prompts.
# Afterwards:
#   python src/comp_stats.py --runs llama=runs/stats/llama31-8b q7=runs/stats/qwen25-7b \
#       q14=runs/stats/qwen25-14b --summary runs/comp_stats_summary.csv \
#       --per-layer runs/comp_stats_layers.csv
#   qsub jobs/w23_mechstats.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
mkdir -p runs/stats

stats_arm () {  # $1 model  $2 stats subdir
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --stats-dir "runs/stats/$2" --stats-chat-n 64 --stats-eig \
    --no-save --out "$STORE/models/_unused_$2"
}

case "$SGE_TASK_ID" in
  1) stats_arm "$LLAMA" llama31-8b ;;
  2) stats_arm "$Q7"    qwen25-7b ;;
  3) stats_arm "$Q14"   qwen25-14b ;;
  4) python src/divergence.py --fp16 "$LLAMA" --quant "$STORE/models/llama3.1-8b-v2gptq3-none" \
       --prompts "$FULL" --n 32 --out runs/div_l_none.csv
     python src/divergence.py --fp16 "$LLAMA" --quant "$STORE/models/llama3.1-8b-rtn3-none" \
       --prompts "$FULL" --n 32 --out runs/div_l_rtn.csv ;;
  5) python src/divergence.py --fp16 "$Q7" --quant "$STORE/models/qwen2.5-7b-v2gptq3-none" \
       --prompts "$FULL" --n 32 --out runs/div_q7_none.csv ;;
  6) python src/divergence.py --fp16 "$Q14" --quant "$STORE/models/qwen2.5-14b-v2gptq3-none" \
       --prompts "$FULL" --n 32 --out runs/div_q14_none.csv
     python src/divergence.py --fp16 "$Q14" --quant "$STORE/models/qwen2.5-14b-rtn3-none" \
       --prompts "$FULL" --n 32 --out runs/div_q14_rtn.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W23] done task $SGE_TASK_ID"
