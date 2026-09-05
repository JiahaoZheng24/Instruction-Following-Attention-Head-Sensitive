#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W25
#$ -t 1-3
# W25: GRADIENT-FREE CRITERION (Tier 1 #7). Protect the top weights by
# |W| * sqrt(H_ii) - activation-aware magnitude, available inside the GPTQ
# pass for free (no 93 GB gradient job). The c4-mask result (15 % identity
# overlap with the IF mask, equal rescue) predicts this works: rescue is
# about structural TYPE, not weight identity. If it rescues:
#   practical fix = one line inside GPTQ; mechanism is Hessian-side.
# Budgets at the phase transition: Llama 1e5, Q14 1e6; plus Llama at the
# 0.55 % headline budget for a like-for-like with tacq (0.643).
#   qsub jobs/w25_gradfree.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-v2gptq3-hmag1e5"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect hmag --budget-params 100000 --out "$CKPT"
     run_ifeval "$CKPT" v2l_hmag1e5 ;;
  2) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-hmag1e6"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect hmag --budget-params 1000000 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_hmag1e6 ;;
  3) CKPT="$STORE/models/llama3.1-8b-v2gptq3-hmag40m"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect hmag --budget-params 40200000 --out "$CKPT"
     run_ifeval "$CKPT" v2l_hmag40m ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W25] done task $SGE_TASK_ID"
