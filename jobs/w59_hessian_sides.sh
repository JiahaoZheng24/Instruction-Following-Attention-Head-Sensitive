#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=4:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W59
#$ -t 1-14
# W59: MEASURE THE OUTPUT SIDE OF THE HESSIAN (no quantisation, no IFEval).
# GPTQ uses the input covariance A = X X^T and drops the output-side factor G
# of the loss Hessian (K-FAC: H ~ A (x) G). Per module and per position class
# (BOS / positions 1-7 / rest) we record the share of A (sum ||x_t||^2), of
# G (sum ||dL/dy_t||^2) and of A.G. Pre-registered in RESULTS 9.10aq:
#   collapsing models (Llama, Q14, Mistral-v0.3) have a sink-forming down_proj
#   where BOS carries >= 90% of A but a far smaller share of G (G/A < 0.1), and
#   under chat prompts positions 1-7 carry a much larger share of G than of A;
#   Q7 / Nemo / Falcon3 have no module with that mismatch.
# Task 1 is a 4-sample smoke test; W60 refuses to start until its output exists.
#   qsub jobs/w59_hessian_sides.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 3h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs/sides

sd () {  # $1 model $2 calib $3 n $4 out-tag [$5 seqlen]
  $T python src/grad_weights.py --model "$1" --calib "$2" --n-calib "$3" --seqlen "${5:-2048}" --out "runs/sides/$4.csv"
}

case "$SGE_TASK_ID" in
   1) sd "$LLAMA" c4       4   smoke_l_c4 512 ;;
   2) sd "$LLAMA" c4       128 l_c4 ;;
   3) sd "$LLAMA" c4win    128 l_c4win ;;
   4) sd "$LLAMA" pile     128 l_pile ;;
   5) sd "$LLAMA" instruct 64  l_chat ;;
   6) sd "$Q14"   c4       128 q14_c4 ;;
   7) sd "$Q14"   c4win    128 q14_c4win ;;
   8) sd "$Q14"   instruct 64  q14_chat ;;
   9) sd "$M7"    c4       128 m7_c4 ;;
  10) sd "$M7"    c4win    128 m7_c4win ;;
  11) sd "$Q7"    c4       128 q7_c4 ;;
  12) sd "$Q7"    instruct 64  q7_chat ;;
  13) sd mistralai/Mistral-Nemo-Instruct-2407 c4 128 nemo_c4 ;;
  14) sd tiiuae/Falcon3-7B-Instruct           c4 128 f3_c4 ;;
   *) echo "bad task id"; exit 1 ;;
esac
echo "[W59] done task $SGE_TASK_ID"
