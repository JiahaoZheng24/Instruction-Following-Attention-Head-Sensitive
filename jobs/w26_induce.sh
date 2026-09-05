#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W26
#$ -t 1-8
# W26: INDUCE COLLAPSE in graceful / middle models (the answer to "what if
# the census stays at 2 collapses"). If collapse = compensation x critical
# structure, pushing compensation harder or making the Hessian worse should
# drive a graceful model over the threshold. Three levers, each a one-flag
# change to the frozen protocol:
#   damp0p001  percdamp 0.001 (50x less dampening -> stronger, less regularised OBS update)
#   ncal8      only 8 calibration docs -> rank-deficient / ill-conditioned H
#   g-1        per-channel, no grouping (2404.14047: Llama-3-8B W3 PPL 8.2 -> 13.0)
# Readings (pre-registered):
#   graceful model collapses (loop rate >50%, RTN sign flips) -> collapse is a
#       THRESHOLD phenomenon in (model x config) space; 2/17 = where defaults sit.
#   nothing collapses -> critical structure is a model-side necessary condition;
#       mechanism claim gets stronger, prevalence stays 2/17 (report honestly).
# RTN references already exist for Qwen-7B (0.500) and Mistral-7B (0.420); the
# g-1 arms get their own RTN g-1 reference (tasks 7-8) so the sign test is
# like-for-like.
#   qsub jobs/w26_induce.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
M7="mistralai/Mistral-7B-Instruct-v0.3"

arm () {  # $1 model $2 ckpt $3 tag $4.. extra flags
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 \
    --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
}

case "$SGE_TASK_ID" in
  1) arm "$Q7" qwen2.5-7b-v2gptq3-damp0p001 v2q_damp0p001 --percdamp 0.001 ;;
  2) arm "$Q7" qwen2.5-7b-v2gptq3-ncal8     v2q_ncal8     --n-calib 8 ;;
  3) arm "$Q7" qwen2.5-7b-v2gptq3-g-1       v2q_gm1       --group-size -1 ;;
  4) arm "$M7" mistral-7b-v2gptq3-damp0p001 v2m_damp0p001 --percdamp 0.001 ;;
  5) arm "$M7" mistral-7b-v2gptq3-ncal8     v2m_ncal8     --n-calib 8 ;;
  6) arm "$M7" mistral-7b-v2gptq3-g-1       v2m_gm1       --group-size -1 ;;
  7) arm "$Q7" qwen2.5-7b-rtn3-g-1          rtn3q_gm1     --group-size -1 --quantizer rtn ;;
  8) arm "$M7" mistral-7b-rtn3-g-1          rtn3m_gm1     --group-size -1 --quantizer rtn ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W26] done task $SGE_TASK_ID"
