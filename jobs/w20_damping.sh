#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W20
#$ -t 1-8
# W20: DAMPING SWEEP - the compensation-strength knob (Tier 0 #1).
# GPTQ adds percdamp*mean(diag H) to the Hessian diagonal; damp -> inf makes
# the OBS update vanish and GPTQ degenerates into RTN. Frozen protocol used
# 0.05 (Llama 0.150 / Q14 0.412); RTN is the limit (0.565 / 0.697).
# Pre-registered readings:
#   monotone recovery toward RTN as damp grows  -> collapse = ill-conditioned
#       compensation; single-model phase curve; answers "n=2" and "why".
#   non-monotone / no recovery                  -> mechanism is not conditioning
#       (look at clipping, W23) - still informative, story adjusts.
#   0.01 (GPTQ reference default) worse than 0.05 -> the frozen protocol was
#       already CONSERVATIVE; reviewers cannot blame dampening.
#   qsub jobs/w20_damping.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }

damp_arm () {  # $1 model  $2 ckpt-name  $3 tag  $4 percdamp
  CKPT="$STORE/models/$2"
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --percdamp "$4" --out "$CKPT"
  run_ifeval "$CKPT" "$3"
}

case "$SGE_TASK_ID" in
  1) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp0p01 v2l_damp0p01 0.01 ;;
  2) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp0p2  v2l_damp0p2  0.2  ;;
  3) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp1    v2l_damp1    1.0  ;;
  4) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp5    v2l_damp5    5.0  ;;
  5) damp_arm "$Q14"   qwen2.5-14b-v2gptq3-damp0p01 v2q14_damp0p01 0.01 ;;
  6) damp_arm "$Q14"   qwen2.5-14b-v2gptq3-damp0p2  v2q14_damp0p2  0.2  ;;
  7) damp_arm "$Q14"   qwen2.5-14b-v2gptq3-damp1    v2q14_damp1    1.0  ;;
  8) damp_arm "$Q14"   qwen2.5-14b-v2gptq3-damp5    v2q14_damp5    5.0  ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W20] done task $SGE_TASK_ID"
