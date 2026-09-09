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
#$ -N IFH_W49
#$ -t 1-9
# W49: the theory table. For each Llama recipe of the threshold table
# (RESULTS 9.10q; deck slide 12) log the LINEAR-LAYER error of layer-1
# down_proj at the template positions (new stats columns tplpos_err_gptq /
# tplpos_err_rtn, plus the ratio). Theory: e(z) = delta(bits, group) x
# R(H, lambda, z); R is independent of the bit-width, delta doubles per bit
# removed. Prediction to check against the measured layer-2 residual error
# (3.85 / 2.43 / 1.54 / 1.37 / 1.27 / 1.33 / 1.20 / 0.45) and IFEval:
#   - ratio (=R^2) about the same at 4-bit as at 3-bit; absolute error halves
#   - group 32: error 20-40% lower than group 128, ratio unchanged
#   - damping 1/2/5: ratio falls monotonically; error crosses the threshold
#     between rho=1 and 2
#   - token-norm and c4chat: ratio near 1
# No checkpoints. ~40 min each.
#   qsub jobs/w49_theory_stats.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
mkdir -p runs/stats

st () {  # $1 stats-dir $2 bits $3 group $4.. flags
  local sd="$1" bits="$2" grp="$3"; shift 3
  $T python src/quantize_protected.py --model "$LLAMA" --bits "$bits" --group-size "$grp" --protect none \
     --stats-dir "$sd" --stats-chat-n 64 --no-save "$@" --out "$STORE/models/_unused_theory"
}

case "$SGE_TASK_ID" in
  1) st runs/stats/theory-l-3b-g32     3 32  ;;
  2) st runs/stats/theory-l-3b-gm1     3 -1  ;;
  3) st runs/stats/theory-l-4b-g128    4 128 ;;
  4) st runs/stats/theory-l-4b-gm1     4 -1  ;;
  5) st runs/stats/theory-l-3b-damp1   3 128 --percdamp 1 ;;
  6) st runs/stats/theory-l-3b-damp2   3 128 --percdamp 2 ;;
  7) st runs/stats/theory-l-3b-damp5   3 128 --percdamp 5 ;;
  8) st runs/stats/theory-l-3b-tnorm   3 128 --hess-token-norm ;;
  9) st runs/stats/theory-l-3b-c4chat  3 128 --calib c4chat ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W49] done task $SGE_TASK_ID"
