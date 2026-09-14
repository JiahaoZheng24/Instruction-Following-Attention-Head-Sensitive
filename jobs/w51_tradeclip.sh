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
#$ -N IFH_W51
#$ -t 1-5
# W51: the micro-mechanism of the bit-independent template error at Llama
# layer-1 down_proj. W50 located it: the error sits in the super-weight output
# rows (788 / 1384) and comes from the BOS massive-activation input columns
# (198, 2427, 6412, 12638, 12657) which carry 54-93% of GPTQ's error at
# <|end_header_id|>; "GPTQ zeroes more weights there" was refuted (RTN zeroes
# more). Hypothesis now: those columns are collinear in the c4 Hessian (they
# all fire only at BOS), so GPTQ trades weight between them to cancel the
# super weight's rounding error; in the super-weight row the group range is
# set by the super weight itself, so the traded weight is CLIPPED at that
# range -> a fixed displacement of order |super weight| that does not scale
# with the bit-width or the damping. Logged per arm: W, Q_gptq, Q_rtn of the
# top-error row at the top input columns, the row's max |W| / max |Q|, and
# how many entries of that row sit at the clip boundary.
#  1 default 3-bit  2 4-bit g128  3 damping 5  4 token-norm  5 no act-order
#   qsub jobs/w51_tradeclip.sh
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
  1) st runs/stats/trade-l-3b-g128  3 128 ;;
  2) st runs/stats/trade-l-4b-g128  4 128 ;;
  3) st runs/stats/trade-l-3b-damp5 3 128 --percdamp 5 ;;
  4) st runs/stats/trade-l-3b-tnorm 3 128 --hess-token-norm ;;
  5) st runs/stats/trade-l-3b-noao  3 128 --no-actorder ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W51] done task $SGE_TASK_ID"
