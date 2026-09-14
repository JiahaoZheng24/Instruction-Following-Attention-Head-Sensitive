#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=8:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W57
#$ -t 1-6
# W57: STATS ONLY (no checkpoint, no IFEval). W56c showed the collapse is a
# property of the calibration SAMPLE DISTRIBUTION, not of GPTQ or of the c4
# corpus: Llama/Q14 GPTQ with c4 128x2048 windows .643/.753 (no collapse),
# with our 128 short c4 docs .150/.412 (collapse), with pile 128x2048 windows
# .402/.291 (collapse), wikitext 128x2048 .624 (cure) but wikitext 32x2048
# .246 (collapse), c4 short x512 .151 (collapse).
# Hypothesis (RESULTS 9.10al): one Hessian quantity orders all of these — the
# off-BOS curvature of the sink channels relative to the damping floor, which
# the per-position detector / product law already measure. Prospective test:
# the product law calibrated on 13 Llama recipes (threshold 8.6-9.5) must put
# c4win and wikitext128 BELOW and pile / wikitext32 ABOVE the line.
#   qsub jobs/w57_calib_stats.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"

st () {  # $1 model $2 stats-dir $3 calib $4 n-calib
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 --protect none \
     --calib "$3" --n-calib "$4" --stats-dir "$2" --stats-chat-n 64 --no-save --out "$STORE/models/_unused_theory"
}

case "$SGE_TASK_ID" in
  1) st "$LLAMA" runs/stats/theory-l-3b-c4win      c4win    128 ;;
  2) st "$LLAMA" runs/stats/theory-l-3b-pile       pile     128 ;;
  3) st "$LLAMA" runs/stats/theory-l-3b-wiki128    wikitext 128 ;;
  4) st "$LLAMA" runs/stats/theory-l-3b-wiki32     wikitext 32  ;;
  5) st "$Q14"   runs/stats/theory-q14-3b-c4win    c4win    128 ;;
  6) st "$Q14"   runs/stats/theory-q14-3b-pile     pile     128 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W57] done task $SGE_TASK_ID"
