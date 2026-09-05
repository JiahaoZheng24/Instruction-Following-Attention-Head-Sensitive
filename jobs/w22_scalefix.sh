#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W22
#$ -t 1-4
# W22: GROUP-SCALE ARTIFACT CHECK (Tier 0 #3).
# Frozen gptq_core computed each group's scale from the max |W| INCLUDING
# protected entries. A protected large weight therefore coarsens the grid of
# its 127 neighbours. --scale-excl-mask removes protected entries from the
# range statistic (SpQR/TaCQ semantics). Two things to learn:
#   (a) does "under-budgeted protection hurts" (Q14 tacq@1e5 0.216 < none
#       0.412, both seeds) survive?  If not -> drop that claim (artifact).
#   (b) were the rescue arms UNDER-estimated? (Llama tacq 0.643 / @1e5 0.671)
#   qsub jobs/w22_scalefix.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq1e5_sx"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 100000 \
       --scale-excl-mask --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq1e5_sx ;;
  2) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq1e5_sx_cs1"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 100000 \
       --scale-excl-mask --calib-seed 1 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq1e5_sx_cs1 ;;
  3) CKPT="$STORE/models/llama3.1-8b-v2gptq3-tacq_sx"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL_L" --budget-params 40200000 \
       --scale-excl-mask --out "$CKPT"
     run_ifeval "$CKPT" v2l_tacq_sx ;;
  4) CKPT="$STORE/models/llama3.1-8b-v2gptq3-tacq1e5_sx"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL_L" --budget-params 100000 \
       --scale-excl-mask --out "$CKPT"
     run_ifeval "$CKPT" v2l_tacq1e5_sx ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W22] done task $SGE_TASK_ID"
