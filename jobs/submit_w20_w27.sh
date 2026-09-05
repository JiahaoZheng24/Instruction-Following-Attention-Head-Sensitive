#!/bin/bash
# One-shot submission of the whole W20-W27 batch (56 array tasks).
# Run from the repo root on the cluster login node:
#   bash jobs/submit_w20_w27.sh
# W20-W26 are independent; W27 holds on W20/W21/W23 by job name.
set -e
cd /store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive
mkdir -p logs runs/stats
[ -f jobs/_w2x_header.sh ] || { echo "jobs/_w2x_header.sh missing: git pull first"; exit 1; }
for j in w20_damping w21_config w22_scalefix w23_mechstats w24_census2 w25_gradfree w26_induce; do
  echo "== qsub jobs/$j.sh"; qsub "jobs/$j.sh"
done
echo "== qsub jobs/w27_followup.sh (held on W20/W21/W23)"; qsub jobs/w27_followup.sh
qstat -u "$USER" | head -40
