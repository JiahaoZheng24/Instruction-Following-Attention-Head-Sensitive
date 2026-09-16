#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=01:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W72z
#$ -t 1-2
# W72z (2026-09-15): PROPAGATION CHECK, run BEFORE W72a-d. Emulates the layer-wise
# loop of quantize_protected.py exactly (capture, two calls per layer per sample,
# RTN of the layer in between) and compares the propagated activations at every
# layer with a full forward of the same quantised model. The W71 defect would have
# failed this check. Prints PASS / FAIL; ~5-10 min per model.
#   1 Llama-3.1-8B-Instruct    2 Qwen2.5-14B-Instruct
#   qsub jobs/w72z_check.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
case "$SGE_TASK_ID" in
  1) python src/check_propagation.py --model "$LLAMA" --n 4 ;;
  2) python src/check_propagation.py --model "$Q14"   --n 4 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W72z] done task $SGE_TASK_ID"
