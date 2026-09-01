#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W6_detect
#$ -t 1-2
# W6 stage 1: detect super weights (activation-spike method, Yu et al.
# 2411.07191) on both models + report where they rank inside our TaCQ
# salience (the rescue-curve attribution check).
#   qsub jobs/w6_detect.sh
#   qsub -hold_jid IFH_W6_detect jobs/w6_rescue.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) python src/detect_super_weights.py --model meta-llama/Llama-3.1-8B-Instruct \
       --out runs/super_weights_llama.csv --salience-dir "$STORE/salience/llama31-8b-if" ;;
  2) python src/detect_super_weights.py --model Qwen/Qwen2.5-7B-Instruct \
       --out runs/super_weights_qwen.csv --salience-dir "$STORE/salience/qwen25-7b-if" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W6-detect] ✅ task $SGE_TASK_ID done"
