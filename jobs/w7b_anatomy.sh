#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W7b_anat
#$ -t 1-2
# W7b: critical-set anatomy (needs the c4-conditioned salience from W7 4/5).
#  1: llama @1e5   — channel concentration + activation-channel overlap +
#                    IF-vs-c4 mask Jaccard (task-generality)
#  2: qwen  @37.6M — same, for the graceful-regime mask (contrast figure)
#   qsub -hold_jid IFH_W7_crit jobs/w7b_anatomy.sh

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
  1) python src/critical_anatomy.py \
       --salience-dir "$STORE/salience/llama31-8b-if" \
       --budget-params 100000 \
       --model meta-llama/Llama-3.1-8B-Instruct \
       --compare-dir "$STORE/salience/llama31-8b-c4" \
       --out runs/critical_anatomy_llama.csv ;;
  2) python src/critical_anatomy.py \
       --salience-dir "$STORE/salience/qwen25-7b-if" \
       --budget-params 37624064 \
       --model Qwen/Qwen2.5-7B-Instruct \
       --compare-dir "$STORE/salience/qwen25-7b-c4" \
       --out runs/critical_anatomy_qwen.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W7b] ✅ task $SGE_TASK_ID done"
