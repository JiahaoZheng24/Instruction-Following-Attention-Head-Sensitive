#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W14
#$ -t 1-2
# W14 micro: insurance for the two spiciest W13 findings at 14B.
#  1: tacq@1e5 calib-seed 1  — replicate "under-budget protection HURTS"
#     (0.216 < none 0.412; collapse-arm seed variance is +-10, so confirm)
#  2: tacq@3e5              — locate the 14B transition (1e5 fails, 1e6 works)
#   qsub jobs/w14_micro.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
Q14="Qwen/Qwen2.5-14B-Instruct"
FULL="data/ifeval_input_data.jsonl"
SAL14="$STORE/salience/qwen25-14b-if"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

run_ifeval () {
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq1e5_cs1"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 100000 \
       --calib-seed 1 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq1e5_cs1 ;;
  2) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq3e5"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$SAL14" --budget-params 300000 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq3e5 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W14] ✅ task $SGE_TASK_ID done"
