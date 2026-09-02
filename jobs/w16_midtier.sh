#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W16
#$ -t 1-4
# W16: the census found a MIDDLE TIER (-15..-25 drop, 8-22% loops). Question:
# does protection gain scale with damage (dose-response) or is there a sharp
# threshold (mid-tier gains ~0, like the graceful band)?
# Representatives: Qwen2.5-3B (heaviest drop, -24.7) and Yi-1.5-9B (highest
# loop rate, 21.6%).
#  1: q25_3b IF salience -> tacq@17M -> IFEval   (salience+tacq in one task)
#  2: yi_9b  IF salience -> tacq@45M -> IFEval
#  3: q25_3b RTN3 none (is mid-tier damage also compensation-driven?)
#  4: yi_9b  RTN3 none
# Budgets ~0.55% of linear params (3B -> 17M, 9B -> 45M).
#   qsub jobs/w16_midtier.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
Q3="Qwen/Qwen2.5-3B-Instruct"
YI="01-ai/Yi-1.5-9B-Chat"
FULL="data/ifeval_input_data.jsonl"

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
  1) mkdir -p "$STORE/salience/qwen25-3b-if"
     python src/tacq_salience.py --model "$Q3" \
       --calib-file data/calib_prompts.jsonl --bits 3 \
       --out-dir "$STORE/salience/qwen25-3b-if"
     CKPT="$STORE/models/qwen2.5-3b-v2gptq3-tacq"
     python src/quantize_protected.py --model "$Q3" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$STORE/salience/qwen25-3b-if" \
       --budget-params 17000000 --out "$CKPT"
     run_ifeval "$CKPT" v2q3_tacq ;;
  2) mkdir -p "$STORE/salience/yi15-9b-if"
     python src/tacq_salience.py --model "$YI" \
       --calib-file data/calib_prompts.jsonl --bits 3 \
       --out-dir "$STORE/salience/yi15-9b-if"
     CKPT="$STORE/models/yi1.5-9b-v2gptq3-tacq"
     python src/quantize_protected.py --model "$YI" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$STORE/salience/yi15-9b-if" \
       --budget-params 45000000 --out "$CKPT"
     run_ifeval "$CKPT" v2yi_tacq ;;
  3) CKPT="$STORE/models/qwen2.5-3b-rtn3-none"
     python src/quantize_protected.py --model "$Q3" --bits 3 --group-size 128 \
       --rtn --protect none --out "$CKPT"
     run_ifeval "$CKPT" rtn3q3_none ;;
  4) CKPT="$STORE/models/yi1.5-9b-rtn3-none"
     python src/quantize_protected.py --model "$YI" --bits 3 --group-size 128 \
       --rtn --protect none --out "$CKPT"
     run_ifeval "$CKPT" rtn3yi_none ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W16] ✅ task $SGE_TASK_ID done"
