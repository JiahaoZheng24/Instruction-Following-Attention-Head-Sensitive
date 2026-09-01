#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W10b
#$ -hold_jid IFH_W10a
#$ -t 1-2
# W10b (UNFREEZE, depends on W10a salience outputs):
#  1: llama rot+gptq3 tacq@40.2M (rotated-basis salience from W10a task 4)
#     -> IF collapse survives rotation, does the rescue too?
#  2: qwen2.5-14b gptq3 tacq@70M (salience from W10a task 6; budget scales
#     with linear-param count, same ~0.55% ratio as the 7B/8B arms)
#   qsub jobs/w10b_rot_tacq.sh   (safe to submit together with w10a)

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
LLAMA="meta-llama/Llama-3.1-8B-Instruct"
Q14="Qwen/Qwen2.5-14B-Instruct"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

run_ifeval () {  # $1 ckpt, $2 tag
  python src/diagnose_heads.py ablate --model "$1" --prompts "$FULL" --tag "$2" --batch 16
  python src/score_ifeval.py --responses "runs/$(basename $1)/$2/responses.jsonl" \
    --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-rot-gptq3-tacq"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --rotate --rotate-seed 0 --protect tacq \
       --salience-dir "$STORE/salience/llama31-8b-if-rot" \
       --budget-params 40200000 --out "$CKPT"
     run_ifeval "$CKPT" v2l_rot_tacq ;;
  2) CKPT="$STORE/models/qwen2.5-14b-v2gptq3-tacq"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect tacq --salience-dir "$STORE/salience/qwen25-14b-if" \
       --budget-params 70000000 --out "$CKPT"
     run_ifeval "$CKPT" v2q14_tacq ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W10b] ✅ task $SGE_TASK_ID done"
