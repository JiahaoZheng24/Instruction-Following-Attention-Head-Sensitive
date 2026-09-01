#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W10c
#$ -t 1-2
# W10c (UNFREEZE): Mistral-Small-24B regime check — second-family scale point
# (Llama has no ~30B model; 70B needs 2 GPUs we may not get -> demoted to
# rebuttal-optional). Together with Qwen 14B/32B (W10a) this gives TWO
# within-family scaling lines. Apache 2.0, ungated, single H200.
#  1: mistral-24b fp16 IFEval baseline
#  2: mistral-24b gptq3 none quantize + IFEval (which regime at 24B?)
# If task 2 collapses, add salience+tacq arms (rescue-at-scale headline).
#   qsub jobs/w10c_mistral24b.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
M24="mistralai/Mistral-Small-24B-Instruct-2501"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1) python src/diagnose_heads.py ablate --model "$M24" --prompts "$FULL" \
       --tag qbase_mistral24_fp16 --batch 16
     python src/score_ifeval.py \
       --responses "runs/Mistral-Small-24B-Instruct-2501/qbase_mistral24_fp16/responses.jsonl" \
       --input-data "$FULL" --tag qbase_mistral24_fp16 \
       --scores-csv runs/scores_qbase_mistral24_fp16.csv ;;
  2) CKPT="$STORE/models/mistral-24b-v2gptq3-none"
     python src/quantize_protected.py --model "$M24" --bits 3 --group-size 128 \
       --protect none --out "$CKPT"
     python src/diagnose_heads.py ablate --model "$CKPT" --prompts "$FULL" \
       --tag v2m24_none --batch 16
     python src/score_ifeval.py --responses "runs/$(basename $CKPT)/v2m24_none/responses.jsonl" \
       --input-data "$FULL" --tag v2m24_none --scores-csv runs/scores_v2m24_none.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W10c] ✅ task $SGE_TASK_ID done"
