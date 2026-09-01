#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -o logs/
#$ -N IFH_W5_prep
#$ -t 1-3
# W5 prep: Llama-3.1-8B-Instruct replication line, stage 1.
#  1: GPTQ 3-bit c4-g128 + smoke + full-541 IFEval baseline
#  2: fp16 full-541 IFEval baseline
#  3: TaCQ saliency (IF-conditioned)
# NOTE meta-llama/Llama-3.1-8B-Instruct is GATED: needs an approved HF token
# (huggingface-cli login, or export HF_TOKEN=...) on the cluster BEFORE this.
# Then: qsub -hold_jid IFH_W5_prep jobs/w5_llama_screen.sh
#       qsub -hold_jid IFH_W5_screen jobs/w5_llama_inloop.sh

set -e
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
export HF_HOME="$STORE/hf_cache"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
QUANT="$STORE/models/llama3.1-8b-gptq3-c4-g128"
FULL="data/ifeval_input_data.jsonl"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
source jobs/hf.env 2>/dev/null || source "$BASE_DIR/jobs/hf.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"
cd "$BASE_DIR"

case "$SGE_TASK_ID" in
  1)
    python src/quantize_gptq.py --model "$MODEL_ID" \
      --bits 3 --group-size 128 --calib c4 --out "$QUANT"
    python src/diagnose_heads.py ablate --model "$QUANT" \
      --prompts "$FULL" --tag qbase_llama_gptq3 --batch 16
    python src/score_ifeval.py \
      --responses "runs/$(basename $QUANT)/qbase_llama_gptq3/responses.jsonl" \
      --input-data "$FULL" --tag qbase_llama_gptq3 \
      --scores-csv runs/scores_qbase_llama_gptq3.csv ;;
  2)
    python src/diagnose_heads.py ablate --model "$MODEL_ID" \
      --prompts "$FULL" --tag qbase_llama_fp16 --batch 16
    python src/score_ifeval.py \
      --responses "runs/Llama-3.1-8B-Instruct/qbase_llama_fp16/responses.jsonl" \
      --input-data "$FULL" --tag qbase_llama_fp16 \
      --scores-csv runs/scores_qbase_llama_fp16.csv ;;
  3)
    SAL_DIR="$STORE/salience/llama31-8b-if"
    mkdir -p "$SAL_DIR"
    python src/tacq_salience.py --model "$MODEL_ID" \
      --calib-file data/calib_prompts.jsonl --bits 3 --out-dir "$SAL_DIR" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W5-prep] ✅ task $SGE_TASK_ID done"
