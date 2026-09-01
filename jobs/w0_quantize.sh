#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_W0_quantize
#$ -t 1-4
# Fresh, protocol-controlled GPTQ checkpoints (old ones deleted / possibly
# corrupted; old 3-bit output was garbled — smoke test at the end of each task
# tells us whether that was real 3-bit damage or a broken checkpoint/kernel).
# Run FIRST:  qsub jobs/w0_quantize.sh
# Then:       qsub -hold_jid IFH_W0_quantize jobs/w1_calib.sh
#             qsub -hold_jid IFH_W1_calib   jobs/w1_ablate.sh

set -e

STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"
MODELS_DIR="$STORE/models"
export HF_HOME="$STORE/hf_cache"
mkdir -p "$MODELS_DIR"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
# deps: pip install gptqmodel datasets   (plus torch/transformers already in env)
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"

# gptqmodel offloads the bf16 base model to disk during quantization. The
# default ./gptqmodel_offload/ lands on NFS and is shared by all array tasks
# -> concurrent tasks race (rmtree FileNotFoundError on task 1, truncated
# safetensors read on task 2; job 1385695). Use node-local scratch, unique
# per task ($TMPDIR is SGE's per-job local dir, auto-cleaned).
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

case "$SGE_TASK_ID" in
  1) BITS=2 ;;
  2) BITS=3 ;;
  3) BITS=4 ;;
  4) BITS=8 ;;   # 8-bit = near-FP reference point
  *) echo "bad task id"; exit 1 ;;
esac

OUT="$MODELS_DIR/qwen2.5-7b-gptq${BITS}-c4-g128"
echo "[W0-quantize] ${BITS}-bit -> $OUT"

python src/quantize_gptq.py \
  --model "$MODEL_ID" \
  --bits "$BITS" \
  --group-size 128 \
  --calib c4 \
  --out "$OUT"

echo "[W0-quantize] ✅ ${BITS}-bit done. CHECK THE SMOKE TEST OUTPUT ABOVE."
