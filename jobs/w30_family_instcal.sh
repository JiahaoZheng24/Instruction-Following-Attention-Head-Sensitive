#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=10:00:00
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W30
#$ -t 1-19
# W30: (a) OBS-FAMILY arm + real packed kernels, (b) instruct-calibration census.
# Storage-safe: every fake-quant checkpoint is DELETED right after its IFEval
# run (scores/responses stay under runs/). Peak usage ~ one checkpoint per
# running task (<=64 GB for the 32B). Packed gptqmodel checkpoints (~4-8 GB)
# are kept: they are the deployable artifacts.
#
#  1-2  GPTAQ / GPTQv2 (gptqmodel v2=True), 3-bit, c4 x128, Llama & Q14.
#       The 2025 OBS descendant: does asymmetric calibration also collapse?
#       Either answer is a paragraph ("whole OBS family" vs "GPTAQ's own
#       correction is a regulariser").
#  3-4  gptqmodel v1 packed, damp_percent 5.0 (real kernel, real checkpoint):
#       converts the damping cure from fake-quant to deployable. gptqmodel
#       may clamp damp_percent — the log prints the value in effect; if it
#       is clamped, that is itself worth a sentence.
#  5-19 instruct (chat-format) calibration for the 15 census models that
#       lack it (Llama/Q14 have it). If instruct calibration never costs
#       >1 pt on graceful models, "calibrate instruct models on instructions"
#       becomes a zero-cost, kernel-free recommendation backed by 17 models.
#   qsub jobs/w30_family_instcal.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"
export TOKENIZERS_PARALLELISM=false
# GPU-hang guards: h_rt=10h hard wall (longest task ~5h) so a stuck task is
# killed by SGE and the card is released; every python step is also wrapped in
# `timeout` so a hung dataloader/network call cannot outlive its budget.
T="timeout --signal=TERM --kill-after=120 6h"

packed () {  # $1 model $2 ckpt $3 tag $4.. flags for quantize_gptq.py
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_gptq.py --model "$model" --bits 3 --group-size 128 --calib c4 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
}
instcal () {  # $1 model $2 ckpt-stem $3 tag   (fake-quant, deleted after eval)
  CKPT="$STORE/models/$2-v2gptq3-calinst"
  $T python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --calib instruct --out "$CKPT"
  run_ifeval "$CKPT" "$3"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) packed "$LLAMA" llama3.1-8b-gptaq3-c4-g128 gptaq3_llama --v2 ;;
  2) packed "$Q14"   qwen2.5-14b-gptaq3-c4-g128 gptaq3_q14  --v2 ;;
  3) packed "$LLAMA" llama3.1-8b-gptq3-damp5-packed gptq3_llama_damp5_packed --damp-percent 5.0 ;;
  4) packed "$Q14"   qwen2.5-14b-gptq3-damp5-packed gptq3_q14_damp5_packed  --damp-percent 5.0 ;;
  5)  instcal "$Q7"                                          qwen2.5-7b       v2q_calinst ;;
  6)  instcal Qwen/Qwen2.5-3B-Instruct                       qwen2.5-3b       v2q3_calinst ;;
  7)  instcal Qwen/Qwen2.5-32B-Instruct                      qwen2.5-32b      v2q32_calinst ;;
  8)  instcal mistralai/Mistral-7B-Instruct-v0.3             mistral-7b       v2m_calinst ;;
  9)  instcal mistralai/Mistral-Small-24B-Instruct-2501      mistral-24b      v2m24_calinst ;;
  10) instcal mistralai/Mistral-Nemo-Instruct-2407           mistral-nemo-12b v2nemo_calinst ;;
  11) instcal mistralai/Mistral-7B-Instruct-v0.2             mistral-7b-v02   v2m7v02_calinst ;;
  12) instcal allenai/OLMo-2-1124-7B-Instruct                olmo2-7b         v2olmo_calinst ;;
  13) instcal meta-llama/Meta-Llama-3-8B-Instruct            llama3-8b        v2l3_calinst ;;
  14) instcal meta-llama/Llama-3.2-3B-Instruct               llama3.2-3b      v2l32_calinst ;;
  15) instcal meta-llama/Llama-3.2-1B-Instruct               llama3.2-1b      v2l321b_calinst ;;
  16) instcal tiiuae/Falcon3-7B-Instruct                     falcon3-7b       v2f3_calinst ;;
  17) instcal HuggingFaceTB/SmolLM2-1.7B-Instruct            smollm2-1.7b     v2sm_calinst ;;
  18) instcal google/gemma-2-9b-it                           gemma2-9b        v2g29_calinst ;;
  19) instcal google/gemma-2-2b-it                           gemma2-2b        v2g22_calinst ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W30] done task $SGE_TASK_ID"
