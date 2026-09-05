#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=10:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W31
#$ -t 1-24
# W31: THE PRACTICAL RECIPE, tested on the whole census (after W30 verdicts).
# W30 showed prompt-only instruct calibration is NOT a safe recipe (−10 on
# Nemo, −7.5 on Llama-3.2-3B, −5.4 on gemma-2-2b) even though it cures both
# collapses; and gptqmodel rejects damp_percent>1 and has no `v2` kwarg.
#  1-2   GPTAQ (gptqmodel METHOD.GPTAQ, alpha 0.25) packed, Llama & Q14:
#        does the 2025 OBS descendant collapse on c4?
#  3-4   gptqmodel packed at its MAXIMUM allowed dampening (0.99), Llama & Q14:
#        real-kernel checkpoints; prediction: Q14 cured (rho=1 -> .748 in
#        fake-quant), Llama still collapsed (rho=1 -> .152; needs >=2, which
#        the library forbids) -> "the library cap is below the cure".
#  5-10  token-matched CHAT calibration (ultrachat_200k, 128 conversations
#        with responses, 2048 tokens each) on the two collapses, Qwen-7B, and
#        the three models the prompt-only set hurt most. Separates
#        "in-distribution" from "too few tokens".
#  11-24 dampening rho=5 on the 14 census models that lack it: does the
#        cure ever hurt? If never by more than noise, "set damping >= 2" is a
#        zero-cost recommendation backed by 17 models.
# Fake-quant checkpoints are deleted after eval; packed ones are kept.
#   qsub jobs/w31_recipe.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"

packed () {  # $1 model $2 ckpt $3 tag $4.. flags for quantize_gptq.py
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_gptq.py --model "$model" --bits 3 --group-size 128 --calib c4 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
}
fq () {  # $1 model $2 ckpt $3 tag $4.. flags for quantize_protected.py  (deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}

case "$SGE_TASK_ID" in
  1) packed "$LLAMA" llama3.1-8b-gptaq3-c4-g128 gptaq3_llama --v2 ;;
  2) packed "$Q14"   qwen2.5-14b-gptaq3-c4-g128 gptaq3_q14  --v2 ;;
  3) packed "$LLAMA" llama3.1-8b-gptq3-damp0p99-packed gptq3_llama_damp0p99_packed --damp-percent 0.99 ;;
  4) packed "$Q14"   qwen2.5-14b-gptq3-damp0p99-packed gptq3_q14_damp0p99_packed  --damp-percent 0.99 ;;
  5)  fq "$LLAMA" llama3.1-8b-v2gptq3-calchat   v2l_calchat   --calib ultrachat ;;
  6)  fq "$Q14"   qwen2.5-14b-v2gptq3-calchat   v2q14_calchat --calib ultrachat ;;
  7)  fq "$Q7"    qwen2.5-7b-v2gptq3-calchat    v2q_calchat   --calib ultrachat ;;
  8)  fq mistralai/Mistral-Nemo-Instruct-2407  mistral-nemo-12b-v2gptq3-calchat v2nemo_calchat --calib ultrachat ;;
  9)  fq meta-llama/Llama-3.2-3B-Instruct      llama3.2-3b-v2gptq3-calchat      v2l32_calchat  --calib ultrachat ;;
  10) fq google/gemma-2-2b-it                  gemma2-2b-v2gptq3-calchat        v2g22_calchat  --calib ultrachat ;;
  11) fq Qwen/Qwen2.5-3B-Instruct                  qwen2.5-3b-v2gptq3-damp5       v2q3_damp5     --percdamp 5 ;;
  12) fq Qwen/Qwen2.5-32B-Instruct                 qwen2.5-32b-v2gptq3-damp5      v2q32_damp5    --percdamp 5 ;;
  13) fq mistralai/Mistral-7B-Instruct-v0.3        mistral-7b-v2gptq3-damp5       v2m_damp5      --percdamp 5 ;;
  14) fq mistralai/Mistral-Small-24B-Instruct-2501 mistral-24b-v2gptq3-damp5      v2m24_damp5    --percdamp 5 ;;
  15) fq mistralai/Mistral-Nemo-Instruct-2407      mistral-nemo-12b-v2gptq3-damp5 v2nemo_damp5   --percdamp 5 ;;
  16) fq mistralai/Mistral-7B-Instruct-v0.2        mistral-7b-v02-v2gptq3-damp5   v2m7v02_damp5  --percdamp 5 ;;
  17) fq allenai/OLMo-2-1124-7B-Instruct           olmo2-7b-v2gptq3-damp5         v2olmo_damp5   --percdamp 5 ;;
  18) fq meta-llama/Meta-Llama-3-8B-Instruct       llama3-8b-v2gptq3-damp5        v2l3_damp5     --percdamp 5 ;;
  19) fq meta-llama/Llama-3.2-3B-Instruct          llama3.2-3b-v2gptq3-damp5      v2l32_damp5    --percdamp 5 ;;
  20) fq meta-llama/Llama-3.2-1B-Instruct          llama3.2-1b-v2gptq3-damp5      v2l321b_damp5  --percdamp 5 ;;
  21) fq tiiuae/Falcon3-7B-Instruct                falcon3-7b-v2gptq3-damp5       v2f3_damp5     --percdamp 5 ;;
  22) fq HuggingFaceTB/SmolLM2-1.7B-Instruct       smollm2-1.7b-v2gptq3-damp5     v2sm_damp5     --percdamp 5 ;;
  23) fq google/gemma-2-9b-it                      gemma2-9b-v2gptq3-damp5        v2g29_damp5    --percdamp 5 ;;
  24) fq google/gemma-2-2b-it                      gemma2-2b-v2gptq3-damp5        v2g22_damp5    --percdamp 5 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W31] done task $SGE_TASK_ID"
