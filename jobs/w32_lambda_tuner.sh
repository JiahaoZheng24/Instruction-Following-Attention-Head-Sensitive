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
#$ -N IFH_W32
#$ -t 1-20
# W32: after W31 — dampening is NOT a universal recipe (Mistral-7B v0.3/v0.2
# at rho=5 produce byte garbage, .167/.126, far BELOW their RTN .420/.440),
# while token-matched chat calibration was safe everywhere and cured both
# collapses. Two questions:
#  (A) What breaks Mistral at rho=5? Candidates: (i) numerical (non-finite /
#      exploding weights — the outputs look like NaN-style garbage), (ii) the
#      act-order GROUPING: with compensation switched off by heavy damping,
#      GPTQ's limit is "RTN with act-order groups", not plain RTN. Tasks 3-7.
#  (B) Can the deployment-objective ratio E_d(GPTQ)/E_d(RTN) — measured at
#      quantization time from 64 chat prompts — SELECT lambda per model?
#      Stats-only runs at rho in {0.5, 2} (+ existing 0.05/5) for four models,
#      paired with IFEval at the same rho. If argmin E_d tracks the IFEval
#      optimum, the paper's recipe is "validate lambda on a chat Hessian",
#      not "set lambda=5". Tasks 8-19.
#  1-2  GPTAQ packed, option name auto-detected (prints available names and
#       exits 3 if the installed gptqmodel has no GPTAQ at all).
#  20   SmolLM2 rho=5 stats-only + finite check (it lost 4.7 at rho=5).
# Fake-quant checkpoints are deleted after eval.
#   qsub jobs/w32_lambda_tuner.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/gptq_offload_${JOB_ID}_${SGE_TASK_ID}"
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 6h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
mkdir -p runs/stats

packed () {  # $1 model $2 ckpt $3 tag $4.. flags
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_gptq.py --model "$model" --bits 3 --group-size 128 --calib c4 "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
}
fq () {  # $1 model $2 ckpt $3 tag $4.. flags   (deleted after eval)
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
  cp "$CKPT/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CKPT")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CKPT"
}
st () {  # $1 model $2 stats-subdir $3.. flags   (stats only, chat Hessian, finite check)
  local model="$1" sub="$2"; shift 2
  $T python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 --protect none \
    --stats-dir "runs/stats/$sub" --stats-chat-n 64 --no-save "$@" --out "$STORE/models/_unused_$sub"
}

case "$SGE_TASK_ID" in
  1) packed "$LLAMA" llama3.1-8b-gptaq3-c4-g128 gptaq3_llama --v2 ;;
  2) packed "$Q14"   qwen2.5-14b-gptaq3-c4-g128 gptaq3_q14  --v2 ;;
  # (A) Mistral-7B v0.3: ladder + grouping/numerics diagnosis
  3) fq "$M7" mistral-7b-v2gptq3-damp0p5   v2m_damp0p5   --percdamp 0.5 ;;
  4) fq "$M7" mistral-7b-v2gptq3-damp2     v2m_damp2     --percdamp 2 ;;
  5) fq "$M7" mistral-7b-v2gptq3-damp5-noao v2m_damp5_noao --percdamp 5 --no-actorder ;;
  6) fq "$M7" mistral-7b-v2gptq3-damp1e6   v2m_damp1e6   --percdamp 1000000 ;;   # = RTN with act-order groups
  7) st "$M7" mistral-7b-damp5 --percdamp 5 ;;
  # (B) E_d(lambda) vs IFEval(lambda): stats at 0.5 and 2 for four models
  8)  st "$LLAMA" llama31-8b-damp0p5 --percdamp 0.5 ;;
  9)  st "$LLAMA" llama31-8b-damp2   --percdamp 2 ;;
  10) st "$Q14"   qwen25-14b-damp0p5 --percdamp 0.5 ;;
  11) st "$Q14"   qwen25-14b-damp2   --percdamp 2 ;;
  12) st "$Q7"    qwen25-7b-damp0p5  --percdamp 0.5 ;;
  13) st "$Q7"    qwen25-7b-damp5    --percdamp 5 ;;
  14) st "$M7"    mistral-7b-damp0p5 --percdamp 0.5 ;;
  15) st "$M7"    mistral-7b-damp2   --percdamp 2 ;;
  # IFEval at the same rho where missing
  16) fq "$Q14" qwen2.5-14b-v2gptq3-damp0p5 v2q14_damp0p5 --percdamp 0.5 ;;
  17) fq "$Q14" qwen2.5-14b-v2gptq3-damp2   v2q14_damp2   --percdamp 2 ;;
  18) fq "$Q7"  qwen2.5-7b-v2gptq3-damp0p5  v2q_damp0p5   --percdamp 0.5 ;;
  19) fq "$Q7"  qwen2.5-7b-v2gptq3-damp2    v2q_damp2     --percdamp 2 ;;
  20) st HuggingFaceTB/SmolLM2-1.7B-Instruct smollm2-1.7b-damp5 --percdamp 5 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W32] done task $SGE_TASK_ID"
