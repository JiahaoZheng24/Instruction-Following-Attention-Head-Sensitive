#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=12:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W71
#$ -t 1-9
# W71 (2026-09-15). Pre-registered in RESULTS 9.10bl.
# quantize_protected.py carried a DynamicCache into the layer-wise loop (W70 OOM
# diagnosis): the propagation pass of every layer attended the layer's UNQUANTISED
# K/V. Fixed (use_cache=False). This batch (a) finishes W70 for Q14 and (b) re-runs
# the headline arms with the fix to measure how much the numbers move.
#   1  Q14   c4 x1024 rho 0.10   (W70 task 2, GPTQModel README defaults)
#   2  Q14   c4 x1024 rho 0.05   (W70 task 4, control)
#   3  Llama c4 x128  GPTQ3 none          stored .150
#   4  Llama c4 x128  GPTQ3 gw+tn         stored .651
#   5  Q14   c4 x128  GPTQ3 none          stored .412
#   6  Q14   c4 x128  GPTQ3 gw+tn         stored .728
#   7  Llama c4win    GPTQ3 none          stored .643
#   8  Mistral c4win  GPTQ3 none          stored .172
#   9  Llama pile     GPTQ3 none          stored .402
# Checkpoints deleted after scoring.
#   qsub jobs/w71_cache_fix.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 11h"
M7="mistralai/Mistral-7B-Instruct-v0.3"

qs () {   # $1 model $2 ckpt-name $3 tag $4... quantizer flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --out "$CK" "$@"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}

case "$SGE_TASK_ID" in
  1) qs "$Q14"   qwen2.5-14b-v2gptq3-c4x1024-d10  v2q14_c4x1024_d10 --calib c4 --n-calib 1024 --percdamp 0.10 ;;
  2) qs "$Q14"   qwen2.5-14b-v2gptq3-c4x1024-d05  v2q14_c4x1024_d05 --calib c4 --n-calib 1024 --percdamp 0.05 ;;
  3) qs "$LLAMA" llama3.1-8b-v2gptq3-nc-none       v2l_nc_none       --calib c4 ;;
  4) qs "$LLAMA" llama3.1-8b-v2gptq3-nc-gwtn       v2l_nc_gwtn       --calib c4 --hess-grad-weight --hess-token-norm ;;
  5) qs "$Q14"   qwen2.5-14b-v2gptq3-nc-none       v2q14_nc_none     --calib c4 ;;
  6) qs "$Q14"   qwen2.5-14b-v2gptq3-nc-gwtn       v2q14_nc_gwtn     --calib c4 --hess-grad-weight --hess-token-norm ;;
  7) qs "$LLAMA" llama3.1-8b-v2gptq3-nc-c4win      v2l_nc_c4win      --calib c4win ;;
  8) qs "$M7"    mistral7b-v2gptq3-nc-c4win        v2m7_nc_c4win     --calib c4win ;;
  9) qs "$LLAMA" llama3.1-8b-v2gptq3-nc-pile       v2l_nc_pile       --calib pile ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W71] done task $SGE_TASK_ID"
