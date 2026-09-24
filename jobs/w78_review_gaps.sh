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
#$ -N IFH_W78
#$ -t 1-15
# W78 (2026-09-17). Reviewer-facing gaps, pre-registered in RESULTS 9.10bq.
# Every arm logs wall-clock (seconds) and peak GPU memory of the quantisation.
#   1-3    second calibration seed for the three COLLAPSES themselves (only the cures had seeds)
#   4-5    chat-formatted calibration on the Mistral c4win collapse: c4chat, ultrachat
#   6-8    4-bit chat-formatted calibration where GPTQ4 loses to RTN4: Mistral-v0.3, Qwen3-14B, Hermes-3
#   9-10   corrected objective at rho = 0.01 and 0.2 (damping sensitivity), Llama short-c4
#   11-15  Table 1 gaps: Llama pile tnorm; Qwen pile G-only; capped corrected objective on Qwen, Mistral c4win, Llama
#   qsub jobs/w78_review_gaps.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 9h"
M7="mistralai/Mistral-7B-Instruct-v0.3"
Q3="Qwen/Qwen3-14B"
HERMES="NousResearch/Hermes-3-Llama-3.1-8B"
GWTN="--hess-grad-weight --hess-token-norm"
mkdir -p runs/protocols

qz () {   # $1 model $2 ckpt $3 tag $4 bits $5.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; bits="$4"; shift 4
  t0=$SECONDS
  $T python src/quantize_protected.py --model "$m" --bits "$bits" --group-size 128 --protect none --out "$CK" "$@"
  echo "[W78-timing] tag=$tag quantise_seconds=$((SECONDS - t0))"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}

case "$SGE_TASK_ID" in
  1) qz "$LLAMA"  f-l-gptq3-none-cs1        f_l_none_cs1        3 --calib c4    --calib-seed 1 ;;
  2) qz "$Q14"    f-q14-gptq3-none-cs1      f_q14_none_cs1      3 --calib c4    --calib-seed 1 ;;
  3) qz "$M7"     f-m7-gptq3-c4win-cs1      f_m7_c4win_cs1      3 --calib c4win --calib-seed 1 ;;
  4) qz "$M7"     f-m7-gptq3-c4chat         f_m7_c4chat         3 --calib c4chat ;;
  5) qz "$M7"     f-m7-gptq3-ultrachat      f_m7_ultrachat      3 --calib ultrachat ;;
  6) qz "$M7"     f-m7-gptq4-c4chat         f_m7_gptq4_c4chat   4 --calib c4chat ;;
  7) qz "$Q3"     f-qwen3_14b-gptq4-c4chat  f_qwen3_14b_gptq4_c4chat 4 --calib c4chat ;;
  8) qz "$HERMES" f-hermes3-gptq4-c4chat    f_hermes3_gptq4_c4chat 4 --calib c4chat ;;
  9) qz "$LLAMA"  f-l-gptq3-gwtn-d0p01      f_l_gwtn_d0p01      3 --calib c4 $GWTN --percdamp 0.01 ;;
 10) qz "$LLAMA"  f-l-gptq3-gwtn-d0p2       f_l_gwtn_d0p2       3 --calib c4 $GWTN --percdamp 0.2 ;;
 11) qz "$LLAMA"  f-l-gptq3-pile-tnorm      f_l_pile_tnorm      3 --calib pile --hess-token-norm ;;
 12) qz "$Q14"    f-q14-gptq3-pile-gw       f_q14_pile_gw       3 --calib pile --hess-grad-weight ;;
 13) qz "$Q14"    f-q14-gptq3-gwtncap       f_q14_gwtncap        3 --calib c4 $GWTN --hess-grad-cap 100 ;;
 14) qz "$M7"     f-m7-gptq3-c4win-gwtncap f_m7_c4win_gwtncap 3 --calib c4win $GWTN --hess-grad-cap 100 ;;
 15) qz "$LLAMA"  f-l-gptq3-gwtncap         f_l_gwtncap         3 --calib c4 $GWTN --hess-grad-cap 100 ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W78] done task $SGE_TASK_ID"
