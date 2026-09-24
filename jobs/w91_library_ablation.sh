#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=03:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W91
#$ -t 1-7
# W91 (2026-09-23). W90 refuted the pre-registered prediction: GPTQModel 5.6.12's own default path does NOT
# collapse Llama (0.646 / 0.665) or Qwen (0.759 / 0.745), where our loop on the same sampling shape gives
# 0.155 / 0.426. One setting at a time, Llama-3.1-8B-Instruct, 3 bits, group 128, 128 rows, everything else
# at the library default; IFEval through the usual harness (TORCH backend).
#   1  desc_act=True                      (our act-order)
#   2  true_sequential=False              (our block-input Hessians)
#   3  act_group_aware=False              (library's group-aware reordering off)
#   4  damp_auto_increment=0              (fixed rho, as in our loop)
#   5  desc_act=True true_sequential=False act_group_aware=False   (our loop's settings inside the library)
#   6  default settings, calibration = the paper's C4 short-document stream instead of the README shard
#   7  default settings, README rows pre-tokenized by us with BOS (bypasses the library's text handling)
# Predictions (written before running): if the cause is a quantizer setting, arm 5 collapses and one of 1-4
# names it; if arms 1-5 stay healthy and 6 or 7 collapses, the cause is in how the rows are fed (BOS or
# sampling), not in the loop. If everything stays healthy, the collapse is specific to our loop and the paper
# must say so.
#   qsub jobs/w91_library_ablation.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/ifh_w91_$SGE_TASK_ID"
mkdir -p "$IFH_OFFLOAD_DIR" "$STORE/models"
T="timeout --signal=TERM --kill-after=60 150m"
Q="python src/quantize_gptqmodel_ablate.py --model $LLAMA --bits 3 --group-size 128 --n-rows 128"

case "$SGE_TASK_ID" in
  1) TAG="lib_l_descact";  ARGS="--set desc_act=True" ;;
  2) TAG="lib_l_notseq";   ARGS="--set true_sequential=False" ;;
  3) TAG="lib_l_nogar";    ARGS="--set act_group_aware=False" ;;
  4) TAG="lib_l_fixdamp";  ARGS="--set damp_auto_increment=0" ;;
  5) TAG="lib_l_ourcfg";   ARGS="--set desc_act=True --set true_sequential=False --set act_group_aware=False" ;;
  6) TAG="lib_l_papercal"; ARGS="--calib paper" ;;
  7) TAG="lib_l_pretok";   ARGS="--pretokenize" ;;
  *) echo "bad task id"; exit 1 ;;
esac
CK="$STORE/models/w91_$TAG"

if [ ! -f "$CK/W91_PROTOCOL.json" ]; then
  $T $Q $ARGS --out "$CK"
fi
cat "$CK/W91_PROTOCOL.json"
run_ifeval "$CK" "$TAG"
echo "[W91] done task $SGE_TASK_ID ($TAG): $(tail -1 runs/scores_$TAG.csv)"
du -sh "$CK" 2>/dev/null; df -h "$STORE" | tail -1
if [ "${W91_KEEP:-0}" = "0" ]; then rm -rf "$CK"; echo "[W91] checkpoint removed (W91_KEEP=1 keeps it)"; fi
