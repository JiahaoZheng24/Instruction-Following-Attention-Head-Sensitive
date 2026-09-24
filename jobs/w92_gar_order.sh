#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=04:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W92
#$ -t 1-3
# W92 (2026-09-24). Does GPTQModel's group-aware column order (act_group_aware; W91: no collapse inside the
# library) also avoid the collapse inside OUR loop, and does it still write the lesion? Our loop, 3 bits, group
# 128, short documents (128 C4 docs), everything as in the main arms except the column order, with the
# per-matrix mechanism statistics (template-position error GPTQ/RTN, row share, overshoot) that Section 7
# reports for the census, plus IFEval.
#   1  Llama-3.1-8B-Instruct, GAR order        -> runs/stats/f-l-3b-gar,   tag f_l_gar
#   2  Qwen2.5-14B-Instruct,  GAR order        -> runs/stats/f-q14-3b-gar, tag f_q14_gar
#   3  Llama-3.1-8B-Instruct, natural order (fixed loop; the pre-fix run v2l_noactorder gave 0.156)
#                                              -> runs/stats/f-l-3b-noao,  tag f_l_noao
# Predictions (written before running): (i) GAR in our loop is healthy on both models, as in the library;
# (ii) the lesion statistics under GAR are smaller than under act-order (template error ratio below the
# census minimum of 3, overshoot below 2x) - if instead they are as large as under act-order, the
# absorption, not the lesion, is what the column order changes; (iii) natural order collapses Llama
# (0.13-0.18) with a larger lesion than act-order (theory-l-3b-noao: detector 1500 vs 105).
# Reference: act-order arms f-l-3b / f-q14-3b (IFEval 0.155 / 0.426); RTN 0.565 / 0.697.
#   qsub jobs/w92_gar_order.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 220m"
mkdir -p runs/stats runs/protocols

case "$SGE_TASK_ID" in
  1) MODEL="$LLAMA"; ORDER=gar;  TAG="f_l_gar";   ST="runs/stats/f-l-3b-gar" ;;
  2) MODEL="$Q14";   ORDER=gar;  TAG="f_q14_gar"; ST="runs/stats/f-q14-3b-gar" ;;
  3) MODEL="$LLAMA"; ORDER=none; TAG="f_l_noao";  ST="runs/stats/f-l-3b-noao" ;;
  *) echo "bad task id"; exit 1 ;;
esac
CK="$STORE/models/w92_$TAG"

$T python src/quantize_protected.py --model "$MODEL" --bits 3 --group-size 128 --protect none --calib c4 \
   --order "$ORDER" --stats-dir "$ST" --stats-chat-n 64 --out "$CK"
cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
run_ifeval "$CK" "$TAG"
echo "[W92] done task $SGE_TASK_ID ($TAG): $(tail -1 runs/scores_$TAG.csv)"
rm -rf "$CK"   # fake-quant checkpoint, fp16 size; the stats dir and the scores are what we keep
