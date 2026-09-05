#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W28
#$ -t 1-19
# W28: CLOSURE batch after the W20-W27 verdicts (2026-09-04 evening).
# What W20-W27 established: dampening alone cures BOTH collapses (Llama damp5
# 0.642, Q14 damp5 0.786 = the best protection arms); calibration corpus
# alone cures Llama (wikitext 0.624, instruct 0.611 vs c4 0.150); AWQ never
# collapses; gradient-free |W|sqrt(H_ii) rescues like TaCQ; the group-scale
# artifact explained "under-budget hurts"; graceful models cannot be pushed
# into collapse; under the CHAT Hessian GPTQ loses to RTN in 124/224 Llama
# modules (median ratio 1.02) vs 34/196 in Qwen-7B -> mechanism = the
# compensation over-fits a calibration Hessian that does not transfer.
# This batch closes the remaining gaps:
#   1-5   Llama damping ladder completion {2,3,10,20,50}: locate the
#         threshold between 1 (0.152) and 5 (0.642) and show the descent
#         back toward RTN (0.565) -> the inverted-U "regularisation" curve
#   6-8   Q14 ladder completion {0.1,20,100} (threshold in (0.05,0.2]; descent)
#   9-10  Qwen-7B damp {5,20}: does damping HELP a graceful model (like tacq
#         +3 n.s.) or hurt it? -> "protection ~ regularisation" symmetry test
#   11-13 Calibration disentangling on Llama: c4 x 512 docs (more tokens,
#         same corpus); c4 seed 2 (third disjoint sample); wikitext x 32
#         windows (token count matched to c4's ~64k)
#   14-15 Q14 with wikitext / instruct calibration (does the corpus cure
#         replicate on collapse #2?)
#   16    re-create the deleted Llama RTN checkpoint + divergence curves for
#         RTN, damp5, calwiki (do the cures restore the early layers?)
#   17-18 mechanism logs for the CURED Llama arms (damp5, wikitext): the
#         chat-Hessian objective ratio should drop below 1 -> closes the loop
#   19    Q14 RTN divergence (checkpoint exists) + comp_stats re-aggregation
#   qsub jobs/w28_closure.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
mkdir -p runs/stats

damp_arm () {  # $1 model $2 ckpt $3 tag $4 percdamp
  CKPT="$STORE/models/$2"
  python src/quantize_protected.py --model "$1" --bits 3 --group-size 128 \
    --protect none --percdamp "$4" --out "$CKPT"
  run_ifeval "$CKPT" "$3"
}
arm () {  # $1 model $2 ckpt $3 tag $4.. flags
  local model="$1" ckpt="$2" tag="$3"; shift 3
  CKPT="$STORE/models/$ckpt"
  python src/quantize_protected.py --model "$model" --bits 3 --group-size 128 \
    --protect none "$@" --out "$CKPT"
  run_ifeval "$CKPT" "$tag"
}
div () {  # $1 fp16 $2 ckpt $3 out
  python src/divergence.py --fp16 "$1" --quant "$2" --prompts "$FULL" --n 32 --out "$3"
}

case "$SGE_TASK_ID" in
  1) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp2  v2l_damp2  2 ;;
  2) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp3  v2l_damp3  3 ;;
  3) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp10 v2l_damp10 10 ;;
  4) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp20 v2l_damp20 20 ;;
  5) damp_arm "$LLAMA" llama3.1-8b-v2gptq3-damp50 v2l_damp50 50 ;;
  6) damp_arm "$Q14" qwen2.5-14b-v2gptq3-damp0p1 v2q14_damp0p1 0.1 ;;
  7) damp_arm "$Q14" qwen2.5-14b-v2gptq3-damp20  v2q14_damp20  20 ;;
  8) damp_arm "$Q14" qwen2.5-14b-v2gptq3-damp100 v2q14_damp100 100 ;;
  9) damp_arm "$Q7" qwen2.5-7b-v2gptq3-damp5  v2q_damp5  5 ;;
  10) damp_arm "$Q7" qwen2.5-7b-v2gptq3-damp20 v2q_damp20 20 ;;
  11) arm "$LLAMA" llama3.1-8b-v2gptq3-c4x512  v2l_c4x512  --n-calib 512 ;;
  12) arm "$LLAMA" llama3.1-8b-v2gptq3-none_cs2 v2l_none_cs2 --calib-seed 2 ;;
  13) arm "$LLAMA" llama3.1-8b-v2gptq3-calwiki32 v2l_calwiki32 --calib wikitext --n-calib 32 ;;
  14) arm "$Q14" qwen2.5-14b-v2gptq3-calwiki v2q14_calwiki --calib wikitext ;;
  15) arm "$Q14" qwen2.5-14b-v2gptq3-calinst v2q14_calinst --calib instruct ;;
  16) CKPT="$STORE/models/llama3.1-8b-rtn3-none"
      [ -d "$CKPT" ] || python src/quantize_protected.py --model "$LLAMA" --bits 3 \
        --group-size 128 --protect none --quantizer rtn --out "$CKPT"
      div "$LLAMA" "$CKPT" runs/div_l_rtn.csv
      div "$LLAMA" "$STORE/models/llama3.1-8b-v2gptq3-damp5"   runs/div_l_damp5.csv
      div "$LLAMA" "$STORE/models/llama3.1-8b-v2gptq3-calwiki" runs/div_l_calwiki.csv ;;
  17) python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
        --protect none --percdamp 5 --stats-dir runs/stats/llama31-8b-damp5 \
        --stats-chat-n 64 --no-save --out "$STORE/models/_unused_l_damp5" ;;
  18) python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
        --protect none --calib wikitext --stats-dir runs/stats/llama31-8b-calwiki \
        --stats-chat-n 64 --no-save --out "$STORE/models/_unused_l_calwiki" ;;
  19) div "$Q14" "$STORE/models/qwen2.5-14b-rtn3-none" runs/div_q14_rtn.csv
      python src/comp_stats.py --runs llama=runs/stats/llama31-8b q7=runs/stats/qwen25-7b \
        q14=runs/stats/qwen25-14b --summary runs/comp_stats_summary.csv \
        --per-layer runs/comp_stats_layers.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W28] done task $SGE_TASK_ID"
