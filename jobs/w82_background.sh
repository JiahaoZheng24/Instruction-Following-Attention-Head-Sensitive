#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=08:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W82
#$ -t 1-9
# W82 (2026-09-19). WHERE IS THE BACKGROUND? Pre-registered in RESULTS 9.10bu.
#   1-5  RTN tail: the collapsed Llama configuration (GPTQ3 short-c4, lesion kept) with every
#        matrix from layer k onward re-rounded RTN, k in {2, 4, 8, 16, 24} -> IFEval tags f_l_rtntail_k
#   6    seed-1 Llama (the draw that does NOT collapse, .531): lesion statistics (stats mode)
#   7    seed-1 Llama: per-token readout on the real seed-1 checkpoint -> runs/sink/f_l_cs1_readout.csv
#   8    composition of every calibration draw (Llama tokenizer): newline / special / doc-initial shares
#   9    the same with the Qwen tokenizer
#   qsub jobs/w82_background.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 7h"
ALL="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
mkdir -p runs/theory runs/sink runs/stats runs/protocols

qz () {   # $1 model $2 ckpt $3 tag $4.. flags
  CK="$STORE/models/$2"; m="$1"; tag="$3"; shift 3
  $T python src/quantize_protected.py --model "$m" --bits 3 --group-size 128 --protect none --calib c4 --out "$CK" "$@"
  run_ifeval "$CK" "$tag"
  cp "$CK/PROTECT_PROTOCOL.json" "runs/protocols/$(basename "$CK")_PROTECT_PROTOCOL.json" 2>/dev/null || true
  rm -rf "$CK"
}

id=$SGE_TASK_ID
if [ "$id" -le 5 ]; then
  KS=(2 4 8 16 24); k="${KS[$((id - 1))]}"
  qz "$LLAMA" "f-l-gptq3-rtntail-$k" "f_l_rtntail_$k" --rtn-modules "$k-31:$ALL"
elif [ "$id" -eq 6 ]; then
  $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --calib c4 --calib-seed 1 \
     --stats-dir runs/stats/f-l-3b-cs1 --stats-chat-n 64 --no-save --out "$STORE/models/_unused_f_stats_l_cs1"
elif [ "$id" -eq 7 ]; then
  WH="runs/theory/f_l_c4_cs1_whiten.pt"
  $T python src/theory_tests.py --model "$LLAMA" --calib c4 --calib-seed 1 --tag f_l_c4_cs1 --save-whiten "$WH"
  CK="$STORE/models/f-l-gptq3-cs1-readout"
  $T python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 --protect none --calib c4 --calib-seed 1 --out "$CK"
  $T python src/inject_sink.py artifact --model "$LLAMA" --quant "$CK" --prompts "$FULL" --n 16 --layer 1 --continuation 128 \
     --whiten "$WH" --out runs/sink/f_l_cs1_readout.csv --out-tokens runs/sink/f_l_cs1_readout_tokens.csv
  rm -rf "$CK"
elif [ "$id" -eq 8 ]; then
  $T python src/calib_composition.py --model "$LLAMA" --arms "c4:0,c4:1,c4:2,c4:3,c4:4,c4win:0,pile:0,wikitext:0,pileshort:0" \
     --out runs/theory/f_l_calib_composition.csv
else
  $T python src/calib_composition.py --model "$Q14" --arms "c4:0,c4:1,c4:2,c4:3,c4win:0,pile:0,wikitext:0,pileshort:0" \
     --out runs/theory/f_q14_calib_composition.csv
fi
echo "[W82] done task $SGE_TASK_ID"
