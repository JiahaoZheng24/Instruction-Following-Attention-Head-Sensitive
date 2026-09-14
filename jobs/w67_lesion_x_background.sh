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
#$ -N IFH_W67
#$ -t 1-8
# W67: lesion x background, causally. W66 showed the fp16 network TOLERATES the
# synthetic lesion (BOS-direction leakage at template positions after the sink
# layer) up to beta=4 (Llama .770, Q14 .821), although the real GPTQ artifact at
# Llama position 4 is exactly that (cos .90, 3.85x). So the lesion alone is not
# fatal; W57 says the decision is the quantised background of layers 2-4. Test:
# inject the synthetic lesion into models that HAVE the quantised background but
# NOT the lesion (RTN3; the zero-bit cure = GPTQ3 with the sink down_proj RTN),
# and into the c4win GPTQ3 model (lesion present, no collapse). Pre-registered in
# RESULTS 9.10bd. Checkpoints deleted after use.
#   qsub jobs/w67_lesion_x_background.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=120 10h"

score () {   # $1 model-or-ckpt  $2 tag
  $T python src/score_ifeval.py --responses "runs/$(basename "$1")/$2/responses.jsonl" \
     --input-data "$FULL" --tag "$2" --scores-csv "runs/scores_$2.csv"
}
inj () {   # $1 ckpt $2 layer $3 beta $4 tag
  $T python src/inject_sink.py inject --model "$1" --prompts "$FULL" --layer "$2" --beta "$3" --tag "$4" --batch 16
  score "$1" "$4"
}
q () {   # $1 model $2 ckpt-name $3... quantize flags   -> echoes ckpt path
  CK="$STORE/models/$2"; m="$1"; shift 2
  $T python src/quantize_protected.py --model "$m" --group-size 128 --protect none --out "$CK" "$@" >&2
  echo "$CK"
}

case "$SGE_TASK_ID" in
  1) CK=$(q "$LLAMA" llama3.1-8b-rtn3-inj --bits 3 --rtn);                           inj "$CK" 1 2 inj_l_rtn_b2;    rm -rf "$CK" ;;
  2) CK=$(q "$LLAMA" llama3.1-8b-rtn3-inj4 --bits 3 --rtn);                          inj "$CK" 1 4 inj_l_rtn_b4;    rm -rf "$CK" ;;
  3) CK=$(q "$LLAMA" llama3.1-8b-cure-inj --bits 3 --calib c4 --rtn-modules "1:down_proj");  inj "$CK" 1 2 inj_l_cure_b2; rm -rf "$CK" ;;
  4) CK=$(q "$LLAMA" llama3.1-8b-cure-inj4 --bits 3 --calib c4 --rtn-modules "1:down_proj"); inj "$CK" 1 4 inj_l_cure_b4; rm -rf "$CK" ;;
  5) CK=$(q "$LLAMA" llama3.1-8b-cure-inj8 --bits 3 --calib c4 --rtn-modules "1:down_proj"); inj "$CK" 1 8 inj_l_cure_b8; rm -rf "$CK" ;;
  6) CK=$(q "$LLAMA" llama3.1-8b-c4win-inj --bits 3 --calib c4win);                 inj "$CK" 1 2 inj_l_c4win_b2;  rm -rf "$CK" ;;
  7) CK=$(q "$Q14"   qwen2.5-14b-rtn3-inj --bits 3 --rtn);                           inj "$CK" 4 4 inj_q14_rtn_b4;  rm -rf "$CK" ;;
  8) CK=$(q "$Q14"   qwen2.5-14b-cure-inj --bits 3 --calib c4 --rtn-modules "4:down_proj");  inj "$CK" 4 4 inj_q14_cure_b4; rm -rf "$CK" ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W67] done task $SGE_TASK_ID"
