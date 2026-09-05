#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W21
#$ -t 1-7
# W21: CONFIG CONFOUNDS + AWQ prediction (Tier 0 #2 and #4).
# Our RTN>GPTQ reversal contradicts the published picture (2404.14047,
# 2608.08188: GPTQ always beats RTN). A quantization reviewer will first ask
# about the config. Already excluded: two implementations, g64, no-actorder,
# calib seed. Still open, closed here:
#   1 asym grid            (sym 3-bit has an asymmetric range: -4..+3 levels)
#   2 instruct calibration (chat-formatted prompts, distribution-shift arm)
#   3 wikitext calibration (the GPTQ paper's alternative corpus)
#   4-7 AWQ-style scaling + RTN (NO compensation), sym and asym, Llama & Q14.
#       Prediction on record: AWQ does NOT collapse. If it does, the story
#       becomes "Llama-3.1 3-bit fragility", not "compensation collapse".
#   qsub jobs/w21_config.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }

case "$SGE_TASK_ID" in
  1) CKPT="$STORE/models/llama3.1-8b-v2gptq3-asym"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect none --asym --out "$CKPT"
     run_ifeval "$CKPT" v2l_asym ;;
  2) CKPT="$STORE/models/llama3.1-8b-v2gptq3-calinst"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect none --calib instruct --out "$CKPT"
     run_ifeval "$CKPT" v2l_calinst ;;
  3) CKPT="$STORE/models/llama3.1-8b-v2gptq3-calwiki"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect none --calib wikitext --out "$CKPT"
     run_ifeval "$CKPT" v2l_calwiki ;;
  4) CKPT="$STORE/models/llama3.1-8b-awq3-none"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect none --quantizer awq --out "$CKPT"
     run_ifeval "$CKPT" awq3l_none ;;
  5) CKPT="$STORE/models/llama3.1-8b-awq3asym-none"
     python src/quantize_protected.py --model "$LLAMA" --bits 3 --group-size 128 \
       --protect none --quantizer awq --asym --out "$CKPT"
     run_ifeval "$CKPT" awq3l_asym_none ;;
  6) CKPT="$STORE/models/qwen2.5-14b-awq3-none"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect none --quantizer awq --out "$CKPT"
     run_ifeval "$CKPT" awq3q14_none ;;
  7) CKPT="$STORE/models/qwen2.5-14b-awq3asym-none"
     python src/quantize_protected.py --model "$Q14" --bits 3 --group-size 128 \
       --protect none --quantizer awq --asym --out "$CKPT"
     run_ifeval "$CKPT" awq3q14_asym_none ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W21] done task $SGE_TASK_ID"
