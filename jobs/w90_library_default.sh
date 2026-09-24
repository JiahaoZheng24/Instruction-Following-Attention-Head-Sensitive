#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=06:00:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W90
#$ -t 1-4
# W90 (2026-09-23). Library default, end to end. The pre-submission review asked whether the collapse
# survives GPTQModel's own default code path (desc_act unset, true_sequential on, damp 0.05, packed
# weights) with the README's calibration recipe (raw C4 rows tokenised one at a time), rather than our
# loop's reproduction of the sampling shape. Four arms, 3 bits, group 128:
#   1  Llama-3.1-8B-Instruct, 1,024 rows (README example)
#   2  Llama-3.1-8B-Instruct,   128 rows (the paper's sample count)
#   3  Qwen2.5-14B-Instruct,  1,024 rows
#   4  Qwen2.5-14B-Instruct,    128 rows
# Reference values from our loop: Llama 0.155 (RTN 0.565), Qwen 0.426 (RTN 0.697); our 1,024-row run 0.148 / 0.435.
# IFEval loads the packed checkpoint through gptqmodel's TORCH backend (the same path every GPTQ arm used).
# Pre-registered prediction (before running): both models collapse under the library default as well
# (IFEval more than 4 points below RTN); if either does not, the paper's "library default" wording is
# narrowed to the sampling shape plus our quantiser settings (Table 1), and the arm is reported either way.
#   qsub jobs/w90_library_default.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
export IFH_OFFLOAD_DIR="${TMPDIR:-/tmp}/ifh_w90_$SGE_TASK_ID"
mkdir -p "$IFH_OFFLOAD_DIR" "$STORE/models"
T="timeout --signal=TERM --kill-after=60 340m"

case "$SGE_TASK_ID" in
  1) MODEL="$LLAMA"; ROWS=1024; TAG="lib_l_1024";   CK="$STORE/models/w90_llama31_gptqmodel_default_1024" ;;
  2) MODEL="$LLAMA"; ROWS=128;  TAG="lib_l_128";    CK="$STORE/models/w90_llama31_gptqmodel_default_128" ;;
  3) MODEL="$Q14";   ROWS=1024; TAG="lib_q14_1024"; CK="$STORE/models/w90_qwen25_14b_gptqmodel_default_1024" ;;
  4) MODEL="$Q14";   ROWS=128;  TAG="lib_q14_128";  CK="$STORE/models/w90_qwen25_14b_gptqmodel_default_128" ;;
  *) echo "bad task id"; exit 1 ;;
esac

python -c "import gptqmodel, transformers, torch; print('[W90] gptqmodel', gptqmodel.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
if [ ! -f "$CK/W90_PROTOCOL.json" ]; then
  $T python src/quantize_gptqmodel_default.py --model "$MODEL" --bits 3 --group-size 128 --n-rows "$ROWS" --out "$CK"
fi
cat "$CK/W90_PROTOCOL.json"
run_ifeval "$CK" "$TAG"
echo "[W90] done task $SGE_TASK_ID ($TAG): $(tail -1 runs/scores_$TAG.csv)"
# Disk: the packed 3-bit checkpoints are 4-7 GB each (about 20 GB for the four), kept by default because
# they are the library's own artefacts and may be re-evaluated in a rebuttal. W90_KEEP=0 deletes after eval.
du -sh "$CK" 2>/dev/null; df -h "$STORE" | tail -1
if [ "${W90_KEEP:-1}" = "0" ]; then rm -rf "$CK"; echo "[W90] checkpoint removed"; fi
