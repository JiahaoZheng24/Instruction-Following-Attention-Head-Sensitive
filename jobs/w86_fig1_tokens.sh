#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -l h_rt=00:30:00
#$ -notify
#$ -j y
#$ -cwd
#$ -o logs/
#$ -N IFH_W86
#$ -t 1-2
# W86 (2026-09-21). Real per-token data for Figure 1(b): along ONE chat exchange, at the sink-forming
# down_proj, the input energy ||x_t||^2 (what the layer-wise objective weights by) and the output
# sensitivity g_t = ||dL/dy_t||^2 (what the loss depends on), fp16 model, loss = summed next-token
# loss over prompt and response. Minutes on one card. The prompt is short on purpose: the figure
# shows the ten tokens up to <|eot_id|>; the assistant header and a short reply follow so that the
# prompt tokens' g_t is measured with something after them. No pre-registration: this is a measurement
# for a figure, not a test; the calibration-set shares are already in runs/sides/l_c4.csv (.998/.0003).
#   1  Llama-3.1-8B-Instruct, its own chat template, layer auto (expected 1)
#   2  Qwen2.5-14B-Instruct, its own chat template, layer auto (expected 4)
#   qsub jobs/w86_fig1_tokens.sh
source "/store01/yshi4/jzheng7/Instruction-Following-Attention-Head-Sensitive/jobs/_w2x_header.sh" || { echo "header not found"; exit 3; }
export TOKENIZERS_PARALLELISM=false
T="timeout --signal=TERM --kill-after=60 25m"
mkdir -p runs/sides

# the tokenizer adds BOS; \n is expanded by the script
L_TEXT='<|start_header_id|>user<|end_header_id|>\n\nWrite a poem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe sea is blue.'
Q_TEXT='<|im_start|>user\nWrite a poem.<|im_end|>\n<|im_start|>assistant\nThe sea is blue.'

case "$SGE_TASK_ID" in
  1) $T python src/grad_weights.py --model "$LLAMA" --tokens-text "$L_TEXT" --out runs/sides/f_l_fig1_tokens.csv ;;
  2) $T python src/grad_weights.py --model "$Q14"   --tokens-text "$Q_TEXT" --out runs/sides/f_q14_fig1_tokens.csv ;;
  *) echo "bad task id"; exit 1 ;;
esac
echo "[W86] done task $SGE_TASK_ID"
