#!/bin/bash
#$ -M jzheng7@nd.edu
#$ -m abe
#$ -pe smp 4
#$ -q gpu@@zzheng3_Lab
#$ -l gpu_card=1
#$ -N IFH_W1_calib
# gpu@@zzheng3_Lab = H200 hostgroup; other flags copied from old if_attn_pipeline.sh

set -e

#############################
# 0) Paths
#############################
STORE="/store01/yshi4/jzheng7"
BASE_DIR="$STORE/Instruction-Following-Attention-Head-Sensitive"   # code lives here
export HF_HOME="$STORE/hf_cache"                                   # models download here
mkdir -p "$HF_HOME"

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

CONDA_ENV="IFEval"
source ~/.bashrc 2>/dev/null || true
conda activate "$CONDA_ENV"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$BASE_DIR"
mkdir -p runs

#############################
# 1) Calibration: per-head mean vectors + activation salience
#############################
python src/diagnose_heads.py calib \
  --model "$MODEL_ID" \
  --calib-file data/calib_prompts.jsonl \
  --out-dir runs/diag

#############################
# 2) Screening rankings: FRESH per-head FP-vs-quant output deviation
#    (quantizer-agnostic; uses existing GPTQ checkpoints on /store01)
#    3-bit = primary battleground; 4-bit for the tail-failure story.
#############################
# Fresh checkpoints from jobs/w0_quantize.sh (old Quant_Lib ones deleted;
# HF mirror irish-quant/qwen25 exists but integrity unverified — do not use)
GPTQ3="$STORE/models/qwen2.5-7b-gptq3-c4-g128"
GPTQ4="$STORE/models/qwen2.5-7b-gptq4-c4-g128"

python src/diagnose_heads.py screen --model "$MODEL_ID" \
  --quant-model "$GPTQ3" --prompts data/calib_prompts.jsonl \
  --tag gptq3 --out-dir runs
python src/diagnose_heads.py screen --model "$MODEL_ID" \
  --quant-model "$GPTQ4" --prompts data/calib_prompts.jsonl \
  --tag gptq4 --out-dir runs
# -> runs/dev_ranking_gptq3.csv, runs/dev_ranking_gptq4.csv

# Optional sanity cross-check vs the OLD 10-sample attention-deviation signal
# (robustness footnote only; not used downstream):
python - <<'EOF' || echo "[warn] old-ISI cross-check skipped"
import csv, numpy as np
fp = np.load('backup/artifacts/attn_fp16.npz')['mean_layer_head']
q4 = np.load('backup/artifacts/attn_gptq4.npz')['mean_layer_head']
delta = np.abs(fp - q4)
with open('runs/isi_old_ranking.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['layer', 'head', 'score'])
    for l in range(delta.shape[0]):
        for h in range(delta.shape[1]):
            w.writerow([l, h, float(delta[l, h])])
print('wrote runs/isi_old_ranking.csv (cross-check only)')
EOF

#############################
# 3) Stratified 100-prompt screening subset (by first instruction type, seed fixed)
#############################
python - <<'EOF'
import json, random
from collections import defaultdict
rng = random.Random(20260825)
rows = [json.loads(l) for l in open('data/ifeval_input_data.jsonl', encoding='utf-8')]
by_type = defaultdict(list)
for r in rows:
    by_type[r['instruction_id_list'][0].split(':')[0]].append(r)
target, picked = 100, []
for t, grp in sorted(by_type.items()):
    k = max(1, round(target * len(grp) / len(rows)))
    picked += rng.sample(grp, min(k, len(grp)))
picked = picked[:target]
with open('data/ifeval_screen100.jsonl', 'w', encoding='utf-8') as f:
    for r in picked:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('wrote data/ifeval_screen100.jsonl:', len(picked))
EOF

#############################
# 4) Gradient saliency + dissociation overlap (act vs grad vs isi)
#    (causal ranking joins this comparison after W1 ablations finish)
#############################
python src/dissociation.py \
  --model "$MODEL_ID" \
  --grad-calib data/calib_prompts.jsonl \
  --rankings act=runs/diag/act_salience.csv dev3=runs/dev_ranking_gptq3.csv dev4=runs/dev_ranking_gptq4.csv \
  --out runs/dissociation

echo "[W1-calib] done. Next: qsub -hold_jid IFH_W1_calib jobs/w1_ablate.sh"
