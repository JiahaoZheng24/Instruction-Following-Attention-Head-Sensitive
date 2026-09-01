# Instruction-Preserving Quantization (working title)

Diagnose which attention heads causally carry instruction following (IF) in
instruct LLMs, show weight PTQ disproportionately damages them, and protect
those heads' weight slices at higher precision during quantization.
Target: ARR 2026-10-12 → NAACL 2027. Full plan: [NAACL2027_proposal.md](NAACL2027_proposal.md).

## Layout

```
src/                    new pipeline (W1: diagnosis + dissociation; W2: protection)
  common.py             model loading (GPTQ-aware), head geometry, ablation hooks
  diagnose_heads.py     stages: calib / screen (FP-vs-quant head deviation) / ablate
  dissociation.py       overlap of head rankings: causal vs act vs grad vs dev
  quantize_gptq.py      GPTQModel quantization + smoke test
  protect_eval.py       W2: head-slice protection (post-hoc fp restoration) + eval
  gptq_core.py          W3/v2: masked GPTQ (in-loop protection hold-out)
  quantize_protected.py W3/v2: protected quantization driver -> fake-quant ckpt
  tacq_salience.py      W3/v2: TaCQ saliency |W|*|grad|*|dW| (IF-conditioned)
  score_ifeval.py       official-checker scoring wrapper -> scores CSV
jobs/                   SGE submission scripts (H200 hostgroup gpu@@zzheng3_Lab)
backup/                 old DAS-era pipeline + artifacts (reference only)
paper/                  last submission: latex, figures, OpenReview reviews PDF
third_party/            vendored official IFEval evaluator (google-research)
data/                   ifeval_input_data.jsonl (541), calib_prompts.jsonl (512)
runs/                   experiment outputs (create as needed)
```

## Cluster (CRC, qsub)

Code + models + caches live under `/store01/yshi4/jzheng7` (`HF_HOME` is set by
the job scripts; conda env `IFEval`). All jobs run on the H200 hostgroup
`gpu@@zzheng3_Lab` (set in every `jobs/*.sh`).

```
qsub jobs/w0_quantize.sh                          # fresh GPTQ 2/3/4/8-bit + smoke test
qsub -hold_jid IFH_W0_quantize jobs/w1_quant_baselines.sh  # full-541 IFEval per bit
qsub -hold_jid IFH_W0_quantize jobs/w1_calib.sh   # means + salience + screen + dissociation
qsub -hold_jid IFH_W1_calib jobs/w1_ablate.sh     # task array (11 configs in parallel)
qsub jobs/w2_protect.sh                           # W2: head-slice protection arms (9 tasks, full 541)
qsub jobs/w2b_decompose.sh                        # W2b: qkv/o decomposition + MLP control
qsub jobs/w2c_budget.sh                           # W2c: concentration curve (attn_all, 2-25%)
qsub jobs/w3_salience.sh                          # W3: TaCQ saliency (needed by w3_inloop arm 3/5)
qsub -hold_jid IFH_W3_salience jobs/w3_inloop.sh  # W3/v2: IN-LOOP protected GPTQ, 5 arms
# W2 verdict (2026-08-27): post-hoc restoration at 0.5% budget = null across
# 16 arms (noop bit-identical to baseline). See NAACL2027_proposal.md ⚡ section.
awk 'FNR==1 && NR!=1 {next} 1' runs/scores_*.csv > runs/scores.csv   # merge
```

Quantized checkpoints: always OURS from `jobs/w0_quantize.sh` (GPTQModel,
c4-128×2048, g128, sym, desc_act; protocol saved in each checkpoint's
QUANT_PROTOCOL.json). Old Quant_Lib checkpoints are deleted; the HF mirror
(irish-quant/qwen25) is integrity-unverified — don't use it.
(extra deps: `pip install gptqmodel datasets absl-py langdetect nltk immutabledict`)

⚠ **3-bit garbage bug — root cause found (2026-08-25):** the 3-bit checkpoint
emitted *identical* garbage for different prompts under BOTH Triton and TORCH
backends → the packed weights themselves were corrupt. This matches known
upstream gptqmodel bugs: 3-bit packing regression (fixed v2.0.0) and 3-bit
Triton dequant issues (fixed v5.6.2-12); see
[GPTQModel#1278](https://github.com/ModelCloud/GPTQModel/issues/1278).
**Rules:** (1) pinned stack, verified importable 2026-08-25 — do NOT run bare
`pip install -U` in this env again; any change goes in with `--no-deps` and
must pass the four-package import check:
`torch==2.9.0+cu128` (never touch) · `gptqmodel==5.6.12` (has the 3-bit fix) ·
`transformers==4.57.6` · `huggingface_hub==0.36.2` · `torchao==0.16.0`
(cpp-ext skip warning is harmless) · `kernels` UNINSTALLED (0.16.1 needs
hub≥1.0 and crashes transformers 4.x import). Re-quantize ALL bits on this
stack (recorded in each checkpoint's QUANT_PROTOCOL.json);
(2) every GPTQ load still goes through `common.load_model` with the
plain-PyTorch backend (`IFH_GPTQ_BACKEND=TORCH` default) as belt-and-braces;
(3) never trust a checkpoint whose smoke test wasn't inspected. The old
"gptq3 garbled / drops 15 pts" numbers remain untrusted; true numbers come
from `jobs/w1_quant_baselines.sh` after re-quantization.

Selectivity read-out: Δ(top32_dev3 − screen_base) vs Δ(rand32_s* − screen_base).

Bit-width strategy: 3-bit GPTQ = primary quantitative battleground (pending
re-measurement on clean checkpoints); 2-bit = boundary analysis only (CASIA:
computation collapse, training-free repair hopeless); 4-bit = tail-failure
story (flips / per-constraint metrics, not averages). AWQ ecosystem is
4-bit-only, so low-bit runs use GPTQ + RTN; AWQ enters at 4-bit for quantizer
generality.

## W1 workflow

1. **Data (prepared 2026-08-25).**
   `data/ifeval_input_data.jsonl`: official IFEval input, 541 prompts.
   `data/calib_prompts.jsonl`: 512 Alpaca instructions (no-input subset,
   30–1200 chars, seed 20260825), prefix-deduplicated against IFEval.
2. **Calibrate:** `python src/diagnose_heads.py calib --calib-file data/calib_prompts.jsonl`
3. **Baseline + ablations:**
   ```
   python src/diagnose_heads.py ablate --prompts data/ifeval_input_data.jsonl --tag baseline
   python src/diagnose_heads.py ablate --prompts data/ifeval_input_data.jsonl \
       --topk-from runs/dev_ranking_gptq3.csv --k 32 --tag top32_dev3
   python src/diagnose_heads.py ablate --prompts data/ifeval_input_data.jsonl \
       --random 32 --seed 0 --tag rand32_s0
   ```
   Screening ranking: `runs/dev_ranking_gptq3.csv` — fresh per-head FP-vs-quant
   output deviation from `diagnose_heads.py screen` (quantizer-agnostic; works
   with any GPTQ/AWQ/RTN checkpoint). Old 10-sample ISI artifacts are kept only
   as a robustness cross-check (`runs/isi_old_ranking.csv`), never used downstream.
4. **Score with the OFFICIAL checker** (vendored in `third_party/`, smoke-tested):
   ```
   python src/score_ifeval.py --responses runs/<model>/<tag>/responses.jsonl --tag <tag>
   ```
   Appends prompt/inst × strict/loose + avg4 to `runs/scores.csv`.
5. **Dissociation (key insight experiment — run this first):**
   ```
   python src/dissociation.py --grad-calib data/calib_prompts.jsonl \
     --rankings act=runs/diag/act_salience.csv dev3=runs/dev_ranking_gptq3.csv \
     --out runs/dissociation
   ```
   Low Jaccard between causal and act/grad ⇒ salience-based protection cannot
   find IF-heads ⇒ targeted protection is necessary (the paper's core card).

## Protocol (fixed 2026-08, do not tune post-hoc)

- Head unit = query head (GQA: k/v handled per kv-group; see common.py).
- Mean ablation with calibration-set means; greedy decoding; max_new_tokens=1280.
- Screening on 100 stratified IFEval prompts; validation on full 541.
- IF-head criterion: Δ(prompt strict acc) under ablation, averaged over ≥3
  random-control comparisons; report per-constraint-type breakdown.
- GPTQ inference backend: TORCH only (see kernel-bug rule above).
- Go/No-Go (9/7): top-k causal heads show selective IF damage vs random-k
  (else pivot to the systematic-analysis paper, plan §6).
