# Attention Compensation for Quantized Models

This repository contains code to test whether **attention compensation** can repair instruction-following degradation in quantized language models.

## 📋 Overview

**Problem:** Quantization (4-bit, 3-bit) degrades instruction-following capabilities.

**Hypothesis:** Degradation is caused by attention weight changes in critical heads.

**Solution:** Artificially boost attention weights on degraded heads during inference.

## 🚀 Quick Start

### Prerequisites

1. **Environment setup** (one-time):
   ```bash
   conda create -n attention python=3.10
   conda activate attention
   pip install torch transformers accelerate auto-gptq optimum
   pip install pandas numpy scipy matplotlib tqdm pyyaml
   ```

2. **Prepare data**:
   - Run IFEval benchmark on FP16 model → `fp16_samples.jsonl`
   - Run IFEval benchmark on quantized model → `quant_samples.jsonl`

### Running Experiments

**Option 1: Automated pipeline (recommended)**
```bash
# 1. Edit configuration section at top of script
vim run_compensation_experiment.sh

# 2. Submit to cluster
qsub run_compensation_experiment.sh

# 3. Check results
cat artifacts/your_experiment_name/summary.csv
```

**Option 2: Manual step-by-step execution**

Useful for debugging or running specific stages:

```bash
# Setup environment
conda activate attention
export BASE_DIR="/users/jzheng7/ifattn"
export OUTPUT_DIR="$BASE_DIR/artifacts/my_test"
mkdir -p "$OUTPUT_DIR"

# Step 1: Select samples
python select_samples_for_compensation.py \
  --fp16_samples /path/to/fp16_samples.jsonl \
  --quant_samples /path/to/quant_samples.jsonl \
  --strategy failure_only \
  --max_samples 5 \
  --output $OUTPUT_DIR/selected_prompts.jsonl

# Step 2: Extract FP16 attention
python dump_attn.py \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --run_tag "fp16" \
  --prompts_jsonl $OUTPUT_DIR/selected_prompts.jsonl \
  --out_dir $OUTPUT_DIR

# Step 3: Extract quantized attention
python dump_attn.py \
  --model_id "/path/to/quantized_model" \
  --run_tag "gptq4" \
  --prompts_jsonl $OUTPUT_DIR/selected_prompts.jsonl \
  --out_dir $OUTPUT_DIR

# Step 4: Identify critical heads
python identify_critical_heads.py \
  --fp16_attn $OUTPUT_DIR/attn_fp16.npz \
  --quant_attn $OUTPUT_DIR/attn_gptq4.npz \
  --top_k 10 \
  --out_dir $OUTPUT_DIR

# Step 5: Run compensation (repeat for each alpha)
python eval_ifeval_with_compensation.py \
  --model_path "/path/to/quantized_model" \
  --ifeval_data $OUTPUT_DIR/selected_prompts.jsonl \
  --attn_fp16 $OUTPUT_DIR/attn_fp16.npz \
  --attn_quant $OUTPUT_DIR/attn_gptq4.npz \
  --top_heads $OUTPUT_DIR/critical_heads_gptq4.json \
  --alpha_list 0.0 \
  --max_samples 5 \
  --max_new_tokens 1280 \
  --output $OUTPUT_DIR/compensation_alpha0.0.jsonl

# Step 6: Analyze results
python analyze_compensation_results.py \
  --results $OUTPUT_DIR/compensation_alpha*.jsonl \
  --output $OUTPUT_DIR/summary.csv \
  --detailed_output $OUTPUT_DIR/detailed_analysis.csv
```

## 📂 File Structure

```
ifattn/
├── run_compensation_experiment.sh    # Main pipeline (edit CONFIGURATION section)
├── select_samples_for_compensation.py
├── dump_attn.py
├── identify_critical_heads.py
├── eval_ifeval_with_compensation.py
├── analyze_compensation_results.py
└── artifacts/
    └── {experiment_name}/
        ├── selected_prompts.jsonl
        ├── attn_fp16.npz
        ├── attn_{quant_method}.npz
        ├── critical_heads_{quant_method}.json
        ├── compensation_alpha*.jsonl
        ├── summary.csv
        └── detailed_analysis.csv
```

## 🔧 Pipeline Stages

### Stage 1: Sample Selection
Selects test cases based on strategy:
- `failure_only`: FP16✓ but Quant✗ (recommended)
- `both_wrong`: Both models fail
- `all`: All samples

**Output:** `selected_prompts.jsonl`

### Stage 2: Attention Extraction
Extracts attention weights on instruction tokens for both FP16 and quantized models.

**Output:** `attn_fp16.npz`, `attn_{quant_method}.npz`

### Stage 3: Critical Head Identification
Computes degradation Δ = A_fp16 - A_quant for each attention head.
Identifies top-K heads with highest degradation.

**Output:** `critical_heads_{quant_method}.json`

### Stage 4: Compensation Testing
Tests multiple compensation strengths (alpha values):
- α=0.0: Baseline (no compensation)
- α=5.0: Moderate compensation
- α=10.0: Strong compensation
- α=20.0: Aggressive compensation

**Formula:** A'[head] = A[head] + α × (A_fp16[head] - A_quant[head])

**Output:** `compensation_alpha{X}.jsonl` for each alpha

### Stage 5: Result Analysis
Compares outputs across alpha values:
- Summary statistics (pass rates, improvements)
- Language error detection
- Quality collapse detection (token repetition, degeneration)

**Output:** `summary.csv`, `detailed_analysis.csv`

## 📊 Configuration Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `FP16_MODEL` | Full-precision model | `"Qwen/Qwen2.5-7B-Instruct"` |
| `QUANT_MODEL` | Quantized model path | `"/path/to/gptq_4bit"` |
| `QUANT_METHOD` | Quantization tag | `"gptq4"`, `"awq4"` |
| `FP16_SAMPLES` | FP16 IFEval results | `"fp16_samples.jsonl"` |
| `QUANT_SAMPLES` | Quant IFEval results | `"quant_samples.jsonl"` |

### Optional Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `SAMPLE_STRATEGY` | `"failure_only"` | Sample selection strategy |
| `MAX_SAMPLES` | `5` | Number of samples to test |
| `TOP_K_HEADS` | `10` | Number of heads to compensate |
| `ALPHA_VALUES` | `"0.0 5.0 10.0 20.0"` | Compensation strengths |
| `MAX_GEN_TOKENS` | `1280` | Max generation length |

## 🧪 Example Experiments

### Quick Configuration Reference

Edit the `CONFIGURATION` section in `run_compensation_experiment.sh`:

### Experiment 1: GPTQ 4-bit
```bash
EXPERIMENT_NAME="gptq4_compensation"
QUANT_MODEL="/path/to/gptq_4bit"
QUANT_METHOD="gptq4"
ALPHA_VALUES="0.0 5.0 10.0 20.0"
MAX_SAMPLES=5
```

### Experiment 2: AWQ 4-bit
```bash
EXPERIMENT_NAME="awq4_compensation"
QUANT_MODEL="/path/to/awq_4bit"
QUANT_METHOD="awq4"
ALPHA_VALUES="0.0 5.0 10.0 20.0"
MAX_SAMPLES=5
```

### Experiment 3: Large-scale test
```bash
EXPERIMENT_NAME="gptq4_fullscale"
QUANT_MODEL="/path/to/gptq_4bit"
QUANT_METHOD="gptq4"
MAX_SAMPLES=24  # All failure cases
ALPHA_VALUES="0.0 2.5 5.0 7.5 10.0 15.0 20.0"
MAX_GEN_TOKENS=2048
```

### Experiment 4: Test different models
```bash
# Llama
FP16_MODEL="meta-llama/Llama-3.1-8B-Instruct"
QUANT_MODEL="/path/to/llama_gptq4"

# Mistral
FP16_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
QUANT_MODEL="/path/to/mistral_gptq4"
```

## 📈 Understanding Results

### Summary Table (summary.csv)

| Column | Meaning |
|--------|---------|
| `alpha` | Compensation strength |
| `avg_pass_rate` | Average IFEval pass rate |
| `samples_improved` | Number of samples that improved vs baseline |
| `language_errors` | Number of language confusion cases |
| `quality_issues` | Number of degenerate outputs |

### Key Findings Indicators

- 🎯 **Language Fix**: Repairs language confusion (e.g., German→English)
- ⚠️ **Quality Collapse**: Over-compensation causes degeneration
- ✅ **Improved**: Higher pass rate than baseline
- ❌ **Degraded**: Lower pass rate than baseline

## 🔬 Research Use

### For Paper Submissions

1. **Run full-scale experiments**:
   - Test all failure cases (`MAX_SAMPLES=24+`)
   - Test multiple quantization methods (GPTQ, AWQ)
   - Test multiple models (Qwen, Llama, Mistral)

2. **Generate figures**:
   - Heatmaps: `attn_fp16.npz` vs `attn_quant.npz`
   - Line plots: Pass rate vs alpha
   - Bar charts: Per-category improvements

3. **Report metrics**:
   - Optimal alpha value
   - Improvement rate
   - Failure mode analysis

### Citation

```bibtex
@inproceedings{yourname2025attention,
  title={Attention Compensation for Repairing Instruction-Following in Quantized LLMs},
  author={Your Name},
  booktitle={Proceedings of ACL},
  year={2025}
}
```

## 🐛 Troubleshooting

### Issue: "No samples selected"
**Solution:** Check that FP16 and quant samples have overlapping keys and different pass/fail statuses.

### Issue: "Attention is None"
**Solution:** Ensure model uses eager attention mode (not Flash Attention). See `dump_attn.py` for fixes.

### Issue: "Out of memory"
**Solution:** Reduce `MAX_SAMPLES` or `MAX_GEN_TOKENS`, or use a GPU with more VRAM.

### Issue: "All alpha values give same output"
**Solution:** Increase alpha values (try 10.0, 20.0, 50.0) or check if compensation hooks are registered correctly.

### Issue: "Script takes too long"
**Solution:** Run stages manually (Option 2) and use previously computed attention files to skip extraction steps.

## 💡 Tips for Manual Execution

When running steps manually (Option 2), you can:

1. **Reuse attention files across experiments**
   ```bash
   # Extract attention once
   python dump_attn.py --model_id "..." --run_tag "fp16" ...
   
   # Use same attention for multiple alpha tests
   python eval_ifeval_with_compensation.py --alpha_list 10.0 ...
   python eval_ifeval_with_compensation.py --alpha_list 15.0 ...
   ```

2. **Test single alpha value quickly**
   ```bash
   # Skip analysis, just test one alpha
   python eval_ifeval_with_compensation.py \
     --alpha_list 10.0 \
     --max_samples 3 \
     ...
   ```

3. **Debug attention extraction**
   ```bash
   # Test on 1 sample first
   python dump_attn.py \
     --prompts_jsonl <(head -1 selected_prompts.jsonl) \
     ...
   ```

4. **Parallel alpha testing**
   ```bash
   # Run different alphas in parallel on different GPUs
   CUDA_VISIBLE_DEVICES=0 python eval_ifeval_with_compensation.py --alpha_list 5.0 &
   CUDA_VISIBLE_DEVICES=1 python eval_ifeval_with_compensation.py --alpha_list 10.0 &
   CUDA_VISIBLE_DEVICES=2 python eval_ifeval_with_compensation.py --alpha_list 20.0 &
   wait
   ```

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

## 📄 License

MIT License - see LICENSE file for details.