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

**Option 1: Direct execution**
```bash
# Edit configuration in the script
vim run_compensation_experiment.sh

# Submit to cluster
qsub run_compensation_experiment.sh
```

**Option 2: Using config file (recommended)**
```bash
# Create custom config
cp config.template.sh my_experiment.config.sh
vim my_experiment.config.sh

# Run with config
source my_experiment.config.sh
qsub run_compensation_experiment.sh
```

## 📂 File Structure

```
ifattn/
├── run_compensation_experiment.sh    # Main experiment pipeline
├── config.template.sh                # Configuration template
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

### Experiment 1: GPTQ 4-bit
```bash
export QUANT_MODEL="/path/to/gptq_4bit"
export QUANT_METHOD="gptq4"
export ALPHA_VALUES="0.0 5.0 10.0 20.0"
```

### Experiment 2: AWQ 4-bit
```bash
export QUANT_MODEL="/path/to/awq_4bit"
export QUANT_METHOD="awq4"
export ALPHA_VALUES="0.0 5.0 10.0 20.0"
```

### Experiment 3: Large-scale test
```bash
export MAX_SAMPLES=24  # All failure cases
export ALPHA_VALUES="0.0 2.5 5.0 7.5 10.0 15.0 20.0"
export MAX_GEN_TOKENS=2048
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

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [your email].

## 📄 License

MIT License - see LICENSE file for details.