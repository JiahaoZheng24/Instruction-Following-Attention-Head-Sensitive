# Attention Compensation Experiment Configuration
#
# Copy this file and edit the variables below for your experiment.
# Then source it before running the main script.
#
# Usage:
#   1. cp config.template.sh my_experiment.config.sh
#   2. Edit my_experiment.config.sh
#   3. source my_experiment.config.sh && qsub run_compensation_experiment.sh

################################################################################
# Project Paths
################################################################################
export BASE_DIR="/users/jzheng7/ifattn"
export EXPERIMENT_NAME="gptq4_compensation"
export CONDA_ENV="attention"

################################################################################
# Models
################################################################################
# Baseline full-precision model
export FP16_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Quantized model to test
export QUANT_MODEL="/store01/yshi4/Quant_Lib/quantized_models_gptq/quantized_Qwen_Qwen2.5-7B-Instruct_4bit"

# Tag for this quantization method (used in output filenames)
export QUANT_METHOD="gptq4"

################################################################################
# IFEval Benchmark Data
################################################################################
# Full-precision IFEval results
export FP16_SAMPLES="/users/jzheng7/result/ifeval/qwen/Qwen__Qwen2.5-7B-Instruct/samples_ifeval_2025-10-09T13-58-12.528717.jsonl"

# Quantized model IFEval results
export QUANT_SAMPLES="/users/jzheng7/result/ifeval/qwen/__store01__yshi4__Quant_Lib__quantized_models_gptq__quantized_Qwen_Qwen2.5-7B-Instruct_4bit/samples_ifeval_2025-10-14T16-06-35.354326.jsonl"

################################################################################
# Experiment Parameters
################################################################################
# Sample selection strategy
#   - failure_only: FP16✓ but Quant✗ (target failures)
#   - both_wrong: Both FP16✗ and Quant✗
#   - all: All samples
export SAMPLE_STRATEGY="failure_only"

# Number of samples to test (for quick experiments, use 5-10)
export MAX_SAMPLES=5

# Number of critical heads to compensate
export TOP_K_HEADS=10

# Alpha values to test (space-separated)
# Recommended: 0.0 (baseline) + 5.0, 10.0, 20.0
export ALPHA_VALUES="0.0 5.0 10.0 20.0"

# Maximum generation length (tokens)
export MAX_GEN_TOKENS=1280

################################################################################
# Example Alternative Configurations
################################################################################

# Example 1: AWQ 4-bit quantization
# export QUANT_MODEL="/path/to/awq_4bit_model"
# export QUANT_METHOD="awq4"
# export QUANT_SAMPLES="/path/to/awq_ifeval_results.jsonl"

# Example 2: GPTQ 3-bit quantization
# export QUANT_MODEL="/path/to/gptq_3bit_model"
# export QUANT_METHOD="gptq3"
# export QUANT_SAMPLES="/path/to/gptq3_ifeval_results.jsonl"

# Example 3: Test more alpha values
# export ALPHA_VALUES="0.0 2.5 5.0 7.5 10.0 15.0 20.0"

# Example 4: Large-scale experiment
# export MAX_SAMPLES=24  # Test all failure cases
# export MAX_GEN_TOKENS=2048