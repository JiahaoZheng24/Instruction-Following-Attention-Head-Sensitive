"""
Select samples where FP16 is correct but quantized model fails
Perfect for testing compensation effectiveness
"""

import argparse
import json
from collections import defaultdict

def load_samples(jsonl_path):
    """Load samples from lm-eval output jsonl"""
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def check_correct(sample):
    """
    Check if a sample is correct
    For IFEval, we check if all instructions are followed

    The lm-eval format has:
    - 'prompt_level_strict_acc': bool - whether ALL instructions passed (strict)
    - 'inst_level_strict_acc': list of bool - per-instruction results
    """

    # Method 1: Check 'prompt_level_strict_acc' (most reliable for lm-eval)
    # This is True only if ALL instructions are followed
    if 'prompt_level_strict_acc' in sample:
        return sample['prompt_level_strict_acc']

    # Method 2: Check if all inst_level_strict_acc are True
    if 'inst_level_strict_acc' in sample:
        inst_acc = sample['inst_level_strict_acc']
        if isinstance(inst_acc, list):
            return all(inst_acc)  # All instructions must pass

    # Method 3: Fallback to loose accuracy
    if 'prompt_level_loose_acc' in sample:
        return sample['prompt_level_loose_acc']

    # If we can't determine, print warning and return None
    print(f"[WARNING] Cannot determine correctness for doc_id={sample.get('doc_id', 'unknown')}")
    print(f"[DEBUG] Available fields: {list(sample.keys())}")
    return None

def select_failure_cases(fp16_samples, quant_samples, strategy='failure_only'):
    """
    Select samples based on strategy

    Args:
        strategy:
            - 'failure_only': FP16✓ & Quant✗ only
            - 'mixed': Include all categories
            - 'both_correct': FP16✓ & Quant✓ (for testing no harm)
    """
    # Create lookup by doc_id
    fp16_dict = {s.get('doc_id', s.get('key', i)): s for i, s in enumerate(fp16_samples)}
    quant_dict = {s.get('doc_id', s.get('key', i)): s for i, s in enumerate(quant_samples)}

    # Find common doc_ids
    common_ids = set(fp16_dict.keys()) & set(quant_dict.keys())

    print(f"\nFound {len(common_ids)} common samples between FP16 and Quant")

    # Categorize samples
    categories = {
        'fp16_correct_quant_wrong': [],  # FP16✓ Quant✗ - Main target!
        'both_correct': [],               # FP16✓ Quant✓ - Test no harm
        'both_wrong': [],                 # FP16✗ Quant✗ - Baseline
        'fp16_wrong_quant_correct': [],   # FP16✗ Quant✓ - Unexpected
        'unknown': []                     # Cannot determine
    }

    for doc_id in common_ids:
        fp16_sample = fp16_dict[doc_id]
        quant_sample = quant_dict[doc_id]

        fp16_correct = check_correct(fp16_sample)
        quant_correct = check_correct(quant_sample)

        if fp16_correct is None or quant_correct is None:
            categories['unknown'].append({
                'doc_id': doc_id,
                'fp16_sample': fp16_sample,
                'quant_sample': quant_sample
            })
        elif fp16_correct and not quant_correct:
            # Extract prompt - handle different formats
            prompt = ''
            if 'doc' in fp16_sample and 'prompt' in fp16_sample['doc']:
                prompt = fp16_sample['doc']['prompt']
            elif 'prompt' in fp16_sample:
                prompt = fp16_sample['prompt']
            elif 'arguments' in fp16_sample:
                try:
                    prompt = fp16_sample['arguments']['gen_args_0']['arg_0']
                except:
                    prompt = str(fp16_sample.get('arguments', ''))[:100]

            categories['fp16_correct_quant_wrong'].append({
                'doc_id': doc_id,
                'fp16_sample': fp16_sample,
                'quant_sample': quant_sample,
                'prompt': prompt,
            })
        elif fp16_correct and quant_correct:
            prompt = ''
            if 'doc' in fp16_sample and 'prompt' in fp16_sample['doc']:
                prompt = fp16_sample['doc']['prompt']
            elif 'prompt' in fp16_sample:
                prompt = fp16_sample['prompt']

            categories['both_correct'].append({
                'doc_id': doc_id,
                'fp16_sample': fp16_sample,
                'quant_sample': quant_sample,
                'prompt': prompt,
            })
        elif not fp16_correct and not quant_correct:
            prompt = ''
            if 'doc' in fp16_sample and 'prompt' in fp16_sample['doc']:
                prompt = fp16_sample['doc']['prompt']
            elif 'prompt' in fp16_sample:
                prompt = fp16_sample['prompt']

            categories['both_wrong'].append({
                'doc_id': doc_id,
                'fp16_sample': fp16_sample,
                'quant_sample': quant_sample,
                'prompt': prompt,
            })
        else:  # fp16 wrong but quant correct (very rare)
            prompt = ''
            if 'doc' in fp16_sample and 'prompt' in fp16_sample['doc']:
                prompt = fp16_sample['doc']['prompt']
            elif 'prompt' in fp16_sample:
                prompt = fp16_sample['prompt']

            categories['fp16_wrong_quant_correct'].append({
                'doc_id': doc_id,
                'fp16_sample': fp16_sample,
                'quant_sample': quant_sample,
                'prompt': prompt,
            })

    # Print statistics
    print("\n" + "="*60)
    print("SAMPLE CATEGORIZATION")
    print("="*60)
    for cat_name, samples in categories.items():
        print(f"{cat_name:35s}: {len(samples):4d} samples")
    print("="*60)

    # Select based on strategy
    selected = []

    if strategy == 'failure_only':
        selected = categories['fp16_correct_quant_wrong']
        print(f"\nStrategy: failure_only")
        print(f"Selected: {len(selected)} samples (FP16✓ Quant✗)")

    elif strategy == 'mixed':
        # 60% failure cases, 20% both correct, 20% both wrong
        failure = categories['fp16_correct_quant_wrong']
        correct = categories['both_correct']
        wrong = categories['both_wrong']

        n_failure = min(len(failure), 60)
        n_correct = min(len(correct), 20)
        n_wrong = min(len(wrong), 20)

        selected = failure[:n_failure] + correct[:n_correct] + wrong[:n_wrong]

        print(f"\nStrategy: mixed")
        print(f"Selected: {len(selected)} samples")
        print(f"  - Failure cases (FP16✓ Quant✗): {n_failure}")
        print(f"  - Both correct (FP16✓ Quant✓): {n_correct}")
        print(f"  - Both wrong (FP16✗ Quant✗): {n_wrong}")

    elif strategy == 'both_correct':
        selected = categories['both_correct']
        print(f"\nStrategy: both_correct")
        print(f"Selected: {len(selected)} samples (FP16✓ Quant✓)")

    return selected, categories

def save_selected_prompts(selected, output_path):
    """Save selected prompts to jsonl file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in selected:
            # Extract prompt text from 'doc' field
            fp16_sample = item['fp16_sample']

            # Get the doc field which contains the actual prompt
            doc = fp16_sample.get('doc', {})

            # Extract prompt
            prompt = doc.get('prompt', '')

            # If still no prompt, try other fields
            if not prompt:
                prompt = item.get('prompt', '')

            # Get instruction info
            output = {
                'key': item['doc_id'],
                'prompt': prompt,
                'instruction_id_list': doc.get('instruction_id_list', []),
                'kwargs': doc.get('kwargs', []),
            }

            f.write(json.dumps(output, ensure_ascii=False) + '\n')

def main():
    ap = argparse.ArgumentParser(description='Select samples for compensation testing')
    ap.add_argument('--fp16_samples', required=True,
                    help='Path to FP16 samples JSONL (from lm-eval output)')
    ap.add_argument('--quant_samples', required=True,
                    help='Path to quantized model samples JSONL')
    ap.add_argument('--output', required=True,
                    help='Output JSONL file for selected prompts')
    ap.add_argument('--strategy', default='failure_only',
                    choices=['failure_only', 'mixed', 'both_correct'],
                    help='Sample selection strategy')
    ap.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to select')
    args = ap.parse_args()

    print("="*60)
    print("SAMPLE SELECTION FOR COMPENSATION TESTING")
    print("="*60)
    print(f"FP16 samples: {args.fp16_samples}")
    print(f"Quant samples: {args.quant_samples}")
    print(f"Strategy: {args.strategy}")
    print(f"Output: {args.output}")

    # Load samples
    print("\nLoading samples...")
    fp16_samples = load_samples(args.fp16_samples)
    quant_samples = load_samples(args.quant_samples)

    print(f"Loaded {len(fp16_samples)} FP16 samples")
    print(f"Loaded {len(quant_samples)} Quant samples")

    # Select samples
    selected, categories = select_failure_cases(fp16_samples, quant_samples, args.strategy)

    # Limit if needed
    if args.max_samples and len(selected) > args.max_samples:
        print(f"\nLimiting to {args.max_samples} samples")
        selected = selected[:args.max_samples]

    # Save
    print(f"\nSaving {len(selected)} selected prompts to {args.output}")
    save_selected_prompts(selected, args.output)

    print("\n" + "="*60)
    print("✅ DONE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Use this file for compensation testing:")
    print(f"   --ifeval_data {args.output}")
    print(f"\n2. Run compensation experiment:")
    print(f"   qsub run_compensation_exp.sh")

if __name__ == "__main__":
    main()