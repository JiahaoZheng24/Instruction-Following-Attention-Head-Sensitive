#!/usr/bin/env python3
"""
Analyze Attention Compensation Results

Compare baseline (alpha=0.0) vs compensated outputs across different alpha values.
Generate summary statistics and detailed comparison tables.

Usage:
    python analyze_compensation_results.py \
        --results compensation_alpha*.json \
        --output compensation_analysis.csv
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_results(filepath):
    """Load a result file that may be JSON or JSONL.
    Returns: list[dict] of per-sample results.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # Peek first non-empty char
        pos = f.tell()
        first = ''
        while True:
            ch = f.read(1)
            if not ch:
                break
            if not ch.isspace():
                first = ch
                break
        f.seek(pos)

        # Case 1: JSON (object or array)
        if first in ['{', '[']:
            obj = json.load(f)
            if isinstance(obj, dict):
                # Common patterns: {"results":[...]} or {"samples":[...]}
                for k in ['results', 'samples', 'data']:
                    if k in obj and isinstance(obj[k], list):
                        return obj[k]
                # If dict but no obvious list key, fail loudly
                raise ValueError(f"{filepath} looks like JSON dict but no results list found.")
            elif isinstance(obj, list):
                return obj
            else:
                raise ValueError(f"Unexpected JSON type: {type(obj)} in {filepath}")

        # Case 2: JSONL
        data = []
        with open(filepath, 'r', encoding='utf-8') as f2:
            for line in f2:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data


def extract_alpha_from_filename(filepath):
    basename = os.path.basename(filepath)
    if 'alpha' in basename:
        alpha_part = basename.split('alpha', 1)[1]
        alpha_str = alpha_part.replace('.jsonl', '').replace('.json', '')
        return float(alpha_str)
    return None


def compute_pass_rate(results):
    """Compute pass rate from follow_instruction_list."""
    if not results:
        return 0.0

    total = 0
    passed = 0
    for item in results:
        follow_list = item.get('follow_instruction_list', [])
        if follow_list:
            total += len(follow_list)
            passed += sum(follow_list)

    return passed / total if total > 0 else 0.0


def analyze_sample_changes(baseline, compensated_dict):
    """
    Compare each sample across different alpha values.

    Returns:
        pd.DataFrame with columns: key, prompt_snippet, alpha, answer_snippet,
                                   follow_instruction_list, pass_rate
    """
    rows = []

    # Get all keys from baseline
    baseline_by_key = {item['key']: item for item in baseline}

    for key, base_item in baseline_by_key.items():
        base_prompt = base_item.get('prompt', '')[:100]  # First 100 chars
        base_answer = base_item.get('response', '')[:100]
        base_follow = base_item.get('follow_instruction_list', [])
        base_rate = sum(base_follow) / len(base_follow) if base_follow else 0.0

        # Baseline row
        rows.append({
            'key': key,
            'prompt_snippet': base_prompt,
            'alpha': 0.0,
            'answer_snippet': base_answer,
            'follow_instruction_list': str(base_follow),
            'pass_rate': base_rate,
            'language': detect_language(base_answer),
            'quality': assess_quality(base_answer)
        })

        # Compensated rows
        for alpha in sorted(compensated_dict.keys()):
            comp_results = compensated_dict[alpha]
            comp_by_key = {item['key']: item for item in comp_results}

            if key in comp_by_key:
                comp_item = comp_by_key[key]
                comp_answer = comp_item.get('response', '')[:100]
                comp_follow = comp_item.get('follow_instruction_list', [])
                comp_rate = sum(comp_follow) / len(comp_follow) if comp_follow else 0.0

                rows.append({
                    'key': key,
                    'prompt_snippet': base_prompt,
                    'alpha': alpha,
                    'answer_snippet': comp_answer,
                    'follow_instruction_list': str(comp_follow),
                    'pass_rate': comp_rate,
                    'language': detect_language(comp_answer),
                    'quality': assess_quality(comp_answer)
                })

    return pd.DataFrame(rows)


def detect_language(text):
    """Simple language detection based on keywords."""
    if not text:
        return 'empty'

    text_lower = text.lower()

    # German indicators
    german_words = ['hallo', 'ich bin', 'mein name', 'und', 'der', 'die', 'das', 'über']
    german_score = sum(1 for word in german_words if word in text_lower)

    # English indicators
    english_words = ['hello', 'i am', 'my name', 'the', 'and', 'of', 'to', 'in']
    english_score = sum(1 for word in english_words if word in text_lower)

    if german_score > english_score and german_score >= 2:
        return 'german'
    elif english_score >= german_score:
        return 'english'
    else:
        return 'unknown'


def assess_quality(text):
    """Assess output quality for degenerate patterns."""
    if not text:
        return 'empty'

    text_lower = text.lower()

    # Check for repetition loops
    words = text.split()[:50]  # First 50 words
    if len(words) > 10:
        # Check if >50% of words are identical
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            return 'collapsed_repetition'

    # Check for specific degenerate patterns
    if '- 9 - 9 - 9' in text or '9 - 9 - 9 - 9' in text:
        return 'collapsed_tokens'
    if text.count('and, and, and') > 2:
        return 'collapsed_tokens'
    if '!' * 10 in text:
        return 'collapsed_exclamation'
    if 'ospel' in text_lower * 3:  # Repeated pattern
        return 'collapsed_repetition'

    # Check for meta-commentary
    if 'create a' in text_lower[:50] or 'write a' in text_lower[:50]:
        return 'meta_instruction'

    return 'normal'


def generate_summary_table(baseline, compensated_dict):
    """
    Generate high-level summary statistics.

    Returns:
        pd.DataFrame with columns: alpha, total_samples, avg_pass_rate,
                                   samples_improved, samples_degraded,
                                   language_errors, quality_issues
    """
    rows = []

    baseline_by_key = {item['key']: item for item in baseline}

    for alpha in [0.0] + sorted(compensated_dict.keys()):
        if alpha == 0.0:
            results = baseline
        else:
            results = compensated_dict[alpha]

        total = len(results)
        avg_rate = compute_pass_rate(results)

        # Compare to baseline
        improved = 0
        degraded = 0
        language_errors = 0
        quality_issues = 0

        for item in results:
            key = item['key']
            answer = item.get('response', '')

            # Language check
            if detect_language(answer) == 'german' and alpha > 0:
                language_errors += 1

            # Quality check
            quality = assess_quality(answer)
            if quality != 'normal':
                quality_issues += 1

            # Compare to baseline
            if alpha > 0 and key in baseline_by_key:
                base_item = baseline_by_key[key]
                base_follow = base_item.get('follow_instruction_list', [])
                comp_follow = item.get('follow_instruction_list', [])

                if comp_follow and base_follow:
                    base_rate = sum(base_follow) / len(base_follow)
                    comp_rate = sum(comp_follow) / len(comp_follow)

                    if comp_rate > base_rate:
                        improved += 1
                    elif comp_rate < base_rate:
                        degraded += 1

        rows.append({
            'alpha': alpha,
            'total_samples': total,
            'avg_pass_rate': avg_rate,
            'samples_improved': improved if alpha > 0 else '-',
            'samples_degraded': degraded if alpha > 0 else '-',
            'language_errors': language_errors,
            'quality_issues': quality_issues
        })

    return pd.DataFrame(rows)


def generate_detailed_comparison(baseline, compensated_dict):
    """
    Generate detailed per-sample comparison showing answer changes.

    Returns:
        pd.DataFrame with one row per (sample, alpha) combination
    """
    rows = []
    baseline_by_key = {item['key']: item for item in baseline}

    for key, base_item in baseline_by_key.items():
        prompt = base_item.get('prompt', '')
        instruction_ids = base_item.get('instruction_id_list', [])

        # Baseline
        base_answer = base_item.get('response', '')
        base_follow = base_item.get('follow_instruction_list', [])

        rows.append({
            'key': key,
            'instruction_ids': ','.join(instruction_ids),
            'prompt': prompt[:150] + '...' if len(prompt) > 150 else prompt,
            'alpha': 0.0,
            'answer': base_answer[:200] + '...' if len(base_answer) > 200 else base_answer,
            'follow_list': str(base_follow),
            'pass_rate': sum(base_follow) / len(base_follow) if base_follow else 0.0,
            'language': detect_language(base_answer),
            'quality': assess_quality(base_answer),
            'vs_baseline': 'N/A'
        })

        # Compensated
        for alpha in sorted(compensated_dict.keys()):
            comp_results = compensated_dict[alpha]
            comp_by_key = {item['key']: item for item in comp_results}

            if key in comp_by_key:
                comp_item = comp_by_key[key]
                comp_answer = comp_item.get('response', '')
                comp_follow = comp_item.get('follow_instruction_list', [])
                comp_rate = sum(comp_follow) / len(comp_follow) if comp_follow else 0.0
                base_rate = sum(base_follow) / len(base_follow) if base_follow else 0.0

                # Determine change
                if comp_rate > base_rate:
                    vs_baseline = '✅ Improved'
                elif comp_rate < base_rate:
                    vs_baseline = '❌ Degraded'
                else:
                    vs_baseline = '➖ Same'

                # Check for language fix (Sample 97)
                if detect_language(base_answer) == 'german' and detect_language(comp_answer) == 'english':
                    vs_baseline = '🎯 LANGUAGE FIX'

                # Check for quality degradation
                if assess_quality(base_answer) == 'normal' and assess_quality(comp_answer) != 'normal':
                    vs_baseline = '⚠️ QUALITY COLLAPSE'

                rows.append({
                    'key': key,
                    'instruction_ids': ','.join(instruction_ids),
                    'prompt': prompt[:150] + '...' if len(prompt) > 150 else prompt,
                    'alpha': alpha,
                    'answer': comp_answer[:200] + '...' if len(comp_answer) > 200 else comp_answer,
                    'follow_list': str(comp_follow),
                    'pass_rate': comp_rate,
                    'language': detect_language(comp_answer),
                    'quality': assess_quality(comp_answer),
                    'vs_baseline': vs_baseline
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Analyze attention compensation results')
    parser.add_argument('--results', nargs='+', required=True,
                        help='List of compensation result files (compensation_alpha*.jsonl)')
    parser.add_argument('--output', required=True,
                        help='Output CSV file for summary')
    parser.add_argument('--detailed_output', default=None,
                        help='Optional: Output CSV for detailed per-sample analysis')
    args = parser.parse_args()

    print("=" * 70)
    print("ANALYZING COMPENSATION RESULTS")
    print("=" * 70)

    # Load all results
    print(f"\n1. Loading {len(args.results)} result files...")
    all_results = {}
    baseline = None

    for filepath in args.results:
        if not os.path.exists(filepath):
            print(f"   ⚠️  File not found: {filepath}")
            continue

        alpha = extract_alpha_from_filename(filepath)
        if alpha is None:
            print(f"   ⚠️  Could not extract alpha from: {filepath}")
            continue

        data = load_results(filepath)
        print(f"   Loaded alpha={alpha}: {len(data)} samples")

        if alpha == 0.0:
            baseline = data
        else:
            all_results[alpha] = data

    if baseline is None:
        print("\n❌ ERROR: No baseline (alpha=0.0) found!")
        return

    print(f"\n   Baseline: {len(baseline)} samples")
    print(f"   Compensated: {len(all_results)} alpha values")

    # Generate summary table
    print("\n2. Generating summary statistics...")
    summary_df = generate_summary_table(baseline, all_results)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Save summary
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    summary_df.to_csv(args.output, index=False)
    print(f"\n✅ Saved summary to: {args.output}")

    # Generate detailed comparison if requested
    if args.detailed_output:
        print("\n3. Generating detailed per-sample comparison...")
        detailed_df = generate_detailed_comparison(baseline, all_results)
        detailed_df.to_csv(args.detailed_output, index=False)
        print(f"✅ Saved detailed analysis to: {args.detailed_output}")

        # Show key findings
        print("\n" + "=" * 70)
        print("KEY FINDINGS")
        print("=" * 70)

        # Count language fixes
        lang_fixes = detailed_df[detailed_df['vs_baseline'] == '🎯 LANGUAGE FIX']
        if not lang_fixes.empty:
            print(f"\n🎯 Language fixes found: {len(lang_fixes)} instances")
            print(f"   Keys: {lang_fixes['key'].unique().tolist()}")
            print(f"   Alpha values: {lang_fixes['alpha'].unique().tolist()}")

        # Count quality collapses
        collapses = detailed_df[detailed_df['vs_baseline'] == '⚠️ QUALITY COLLAPSE']
        if not collapses.empty:
            print(f"\n⚠️  Quality collapses found: {len(collapses)} instances")
            print(f"   Keys: {collapses['key'].unique().tolist()}")
            print(f"   Alpha values: {collapses['alpha'].unique().tolist()}")

        # Best alpha
        alpha_scores = summary_df.groupby('alpha')['avg_pass_rate'].mean()
        best_alpha = alpha_scores.idxmax()
        print(f"\n🏆 Best performing alpha: {best_alpha} (avg pass rate: {alpha_scores[best_alpha]:.3f})")

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()