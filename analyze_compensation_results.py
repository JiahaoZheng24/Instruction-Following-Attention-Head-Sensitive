#!/usr/bin/env python3
"""
Analyze Attention Compensation Results

Compare baseline (alpha=0.0) vs compensated outputs across different alpha values.
Generate summary statistics and detailed comparison tables.

FIXED VERSION - handles eval_ifeval_with_compensation.py output format correctly
"""

import argparse
import json
import os
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_results(filepath):
    """
    Load results from eval_ifeval_with_compensation.py output.

    Expected structure:
    {
        "model_path": "...",
        "results": {
            "alpha_0.0": {
                "alpha": 0.0,
                "mean_score": 1.0,
                "samples": [{"doc_id": ..., "response": ..., ...}]
            }
        }
    }

    Returns the ENTIRE dict so main() can access both metadata and per-alpha results.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        obj = json.load(f)

    # Validate structure
    if not isinstance(obj, dict):
        raise ValueError(f"{filepath}: Expected JSON object, got {type(obj)}")

    if "results" not in obj:
        raise ValueError(
            f"{filepath}: Missing 'results' key.\n"
            f"Available keys: {list(obj.keys())}\n"
            f"This file may not be from eval_ifeval_with_compensation.py"
        )

    if not isinstance(obj["results"], dict):
        raise ValueError(
            f"{filepath}: 'results' should be dict, got {type(obj['results'])}"
        )

    return obj


def extract_alpha_from_filename(filepath):
    """Extract alpha value from filename like 'compensation_alpha5.0.json'"""
    import re
    basename = os.path.basename(filepath)

    # Fixed regex: non-greedy match, stop at .json or end of string
    # Prevents matching the dot in .json extension
    match = re.search(r'alpha([\d.]+?)(?:\.json|$)', basename)

    if match:
        alpha_str = match.group(1)
        # Remove any trailing dots (safety measure)
        alpha_str = alpha_str.rstrip('.')
        try:
            return float(alpha_str)
        except ValueError:
            return None

    return None


def compare_outputs(baseline_sample, compensated_sample):
    """
    Compare baseline vs compensated output for a single sample.
    Returns dict with comparison metrics.
    """
    # Handle both 'response' and 'answer' keys
    baseline_resp = baseline_sample.get("response") or baseline_sample.get("answer", "")
    comp_resp = compensated_sample.get("response") or compensated_sample.get("answer", "")

    # Length comparison
    len_baseline = len(baseline_resp)
    len_comp = len(comp_resp)
    len_change_pct = ((len_comp - len_baseline) / max(len_baseline, 1)) * 100

    # Exact match
    exact_match = (baseline_resp == comp_resp)

    # Word-level similarity (simple)
    baseline_words = set(baseline_resp.lower().split())
    comp_words = set(comp_resp.lower().split())

    if baseline_words or comp_words:
        jaccard = len(baseline_words & comp_words) / len(baseline_words | comp_words)
    else:
        jaccard = 0.0

    return {
        "exact_match": exact_match,
        "baseline_length": len_baseline,
        "compensated_length": len_comp,
        "length_change_pct": len_change_pct,
        "jaccard_similarity": jaccard,
        "baseline_score": baseline_sample.get("score", 0),
        "compensated_score": compensated_sample.get("score", 0),
    }


def main():
    ap = argparse.ArgumentParser(
        description='Analyze attention compensation results'
    )
    ap.add_argument(
        '--results',
        nargs='+',
        required=True,
        help='Result JSON files (compensation_alpha*.json)'
    )
    ap.add_argument(
        '--output',
        required=True,
        help='Output CSV file for summary statistics'
    )
    ap.add_argument(
        '--detailed_output',
        default=None,
        help='Optional: Detailed per-sample comparison CSV'
    )
    args = ap.parse_args()

    print("=" * 70)
    print("ANALYZING COMPENSATION RESULTS")
    print("=" * 70)

    # Step 1: Load all result files
    print(f"\n1. Loading {len(args.results)} result files...")

    all_results = {}  # alpha -> samples list
    baseline = None
    metadata = {}

    for filepath in args.results:
        if not os.path.exists(filepath):
            print(f"   ⚠️  File not found: {filepath}")
            continue

        print(f"   Loading: {filepath}")

        # Extract alpha from filename
        alpha = extract_alpha_from_filename(filepath)
        if alpha is None:
            print(f"   ⚠️  Could not extract alpha from: {filepath}")
            continue

        # Load the entire result object
        try:
            result_obj = load_results(filepath)
        except Exception as e:
            print(f"   ❌ Failed to load {filepath}: {e}")
            continue

        # Save metadata from first file
        if not metadata:
            metadata = {
                k: v for k, v in result_obj.items()
                if k != 'results'
            }

        # Extract the samples for this alpha
        alpha_key = f"alpha_{alpha}"
        if alpha_key not in result_obj["results"]:
            print(f"   ⚠️  Key '{alpha_key}' not found in results")
            continue

        alpha_data = result_obj["results"][alpha_key]

        # Try both 'samples' and 'per_sample' keys (for compatibility)
        if "per_sample" in alpha_data:
            samples = alpha_data["per_sample"]
        elif "samples" in alpha_data:
            samples = alpha_data["samples"]
        else:
            print(f"   ⚠️  No 'samples' or 'per_sample' in {alpha_key}")
            continue
        print(f"   ✓ Loaded alpha={alpha}: {len(samples)} samples, "
              f"mean_score={alpha_data.get('mean_score', 'N/A')}")

        if alpha == 0.0:
            baseline = samples
        else:
            all_results[alpha] = samples

    if baseline is None:
        print("\n❌ ERROR: No baseline (alpha=0.0) results found!")
        print("Make sure one file has 'alpha0.0' in its name")
        return

    if not all_results:
        print("\n❌ ERROR: No compensated results found!")
        return

    print(f"\n✓ Loaded baseline ({len(baseline)} samples) and "
          f"{len(all_results)} compensation conditions")

    # Step 2: Build comparison table
    print("\n2. Comparing baseline vs compensated outputs...")

    summary_rows = []
    detailed_rows = []

    for alpha in sorted(all_results.keys()):
        compensated = all_results[alpha]

        # Match samples by doc_id or key
        baseline_dict = {}
        comp_dict = {}

        for s in baseline:
            sample_id = s.get('doc_id') or s.get('key')
            if sample_id is not None:
                baseline_dict[sample_id] = s

        for s in compensated:
            sample_id = s.get('doc_id') or s.get('key')
            if sample_id is not None:
                comp_dict[sample_id] = s

        common_ids = set(baseline_dict.keys()) & set(comp_dict.keys())

        if not common_ids:
            print(f"   ⚠️  Alpha={alpha}: No matching doc_ids!")
            continue

        # Per-sample comparisons
        comparisons = []
        for doc_id in sorted(common_ids):
            comp_metrics = compare_outputs(
                baseline_dict[doc_id],
                comp_dict[doc_id]
            )
            comp_metrics['doc_id'] = doc_id
            comp_metrics['alpha'] = alpha

            comparisons.append(comp_metrics)
            detailed_rows.append(comp_metrics)

        # Aggregate statistics for this alpha
        exact_matches = sum(1 for c in comparisons if c['exact_match'])
        avg_jaccard = sum(c['jaccard_similarity'] for c in comparisons) / len(comparisons)
        avg_len_change = sum(c['length_change_pct'] for c in comparisons) / len(comparisons)

        score_improved = sum(
            1 for c in comparisons
            if c['compensated_score'] > c['baseline_score']
        )
        score_degraded = sum(
            1 for c in comparisons
            if c['compensated_score'] < c['baseline_score']
        )
        score_unchanged = len(comparisons) - score_improved - score_degraded

        summary_rows.append({
            'alpha': alpha,
            'num_samples': len(comparisons),
            'exact_matches': exact_matches,
            'exact_match_rate': exact_matches / len(comparisons),
            'avg_jaccard_similarity': avg_jaccard,
            'avg_length_change_pct': avg_len_change,
            'score_improved': score_improved,
            'score_degraded': score_degraded,
            'score_unchanged': score_unchanged,
        })

        print(f"   Alpha={alpha:4.1f}: {exact_matches}/{len(comparisons)} exact matches, "
              f"avg similarity={avg_jaccard:.3f}, "
              f"score: +{score_improved}/-{score_degraded}/={score_unchanged}")

    # Step 3: Save results
    print("\n3. Saving results...")

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values('alpha')
    summary_df.to_csv(args.output, index=False)
    print(f"   ✓ Summary saved to: {args.output}")

    # Detailed CSV (optional)
    if args.detailed_output and detailed_rows:
        detailed_df = pd.DataFrame(detailed_rows)
        detailed_df = detailed_df.sort_values(['alpha', 'doc_id'])
        detailed_df.to_csv(args.detailed_output, index=False)
        print(f"   ✓ Detailed comparison saved to: {args.detailed_output}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()