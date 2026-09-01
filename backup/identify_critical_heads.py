"""
Identify critical attention heads for instruction-following
Based on degradation analysis between FP16 and quantized models
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_npz(path):
    """Load attention data from npz file"""
    z = np.load(path, allow_pickle=True)
    M = z["mean_layer_head"]  # [L,H]
    per_prompt = z["per_prompt"]  # [P,L,H]
    meta = json.loads(str(z["meta"].tolist()))
    return M, per_prompt, meta


def compute_head_degradation(fp16_attn, quant_attn):
    """
    Compute degradation for each head
    Returns: degradation matrix [L, H] where positive = attention decreased
    """
    delta = fp16_attn - quant_attn  # Positive = degradation
    return delta


def identify_top_k_heads(delta_matrix, k=10, method='absolute'):
    """
    Identify top-K most degraded heads

    Args:
        delta_matrix: [L, H] degradation matrix
        k: number of heads to select
        method: 'absolute' (largest |delta|) or 'positive' (largest positive delta)

    Returns:
        List of (layer, head, delta_value) tuples
    """
    L, H = delta_matrix.shape
    heads = []

    for layer in range(L):
        for head in range(H):
            delta_val = delta_matrix[layer, head]

            if method == 'absolute':
                metric = abs(delta_val)
            elif method == 'positive':
                metric = delta_val if delta_val > 0 else 0
            else:
                metric = delta_val

            heads.append((layer, head, delta_val, metric))

    # Sort by metric (descending)
    heads.sort(key=lambda x: x[3], reverse=True)

    # Return top-K
    top_k = [(layer, head, delta) for layer, head, delta, metric in heads[:k]]

    return top_k


def analyze_head_statistics(fp16_attn, quant_attn):
    """
    Compute various statistics for analysis
    """
    delta = fp16_attn - quant_attn

    stats = {
        'mean_degradation': delta.mean(),
        'std_degradation': delta.std(),
        'max_degradation': delta.max(),
        'min_degradation': delta.min(),
        'num_positive': np.sum(delta > 0),
        'num_negative': np.sum(delta < 0),
        'num_near_zero': np.sum(np.abs(delta) < 0.001),
    }

    # Per-layer statistics
    layer_stats = []
    for layer in range(delta.shape[0]):
        layer_stats.append({
            'layer': layer,
            'mean_delta': delta[layer].mean(),
            'std_delta': delta[layer].std(),
            'max_delta': delta[layer].max(),
            'min_delta': delta[layer].min(),
        })

    return stats, layer_stats


def visualize_degradation(delta_matrix, top_k_heads, out_dir, tag):
    """Create visualizations of degradation patterns"""

    # 1. Heatmap of degradation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full degradation heatmap
    ax = axes[0]
    abs_max = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    im = ax.imshow(delta_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-abs_max, vmax=abs_max)
    plt.colorbar(im, ax=ax)
    ax.set_title('Attention Degradation (FP16 - Quant)\nPositive = Loss of Attention')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')

    # Mark top-K heads
    for layer, head, delta in top_k_heads:
        ax.plot(head, layer, 'y*', markersize=15, markeredgecolor='black', markeredgewidth=1)

    # Histogram of degradation values
    ax = axes[1]
    ax.hist(delta_matrix.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='No change')
    ax.set_xlabel('Degradation (FP16 - Quant)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Attention Degradation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'degradation_analysis_{tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Per-layer degradation profile
    fig, ax = plt.subplots(figsize=(10, 6))

    layer_means = delta_matrix.mean(axis=1)
    layer_stds = delta_matrix.std(axis=1)
    layers = np.arange(len(layer_means))

    ax.plot(layers, layer_means, 'o-', linewidth=2, markersize=8, label='Mean degradation')
    ax.fill_between(layers,
                    layer_means - layer_stds,
                    layer_means + layer_stds,
                    alpha=0.3, label='±1 std')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='No change')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mean Degradation', fontsize=12)
    ax.set_title('Layer-wise Attention Degradation Profile', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'layer_profile_{tag}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Identify critical attention heads')
    ap.add_argument('--fp16_attn', required=True, help='Path to FP16 attention npz')
    ap.add_argument('--quant_attn', required=True, help='Path to quantized attention npz')
    ap.add_argument('--out_dir', required=True, help='Output directory')
    ap.add_argument('--top_k', type=int, default=10, help='Number of top heads to identify')
    ap.add_argument('--method', default='positive',
                    choices=['absolute', 'positive', 'negative'],
                    help='Method to rank heads')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("IDENTIFYING CRITICAL ATTENTION HEADS")
    print("=" * 80)

    # Load attention data
    print("\n1. Loading attention data...")
    fp16_mean, fp16_per_prompt, fp16_meta = load_npz(args.fp16_attn)
    quant_mean, quant_per_prompt, quant_meta = load_npz(args.quant_attn)

    print(f"   FP16:  {fp16_meta['run_tag']}, shape {fp16_mean.shape}")
    print(f"   Quant: {quant_meta['run_tag']}, shape {quant_mean.shape}")

    # Compute degradation
    print("\n2. Computing degradation...")
    delta = compute_head_degradation(fp16_mean, quant_mean)

    # Overall statistics
    print("\n3. Overall statistics:")
    stats, layer_stats = analyze_head_statistics(fp16_mean, quant_mean)

    print(f"   Mean degradation:     {stats['mean_degradation']:+.6f}")
    print(f"   Std degradation:      {stats['std_degradation']:.6f}")
    print(f"   Max degradation:      {stats['max_degradation']:+.6f}")
    print(f"   Min degradation:      {stats['min_degradation']:+.6f}")
    print(f"   Heads with positive Δ: {stats['num_positive']} ({stats['num_positive'] / delta.size * 100:.1f}%)")
    print(f"   Heads with negative Δ: {stats['num_negative']} ({stats['num_negative'] / delta.size * 100:.1f}%)")
    print(f"   Heads near zero:      {stats['num_near_zero']} ({stats['num_near_zero'] / delta.size * 100:.1f}%)")

    # Identify top-K heads
    print(f"\n4. Identifying top-{args.top_k} critical heads (method={args.method})...")
    top_k_heads = identify_top_k_heads(delta, k=args.top_k, method=args.method)
    critical_set = {(l, h) for (l, h, _) in top_k_heads}

    print(f"\n   Top-{args.top_k} critical heads:")
    print("   " + "-" * 60)
    print(f"   {'Rank':<6} {'Layer':<8} {'Head':<8} {'Degradation':<15}")
    print("   " + "-" * 60)

    for rank, (layer, head, delta_val) in enumerate(top_k_heads, 1):
        print(f"   {rank:<6} {layer:<8} {head:<8} {delta_val:+.6f}")

    # Save results
    print("\n5. Saving results...")

    # Save top-K heads to JSON
    top_k_json = {
        'top_k': args.top_k,
        'method': args.method,
        'fp16_model': fp16_meta['run_tag'],
        'quant_model': quant_meta['run_tag'],
        'critical_heads': [
            {
                'rank': rank,
                'layer': int(layer),
                'head': int(head),
                'degradation': float(delta_val),
                'fp16_attn': float(fp16_mean[layer, head]),
                'quant_attn': float(quant_mean[layer, head]),
            }
            for rank, (layer, head, delta_val) in enumerate(top_k_heads, 1)
        ],
        'statistics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in stats.items()}
    }

    json_path = os.path.join(args.out_dir,
                             f'critical_heads_{quant_meta["run_tag"]}.json')
    with open(json_path, 'w') as f:
        json.dump(top_k_json, f, indent=2)
    print(f"   Saved: {json_path}")

    # Save full degradation matrix
    np.save(os.path.join(args.out_dir, f'degradation_{quant_meta["run_tag"]}.npy'),
            delta)

    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_csv = os.path.join(args.out_dir, f'statistics_{quant_meta["run_tag"]}.csv')
    stats_df.to_csv(stats_csv, index=False)
    print(f"   Saved: {stats_csv}")

    # Save per-layer statistics
    layer_df = pd.DataFrame(layer_stats)
    layer_csv = os.path.join(args.out_dir,
                             f'layer_statistics_{quant_meta["run_tag"]}.csv')
    layer_df.to_csv(layer_csv, index=False)
    print(f"   Saved: {layer_csv}")

    # Create visualizations
    print("\n6. Creating visualizations...")
    visualize_degradation(delta, top_k_heads, args.out_dir, quant_meta['run_tag'])
    print(f"   Saved: degradation_analysis_{quant_meta['run_tag']}.png")
    print(f"   Saved: layer_profile_{quant_meta['run_tag']}.png")

    # Save per-head data for further analysis
    L, H = delta.shape
    per_head_data = []
    for layer in range(L):
        for head in range(H):
            per_head_data.append({
                'layer': layer,
                'head': head,
                'fp16_attn': fp16_mean[layer, head],
                'quant_attn': quant_mean[layer, head],
                'degradation': delta[layer, head],
                'abs_degradation': abs(delta[layer, head]),
                'is_critical': (layer, head) in critical_set
            })

    per_head_df = pd.DataFrame(per_head_data)
    per_head_csv = os.path.join(args.out_dir,
                                f'per_head_analysis_{quant_meta["run_tag"]}.csv')
    per_head_df.to_csv(per_head_csv, index=False)
    print(f"   Saved: {per_head_csv}")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nCritical heads identified and saved to: {args.out_dir}")
    print(f"Use these heads for attention compensation experiments.")


if __name__ == "__main__":
    main()