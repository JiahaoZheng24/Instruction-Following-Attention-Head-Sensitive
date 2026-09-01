import argparse, os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def load_prompts_with_category(jsonl_path):
    """Load prompts from JSONL with category labels if available"""
    prompts_by_idx = {}
    prompts_by_category = defaultdict(list)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if line.strip():
                data = json.loads(line)
                prompts_by_idx[idx] = data

                # Use existing category if available, otherwise auto-detect
                if 'category' in data:
                    category = data['category']
                else:
                    category = auto_categorize(data['prompt'])

                prompts_by_category[category].append((idx, data))

    return prompts_by_idx, prompts_by_category


def auto_categorize(prompt_text):
    """Auto-categorize prompt if no category label exists"""
    text_lower = prompt_text.lower()

    if 'json' in text_lower:
        return 'json_format'
    elif 'bullet' in text_lower:
        return 'bullet_points'
    elif 'start with' in text_lower:
        return 'start_with'
    elif any(kw in text_lower for kw in ['word', 'less than', 'more than', 'at least']):
        return 'word_limit'
    elif 'format' in text_lower or 'wrap' in text_lower:
        return 'regex_or_format'
    elif 'only output' in text_lower:
        return 'only_output'
    elif 'no explanation' in text_lower or 'do not explain' in text_lower:
        return 'no_explanation'
    else:
        return 'other'


def load_npz(path):
    """Load attention data from npz file"""
    z = np.load(path, allow_pickle=True)
    M = z["mean_layer_head"]  # [L,H]
    per_prompt = z["per_prompt"]  # [P,L,H]
    meta = json.loads(str(z["meta"].tolist()))
    return M, per_prompt, meta


def create_category_grid_plot(category_means, model_tag, out_png, prompts_by_category):
    """
    Create a grid plot showing attention heatmaps for all instruction categories for ONE model
    Similar to the uploaded image
    """
    categories = sorted(category_means.keys())
    n_cats = len(categories)

    # Calculate grid size
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(f'IFEval Instruction Categories - Attention Patterns ({model_tag})',
                 fontsize=16, y=0.995)

    # Flatten axes for easier iteration
    if n_cats == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Find global min/max for consistent colorbar
    all_values = []
    for cat in categories:
        all_values.extend(category_means[cat].flatten())

    # Ensure non-negative values
    vmin = max(0, np.min(all_values))
    vmax = np.max(all_values)

    print(f"  Colorbar range for {model_tag}: [{vmin:.6f}, {vmax:.6f}]")

    # Plot each category
    for idx, cat in enumerate(categories):
        ax = axes[idx]
        attn_matrix = category_means[cat]
        n_prompts = len(prompts_by_category[cat])

        # Ensure no negative values (shouldn't happen but just in case)
        attn_matrix = np.maximum(attn_matrix, 0)

        # Plot heatmap
        im = ax.imshow(attn_matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

        # Format title and labels
        cat_title = cat.replace('_', ' ').title()
        ax.set_title(f'{cat_title}\n(n={n_prompts})', fontsize=11)
        ax.set_xlabel('Head', fontsize=10)
        ax.set_ylabel('Layer', fontsize=10)

        # Add colorbar to each subplot
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention', fontsize=9)

    # Hide unused subplots
    for idx in range(n_cats, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_jsonl", required=True, help="Path to prompts file")
    ap.add_argument("--attn_glob", required=True, help='Glob pattern for attention files')
    ap.add_argument("--out_dir", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load prompts with categories
    prompts_by_idx, prompts_by_category = load_prompts_with_category(args.prompts_jsonl)

    print("\n=== Prompt Categories ===")
    for cat in sorted(prompts_by_category.keys()):
        print(f"{cat}: {len(prompts_by_category[cat])} prompts")
        for idx, data in prompts_by_category[cat]:
            print(f"  - Prompt {idx}: {data.get('key', 'N/A')}")

    # Load all attention files
    files = glob.glob(args.attn_glob)
    if not files:
        print(f"No files found matching: {args.attn_glob}")
        return

    attn_data = {}
    for f in files:
        M, P, meta = load_npz(f)
        tag = meta["run_tag"]
        attn_data[tag] = {"M": M, "P": P, "meta": meta}
        print(f"\nLoaded {tag}: shape {M.shape}, {P.shape[0]} prompts")

    # Sort tags: fp16 first, then gptq by bit number
    def sort_key(tag):
        if tag == 'fp16':
            return (0, 0)
        elif tag.startswith('gptq'):
            bit = int(tag.replace('gptq', ''))
            return (1, bit)
        else:
            return (2, tag)

    sorted_tags = sorted(attn_data.keys(), key=sort_key)
    print(f"\nModel order: {sorted_tags}")

    # For each model, compute mean attention for each instruction category
    for model_tag in sorted_tags:
        print(f"\n=== Processing model: {model_tag} ===")

        per_prompt = attn_data[model_tag]["P"]  # [num_prompts, L, H]

        # Compute mean attention for each category
        category_means = {}
        for cat, prompt_list in prompts_by_category.items():
            category_attns = []
            for prompt_idx, _ in prompt_list:
                if prompt_idx < per_prompt.shape[0]:
                    category_attns.append(per_prompt[prompt_idx])

            if len(category_attns) > 0:
                # Stack and compute mean across prompts
                attn_stack = np.stack(category_attns, axis=0)  # [n_prompts, L, H]
                mean_attn = attn_stack.mean(axis=0)  # [L, H]

                # Check for any negative or invalid values
                if np.any(mean_attn < 0):
                    print(f"  WARNING: {cat} has negative values! Min={mean_attn.min():.6f}")
                    mean_attn = np.maximum(mean_attn, 0)  # Clip to non-negative

                if np.any(np.isnan(mean_attn)):
                    print(f"  WARNING: {cat} has NaN values!")
                    mean_attn = np.nan_to_num(mean_attn, 0)

                category_means[cat] = mean_attn
                print(
                    f"  {cat}: {len(category_attns)} prompts, mean={mean_attn.mean():.6f}, min={mean_attn.min():.6f}, max={mean_attn.max():.6f}")

        # Create grid plot for this model
        out_png = os.path.join(args.out_dir, f"instruction_categories_{model_tag}.png")
        create_category_grid_plot(category_means, model_tag, out_png, prompts_by_category)

        # Save detailed statistics to CSV
        stats_rows = []
        for cat in sorted(category_means.keys()):
            mean_attn = category_means[cat]
            stats_rows.append({
                'model': model_tag,
                'category': cat,
                'num_prompts': len([p for p in prompts_by_category[cat]]),
                'mean_attention': mean_attn.mean(),
                'std_attention': mean_attn.std(),
                'max_attention': mean_attn.max(),
                'min_attention': mean_attn.min()
            })

        stats_df = pd.DataFrame(stats_rows)
        stats_csv = os.path.join(args.out_dir, f"stats_{model_tag}.csv")
        stats_df.to_csv(stats_csv, index=False)
        print(f"Saved stats: {stats_csv}")

        # Save per-head data for this model
        for cat in sorted(category_means.keys()):
            mean_attn = category_means[cat]
            L, H = mean_attn.shape

            rows = []
            for layer in range(L):
                for head in range(H):
                    rows.append({
                        'model': model_tag,
                        'category': cat,
                        'layer': layer,
                        'head': head,
                        'attention': mean_attn[layer, head]
                    })

            df = pd.DataFrame(rows)
            csv_path = os.path.join(args.out_dir, f"per_head_{model_tag}_{cat}.csv")
            df.to_csv(csv_path, index=False)

    # Create comparison across models
    print("\n=== Creating cross-model comparison ===")

    # Collect all data for comparison
    all_stats = []
    for model_tag in sorted_tags:
        per_prompt = attn_data[model_tag]["P"]

        for cat, prompt_list in prompts_by_category.items():
            category_attns = []
            for prompt_idx, _ in prompt_list:
                if prompt_idx < per_prompt.shape[0]:
                    category_attns.append(per_prompt[prompt_idx])

            if len(category_attns) > 0:
                attn_stack = np.stack(category_attns, axis=0)
                mean_attn = attn_stack.mean(axis=0)

                all_stats.append({
                    'model': model_tag,
                    'category': cat,
                    'num_prompts': len(category_attns),
                    'mean_attention': mean_attn.mean(),
                    'std_attention': mean_attn.std()
                })

    summary_df = pd.DataFrame(all_stats)
    summary_csv = os.path.join(args.out_dir, "summary_all_models.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    # Create bar chart comparing attention across models for each category
    categories = sorted(prompts_by_category.keys())
    n_cats = len(categories)

    fig, axes = plt.subplots(1, n_cats, figsize=(5 * n_cats, 5))
    if n_cats == 1:
        axes = [axes]

    for idx, cat in enumerate(categories):
        ax = axes[idx]

        # Filter data for this category
        cat_data = summary_df[summary_df['category'] == cat]

        models = []
        attentions = []
        colors = []

        for model_tag in sorted_tags:
            model_data = cat_data[cat_data['model'] == model_tag]
            if len(model_data) > 0:
                models.append(model_tag)
                attentions.append(model_data['mean_attention'].values[0])

                if model_tag == 'fp16':
                    colors.append('green')
                elif model_tag.startswith('gptq'):
                    bit = int(model_tag.replace('gptq', ''))
                    colors.append(plt.cm.Reds(0.3 + 0.6 * (8 - bit) / 6))
                else:
                    colors.append('gray')

        bars = ax.bar(models, attentions, color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(f'{cat.replace("_", " ").title()}\n({len(prompts_by_category[cat])} prompts)')
        ax.set_ylabel('Mean Instruction Attention')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    comparison_png = os.path.join(args.out_dir, "comparison_across_models.png")
    plt.savefig(comparison_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison: {comparison_png}")

    # Create degradation plot if we have fp16
    if 'fp16' in attn_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        for cat in categories:
            cat_data = summary_df[summary_df['category'] == cat]

            # Get FP16 baseline
            fp16_data = cat_data[cat_data['model'] == 'fp16']
            if len(fp16_data) == 0:
                continue

            fp16_mean = fp16_data['mean_attention'].values[0]

            bits = []
            degradations = []

            for model_tag in sorted_tags:
                if model_tag.startswith('gptq'):
                    model_data = cat_data[cat_data['model'] == model_tag]
                    if len(model_data) > 0:
                        bit = int(model_tag.replace('gptq', ''))
                        quant_mean = model_data['mean_attention'].values[0]
                        degradation = (fp16_mean - quant_mean) / fp16_mean * 100

                        bits.append(bit)
                        degradations.append(degradation)

            if bits:
                ax.plot(bits, degradations, marker='o', label=cat.replace('_', ' ').title(),
                        linewidth=2, markersize=8)

        ax.set_xlabel('Quantization Bits', fontsize=12)
        ax.set_ylabel('Attention Degradation (%)', fontsize=12)
        ax.set_title('Instruction Attention Degradation by Quantization Level', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        if any(tag.startswith('gptq') for tag in sorted_tags):
            gptq_bits = sorted([int(t.replace('gptq', '')) for t in sorted_tags if t.startswith('gptq')])
            ax.set_xticks(gptq_bits)

        plt.tight_layout()
        degradation_png = os.path.join(args.out_dir, "degradation_by_bit.png")
        plt.savefig(degradation_png, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved degradation: {degradation_png}")

    print(f"\n✅ All done! Check outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()