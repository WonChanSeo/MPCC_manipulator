"""
Compare Per-Op vs FMA-level quantization
"""

import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("PER-OP vs FMA-LEVEL QUANTIZATION COMPARISON")
print("="*80)

# Load Per-Op data
perop_file = 'perop_quant_summary.npz'
fma_file = 'fma_quant_summary.npz'

if not os.path.exists(perop_file):
    print(f"Warning: {perop_file} not found. Skipping Per-Op comparison.")
    perop_data = None
else:
    perop_data = np.load(perop_file)
    print(f"✓ Loaded Per-Op data: {len(perop_data['configs'])} configurations")

if not os.path.exists(fma_file):
    print(f"Error: {fma_file} not found. Run validation_fma_quant.py first.")
    exit(1)

fma_data = np.load(fma_file)
print(f"✓ Loaded FMA data: {len(fma_data['configs'])} configurations")

# Create comparison visualization
if perop_data is not None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Operation vs FMA-Level Quantization Comparison',
                 fontsize=16, fontweight='bold')

    # Find common configurations
    common_configs = []
    perop_indices = []
    fma_indices = []

    for i, fma_config in enumerate(fma_data['configs']):
        for j, perop_config in enumerate(perop_data['configs']):
            if (fma_data['exp_bits'][i] == perop_data['exp_bits'][j] and
                fma_data['mant_bits'][i] == perop_data['mant_bits'][j]):
                common_configs.append(fma_config)
                fma_indices.append(i)
                perop_indices.append(j)
                break

    print(f"\nFound {len(common_configs)} common configurations:")
    for config in common_configs:
        print(f"  - {config}")

    if len(common_configs) == 0:
        print("No common configurations found. Cannot create comparison.")
        exit(1)

    # Extract data for common configs
    perop_means = perop_data['mean_errors'][perop_indices]
    fma_means = fma_data['mean_errors'][fma_indices]
    perop_maxs = perop_data['max_errors'][perop_indices]
    fma_maxs = fma_data['max_errors'][fma_indices]
    mantissa_bits = fma_data['mant_bits'][fma_indices]

    labels = [f"E{fma_data['exp_bits'][i]}M{fma_data['mant_bits'][i]}"
              for i in fma_indices]

    # 1. Mean Error Comparison
    ax1 = axes[0, 0]
    x = np.arange(len(common_configs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, perop_means, width, label='Per-Op', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, fma_means, width, label='FMA-Level', color='coral', alpha=0.8)

    ax1.set_xlabel('Configuration', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Mean Error (cm)', fontweight='bold', fontsize=11)
    ax1.set_title('Mean Error Comparison', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    # 2. Max Error Comparison
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, perop_maxs, width, label='Per-Op', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, fma_maxs, width, label='FMA-Level', color='coral', alpha=0.8)

    ax2.set_xlabel('Configuration', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Max Error (cm)', fontweight='bold', fontsize=11)
    ax2.set_title('Maximum Error Comparison', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=8)

    # 3. Error vs Mantissa Bits
    ax3 = axes[1, 0]
    ax3.plot(mantissa_bits, perop_means, 'o-', linewidth=2, markersize=10,
             label='Per-Op Mean', color='blue')
    ax3.plot(mantissa_bits, fma_means, 's-', linewidth=2, markersize=10,
             label='FMA Mean', color='red')
    ax3.set_xlabel('Mantissa Bits', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Mean Error (cm)', fontweight='bold', fontsize=11)
    ax3.set_title('Mean Error vs Precision', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # 4. Error Ratio (FMA / Per-Op)
    ax4 = axes[1, 1]
    error_ratio = fma_means / perop_means
    bars = ax4.bar(x, error_ratio, color='purple', alpha=0.7)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Equal (ratio=1)')

    ax4.set_xlabel('Configuration', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Error Ratio (FMA / Per-Op)', fontweight='bold', fontsize=11)
    ax4.set_title('FMA-Level Error Overhead', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=10)
    ax4.legend(fontsize=10)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, ratio) in enumerate(zip(bars, error_ratio)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{ratio:.2f}x',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save figure
    output_file = 'perop_vs_fma_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Comparison visualization saved to: {output_file}")

    # Print comparison table
    print("\n" + "="*100)
    print("DETAILED COMPARISON TABLE")
    print("="*100)
    print(f"{'Config':<10} {'Per-Op Mean':>12} {'FMA Mean':>12} {'Difference':>12} {'Ratio':>10} "
          f"{'Per-Op Max':>12} {'FMA Max':>12}")
    print("-"*100)
    for i, config in enumerate(common_configs):
        diff = fma_means[i] - perop_means[i]
        ratio = fma_means[i] / perop_means[i] if perop_means[i] > 0 else 0
        print(f"{labels[i]:<10} {perop_means[i]:>12.2f} {fma_means[i]:>12.2f} "
              f"{diff:>12.2f} {ratio:>10.2f}x {perop_maxs[i]:>12.2f} {fma_maxs[i]:>12.2f}")
    print("="*100)

    # Analysis
    print("\nKey Findings:")
    print("-"*80)
    print(f"  • Per-Op quantizes at layer boundaries (~10 quantizations)")
    print(f"  • FMA-Level quantizes every accumulation (~205,569 quantizations)")
    print(f"  • FMA-Level represents worst-case hardware behavior")
    print(f"  • Average error overhead (FMA/Per-Op): {np.mean(error_ratio):.2f}x")
    print(f"  • Per-Op is more realistic for actual GPU/TPU hardware")
    print("="*80)

    plt.show()

else:
    # Just show FMA results
    print("\nShowing FMA-only results (Per-Op data not available)")
    configs = fma_data['configs']
    mean_errors = fma_data['mean_errors']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bars = ax.bar(range(len(configs)), mean_errors,
                  color=['green', 'orange', 'red', 'darkred', 'purple'])

    ax.set_xlabel('Configuration', fontweight='bold')
    ax.set_ylabel('Mean Error (cm)', fontweight='bold')
    ax.set_title('FMA-Level Quantization Results', fontweight='bold')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fma_only_results.png', dpi=300, bbox_inches='tight')
    print("✓ FMA results saved to: fma_only_results.png")
    plt.show()
