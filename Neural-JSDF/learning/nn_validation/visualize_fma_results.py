"""
Visualize FMA-level quantization results
"""

import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("FMA-LEVEL QUANTIZATION RESULTS VISUALIZATION")
print("="*80)

# Load summary data
summary_file = 'fma_quant_summary.npz'
if not os.path.exists(summary_file):
    print(f"Error: {summary_file} not found. Run validation_fma_quant.py first.")
    exit(1)

data = np.load(summary_file)
configs = data['config_names']
mean_errors = data['mean_errors']
max_errors = data['max_errors']
times = data['nn_times']
exp_bits = data['exponent_bits']
mant_bits = data['mantissa_bits']

print(f"\nLoaded {len(configs)} configurations:")
for i, config in enumerate(configs):
    print(f"  {config}: Mean={mean_errors[i]:.2f}cm, Max={max_errors[i]:.2f}cm, Time={times[i]:.1f}s")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('FMA-Level Quantization Analysis\n(Output-Stationary Dataflow with Per-FMA Quantization)',
             fontsize=14, fontweight='bold')

# Prepare labels
labels = [f"{c}\n(E{exp_bits[i]}M{mant_bits[i]})"
          for i, c in enumerate(configs)]

# 1. Mean Error Bar Chart
ax1 = axes[0, 0]
bars = ax1.bar(range(len(configs)), mean_errors, color=['green', 'orange', 'red', 'darkred', 'purple'])
ax1.set_xlabel('Configuration', fontweight='bold')
ax1.set_ylabel('Mean Error (cm)', fontweight='bold')
ax1.set_title('Mean Distance Error', fontweight='bold')
ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, mean_errors)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}cm',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2. Max Error Bar Chart
ax2 = axes[0, 1]
bars = ax2.bar(range(len(configs)), max_errors, color=['green', 'orange', 'red', 'darkred', 'purple'])
ax2.set_xlabel('Configuration', fontweight='bold')
ax2.set_ylabel('Max Error (cm)', fontweight='bold')
ax2.set_title('Maximum Distance Error', fontweight='bold')
ax2.set_xticks(range(len(configs)))
ax2.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, max_errors)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}cm',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 3. Inference Time
ax3 = axes[1, 0]
bars = ax3.bar(range(len(configs)), times, color=['green', 'orange', 'red', 'darkred', 'purple'])
ax3.set_xlabel('Configuration', fontweight='bold')
ax3.set_ylabel('Time (seconds)', fontweight='bold')
ax3.set_title('Inference Time (100 samples)', fontweight='bold')
ax3.set_xticks(range(len(configs)))
ax3.set_xticklabels(labels, rotation=0, ha='center', fontsize=9)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, times)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.1f}s',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 4. Mantissa bits vs Error
ax4 = axes[1, 1]
ax4.plot(mant_bits, mean_errors, 'o-', linewidth=2, markersize=8, label='Mean Error', color='blue')
ax4.plot(mant_bits, max_errors, 's--', linewidth=2, markersize=8, label='Max Error', color='red')
ax4.set_xlabel('Mantissa Bits', fontweight='bold')
ax4.set_ylabel('Error (cm)', fontweight='bold')
ax4.set_title('Error vs Mantissa Precision', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right', fontsize=10)
ax4.invert_xaxis()  # Higher precision on right

# Add annotations for key points
for i, (m, mean, max_e) in enumerate(zip(mant_bits, mean_errors, max_errors)):
    if i % 2 == 0:  # Annotate every other point to avoid clutter
        ax4.annotate(f'{mean:.1f}', xy=(m, mean), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='blue')

plt.tight_layout()

# Save figure
output_file = 'fma_quant_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Create comparison table
print("\n" + "="*80)
print("DETAILED COMPARISON TABLE")
print("="*80)
print(f"{'Config':<12} {'Exp':>4} {'Mant':>5} {'Mean (cm)':>10} {'Max (cm)':>10} {'Time (s)':>10}")
print("-"*80)
for i, config in enumerate(configs):
    print(f"{config:<12} {exp_bits[i]:>4} {mant_bits[i]:>5} "
          f"{mean_errors[i]:>10.2f} {max_errors[i]:>10.2f} {times[i]:>10.1f}")
print("="*80)

# Calculate quantization overhead
baseline_error = mean_errors[0]  # FP32 baseline
print(f"\nQuantization Error Analysis (relative to FP32 baseline: {baseline_error:.2f}cm):")
print("-"*80)
for i, config in enumerate(configs[1:], 1):
    overhead = mean_errors[i] - baseline_error
    overhead_pct = (overhead / baseline_error) * 100 if baseline_error > 0 else 0
    print(f"  {config}: +{overhead:.2f}cm ({overhead_pct:.1f}% increase)")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)

plt.show()
