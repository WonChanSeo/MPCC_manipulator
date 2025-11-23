"""
Visualization script for mantissa precision comparison results
Generates plots comparing accuracy degradation across different mantissa bit configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import sys
import os

# Check if summary file exists
summary_file = "precision_comparison_summary.npz"
if not os.path.exists(summary_file):
    print(f"Error: {summary_file} not found")
    print("Run validation_precision.py first to generate comparison data")
    sys.exit(1)

# Load summary data
print("Loading precision comparison results...")
data = np.load(summary_file)
mantissa_bits = data['mantissa_bits']
mean_errors = data['mean_errors']
std_errors = data['std_errors']
max_errors = data['max_errors']
median_errors = data['median_errors']
p95_errors = data['p95_errors']
p99_errors = data['p99_errors']
nn_times = data['nn_times']

print(f"Loaded results for {len(mantissa_bits)} precision configurations")
print(f"Mantissa bits: {mantissa_bits}")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Mean Error vs Mantissa Bits
ax1 = plt.subplot(3, 3, 1)
ax1.plot(mantissa_bits, mean_errors, 'o-', linewidth=2, markersize=8, color='#2E86AB')
ax1.fill_between(mantissa_bits,
                  np.array(mean_errors) - np.array(std_errors),
                  np.array(mean_errors) + np.array(std_errors),
                  alpha=0.3, color='#2E86AB')
ax1.set_xlabel('Mantissa Bits', fontsize=11)
ax1.set_ylabel('Mean L1 Error (cm)', fontsize=11)
ax1.set_title('Mean Error vs Precision', fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(mantissa_bits)

# 2. Max Error vs Mantissa Bits
ax2 = plt.subplot(3, 3, 2)
ax2.plot(mantissa_bits, max_errors, 'o-', linewidth=2, markersize=8, color='#A23B72')
ax2.set_xlabel('Mantissa Bits', fontsize=11)
ax2.set_ylabel('Max Error (cm)', fontsize=11)
ax2.set_title('Maximum Error vs Precision', fontweight='bold', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(mantissa_bits)

# 3. Percentile Errors
ax3 = plt.subplot(3, 3, 3)
ax3.plot(mantissa_bits, median_errors, 'o-', linewidth=2, markersize=8, label='Median (50th)', color='#18A558')
ax3.plot(mantissa_bits, p95_errors, 's-', linewidth=2, markersize=8, label='95th percentile', color='#F18F01')
ax3.plot(mantissa_bits, p99_errors, '^-', linewidth=2, markersize=8, label='99th percentile', color='#C73E1D')
ax3.set_xlabel('Mantissa Bits', fontsize=11)
ax3.set_ylabel('Error (cm)', fontsize=11)
ax3.set_title('Error Percentiles vs Precision', fontweight='bold', fontsize=12)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(mantissa_bits)

# 4. Relative degradation (compared to FP32 baseline)
ax4 = plt.subplot(3, 3, 4)
# Use FP32 (23-bit mantissa) as baseline
baseline_error = mean_errors[0]  # 23-bit mantissa (fp32)
relative_degradation = [(e - baseline_error) / baseline_error * 100 for e in mean_errors]
ax4.plot(mantissa_bits, relative_degradation, 'o-', linewidth=2, markersize=8, color='#6A4C93')
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax4.set_xlabel('Mantissa Bits', fontsize=11)
ax4.set_ylabel('Relative Error Change (%)', fontsize=11)
ax4.set_title('Error Change vs FP32 Baseline', fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(mantissa_bits)

# 5. Inference time comparison
ax5 = plt.subplot(3, 3, 5)
ax5.bar(mantissa_bits, nn_times, color='#1982C4', alpha=0.7)
ax5.set_xlabel('Mantissa Bits', fontsize=11)
ax5.set_ylabel('Inference Time (s)', fontsize=11)
ax5.set_title('Inference Time vs Precision', fontweight='bold', fontsize=12)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(mantissa_bits)

# 6. Error distribution histogram (all precisions overlaid)
ax6 = plt.subplot(3, 3, 6)
colors = plt.cm.viridis(np.linspace(0, 1, len(mantissa_bits)))
for i, mb in enumerate(mantissa_bits):
    # Load individual precision data
    data_file = f"data_mantissa_{mb}bit.npz"
    if os.path.exists(data_file):
        precision_data = np.load(data_file)
        errors_flat = precision_data['errors'].flatten()
        # Plot histogram with transparency
        ax6.hist(errors_flat, bins=50, alpha=0.5, label=f'{mb}-bit',
                color=colors[i], range=(0, 100), density=True)

ax6.set_xlabel('Error (cm)', fontsize=11)
ax6.set_ylabel('Density', fontsize=11)
ax6.set_title('Error Distribution by Precision', fontweight='bold', fontsize=12)
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xlim(0, 100)

# 7. Per-link error comparison (for each precision)
ax7 = plt.subplot(3, 3, 7)
link_ids = np.arange(9)
width = 0.12
offsets = np.linspace(-width*2.5, width*2.5, len(mantissa_bits))

for i, mb in enumerate(mantissa_bits):
    data_file = f"data_mantissa_{mb}bit.npz"
    if os.path.exists(data_file):
        precision_data = np.load(data_file)
        errors = precision_data['errors']
        per_link_mean = [errors[:, link].mean() for link in range(9)]
        ax7.bar(link_ids + offsets[i], per_link_mean, width,
               label=f'{mb}-bit', color=colors[i], alpha=0.8)

ax7.set_xlabel('Link ID', fontsize=11)
ax7.set_ylabel('Mean Error (cm)', fontsize=11)
ax7.set_title('Per-Link Error by Precision', fontweight='bold', fontsize=12)
ax7.legend(fontsize=8, ncol=2)
ax7.grid(True, alpha=0.3, axis='y')
ax7.set_xticks(link_ids)

# 8. Cumulative distribution function (CDF) of errors
ax8 = plt.subplot(3, 3, 8)
for i, mb in enumerate(mantissa_bits):
    data_file = f"data_mantissa_{mb}bit.npz"
    if os.path.exists(data_file):
        precision_data = np.load(data_file)
        errors_flat = precision_data['errors'].flatten()
        errors_sorted = np.sort(errors_flat)
        cdf = np.arange(1, len(errors_sorted) + 1) / len(errors_sorted)
        ax8.plot(errors_sorted, cdf, linewidth=2, label=f'{mb}-bit', color=colors[i])

ax8.set_xlabel('Error (cm)', fontsize=11)
ax8.set_ylabel('Cumulative Probability', fontsize=11)
ax8.set_title('Cumulative Error Distribution', fontweight='bold', fontsize=12)
ax8.legend(fontsize=9)
ax8.grid(True, alpha=0.3)
ax8.set_xlim(0, 100)
ax8.set_ylim(0, 1)

# 9. Memory/Accuracy Trade-off
ax9 = plt.subplot(3, 3, 9)
# Memory usage proportional to mantissa bits (simplified)
# Actual: sign(1) + exp(8) + mantissa(m) = 9 + m bits
memory_relative = [(9 + mb) / (9 + 7) for mb in mantissa_bits]  # Relative to bf16
ax9.scatter(memory_relative, mean_errors, s=200, c=mantissa_bits,
           cmap='viridis', edgecolors='black', linewidths=1.5)
for i, mb in enumerate(mantissa_bits):
    ax9.annotate(f'{mb}-bit', (memory_relative[i], mean_errors[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)
ax9.set_xlabel('Relative Memory Usage', fontsize=11)
ax9.set_ylabel('Mean Error (cm)', fontsize=11)
ax9.set_title('Memory-Accuracy Trade-off', fontweight='bold', fontsize=12)
ax9.grid(True, alpha=0.3)

# Adjust layout
plt.suptitle('Neural-JSDF Mantissa Precision Comparison',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
output_file = 'precision_comparison_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Create a second figure focusing on key metrics
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error metrics summary
ax1.plot(mantissa_bits, mean_errors, 'o-', linewidth=2.5, markersize=10, label='Mean', color='#2E86AB')
ax1.plot(mantissa_bits, median_errors, 's-', linewidth=2.5, markersize=10, label='Median', color='#18A558')
ax1.fill_between(mantissa_bits,
                  np.array(mean_errors) - np.array(std_errors),
                  np.array(mean_errors) + np.array(std_errors),
                  alpha=0.2, color='#2E86AB', label='±1 std')
ax1.set_xlabel('Mantissa Bits', fontsize=13)
ax1.set_ylabel('Error (cm)', fontsize=13)
ax1.set_title('Central Tendency Errors', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(mantissa_bits)

# Plot 2: Tail errors
ax2.plot(mantissa_bits, max_errors, '^-', linewidth=2.5, markersize=10, label='Max', color='#C73E1D')
ax2.plot(mantissa_bits, p99_errors, 's-', linewidth=2.5, markersize=10, label='99th %ile', color='#F18F01')
ax2.plot(mantissa_bits, p95_errors, 'o-', linewidth=2.5, markersize=10, label='95th %ile', color='#FFA630')
ax2.set_xlabel('Mantissa Bits', fontsize=13)
ax2.set_ylabel('Error (cm)', fontsize=13)
ax2.set_title('Tail Error Metrics', fontweight='bold', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(mantissa_bits)

# Plot 3: Relative degradation with annotations
ax3.plot(mantissa_bits, relative_degradation, 'o-', linewidth=2.5, markersize=10, color='#6A4C93')
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='BF16 baseline')
# Add colored zones
ax3.axhspan(-5, 5, alpha=0.1, color='green', label='Negligible (<5%)')
ax3.axhspan(-10, -5, alpha=0.1, color='yellow')
ax3.axhspan(5, 10, alpha=0.1, color='yellow', label='Acceptable (5-10%)')
# Add value annotations
for i, (mb, rd) in enumerate(zip(mantissa_bits, relative_degradation)):
    ax3.annotate(f'{rd:+.1f}%', (mb, rd), xytext=(0, 10 if rd > 0 else -15),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Mantissa Bits', fontsize=13)
ax3.set_ylabel('Relative Error Change (%)', fontsize=13)
ax3.set_title('Degradation vs FP32 Baseline', fontweight='bold', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(mantissa_bits)

# Plot 4: Summary statistics table
ax4.axis('off')
# Create table data
table_data = []
table_data.append(['Mantissa', 'Mean Err', 'Max Err', '95th %ile', 'vs FP32'])
table_data.append(['Bits', '(cm)', '(cm)', '(cm)', '(%)'])
for i, mb in enumerate(mantissa_bits):
    row = [
        f'{mb}',
        f'{mean_errors[i]:.2f}',
        f'{max_errors[i]:.2f}',
        f'{p95_errors[i]:.2f}',
        f'{relative_degradation[i]:+.1f}'
    ]
    table_data.append(row)

table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.2, 0.2, 0.2, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header rows
for i in range(5):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#A0C4D4')
    table[(1, i)].set_text_props(weight='bold')

# Color code the degradation column
for i in range(2, len(table_data)):
    degradation = relative_degradation[i-2]
    if abs(degradation) < 5:
        color = '#D4EDDA'  # Green
    elif abs(degradation) < 10:
        color = '#FFF3CD'  # Yellow
    else:
        color = '#F8D7DA'  # Red
    table[(i, 4)].set_facecolor(color)

ax4.set_title('Summary Statistics', fontweight='bold', fontsize=14, pad=20)

plt.suptitle('Neural-JSDF Precision Analysis - Key Metrics',
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save second figure
output_file2 = 'precision_comparison_summary.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✓ Summary visualization saved to: {output_file2}")

# Print text summary
print("\n" + "="*80)
print("PRECISION COMPARISON SUMMARY")
print("="*80)
print(f"\n{'Mantissa Bits':<15} {'Mean Error':<15} {'Degradation':<20} {'Quality':<15}")
print(f"{'':15} {'(cm)':<15} {'vs FP32 (%)':<20} {'Assessment':<15}")
print("-"*80)

for i, mb in enumerate(mantissa_bits):
    if abs(relative_degradation[i]) < 5:
        quality = "EXCELLENT"
    elif abs(relative_degradation[i]) < 10:
        quality = "GOOD"
    elif abs(relative_degradation[i]) < 20:
        quality = "ACCEPTABLE"
    else:
        quality = "POOR"

    print(f"{mb:<15} {mean_errors[i]:<15.2f} {relative_degradation[i]:>+8.2f}           {quality:<15}")

print("="*80)
print("\nVisualization complete!")
print(f"  • {output_file}")
print(f"  • {output_file2}")
print("="*80)
