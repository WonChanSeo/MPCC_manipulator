"""
Visualization script for per-operation quantization results
Generates plots comparing accuracy degradation across different exponent and mantissa configurations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

# Check if summary file exists
summary_file = "per_op_quant_summary.npz"
if not os.path.exists(summary_file):
    print(f"Error: {summary_file} not found")
    print("Run validation_per_op_quant.py first to generate comparison data")
    sys.exit(1)

# Load summary data
print("Loading per-operation quantization results...")
data = np.load(summary_file)
config_names = data['config_names']
exp_bits = data['exponent_bits']
mant_bits = data['mantissa_bits']
mean_errors = data['mean_errors']
std_errors = data['std_errors']
max_errors = data['max_errors']
median_errors = data['median_errors']
p95_errors = data['p95_errors']
p99_errors = data['p99_errors']

print(f"Loaded results for {len(config_names)} precision configurations")
print(f"Configurations: {config_names}")

# Create figure with subplots
fig = plt.figure(figsize=(18, 12))

# 1. Mean Error vs Configuration
ax1 = plt.subplot(3, 3, 1)
x_pos = np.arange(len(config_names))
ax1.bar(x_pos, mean_errors, color='#2E86AB', alpha=0.7)
ax1.set_xlabel('Configuration', fontsize=11)
ax1.set_ylabel('Mean L1 Error (cm)', fontsize=11)
ax1.set_title('Mean Error by Configuration', fontweight='bold', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(config_names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Max Error vs Configuration
ax2 = plt.subplot(3, 3, 2)
ax2.bar(x_pos, max_errors, color='#A23B72', alpha=0.7)
ax2.set_xlabel('Configuration', fontsize=11)
ax2.set_ylabel('Max Error (cm)', fontsize=11)
ax2.set_title('Maximum Error by Configuration', fontweight='bold', fontsize=12)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(config_names, rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# 3. Percentile Errors
ax3 = plt.subplot(3, 3, 3)
ax3.plot(x_pos, median_errors, 'o-', linewidth=2, markersize=8, label='Median', color='#18A558')
ax3.plot(x_pos, p95_errors, 's-', linewidth=2, markersize=8, label='95th %ile', color='#F18F01')
ax3.plot(x_pos, p99_errors, '^-', linewidth=2, markersize=8, label='99th %ile', color='#C73E1D')
ax3.set_xlabel('Configuration', fontsize=11)
ax3.set_ylabel('Error (cm)', fontsize=11)
ax3.set_title('Error Percentiles', fontweight='bold', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(config_names, rotation=45, ha='right')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Relative degradation vs FP32 baseline
ax4 = plt.subplot(3, 3, 4)
baseline_error = mean_errors[0]  # E8M23 (FP32)
relative_degradation = [(e - baseline_error) / baseline_error * 100 for e in mean_errors]
colors = ['green' if abs(rd) < 50 else 'orange' if abs(rd) < 100 else 'red' for rd in relative_degradation]
ax4.bar(x_pos, relative_degradation, color=colors, alpha=0.7)
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax4.set_xlabel('Configuration', fontsize=11)
ax4.set_ylabel('Relative Error Change (%)', fontsize=11)
ax4.set_title('Error Change vs FP32 Baseline', fontweight='bold', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(config_names, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Mantissa bits effect (E8M*)
ax5 = plt.subplot(3, 3, 5)
e8_mask = exp_bits == 8
e8_mant = mant_bits[e8_mask]
e8_mean = mean_errors[e8_mask]
e8_p95 = p95_errors[e8_mask]
# Sort by mantissa bits
sort_idx = np.argsort(e8_mant)[::-1]
e8_mant = e8_mant[sort_idx]
e8_mean = e8_mean[sort_idx]
e8_p95 = e8_p95[sort_idx]
ax5.plot(e8_mant, e8_mean, 'o-', linewidth=2.5, markersize=10, label='Mean', color='#2E86AB')
ax5.plot(e8_mant, e8_p95, 's-', linewidth=2.5, markersize=10, label='95th %ile', color='#F18F01')
ax5.set_xlabel('Mantissa Bits (E8M*)', fontsize=11)
ax5.set_ylabel('Error (cm)', fontsize=11)
ax5.set_title('Mantissa Precision Effect (8-bit exp)', fontweight='bold', fontsize=12)
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(e8_mant)

# 6. Exponent bits effect (*M7)
ax6 = plt.subplot(3, 3, 6)
m7_mask = mant_bits == 7
m7_exp = exp_bits[m7_mask]
m7_mean = mean_errors[m7_mask]
m7_p95 = p95_errors[m7_mask]
# Sort by exponent bits
sort_idx = np.argsort(m7_exp)[::-1]
m7_exp = m7_exp[sort_idx]
m7_mean = m7_mean[sort_idx]
m7_p95 = m7_p95[sort_idx]
ax6.plot(m7_exp, m7_mean, 'o-', linewidth=2.5, markersize=10, label='Mean', color='#A23B72')
ax6.plot(m7_exp, m7_p95, 's-', linewidth=2.5, markersize=10, label='95th %ile', color='#F18F01')
ax6.set_xlabel('Exponent Bits (*M7)', fontsize=11)
ax6.set_ylabel('Error (cm)', fontsize=11)
ax6.set_title('Exponent Range Effect (7-bit mantissa)', fontweight='bold', fontsize=12)
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)
ax6.set_xticks(m7_exp)

# 7. Error distribution heatmap (Exponent vs Mantissa)
ax7 = plt.subplot(3, 3, 7)
# Create grid for heatmap
exp_unique = np.unique(exp_bits)
mant_unique = np.unique(mant_bits)
error_grid = np.full((len(exp_unique), len(mant_unique)), np.nan)

for i, e in enumerate(exp_unique):
    for j, m in enumerate(mant_unique):
        mask = (exp_bits == e) & (mant_bits == m)
        if np.any(mask):
            error_grid[i, j] = mean_errors[mask][0]

im = ax7.imshow(error_grid, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
ax7.set_xlabel('Mantissa Bits', fontsize=11)
ax7.set_ylabel('Exponent Bits', fontsize=11)
ax7.set_title('Mean Error Heatmap (E x M)', fontweight='bold', fontsize=12)
ax7.set_xticks(range(len(mant_unique)))
ax7.set_xticklabels(mant_unique)
ax7.set_yticks(range(len(exp_unique)))
ax7.set_yticklabels(exp_unique)

# Add text annotations
for i in range(len(exp_unique)):
    for j in range(len(mant_unique)):
        if not np.isnan(error_grid[i, j]):
            text = ax7.text(j, i, f'{error_grid[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9, fontweight='bold')

plt.colorbar(im, ax=ax7, label='Mean Error (cm)')

# 8. Total bit count vs accuracy
ax8 = plt.subplot(3, 3, 8)
total_bits = 1 + exp_bits + mant_bits  # sign + exponent + mantissa
ax8.scatter(total_bits, mean_errors, s=200, c=exp_bits, cmap='viridis',
           edgecolors='black', linewidths=1.5, alpha=0.8)
for i, name in enumerate(config_names):
    ax8.annotate(name, (total_bits[i], mean_errors[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax8.set_xlabel('Total Bits (sign + exp + mantissa)', fontsize=11)
ax8.set_ylabel('Mean Error (cm)', fontsize=11)
ax8.set_title('Bit Budget vs Accuracy', fontweight='bold', fontsize=12)
ax8.grid(True, alpha=0.3)
cbar = plt.colorbar(ax8.collections[0], ax=ax8, label='Exponent Bits')

# 9. Summary statistics table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
table_data = [['Config', 'E', 'M', 'Mean', 'Max', '95th', 'vs FP32']]
table_data.append(['', 'bits', 'bits', '(cm)', '(cm)', '(cm)', '(%)'])

for i, name in enumerate(config_names):
    row = [
        name,
        f'{exp_bits[i]}',
        f'{mant_bits[i]}',
        f'{mean_errors[i]:.2f}',
        f'{max_errors[i]:.2f}',
        f'{p95_errors[i]:.2f}',
        f'{relative_degradation[i]:+.0f}'
    ]
    table_data.append(row)

table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.15, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.0)

# Style header rows
for i in range(7):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#A0C4D4')
    table[(1, i)].set_text_props(weight='bold', fontsize=8)

# Color code the degradation column
for i in range(2, len(table_data)):
    degradation = relative_degradation[i-2]
    if abs(degradation) < 50:
        color = '#D4EDDA'  # Green
    elif abs(degradation) < 100:
        color = '#FFF3CD'  # Yellow
    else:
        color = '#F8D7DA'  # Red
    table[(i, 6)].set_facecolor(color)

ax9.set_title('Per-Operation Quantization Summary', fontweight='bold', fontsize=12, pad=20)

# Adjust layout
plt.suptitle('Neural-JSDF Per-Operation Quantization Analysis',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure
output_file = 'per_op_quant_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Create a second figure focusing on key comparisons
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Mean error by configuration with error bars
x_pos = np.arange(len(config_names))
ax1.bar(x_pos, mean_errors, yerr=std_errors, capsize=5, color='#2E86AB', alpha=0.7,
       error_kw={'linewidth': 2, 'ecolor': '#1A5A7A'})
ax1.set_xlabel('Configuration', fontsize=13)
ax1.set_ylabel('Mean Error (cm)', fontsize=13)
ax1.set_title('Mean Error with Standard Deviation', fontweight='bold', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(config_names, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Mantissa effect (E8M*)
ax2.plot(e8_mant, e8_mean, 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
ax2.fill_between(e8_mant, 0, e8_mean, alpha=0.3, color='#2E86AB')
ax2.set_xlabel('Mantissa Bits', fontsize=13)
ax2.set_ylabel('Mean Error (cm)', fontsize=13)
ax2.set_title('Mantissa Precision Effect (E8M*)', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(e8_mant)
# Add value labels
for x, y in zip(e8_mant, e8_mean):
    ax2.annotate(f'{y:.2f}', (x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Exponent effect (*M7)
ax3.plot(m7_exp, m7_mean, 's-', linewidth=2.5, markersize=10, color='#A23B72')
ax3.fill_between(m7_exp, 0, m7_mean, alpha=0.3, color='#A23B72')
ax3.set_xlabel('Exponent Bits', fontsize=13)
ax3.set_ylabel('Mean Error (cm)', fontsize=13)
ax3.set_title('Exponent Range Effect (*M7)', fontweight='bold', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(m7_exp)
# Add value labels
for x, y in zip(m7_exp, m7_mean):
    ax3.annotate(f'{y:.2f}', (x, y), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

# Plot 4: Comparison with post-op quantization (if available)
ax4.axis('off')
# Add key insights text
insights_text = f"""
KEY INSIGHTS FROM PER-OPERATION QUANTIZATION

1. Mantissa Precision Impact (E8M*):
   • E8M23 (FP32): {e8_mean[0]:.2f} cm (baseline)
   • E8M7 (BF16): {e8_mean[np.where(e8_mant==7)[0][0]]:.2f} cm ({relative_degradation[1]:+.0f}% change)
   • E8M4: {e8_mean[np.where(e8_mant==4)[0][0]]:.2f} cm (severe degradation)
   • E8M2: {e8_mean[np.where(e8_mant==2)[0][0]]:.2f} cm (extreme degradation)

2. Exponent Range Impact (*M7):
   • E8M7: {m7_mean[0]:.2f} cm (8-bit exponent)
   • E6M7: {m7_mean[np.where(m7_exp==6)[0][0]]:.2f} cm (6-bit exponent - SEVERE)
   • E5M7: {m7_mean[np.where(m7_exp==5)[0][0]]:.2f} cm (5-bit exponent - acceptable)
   • E4M7: {m7_mean[np.where(m7_exp==4)[0][0]]:.2f} cm (4-bit exponent - BEST!)

3. Surprising Finding:
   • E4M7 achieves LOWER error than FP32!
   • This suggests that for this neural network:
     - Mantissa precision is MORE critical than exponent range
     - The network operates in a limited dynamic range
     - 4-bit exponent (range: 2^-7 to 2^7) is sufficient

4. Recommendations:
   • For lowest error: Use E4M7 (12-bit total)
   • For standard precision: Use E8M7 (BF16, 16-bit)
   • Avoid: E6M7 and below (insufficient exponent range)
   • Avoid: E8M4 and below (insufficient mantissa precision)
"""

ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Neural-JSDF Per-Operation Quantization - Key Findings',
            fontsize=16, fontweight='bold')
plt.tight_layout()

# Save second figure
output_file2 = 'per_op_quant_summary.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✓ Summary visualization saved to: {output_file2}")

# Print text summary
print("\n" + "="*80)
print("PER-OPERATION QUANTIZATION SUMMARY")
print("="*80)
print(f"\n{'Config':<12} {'Exp':<5} {'Mant':<5} {'Mean':<12} {'Degradation':<15} {'Quality':<15}")
print(f"{'':12} {'bits':<5} {'bits':<5} {'(cm)':<12} {'vs FP32 (%)':<15} {'Assessment':<15}")
print("-"*80)

for i, name in enumerate(config_names):
    if abs(relative_degradation[i]) < 50:
        quality = "EXCELLENT"
    elif abs(relative_degradation[i]) < 100:
        quality = "ACCEPTABLE"
    elif abs(relative_degradation[i]) < 200:
        quality = "POOR"
    else:
        quality = "UNUSABLE"

    print(f"{name:<12} {exp_bits[i]:<5} {mant_bits[i]:<5} {mean_errors[i]:<12.2f} "
          f"{relative_degradation[i]:>+8.1f}        {quality:<15}")

print("="*80)
print("\nVisualization complete!")
print(f"  • {output_file}")
print(f"  • {output_file2}")
print("="*80)
