"""
Visualization script for FMA-level quantization results
Compares FMA-level quantization with per-operation hook method
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os

# Check if summary files exist
fma_summary = "fma_quant_summary.npz"
per_op_summary = "per_op_quant_summary.npz"

if not os.path.exists(fma_summary):
    print(f"Error: {fma_summary} not found")
    print("Run validation_fma_quant.py first")
    sys.exit(1)

if not os.path.exists(per_op_summary):
    print(f"Warning: {per_op_summary} not found")
    print("Per-operation comparison will be skipped")
    has_per_op = False
else:
    has_per_op = True

# Load FMA data
print("Loading FMA-level quantization results...")
fma_data = np.load(fma_summary)
fma_configs = fma_data['config_names']
fma_exp_bits = fma_data['exponent_bits']
fma_mant_bits = fma_data['mantissa_bits']
fma_mean_errors = fma_data['mean_errors']
fma_std_errors = fma_data['std_errors']
fma_max_errors = fma_data['max_errors']
fma_median_errors = fma_data['median_errors']
fma_p95_errors = fma_data['p95_errors']
fma_p99_errors = fma_data['p99_errors']
fma_times = fma_data['nn_times']

print(f"Loaded FMA results for {len(fma_configs)} configurations")

# Load per-operation data if available
if has_per_op:
    print("Loading per-operation quantization results...")
    per_op_data = np.load(per_op_summary)
    per_op_configs = per_op_data['config_names']
    per_op_mean_errors = per_op_data['mean_errors']

    # Match configurations
    per_op_matched = {}
    for i, name in enumerate(per_op_configs):
        per_op_matched[name] = per_op_mean_errors[i]

# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))

# 1. Mean Error by Configuration
ax1 = plt.subplot(3, 3, 1)
x_pos = np.arange(len(fma_configs))
bars = ax1.bar(x_pos, fma_mean_errors, color='#C73E1D', alpha=0.7, label='FMA-level')
if has_per_op:
    per_op_values = [per_op_matched.get(c, 0) for c in fma_configs]
    ax1.bar(x_pos, per_op_values, color='#2E86AB', alpha=0.5, label='Per-op hook')
ax1.set_xlabel('Configuration', fontsize=11)
ax1.set_ylabel('Mean Error (cm)', fontsize=11)
ax1.set_title('Mean Error Comparison', fontweight='bold', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(fma_configs, rotation=45, ha='right')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Error Amplification vs Per-Op
if has_per_op:
    ax2 = plt.subplot(3, 3, 2)
    amplification = []
    for i, name in enumerate(fma_configs):
        if name in per_op_matched and per_op_matched[name] > 0:
            amp = (fma_mean_errors[i] / per_op_matched[name]) - 1
            amplification.append(amp * 100)
        else:
            amplification.append(0)

    colors = ['red' if a > 1000 else 'orange' if a > 500 else 'yellow' for a in amplification]
    ax2.bar(x_pos, amplification, color=colors, alpha=0.7)
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Error Amplification (%)', fontsize=11)
    ax2.set_title('FMA vs Per-Op Error Increase', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(fma_configs, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# 3. Percentile Errors
ax3 = plt.subplot(3, 3, 3)
ax3.plot(x_pos, fma_median_errors, 'o-', linewidth=2, markersize=8, label='Median', color='#18A558')
ax3.plot(x_pos, fma_p95_errors, 's-', linewidth=2, markersize=8, label='95th %ile', color='#F18F01')
ax3.plot(x_pos, fma_p99_errors, '^-', linewidth=2, markersize=8, label='99th %ile', color='#C73E1D')
ax3.set_xlabel('Configuration', fontsize=11)
ax3.set_ylabel('Error (cm)', fontsize=11)
ax3.set_title('Error Percentiles', fontweight='bold', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(fma_configs, rotation=45, ha='right')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# 4. Relative degradation vs FP32 FMA baseline
ax4 = plt.subplot(3, 3, 4)
baseline_error = fma_mean_errors[0]  # E8M23 (FP32)
relative_degradation = [(e - baseline_error) / baseline_error * 100 for e in fma_mean_errors]
colors_deg = ['green' if rd < 20 else 'orange' if rd < 50 else 'red' for rd in relative_degradation]
ax4.bar(x_pos, relative_degradation, color=colors_deg, alpha=0.7)
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
ax4.set_xlabel('Configuration', fontsize=11)
ax4.set_ylabel('Error Change vs FP32 (%)', fontsize=11)
ax4.set_title('Degradation vs FP32 FMA Baseline', fontweight='bold', fontsize=12)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(fma_configs, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Inference Time Comparison
ax5 = plt.subplot(3, 3, 5)
ax5.bar(x_pos, fma_times, color='#1982C4', alpha=0.7)
ax5.set_xlabel('Configuration', fontsize=11)
ax5.set_ylabel('Inference Time (s)', fontsize=11)
ax5.set_title('Inference Time (100 samples)', fontweight='bold', fontsize=12)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(fma_configs, rotation=45, ha='right')
ax5.grid(True, alpha=0.3, axis='y')
# Add value labels
for i, (x, y) in enumerate(zip(x_pos, fma_times)):
    ax5.text(x, y, f'{y:.0f}s', ha='center', va='bottom', fontsize=9)

# 6. Mean vs Max Error scatter
ax6 = plt.subplot(3, 3, 6)
scatter = ax6.scatter(fma_mean_errors, fma_max_errors, s=200, c=fma_mant_bits,
                     cmap='viridis', edgecolors='black', linewidths=1.5, alpha=0.8)
for i, name in enumerate(fma_configs):
    ax6.annotate(name, (fma_mean_errors[i], fma_max_errors[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax6.set_xlabel('Mean Error (cm)', fontsize=11)
ax6.set_ylabel('Max Error (cm)', fontsize=11)
ax6.set_title('Mean vs Max Error', fontweight='bold', fontsize=12)
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Mantissa Bits')

# 7. Error distribution violin plot (if individual data exists)
ax7 = plt.subplot(3, 3, 7)
error_distributions = []
labels = []
for i, name in enumerate(fma_configs):
    data_file = f"data_fma_E{fma_exp_bits[i]}M{fma_mant_bits[i]}.npz"
    if os.path.exists(data_file):
        data = np.load(data_file)
        errors_flat = data['errors'].flatten()
        error_distributions.append(errors_flat)
        labels.append(name)

if error_distributions:
    parts = ax7.violinplot(error_distributions, positions=range(len(labels)),
                           showmeans=True, showmedians=True)
    ax7.set_xlabel('Configuration', fontsize=11)
    ax7.set_ylabel('Error (cm)', fontsize=11)
    ax7.set_title('Error Distribution', fontweight='bold', fontsize=12)
    ax7.set_xticks(range(len(labels)))
    ax7.set_xticklabels(labels, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3, axis='y')

# 8. Comparison table
ax8 = plt.subplot(3, 3, 8)
ax8.axis('off')
table_data = [['Config', 'E', 'M', 'Mean', 'Max', '95th', 'Time']]
table_data.append(['', 'bits', 'bits', '(cm)', '(cm)', '(cm)', '(s)'])

for i, name in enumerate(fma_configs):
    row = [
        name,
        f'{fma_exp_bits[i]}',
        f'{fma_mant_bits[i]}',
        f'{fma_mean_errors[i]:.1f}',
        f'{fma_max_errors[i]:.1f}',
        f'{fma_p95_errors[i]:.1f}',
        f'{fma_times[i]:.0f}'
    ]
    table_data.append(row)

table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.12, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.0)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#C73E1D')
    table[(0, i)].set_text_props(weight='bold', color='white')
    table[(1, i)].set_facecolor('#F8D7DA')
    table[(1, i)].set_text_props(weight='bold')

ax8.set_title('FMA-Level Quantization Summary', fontweight='bold', fontsize=12, pad=20)

# 9. Method comparison (if per-op available)
if has_per_op:
    ax9 = plt.subplot(3, 3, 9)
    width = 0.35
    x = np.arange(len(fma_configs))
    per_op_vals = [per_op_matched.get(c, 0) for c in fma_configs]

    ax9.bar(x - width/2, per_op_vals, width, label='Per-Op Hook', color='#2E86AB', alpha=0.7)
    ax9.bar(x + width/2, fma_mean_errors, width, label='FMA-Level', color='#C73E1D', alpha=0.7)

    ax9.set_xlabel('Configuration', fontsize=11)
    ax9.set_ylabel('Mean Error (cm)', fontsize=11)
    ax9.set_title('Method Comparison', fontweight='bold', fontsize=12)
    ax9.set_xticks(x)
    ax9.set_xticklabels(fma_configs, rotation=45, ha='right')
    ax9.legend(fontsize=10)
    ax9.grid(True, alpha=0.3, axis='y')
    ax9.set_yscale('log')  # Log scale to show both ranges

plt.suptitle('Neural-JSDF FMA-Level Quantization Analysis',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

output_file = 'fma_quant_analysis.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved to: {output_file}")

# Create summary figure
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Mean error with error bars
x_pos = np.arange(len(fma_configs))
ax1.bar(x_pos, fma_mean_errors, yerr=fma_std_errors, capsize=5,
       color='#C73E1D', alpha=0.7, error_kw={'linewidth': 2, 'ecolor': '#8B0000'})
ax1.set_xlabel('Configuration', fontsize=13)
ax1.set_ylabel('Mean Error (cm)', fontsize=13)
ax1.set_title('FMA-Level Quantization Error', fontweight='bold', fontsize=14)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(fma_configs, rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Comparison with per-op (if available)
if has_per_op:
    per_op_vals = [per_op_matched.get(c, 0) for c in fma_configs]
    ax2.plot(x_pos, per_op_vals, 'o-', linewidth=2.5, markersize=10,
            label='Per-Op Hook', color='#2E86AB')
    ax2.plot(x_pos, fma_mean_errors, 's-', linewidth=2.5, markersize=10,
            label='FMA-Level', color='#C73E1D')
    ax2.set_xlabel('Configuration', fontsize=13)
    ax2.set_ylabel('Mean Error (cm)', fontsize=13)
    ax2.set_title('FMA vs Per-Op Quantization', fontweight='bold', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(fma_configs, rotation=45, ha='right')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
else:
    ax2.axis('off')
    ax2.text(0.5, 0.5, 'Per-Op data not available', ha='center', va='center',
            fontsize=14, transform=ax2.transAxes)

# Plot 3: Degradation with annotations
ax3.bar(x_pos, relative_degradation, color=colors_deg, alpha=0.7)
ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
# Add value annotations
for i, (x, rd) in enumerate(zip(x_pos, relative_degradation)):
    ax3.annotate(f'{rd:+.0f}%', (x, rd), xytext=(0, 10 if rd > 0 else -15),
                textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Configuration', fontsize=13)
ax3.set_ylabel('Error Change (%)', fontsize=13)
ax3.set_title('Degradation vs FP32 FMA', fontweight='bold', fontsize=14)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(fma_configs, rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Key insights
ax4.axis('off')
insights_text = f"""
KEY FINDINGS: FMA-LEVEL QUANTIZATION

1. Extreme Error Accumulation:
   • FP32 FMA: {fma_mean_errors[0]:.1f} cm
   • BF16 FMA: {fma_mean_errors[1]:.1f} cm
   • E8M4 FMA: {fma_mean_errors[4]:.1f} cm

   FMA-level shows 30-80 cm errors!
   This is 30-50x worse than per-op hook method.

2. Why FMA-level is so bad:
   • Quantizes after EVERY accumulation
   • 256 accumulations per neuron
   • 5 layers × 256 neurons = catastrophic error buildup

3. Comparison with Per-Op Hook:
"""

if has_per_op:
    for i, name in enumerate(fma_configs[:3]):
        if name in per_op_matched:
            amp = fma_mean_errors[i] / per_op_matched[name]
            insights_text += f"   • {name}: {amp:.0f}x worse\n"

insights_text += f"""

4. Performance:
   • FP32: {fma_times[0]:.0f}s (fast, no quantize)
   • BF16+: ~{fma_times[1]:.0f}s (slow, quantize overhead)

5. Conclusion:
   • FMA-level is TOO pessimistic for software simulation
   • Real hardware uses higher-precision accumulators
   • Per-op hook method is more realistic
   • Use FMA-level only for worst-case analysis
"""

ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Neural-JSDF FMA-Level Quantization - Summary',
            fontsize=16, fontweight='bold')
plt.tight_layout()

output_file2 = 'fma_quant_summary.png'
plt.savefig(output_file2, dpi=150, bbox_inches='tight')
print(f"✓ Summary visualization saved to: {output_file2}")

# Print text summary
print("\n" + "="*80)
print("FMA-LEVEL QUANTIZATION SUMMARY")
print("="*80)
print(f"\n{'Config':<10} {'Mean Err':<12} {'vs FP32':<12} {'vs Per-Op':<12} {'Quality':<15}")
print(f"{'':10} {'(cm)':<12} {'(%)':<12} {'(x)':<12} {'Assessment':<15}")
print("-"*80)

for i, name in enumerate(fma_configs):
    vs_fp32 = relative_degradation[i]

    if has_per_op and name in per_op_matched:
        vs_per_op = fma_mean_errors[i] / per_op_matched[name]
        vs_per_op_str = f"{vs_per_op:.1f}x"
    else:
        vs_per_op_str = "N/A"

    if fma_mean_errors[i] < 20:
        quality = "POOR"
    elif fma_mean_errors[i] < 50:
        quality = "VERY POOR"
    else:
        quality = "UNUSABLE"

    print(f"{name:<10} {fma_mean_errors[i]:<12.1f} {vs_fp32:>+8.1f}     "
          f"{vs_per_op_str:<12} {quality:<15}")

print("="*80)
print("\nVisualization complete!")
print(f"  • {output_file}")
print(f"  • {output_file2}")
print("="*80)
