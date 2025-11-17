import numpy as np
import matplotlib.pyplot as plt
import sys

print("Loading data for visualization...")

# Load BF16 and FP32 data
try:
    data_bf16 = np.loadtxt('data_bf16.csv', delimiter=',')
    nn_dist_bf16 = data_bf16[:, 10:19]
    mesh_dist_bf16 = data_bf16[:, 19:]
    print(f"✓ Loaded BF16 data: {data_bf16.shape[0]} samples")
except FileNotFoundError:
    print("✗ Error: data_bf16.csv not found")
    sys.exit(1)

try:
    data_fp32 = np.loadtxt('data_fp32.csv', delimiter=',')
    nn_dist_fp32 = data_fp32[:, 10:19]
    mesh_dist_fp32 = data_fp32[:, 19:]
    print(f"✓ Loaded FP32 data: {data_fp32.shape[0]} samples")
except FileNotFoundError:
    print("✗ Error: data_fp32.csv not found")
    sys.exit(1)

# Use same sample size
N = min(data_bf16.shape[0], data_fp32.shape[0])
print(f"Comparing first {N} samples\n")

# Calculate errors
bf16_errors = np.abs(mesh_dist_bf16[:N] - nn_dist_bf16[:N])
fp32_errors = np.abs(mesh_dist_fp32[:N] - nn_dist_fp32[:N])

# Create comprehensive comparison plots
fig = plt.figure(figsize=(16, 12))

# ============================================================================
# Plot 1: Per-Link L1 Error Comparison (Bar Chart)
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
link_names = [f'Link {i}' for i in range(9)]
fp32_l1_per_link = [fp32_errors[:, i].mean() for i in range(9)]
bf16_l1_per_link = [bf16_errors[:, i].mean() for i in range(9)]

x = np.arange(9)
width = 0.35
bars1 = ax1.bar(x - width/2, fp32_l1_per_link, width, label='FP32', alpha=0.8, color='#2E86AB')
bars2 = ax1.bar(x + width/2, bf16_l1_per_link, width, label='BF16', alpha=0.8, color='#A23B72')

ax1.set_xlabel('Link Index')
ax1.set_ylabel('Mean L1 Error (cm)')
ax1.set_title('Per-Link L1 Error Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels([str(i) for i in range(9)])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 2: Per-Link Max Error Comparison (Bar Chart)
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
fp32_max_per_link = [fp32_errors[:, i].max() for i in range(9)]
bf16_max_per_link = [bf16_errors[:, i].max() for i in range(9)]

bars1 = ax2.bar(x - width/2, fp32_max_per_link, width, label='FP32', alpha=0.8, color='#2E86AB')
bars2 = ax2.bar(x + width/2, bf16_max_per_link, width, label='BF16', alpha=0.8, color='#A23B72')

ax2.set_xlabel('Link Index')
ax2.set_ylabel('Max Error (cm)')
ax2.set_title('Per-Link Max Error Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels([str(i) for i in range(9)])
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 3: Error Distribution Histogram
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
ax3.hist(fp32_errors.flatten(), bins=50, alpha=0.6, label='FP32', color='#2E86AB', density=True)
ax3.hist(bf16_errors.flatten(), bins=50, alpha=0.6, label='BF16', color='#A23B72', density=True)
ax3.set_xlabel('Error (cm)')
ax3.set_ylabel('Density')
ax3.set_title('Error Distribution (All Links)')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xlim(0, 10)

# ============================================================================
# Plot 4: Cumulative Distribution Function (CDF)
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
fp32_sorted = np.sort(fp32_errors.flatten())
bf16_sorted = np.sort(bf16_errors.flatten())
fp32_cdf = np.arange(1, len(fp32_sorted) + 1) / len(fp32_sorted)
bf16_cdf = np.arange(1, len(bf16_sorted) + 1) / len(bf16_sorted)

ax4.plot(fp32_sorted, fp32_cdf, label='FP32', color='#2E86AB', linewidth=2)
ax4.plot(bf16_sorted, bf16_cdf, label='BF16', color='#A23B72', linewidth=2)
ax4.set_xlabel('Error (cm)')
ax4.set_ylabel('Cumulative Probability')
ax4.set_title('Cumulative Distribution of Errors')
ax4.legend()
ax4.grid(alpha=0.3)
ax4.set_xlim(0, 10)

# ============================================================================
# Plot 5: Box Plot Comparison
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
data_to_plot = [fp32_errors.flatten(), bf16_errors.flatten()]
bp = ax5.boxplot(data_to_plot, labels=['FP32', 'BF16'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
ax5.set_ylabel('Error (cm)')
ax5.set_title('Error Distribution Box Plot')
ax5.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 6: Percentile Comparison
# ============================================================================
ax6 = plt.subplot(3, 3, 6)
percentiles = [50, 75, 90, 95, 99]
fp32_percentiles = [np.percentile(fp32_errors.flatten(), p) for p in percentiles]
bf16_percentiles = [np.percentile(bf16_errors.flatten(), p) for p in percentiles]

ax6.plot(percentiles, fp32_percentiles, marker='o', linewidth=2, markersize=8, label='FP32', color='#2E86AB')
ax6.plot(percentiles, bf16_percentiles, marker='s', linewidth=2, markersize=8, label='BF16', color='#A23B72')
ax6.set_xlabel('Percentile')
ax6.set_ylabel('Error (cm)')
ax6.set_title('Error Percentiles')
ax6.legend()
ax6.grid(alpha=0.3)

# ============================================================================
# Plot 7: Scatter Plot - FP32 vs BF16 Predictions (Sample)
# ============================================================================
ax7 = plt.subplot(3, 3, 7)
# Sample 1000 random points for clarity
sample_idx = np.random.choice(N * 9, min(1000, N * 9), replace=False)
fp32_flat = nn_dist_fp32[:N].flatten()[sample_idx]
bf16_flat = nn_dist_bf16[:N].flatten()[sample_idx]

ax7.scatter(fp32_flat, bf16_flat, alpha=0.5, s=10, color='#F18F01')
min_val = min(fp32_flat.min(), bf16_flat.min())
max_val = max(fp32_flat.max(), bf16_flat.max())
ax7.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y=x')
ax7.set_xlabel('FP32 Prediction (cm)')
ax7.set_ylabel('BF16 Prediction (cm)')
ax7.set_title('Prediction Comparison (1000 samples)')
ax7.legend()
ax7.grid(alpha=0.3)

# ============================================================================
# Plot 8: Error Difference by Link
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
error_diff_per_link = [bf16_errors[:, i].mean() - fp32_errors[:, i].mean() for i in range(9)]
colors = ['#C1292E' if x > 0 else '#2E86AB' for x in error_diff_per_link]
bars = ax8.bar(x, error_diff_per_link, color=colors, alpha=0.8)

ax8.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
ax8.set_xlabel('Link Index')
ax8.set_ylabel('L1 Error Difference (BF16 - FP32, cm)')
ax8.set_title('L1 Error Difference per Link')
ax8.set_xticks(x)
ax8.set_xticklabels([str(i) for i in range(9)])
ax8.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 9: Summary Statistics Table
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

# Calculate summary statistics
fp32_mean = fp32_errors.mean()
bf16_mean = bf16_errors.mean()
fp32_std = fp32_errors.std()
bf16_std = bf16_errors.std()
fp32_max = fp32_errors.max()
bf16_max = bf16_errors.max()
fp32_median = np.median(fp32_errors)
bf16_median = np.median(bf16_errors)

# Create table data
table_data = [
    ['Metric', 'FP32', 'BF16', 'Diff'],
    ['Mean (cm)', f'{fp32_mean:.2f}', f'{bf16_mean:.2f}', f'{bf16_mean - fp32_mean:.2f}'],
    ['Median (cm)', f'{fp32_median:.2f}', f'{bf16_median:.2f}', f'{bf16_median - fp32_median:.2f}'],
    ['Std Dev (cm)', f'{fp32_std:.2f}', f'{bf16_std:.2f}', f'{bf16_std - fp32_std:.2f}'],
    ['Max (cm)', f'{fp32_max:.2f}', f'{bf16_max:.2f}', f'{bf16_max - fp32_max:.2f}'],
]

table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#CCCCCC')
    table[(0, i)].set_text_props(weight='bold')

ax9.set_title('Summary Statistics', fontsize=12, weight='bold', pad=20)

# ============================================================================
# Overall Title
# ============================================================================
fig.suptitle('BF16 vs FP32 Inference Comparison\nModel: sdf_256x5_mesh_50000.pt',
             fontsize=16, weight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('bf16_fp32_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: bf16_fp32_comparison.png")

plt.show()
