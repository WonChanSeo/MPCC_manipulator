import numpy as np
import matplotlib.pyplot as plt
import sys

print("Loading data for visualization...")

# Load same input data
try:
    data = np.loadtxt('data_same_input.csv', delimiter=',')
    N = data.shape[0]
    print(f"✓ Loaded same input data: {N} samples")
except FileNotFoundError:
    print("✗ Error: data_same_input.csv not found")
    sys.exit(1)

# Parse data: [input(10), fp32_pred(9), bf16_pred(9), mesh(9)]
input_data = data[:, :10]
fp32_pred = data[:, 10:19]
bf16_pred = data[:, 19:28]
mesh_dist = data[:, 28:37]

print("Calculating errors...")

# Calculate errors
fp32_errors = np.abs(mesh_dist - fp32_pred)
bf16_errors = np.abs(mesh_dist - bf16_pred)
pred_diff = np.abs(fp32_pred - bf16_pred)

# Create comprehensive comparison plots
fig = plt.figure(figsize=(20, 12))

# ============================================================================
# Plot 1: Per-Link L1 Error Comparison (Bar Chart)
# ============================================================================
ax1 = plt.subplot(3, 4, 1)
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
# Plot 2: Direct Prediction Difference per Link
# ============================================================================
ax2 = plt.subplot(3, 4, 2)
pred_diff_per_link = [pred_diff[:, i].mean() for i in range(9)]
bars = ax2.bar(x, pred_diff_per_link, color='#F18F01', alpha=0.8)

ax2.set_xlabel('Link Index')
ax2.set_ylabel('Mean Prediction Difference (cm)')
ax2.set_title('BF16 vs FP32 Prediction Difference')
ax2.set_xticks(x)
ax2.set_xticklabels([str(i) for i in range(9)])
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(y=pred_diff.mean(), color='r', linestyle='--', linewidth=1, label=f'Overall: {pred_diff.mean():.3f}cm')
ax2.legend()

# ============================================================================
# Plot 3: Prediction Scatter Plot (SAME INPUT!)
# ============================================================================
ax3 = plt.subplot(3, 4, 3)
# Sample 2000 random points for clarity
sample_idx = np.random.choice(N * 9, min(2000, N * 9), replace=False)
fp32_flat = fp32_pred.flatten()[sample_idx]
bf16_flat = bf16_pred.flatten()[sample_idx]

ax3.scatter(fp32_flat, bf16_flat, alpha=0.3, s=5, color='#F18F01')
min_val = min(fp32_flat.min(), bf16_flat.min())
max_val = max(fp32_flat.max(), bf16_flat.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y=x (perfect agreement)')
correlation = np.corrcoef(fp32_pred.flatten(), bf16_pred.flatten())[0,1]
ax3.set_xlabel('FP32 Prediction (cm)')
ax3.set_ylabel('BF16 Prediction (cm)')
ax3.set_title(f'Prediction Comparison (r={correlation:.6f})')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ============================================================================
# Plot 4: Prediction Difference Distribution
# ============================================================================
ax4 = plt.subplot(3, 4, 4)
ax4.hist(pred_diff.flatten(), bins=50, alpha=0.7, color='#F18F01', edgecolor='black')
ax4.axvline(pred_diff.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {pred_diff.mean():.4f}cm')
ax4.axvline(np.median(pred_diff), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(pred_diff):.4f}cm')
ax4.set_xlabel('Prediction Difference (cm)')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Prediction Differences')
ax4.legend()
ax4.grid(alpha=0.3)

# ============================================================================
# Plot 5: Error Distribution Histogram
# ============================================================================
ax5 = plt.subplot(3, 4, 5)
ax5.hist(fp32_errors.flatten(), bins=50, alpha=0.6, label='FP32', color='#2E86AB', density=True)
ax5.hist(bf16_errors.flatten(), bins=50, alpha=0.6, label='BF16', color='#A23B72', density=True)
ax5.set_xlabel('Error (cm)')
ax5.set_ylabel('Density')
ax5.set_title('Error Distribution (All Links)')
ax5.legend()
ax5.grid(alpha=0.3)
ax5.set_xlim(0, 10)

# ============================================================================
# Plot 6: Cumulative Distribution Function (CDF)
# ============================================================================
ax6 = plt.subplot(3, 4, 6)
fp32_sorted = np.sort(fp32_errors.flatten())
bf16_sorted = np.sort(bf16_errors.flatten())
fp32_cdf = np.arange(1, len(fp32_sorted) + 1) / len(fp32_sorted)
bf16_cdf = np.arange(1, len(bf16_sorted) + 1) / len(bf16_sorted)

ax6.plot(fp32_sorted, fp32_cdf, label='FP32', color='#2E86AB', linewidth=2)
ax6.plot(bf16_sorted, bf16_cdf, label='BF16', color='#A23B72', linewidth=2)
ax6.set_xlabel('Error (cm)')
ax6.set_ylabel('Cumulative Probability')
ax6.set_title('Cumulative Distribution of Errors')
ax6.legend()
ax6.grid(alpha=0.3)
ax6.set_xlim(0, 10)

# ============================================================================
# Plot 7: Box Plot Comparison
# ============================================================================
ax7 = plt.subplot(3, 4, 7)
data_to_plot = [fp32_errors.flatten(), bf16_errors.flatten()]
bp = ax7.boxplot(data_to_plot, tick_labels=['FP32', 'BF16'], patch_artist=True)
bp['boxes'][0].set_facecolor('#2E86AB')
bp['boxes'][1].set_facecolor('#A23B72')
ax7.set_ylabel('Error (cm)')
ax7.set_title('Error Distribution Box Plot')
ax7.grid(axis='y', alpha=0.3)

# ============================================================================
# Plot 8: Percentile Comparison
# ============================================================================
ax8 = plt.subplot(3, 4, 8)
percentiles = [50, 75, 90, 95, 99]
fp32_percentiles = [np.percentile(fp32_errors.flatten(), p) for p in percentiles]
bf16_percentiles = [np.percentile(bf16_errors.flatten(), p) for p in percentiles]

ax8.plot(percentiles, fp32_percentiles, marker='o', linewidth=2, markersize=8, label='FP32', color='#2E86AB')
ax8.plot(percentiles, bf16_percentiles, marker='s', linewidth=2, markersize=8, label='BF16', color='#A23B72')
ax8.set_xlabel('Percentile')
ax8.set_ylabel('Error (cm)')
ax8.set_title('Error Percentiles')
ax8.legend()
ax8.grid(alpha=0.3)

# ============================================================================
# Plot 9: Per-Link Summary Table (FP32)
# ============================================================================
ax9 = plt.subplot(3, 4, 9)
ax9.axis('off')

table_data = [['Link', 'FP32 L1', 'FP32 Max']]
for i in range(9):
    l1 = fp32_errors[:, i].mean()
    max_err = fp32_errors[:, i].max()
    table_data.append([f'{i}', f'{l1:.3f}', f'{max_err:.3f}'])

table = ax9.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.2, 0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

for i in range(3):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax9.set_title('FP32 Per-Link Stats (cm)', fontsize=10, weight='bold', pad=10)

# ============================================================================
# Plot 10: Per-Link Summary Table (BF16)
# ============================================================================
ax10 = plt.subplot(3, 4, 10)
ax10.axis('off')

table_data = [['Link', 'BF16 L1', 'BF16 Max']]
for i in range(9):
    l1 = bf16_errors[:, i].mean()
    max_err = bf16_errors[:, i].max()
    table_data.append([f'{i}', f'{l1:.3f}', f'{max_err:.3f}'])

table = ax10.table(cellText=table_data, cellLoc='center', loc='center',
                   colWidths=[0.2, 0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

for i in range(3):
    table[(0, i)].set_facecolor('#A23B72')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax10.set_title('BF16 Per-Link Stats (cm)', fontsize=10, weight='bold', pad=10)

# ============================================================================
# Plot 11: Per-Link Summary Table (Comparison)
# ============================================================================
ax11 = plt.subplot(3, 4, 11)
ax11.axis('off')

table_data = [['Link', 'Δ L1', 'Pred Δ']]
for i in range(9):
    delta_l1 = bf16_errors[:, i].mean() - fp32_errors[:, i].mean()
    pred_delta = pred_diff[:, i].mean()
    table_data.append([f'{i}', f'{delta_l1:+.3f}', f'{pred_delta:.3f}'])

table = ax11.table(cellText=table_data, cellLoc='center', loc='center',
                   colWidths=[0.2, 0.4, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

for i in range(3):
    table[(0, i)].set_facecolor('#F18F01')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax11.set_title('Difference (BF16-FP32) cm', fontsize=10, weight='bold', pad=10)

# ============================================================================
# Plot 12: Overall Summary Statistics
# ============================================================================
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')

# Calculate summary statistics
fp32_mean = fp32_errors.mean()
bf16_mean = bf16_errors.mean()
fp32_median = np.median(fp32_errors)
bf16_median = np.median(bf16_errors)
fp32_max = fp32_errors.max()
bf16_max = bf16_errors.max()
pred_diff_mean = pred_diff.mean()
pred_diff_max = pred_diff.max()
correlation = np.corrcoef(fp32_pred.flatten(), bf16_pred.flatten())[0,1]

# Create table data
table_data = [
    ['Metric', 'FP32', 'BF16', 'Diff'],
    ['Mean (cm)', f'{fp32_mean:.3f}', f'{bf16_mean:.3f}', f'{bf16_mean - fp32_mean:+.3f}'],
    ['Median (cm)', f'{fp32_median:.3f}', f'{bf16_median:.3f}', f'{bf16_median - fp32_median:+.3f}'],
    ['Max (cm)', f'{fp32_max:.2f}', f'{bf16_max:.2f}', f'{bf16_max - fp32_max:+.2f}'],
    ['', '', '', ''],
    ['Pred Δ Mean', '', '', f'{pred_diff_mean:.4f}'],
    ['Pred Δ Max', '', '', f'{pred_diff_max:.4f}'],
    ['Correlation', '', '', f'{correlation:.6f}'],
]

table = ax12.table(cellText=table_data, cellLoc='center', loc='center',
                  colWidths=[0.35, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.3)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#555555')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style prediction difference rows
for i in [5, 6, 7]:
    table[(i, 0)].set_facecolor('#FFE5CC')
    table[(i, 3)].set_facecolor('#FFE5CC')

ax12.set_title('Overall Summary Statistics', fontsize=11, weight='bold', pad=15)

# ============================================================================
# Overall Title
# ============================================================================
fig.suptitle('BF16 vs FP32 Direct Comparison (Same Input)\nModel: sdf_256x5_mesh_50000.pt',
             fontsize=16, weight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('bf16_fp32_same_input_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved to: bf16_fp32_same_input_comparison.png")

plt.show()
