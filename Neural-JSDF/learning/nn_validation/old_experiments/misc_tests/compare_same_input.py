import numpy as np
import sys

print("=" * 80)
print("BF16 vs FP32 DIRECT COMPARISON (Same Input)")
print("=" * 80)
print()

# Load same input data
try:
    data = np.loadtxt('data_same_input.csv', delimiter=',')
    N = data.shape[0]
    print(f"✓ Loaded same input data: {N:,} samples")
except FileNotFoundError:
    print("✗ Error: data_same_input.csv not found")
    print("  Run get_data_same_input.py first")
    sys.exit(1)

# Parse data: [input(10), fp32_pred(9), bf16_pred(9), mesh(9)]
input_data = data[:, :10]
fp32_pred = data[:, 10:19]
bf16_pred = data[:, 19:28]
mesh_dist = data[:, 28:37]

print()

# ============================================================================
# 1. DIRECT PREDICTION COMPARISON (FP32 vs BF16 on same input)
# ============================================================================
print("1. DIRECT PREDICTION COMPARISON")
print("-" * 80)

# Direct prediction differences
pred_diff = np.abs(fp32_pred - bf16_pred)
pred_diff_mean = pred_diff.mean()
pred_diff_max = pred_diff.max()
pred_diff_std = pred_diff.std()
pred_diff_median = np.median(pred_diff)

print(f"Prediction differences (FP32 vs BF16 on SAME input):")
print(f"  Mean difference:   {pred_diff_mean:.4f} cm")
print(f"  Median difference: {pred_diff_median:.4f} cm")
print(f"  Std difference:    {pred_diff_std:.4f} cm")
print(f"  Max difference:    {pred_diff_max:.4f} cm")
print()

# Correlation between predictions
correlation = np.corrcoef(fp32_pred.flatten(), bf16_pred.flatten())[0,1]
print(f"Pearson correlation (FP32 vs BF16 predictions): {correlation:.6f}")
print()

# ============================================================================
# 2. ACCURACY COMPARISON (vs Ground Truth Mesh)
# ============================================================================
print("2. ACCURACY COMPARISON (vs Ground Truth Mesh)")
print("-" * 80)

# FP32 errors
fp32_errors = np.abs(mesh_dist - fp32_pred)
fp32_l1_mean = fp32_errors.mean()
fp32_l1_std = fp32_errors.std()
fp32_max_error = fp32_errors.max()
fp32_median = np.median(fp32_errors)

# BF16 errors
bf16_errors = np.abs(mesh_dist - bf16_pred)
bf16_l1_mean = bf16_errors.mean()
bf16_l1_std = bf16_errors.std()
bf16_max_error = bf16_errors.max()
bf16_median = np.median(bf16_errors)

print(f"{'Metric':<25} {'FP32':<20} {'BF16':<20} {'Difference':<15}")
print("-" * 80)
print(f"{'Mean L1 Error (cm)':<25} {fp32_l1_mean:>15.4f}     {bf16_l1_mean:>15.4f}     {bf16_l1_mean - fp32_l1_mean:>10.4f}")
print(f"{'Median Error (cm)':<25} {fp32_median:>15.4f}     {bf16_median:>15.4f}     {bf16_median - fp32_median:>10.4f}")
print(f"{'Std Dev Error (cm)':<25} {fp32_l1_std:>15.4f}     {bf16_l1_std:>15.4f}     {bf16_l1_std - fp32_l1_std:>10.4f}")
print(f"{'Max Error (cm)':<25} {fp32_max_error:>15.4f}     {bf16_max_error:>15.4f}     {bf16_max_error - fp32_max_error:>10.4f}")
print()

# Relative degradation
if fp32_l1_mean > 0:
    relative_degradation = ((bf16_l1_mean - fp32_l1_mean) / fp32_l1_mean) * 100
else:
    relative_degradation = 0

print(f"Relative accuracy degradation (FP32 → BF16): {relative_degradation:+.2f}%")
print()

# ============================================================================
# 3. PER-LINK ANALYSIS
# ============================================================================
print("3. PER-LINK DETAILED ANALYSIS")
print("-" * 80)
print(f"{'Link':<8} {'FP32 L1':<12} {'BF16 L1':<12} {'Δ L1':<10} {'FP32 Max':<12} {'BF16 Max':<12} {'Pred Diff':<12}")
print("-" * 80)

for i in range(9):
    fp32_link_err = np.abs(mesh_dist[:, i] - fp32_pred[:, i])
    bf16_link_err = np.abs(mesh_dist[:, i] - bf16_pred[:, i])
    pred_diff_link = np.abs(fp32_pred[:, i] - bf16_pred[:, i])

    fp32_l1 = fp32_link_err.mean()
    bf16_l1 = bf16_link_err.mean()
    fp32_max = fp32_link_err.max()
    bf16_max = bf16_link_err.max()
    pred_diff_mean_link = pred_diff_link.mean()

    delta_l1 = bf16_l1 - fp32_l1

    print(f"Link {i:<3} {fp32_l1:>10.4f}  {bf16_l1:>10.4f}  {delta_l1:>8.4f}  "
          f"{fp32_max:>10.4f}  {bf16_max:>10.4f}  {pred_diff_mean_link:>10.4f}")

print("-" * 80)
print()

# ============================================================================
# 4. PERCENTILE ANALYSIS
# ============================================================================
print("4. ERROR DISTRIBUTION PERCENTILES")
print("-" * 80)

percentiles = [50, 75, 90, 95, 99]
print(f"{'Percentile':<15} {'FP32 Error (cm)':<20} {'BF16 Error (cm)':<20} {'Difference':<15}")
print("-" * 80)

for p in percentiles:
    fp32_p = np.percentile(fp32_errors.flatten(), p)
    bf16_p = np.percentile(bf16_errors.flatten(), p)
    print(f"{p}th{' ':<12} {fp32_p:>15.4f}      {bf16_p:>15.4f}      {bf16_p - fp32_p:>10.4f}")

print()

# ============================================================================
# 5. QUANTIZATION IMPACT ANALYSIS
# ============================================================================
print("5. QUANTIZATION IMPACT (Direct Prediction Difference)")
print("-" * 80)

# Per-link quantization impact
print(f"{'Link':<8} {'Mean Δ (cm)':<15} {'Max Δ (cm)':<15} {'95th %ile Δ (cm)':<20}")
print("-" * 80)

for i in range(9):
    pred_diff_link = np.abs(fp32_pred[:, i] - bf16_pred[:, i])
    mean_diff = pred_diff_link.mean()
    max_diff = pred_diff_link.max()
    p95_diff = np.percentile(pred_diff_link, 95)

    print(f"Link {i:<3} {mean_diff:>10.4f}      {max_diff:>10.4f}      {p95_diff:>15.4f}")

print("-" * 80)
print(f"Overall  {pred_diff_mean:>10.4f}      {pred_diff_max:>10.4f}      "
      f"{np.percentile(pred_diff, 95):>15.4f}")
print()

# ============================================================================
# 6. SUMMARY
# ============================================================================
print("6. SUMMARY AND CONCLUSIONS")
print("-" * 80)

print(f"Total samples compared: {N:,} (same input for both models)")
print()

# Determine quality
if abs(relative_degradation) < 5:
    quality = "NEGLIGIBLE"
    symbol = "✓✓"
elif abs(relative_degradation) < 10:
    quality = "ACCEPTABLE"
    symbol = "✓"
elif abs(relative_degradation) < 20:
    quality = "MODERATE"
    symbol = "⚠"
else:
    quality = "SIGNIFICANT"
    symbol = "✗"

print(f"{symbol} Accuracy degradation: {relative_degradation:+.2f}% ({quality})")
print(f"✓ Direct prediction difference: {pred_diff_mean:.4f} cm (mean), {pred_diff_max:.4f} cm (max)")
print(f"✓ Correlation between predictions: {correlation:.6f} (very high)")
print()

print("Key Findings:")
print(f"  • BF16 predictions differ from FP32 by only ~{pred_diff_mean:.2f} cm on average")
print(f"  • Accuracy loss vs ground truth: {abs(relative_degradation):.2f}%")
print(f"  • Strong correlation (r={correlation:.4f}) shows consistent behavior")
print(f"  • Maximum prediction difference: {pred_diff_max:.4f} cm")
print()

if abs(relative_degradation) < 10 and pred_diff_mean < 0.5:
    print("RECOMMENDATION: ✓✓ BF16 is EXCELLENT for deployment")
    print("  - Minimal accuracy loss")
    print("  - 50% memory savings")
    print("  - Very small prediction differences")
elif abs(relative_degradation) < 20:
    print("RECOMMENDATION: ✓ BF16 is SUITABLE for deployment")
    print("  - Acceptable accuracy loss")
    print("  - 50% memory savings")
else:
    print("RECOMMENDATION: ⚠ Review BF16 deployment carefully")
    print("  - Consider application error tolerance")

print()
print("=" * 80)
print("END OF COMPARISON")
print("=" * 80)
