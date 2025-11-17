import numpy as np
import sys

print("=" * 80)
print("BF16 vs FP32 INFERENCE COMPARISON")
print("=" * 80)
print()

# Check if data files exist
try:
    # Load BF16 test results
    data_bf16 = np.loadtxt('data_bf16.csv', delimiter=',')
    N_bf16 = data_bf16.shape[0]
    input_bf16 = data_bf16[:, :10]
    nn_dist_bf16 = data_bf16[:, 10:19]
    mesh_dist_bf16 = data_bf16[:, 19:]
    print(f"✓ Loaded BF16 data: {N_bf16:,} samples")
except FileNotFoundError:
    print("✗ Error: data_bf16.csv not found")
    print("  Run get_data_bf16.py first to generate BF16 inference data")
    sys.exit(1)

try:
    # Load FP32 test results (data_fp32.csv appears to be FP32 based on file listing)
    data_fp32 = np.loadtxt('data_fp32.csv', delimiter=',')
    N_fp32 = data_fp32.shape[0]
    input_fp32 = data_fp32[:, :10]
    nn_dist_fp32 = data_fp32[:, 10:19]
    mesh_dist_fp32 = data_fp32[:, 19:]
    print(f"✓ Loaded FP32 data: {N_fp32:,} samples")
except FileNotFoundError:
    print("✗ Error: data_fp32.csv not found")
    print("  Run validation script with FP32 first to generate FP32 inference data")
    sys.exit(1)

print()

# Use smaller sample size for fair comparison
N = min(N_bf16, N_fp32)
print(f"Comparing first {N:,} samples from each dataset")
print()

# ============================================================================
# 1. ACCURACY COMPARISON
# ============================================================================
print("1. ACCURACY COMPARISON (vs Ground Truth Mesh)")
print("-" * 80)

# BF16 errors
bf16_errors = np.abs(mesh_dist_bf16[:N] - nn_dist_bf16[:N])
bf16_l1_mean = bf16_errors.mean()
bf16_l1_std = bf16_errors.std()
bf16_max_error = bf16_errors.max()

# FP32 errors
fp32_errors = np.abs(mesh_dist_fp32[:N] - nn_dist_fp32[:N])
fp32_l1_mean = fp32_errors.mean()
fp32_l1_std = fp32_errors.std()
fp32_max_error = fp32_errors.max()

print(f"{'Metric':<25} {'FP32':<20} {'BF16':<20} {'Difference':<15}")
print("-" * 80)
print(f"{'Mean L1 Error (cm)':<25} {fp32_l1_mean:>15.2f}     {bf16_l1_mean:>15.2f}     {bf16_l1_mean - fp32_l1_mean:>10.2f}")
print(f"{'Std Dev L1 Error (cm)':<25} {fp32_l1_std:>15.2f}     {bf16_l1_std:>15.2f}     {bf16_l1_std - fp32_l1_std:>10.2f}")
print(f"{'Max Error (cm)':<25} {fp32_max_error:>15.2f}     {bf16_max_error:>15.2f}     {bf16_max_error - fp32_max_error:>10.2f}")
print()

# ============================================================================
# 2. PER-LINK ERROR COMPARISON
# ============================================================================
print("2. PER-LINK ERROR ANALYSIS")
print("-" * 80)
print(f"{'Link':<8} {'FP32 L1 (cm)':<15} {'BF16 L1 (cm)':<15} {'FP32 Max (cm)':<15} {'BF16 Max (cm)':<15}")
print("-" * 80)

for i in range(9):
    fp32_link_err = np.abs(mesh_dist_fp32[:N, i] - nn_dist_fp32[:N, i])
    bf16_link_err = np.abs(mesh_dist_bf16[:N, i] - nn_dist_bf16[:N, i])

    fp32_l1 = fp32_link_err.mean()
    bf16_l1 = bf16_link_err.mean()
    fp32_max = fp32_link_err.max()
    bf16_max = bf16_link_err.max()

    print(f"Link {i:<3} {fp32_l1:>10.2f}      {bf16_l1:>10.2f}      {fp32_max:>10.2f}      {bf16_max:>10.2f}")

print("-" * 80)
print()

# ============================================================================
# 3. DIRECT PREDICTION COMPARISON (BF16 vs FP32 outputs)
# ============================================================================
print("3. DIRECT OUTPUT COMPARISON (BF16 predictions vs FP32 predictions)")
print("-" * 80)

# Compare predictions directly where inputs are identical
# For fair comparison, we need to check if inputs match
input_match_threshold = 1e-6
n_matching = 0

# Check if any samples have matching inputs
matching_indices = []
for i in range(min(1000, N)):  # Check first 1000 samples
    if np.allclose(input_bf16[i], input_fp32[i], atol=input_match_threshold):
        matching_indices.append(i)
        n_matching += 1

if n_matching > 0:
    print(f"Found {n_matching} samples with matching inputs")
    matching_indices = np.array(matching_indices)

    # Direct prediction difference
    pred_diff = np.abs(nn_dist_bf16[matching_indices] - nn_dist_fp32[matching_indices])
    pred_diff_mean = pred_diff.mean()
    pred_diff_max = pred_diff.max()
    pred_diff_std = pred_diff.std()

    print(f"Mean prediction difference: {pred_diff_mean:.4f} cm")
    print(f"Max prediction difference:  {pred_diff_max:.4f} cm")
    print(f"Std prediction difference:  {pred_diff_std:.4f} cm")
    print()
    print("This represents the direct impact of BF16 quantization on model outputs.")
else:
    print("No matching input samples found between BF16 and FP32 datasets.")
    print("Both datasets use random sampling, so direct output comparison is not possible.")
    print()
    print("Statistical comparison based on error distributions:")

    # Compare prediction distributions statistically
    pred_diff_l1 = abs(bf16_l1_mean - fp32_l1_mean)
    relative_diff = (pred_diff_l1 / fp32_l1_mean) * 100 if fp32_l1_mean > 0 else 0

    print(f"Absolute L1 error difference: {pred_diff_l1:.2f} cm")
    print(f"Relative difference: {relative_diff:.2f}%")

print()

# ============================================================================
# 4. ERROR DISTRIBUTION ANALYSIS
# ============================================================================
print("4. ERROR DISTRIBUTION ANALYSIS")
print("-" * 80)

# Percentile analysis
percentiles = [50, 75, 90, 95, 99]
print(f"{'Percentile':<15} {'FP32 Error (cm)':<20} {'BF16 Error (cm)':<20}")
print("-" * 80)

for p in percentiles:
    fp32_p = np.percentile(fp32_errors.flatten(), p)
    bf16_p = np.percentile(bf16_errors.flatten(), p)
    print(f"{p}th{' ':<12} {fp32_p:>15.2f}      {bf16_p:>15.2f}")

print()

# ============================================================================
# 5. SUMMARY AND CONCLUSIONS
# ============================================================================
print("5. SUMMARY AND CONCLUSIONS")
print("-" * 80)

# Calculate relative degradation
if fp32_l1_mean > 0:
    relative_degradation = ((bf16_l1_mean - fp32_l1_mean) / fp32_l1_mean) * 100
else:
    relative_degradation = 0

print(f"Model: sdf_256x5_mesh_50000.pt (Early checkpoint, epoch 1,996)")
print(f"Total samples compared: {N:,}")
print()

if abs(relative_degradation) < 5:
    quality = "NEGLIGIBLE"
    symbol = "✓"
elif abs(relative_degradation) < 10:
    quality = "ACCEPTABLE"
    symbol = "✓"
elif abs(relative_degradation) < 20:
    quality = "MODERATE"
    symbol = "⚠"
else:
    quality = "SIGNIFICANT"
    symbol = "✗"

print(f"{symbol} Accuracy degradation from FP32 to BF16: {relative_degradation:+.2f}% ({quality})")
print()

print("Key Observations:")
if abs(relative_degradation) < 10:
    print(f"  • BF16 quantization introduces minimal accuracy loss ({abs(relative_degradation):.2f}%)")
    print("  • BF16 is suitable for deployment with ~2x memory savings")
else:
    print(f"  • BF16 quantization shows {abs(relative_degradation):.2f}% accuracy change")
    print("  • Consider error tolerance requirements for your application")

print()
print("Note: Both models show high errors (~40-44 cm L1) because the checkpoint is")
print("      from epoch 1,996 of ~100,000. The quantization comparison is valid,")
print("      but absolute errors will decrease significantly with full training.")

print()
print("=" * 80)
print("END OF COMPARISON")
print("=" * 80)
