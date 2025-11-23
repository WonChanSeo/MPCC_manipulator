"""
Test to demonstrate FP32 vs FP64 difference in quantization
"""

import numpy as np
import torch

def quantize_float(value, mantissa_bits=7):
    """Quantize to E8M7 (BF16-like)"""
    if isinstance(value, float):
        value = np.float32(value)

    bits = np.array([value], dtype=np.float32).view(np.uint32)[0]

    # Quantize mantissa
    mantissa_bits_to_zero = 23 - mantissa_bits
    mantissa_mask = np.uint32(0x007FFFFF >> mantissa_bits_to_zero)
    mantissa_mask = mantissa_mask << mantissa_bits_to_zero

    # Reconstruct
    sign_bit = bits & 0x80000000
    exponent_field = bits & 0x7F800000
    mantissa_field = bits & mantissa_mask
    quantized_bits = sign_bit | exponent_field | mantissa_field

    return np.frombuffer(quantized_bits.tobytes(), dtype=np.float32)[0]

# Test case: accumulate 256 small values
print("Testing FP32 vs FP64 accumulation with quantization:\n")

# Quantized inputs (E8M7)
np.random.seed(42)
a_vals = [quantize_float(np.random.randn() * 0.1) for _ in range(256)]
b_vals = [quantize_float(np.random.randn() * 0.1) for _ in range(256)]

# FP32 accumulation
acc_fp32 = np.float32(0.0)
for i in range(256):
    prod = np.float32(a_vals[i]) * np.float32(b_vals[i])
    acc_fp32 = np.float32(acc_fp32) + np.float32(prod)
    acc_fp32 = quantize_float(acc_fp32)

# FP64 accumulation
acc_fp64 = np.float64(0.0)
for i in range(256):
    prod = np.float64(a_vals[i]) * np.float64(b_vals[i])
    acc_fp64 = np.float64(acc_fp64) + np.float64(prod)
    acc_fp64_quantized = quantize_float(np.float32(acc_fp64))

print(f"Quantized inputs: same for both (first 5):")
for i in range(5):
    print(f"  a[{i}] = {a_vals[i]:.10f}, b[{i}] = {b_vals[i]:.10f}")

print(f"\nAfter 256 FMA operations:")
print(f"  FP32 accumulator: {acc_fp32:.10f}")
print(f"  FP64 accumulator: {acc_fp64_quantized:.10f}")
print(f"  Difference:       {abs(acc_fp32 - acc_fp64_quantized):.10e}")

if abs(acc_fp32 - acc_fp64_quantized) > 1e-10:
    print(f"\n✓ CONFIRMED: FP32 and FP64 give DIFFERENT results!")
    print(f"  Even with identical quantized inputs, the intermediate")
    print(f"  precision affects where quantization boundaries are crossed.")
else:
    print(f"\n✗ Same result (this case didn't hit a quantization boundary)")

# Test multiple random seeds to find differences
print("\n" + "="*70)
print("Testing 100 random cases:")
diff_count = 0
max_diff = 0.0

for seed in range(100):
    np.random.seed(seed)
    a_vals = [quantize_float(np.random.randn() * 0.5) for _ in range(256)]
    b_vals = [quantize_float(np.random.randn() * 0.5) for _ in range(256)]

    # FP32
    acc_fp32 = np.float32(0.0)
    for i in range(256):
        prod = np.float32(a_vals[i]) * np.float32(b_vals[i])
        acc_fp32 = np.float32(acc_fp32) + np.float32(prod)
        acc_fp32 = quantize_float(acc_fp32)

    # FP64
    acc_fp64 = np.float64(0.0)
    for i in range(256):
        prod = np.float64(a_vals[i]) * np.float64(b_vals[i])
        acc_fp64 = np.float64(acc_fp64) + np.float64(prod)
        acc_fp64_quantized = quantize_float(np.float32(acc_fp64))

    diff = abs(acc_fp32 - acc_fp64_quantized)
    if diff > 1e-10:
        diff_count += 1
        max_diff = max(max_diff, diff)

print(f"  Cases with different results: {diff_count}/100")
print(f"  Maximum difference: {max_diff:.10e}")
print(f"\nConclusion: FP64 gives ~{diff_count}% different results than FP32")
print(f"due to quantization boundary effects.")
