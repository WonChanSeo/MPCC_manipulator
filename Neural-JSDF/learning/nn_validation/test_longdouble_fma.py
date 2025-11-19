#!/usr/bin/env python3
"""
Test using numpy.longdouble (80-bit) for FMA operations to match C's fma() behavior.
On x86-64 Linux, np.longdouble is typically 80-bit extended precision.
"""

import numpy as np
import torch

def quantize_float(x):
    """Quantize to E8M7 (bfloat16) format using truncation."""
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.array(x, dtype=np.float32)

    scalar_input = x_np.ndim == 0
    if scalar_input:
        x_np = x_np.reshape(1)

    x_bytes = x_np.view(np.uint32)
    mask = 0xFFFF0000
    x_truncated = (x_bytes & mask).view(np.float32)

    if scalar_input:
        x_truncated = x_truncated[0]

    if isinstance(x, torch.Tensor):
        return torch.from_numpy(np.array(x_truncated)).to(x.device)
    return x_truncated

def check_precision_support():
    """Check what precision types are available."""
    print("Precision Type Information:")
    print("=" * 60)

    print(f"\nfloat32 info:")
    print(f"  itemsize: {np.float32().itemsize} bytes")
    print(f"  finfo: {np.finfo(np.float32)}")

    print(f"\nfloat64 info:")
    print(f"  itemsize: {np.float64().itemsize} bytes")
    print(f"  finfo: {np.finfo(np.float64)}")

    print(f"\nlongdouble info:")
    print(f"  itemsize: {np.longdouble().itemsize} bytes")
    print(f"  finfo: {np.finfo(np.longdouble)}")

    # Test if longdouble is actually 80-bit or 128-bit
    if np.longdouble().itemsize == 16:
        # Could be 80-bit padded to 16 bytes or true 128-bit
        eps = np.finfo(np.longdouble).eps
        if eps < 1e-18:
            print(f"\n  -> longdouble appears to be 80-bit extended precision (eps={eps})")
        else:
            print(f"\n  -> longdouble appears to be 128-bit quad precision (eps={eps})")
    elif np.longdouble().itemsize == 12:
        print(f"\n  -> longdouble is 80-bit extended precision (12 bytes)")

def fma_longdouble(a, b, c):
    """FMA using longdouble for extended precision intermediate."""
    result = np.longdouble(a) * np.longdouble(b) + np.longdouble(c)
    return float(result)

def fma_float64(a, b, c):
    """FMA using float64 for extended precision intermediate."""
    result = np.float64(a) * np.float64(b) + np.float64(c)
    return float(result)

def fma_float32(a, b, c):
    """FMA using pure float32 operations."""
    return float(np.float32(np.float32(a) * np.float32(b)) + np.float32(c))

def test_fma_precision():
    """Test FMA operations with different precision levels."""
    print("\n" + "=" * 60)
    print("FMA Precision Test")
    print("=" * 60)

    # Test values that might show precision differences
    test_cases = [
        # (a, b, c) - designed to trigger precision edge cases
        (0.1, 0.1, 0.0),  # 0.01 - classic float representation issue
        (1.0000001, 1.0000001, 0.0),  # Near 1 multiplication
        (1.23456789, 9.87654321, 0.12345678),  # General case
        (1e-7, 1e-7, 1e-14),  # Small numbers
        (1e7, 1e-7, 1.0),  # Different magnitudes
        # Values from our actual network
        (0.0069274902, 0.7265625, 0.0),
        (-0.0546875, -0.7578125, 0.0050048828),
    ]

    print(f"\n{'a':>14} {'b':>14} {'c':>14} | {'f32':>14} {'f64':>14} {'longdbl':>14}")
    print("-" * 90)

    for a, b, c in test_cases:
        r32 = fma_float32(a, b, c)
        r64 = fma_float64(a, b, c)
        rld = fma_longdouble(a, b, c)

        print(f"{a:>14.10f} {b:>14.10f} {c:>14.10f} | {r32:>14.10f} {r64:>14.10f} {rld:>14.10f}")

        if r32 != r64 or r64 != rld:
            diff_64_32 = abs(r64 - r32)
            diff_ld_64 = abs(rld - r64)
            if diff_64_32 > 0:
                print(f"  -> f64-f32 diff: {diff_64_32:.2e}")
            if diff_ld_64 > 0:
                print(f"  -> ld-f64 diff: {diff_ld_64:.2e}")

def test_accumulated_fma():
    """Test accumulated FMA operations (like matrix multiplication)."""
    print("\n" + "=" * 60)
    print("Accumulated FMA Test (simulating dot product)")
    print("=" * 60)

    # Generate random values for a dot product
    np.random.seed(42)
    n = 30  # Input size for first layer

    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    # Quantize inputs
    a_q = quantize_float(a)
    b_q = quantize_float(b)

    # Compute dot product with different FMA precision
    results = {}

    for name, fma_func in [('f32', fma_float32), ('f64', fma_float64), ('longdouble', fma_longdouble)]:
        acc = 0.0
        for i in range(n):
            acc = fma_func(float(a_q[i]), float(b_q[i]), acc)
            acc_q = float(quantize_float(np.array(acc, dtype=np.float32)))
            acc = acc_q
        results[name] = acc

    print(f"\nDot product results after {n} accumulated FMAs:")
    for name, val in results.items():
        print(f"  {name:>12}: {val:.10f}")

    print(f"\nDifferences:")
    print(f"  f64 - f32:        {abs(results['f64'] - results['f32']):.2e}")
    print(f"  longdouble - f64: {abs(results['longdouble'] - results['f64']):.2e}")
    print(f"  longdouble - f32: {abs(results['longdouble'] - results['f32']):.2e}")

def main():
    check_precision_support()
    test_fma_precision()
    test_accumulated_fma()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
If longdouble shows different results from float64, we can update
the Python validation to use np.longdouble for FMA operations,
which should match C's fma() behavior on x86-64.

If all three precision levels give the same result after quantization,
then the difference must come from somewhere else in the computation.
""")

if __name__ == '__main__':
    main()
