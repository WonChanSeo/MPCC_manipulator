#!/usr/bin/env python3
"""
Test precision degradation compared to FP32 baseline.
This script measures how much each quantized precision differs from FP32 ground truth.
"""

import numpy as np
import os
import subprocess

# Precision configurations to test (FP32 is baseline)
PRECISIONS = [
    (8, 23, "FP32"),
    (8, 7, "BF16"),
    (8, 6, "E8M6"),
    (8, 5, "E8M5"),
    (8, 4, "E8M4"),
]

def build_cpp_with_precision(exponent_bits, mantissa_bits, name):
    """Build C++ with specified precision settings."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(script_dir, '..', '..', '..', 'cpp')

    preset_map = {
        (8, 23): 'fp32',
        (8, 7): 'bf16',
        (8, 6): 'e8m6',
        (8, 5): 'e8m5',
        (8, 4): 'e8m4',
    }

    preset = preset_map.get((exponent_bits, mantissa_bits))
    if not preset:
        print(f"  Unknown precision: E{exponent_bits}M{mantissa_bits}")
        return False

    print(f"  Building C++ with preset '{preset}'...")

    result = subprocess.run(
        ['bash', 'build_with_options.sh', preset],
        cwd=cpp_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Build failed!")
        return False

    return True

def run_cpp_inference():
    """Run C++ inference and load results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(script_dir, '..', '..', '..', 'cpp')
    build_dir = os.path.join(cpp_dir, 'build')

    result = subprocess.run(['./test_validation_inference'], cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Run failed: {result.stderr}")
        return None

    output_file = os.path.join(build_dir, 'cpp_validation_output.txt')
    cpp_outputs = np.loadtxt(output_file)

    return cpp_outputs

def compare_with_baseline(outputs, baseline, precision_name):
    """Compare outputs with FP32 baseline."""
    n_samples = outputs.shape[0]
    n_outputs = outputs.shape[1]
    total = n_samples * n_outputs

    diffs = np.abs(outputs - baseline)
    exact_matches = np.sum(diffs == 0)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)

    # Relative error (avoid division by zero)
    rel_errors = diffs / (np.abs(baseline) + 1e-10)
    max_rel_error = np.max(rel_errors)
    mean_rel_error = np.mean(rel_errors)

    # Per-sample max difference
    sample_diffs = np.max(diffs, axis=1)
    samples_with_diff = np.sum(sample_diffs > 0)

    print(f"\n  {precision_name} vs FP32 Baseline:")
    print(f"    Exact matches: {exact_matches}/{total} ({100*exact_matches/total:.2f}%)")
    print(f"    Max absolute diff: {max_diff:.6f}")
    print(f"    Mean absolute diff: {mean_diff:.6f}")
    print(f"    Max relative error: {100*max_rel_error:.4f}%")
    print(f"    Mean relative error: {100*mean_rel_error:.4f}%")
    print(f"    Samples with differences: {samples_with_diff}/{n_samples}")

    if samples_with_diff > 0 and samples_with_diff <= 10:
        worst_samples = np.argsort(sample_diffs)[-3:][::-1]
        print(f"    Worst samples: {worst_samples.tolist()} (max diffs: {sample_diffs[worst_samples].tolist()})")

    return {
        'precision': precision_name,
        'exact_matches': exact_matches,
        'total': total,
        'match_rate': 100 * exact_matches / total,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'max_rel_error': 100 * max_rel_error,
        'mean_rel_error': 100 * mean_rel_error,
        'samples_with_diff': samples_with_diff
    }

def main():
    print("="*70)
    print("Precision Degradation Test: Quantized vs FP32 Baseline")
    print("="*70)
    print("\nThis test measures how much each quantized precision")
    print("differs from the FP32 ground truth (C++ implementation).\n")

    # First, get FP32 baseline
    print("="*70)
    print("Step 1: Getting FP32 Baseline")
    print("="*70)

    if not build_cpp_with_precision(8, 23, "FP32"):
        print("FAILED to build FP32 baseline!")
        return

    fp32_baseline = run_cpp_inference()
    if fp32_baseline is None:
        print("FAILED to run FP32 baseline!")
        return

    print(f"  FP32 baseline loaded: {fp32_baseline.shape}")
    print(f"  Output range: [{fp32_baseline.min():.6f}, {fp32_baseline.max():.6f}]")
    print(f"  Mean: {fp32_baseline.mean():.6f}")

    # Now test each quantized precision
    results = []

    for exp_bits, mant_bits, name in PRECISIONS:
        if name == "FP32":
            # Skip FP32 (it's the baseline)
            results.append({
                'precision': "FP32",
                'exact_matches': fp32_baseline.size,
                'total': fp32_baseline.size,
                'match_rate': 100.0,
                'max_diff': 0.0,
                'mean_diff': 0.0,
                'max_rel_error': 0.0,
                'mean_rel_error': 0.0,
                'samples_with_diff': 0
            })
            continue

        print(f"\n{'='*70}")
        print(f"Testing {name} (E{exp_bits}M{mant_bits})")
        print("="*70)

        # Build and run C++ with this precision
        if not build_cpp_with_precision(exp_bits, mant_bits, name):
            print(f"  FAILED to build C++ for {name}")
            continue

        cpp_outputs = run_cpp_inference()
        if cpp_outputs is None:
            print(f"  FAILED to run C++ for {name}")
            continue

        # Compare with FP32 baseline
        result = compare_with_baseline(cpp_outputs, fp32_baseline, name)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Precision Degradation vs FP32 Baseline")
    print("="*70)
    print(f"\n{'Precision':<10} {'Match%':>10} {'Max Diff':>12} {'Mean Diff':>12} {'Max Rel%':>10} {'Mean Rel%':>10}")
    print("-"*70)

    for r in results:
        print(f"{r['precision']:<10} {r['match_rate']:>9.2f}% {r['max_diff']:>12.6f} {r['mean_diff']:>12.6f} {r['max_rel_error']:>9.4f}% {r['mean_rel_error']:>9.4f}%")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    for r in results:
        if r['precision'] == "FP32":
            continue
        print(f"\n{r['precision']}:")
        if r['max_diff'] < 0.01:
            print(f"  - Excellent accuracy: max error < 0.01")
        elif r['max_diff'] < 0.1:
            print(f"  - Good accuracy: max error < 0.1")
        elif r['max_diff'] < 1.0:
            print(f"  - Acceptable accuracy: max error < 1.0")
        else:
            print(f"  - Significant degradation: max error = {r['max_diff']:.2f}")

        if r['mean_rel_error'] < 0.1:
            print(f"  - Mean relative error < 0.1% (negligible)")
        elif r['mean_rel_error'] < 1.0:
            print(f"  - Mean relative error < 1% (acceptable)")
        else:
            print(f"  - Mean relative error = {r['mean_rel_error']:.2f}% (significant)")

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)

if __name__ == '__main__':
    main()
