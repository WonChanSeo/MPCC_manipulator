#!/usr/bin/env python3
"""
QDLDL Precision Error Analysis

Compares different precision results against FP64 baseline to calculate errors.
"""

import sys
import os
import struct
import numpy as np
from pathlib import Path

def read_hex_floats(filepath):
    """Read floating point values from hex file."""
    values = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # Read as uint64 hex and convert to float64
                val_int = int(line, 16)
                val_bytes = val_int.to_bytes(8, byteorder='big')
                val_float = struct.unpack('>d', val_bytes)[0]
                values.append(val_float)
    return np.array(values)

def calculate_errors(baseline, test):
    """Calculate various error metrics."""
    # Absolute error
    abs_error = np.abs(test - baseline)

    # Relative error (avoid division by zero)
    rel_error = np.zeros_like(abs_error)
    nonzero_mask = np.abs(baseline) > 1e-100
    rel_error[nonzero_mask] = abs_error[nonzero_mask] / np.abs(baseline[nonzero_mask])

    # Mean squared error
    mse = np.mean((test - baseline) ** 2)
    rmse = np.sqrt(mse)

    # Max errors
    max_abs_error = np.max(abs_error)
    max_abs_idx = np.argmax(abs_error)

    # Mean errors
    mean_abs_error = np.mean(abs_error)
    mean_rel_error = np.mean(rel_error[nonzero_mask]) if np.any(nonzero_mask) else 0.0

    return {
        'abs_error': abs_error,
        'rel_error': rel_error,
        'max_abs_error': max_abs_error,
        'max_abs_idx': max_abs_idx,
        'mean_abs_error': mean_abs_error,
        'mean_rel_error': mean_rel_error,
        'mse': mse,
        'rmse': rmse
    }

def analyze_precision(result_dir, baseline_name='qdldl_samples_double/sample_000000', test_names=None):
    """Analyze precision errors for all test cases."""

    if test_names is None:
        test_names = ['qdldl_test_FP32', 'qdldl_test_E8M23', 'qdldl_test_E8M22',
                      'qdldl_test_E8M21', 'qdldl_test_E8M20', 'qdldl_test_E8M19',
                      'qdldl_test_E8M15', 'qdldl_test_E8M10']

    baseline_dir = Path(result_dir) / baseline_name

    if not baseline_dir.exists():
        print(f"Error: Baseline directory not found: {baseline_dir}")
        return

    # Files to analyze
    files_to_analyze = ['D.hex', 'Dinv.hex', 'Lx.hex', 'x_output.hex']

    print("=" * 80)
    print("QDLDL PRECISION ERROR ANALYSIS")
    print("=" * 80)
    print(f"Baseline: {baseline_name}")
    print(f"Result directory: {result_dir}")
    print("=" * 80)
    print()

    for filename in files_to_analyze:
        baseline_file = baseline_dir / filename

        if not baseline_file.exists():
            print(f"Warning: Baseline file not found: {baseline_file}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Analyzing: {filename}")
        print('=' * 80)

        # Read baseline
        baseline_data = read_hex_floats(baseline_file)
        print(f"Baseline samples: {len(baseline_data)}")
        print(f"Baseline range: [{np.min(baseline_data):.6e}, {np.max(baseline_data):.6e}]")
        print()

        # Compare with each test precision
        for test_name in test_names:
            test_dir = Path(result_dir) / test_name
            test_file = test_dir / filename

            if not test_file.exists():
                print(f"  Warning: Test file not found: {test_file}")
                continue

            # Read test data
            test_data = read_hex_floats(test_file)

            if len(test_data) != len(baseline_data):
                print(f"  Error: Size mismatch for {test_name}: {len(test_data)} vs {len(baseline_data)}")
                continue

            # Calculate errors
            errors = calculate_errors(baseline_data, test_data)

            # Extract precision name
            precision = test_name.replace('qdldl_test_', '')

            print(f"  {precision}:")
            print(f"    Max absolute error:  {errors['max_abs_error']:.6e} (at index {errors['max_abs_idx']})")
            print(f"    Mean absolute error: {errors['mean_abs_error']:.6e}")
            print(f"    Mean relative error: {errors['mean_rel_error']:.6e} ({errors['mean_rel_error']*100:.4f}%)")
            print(f"    RMSE:                {errors['rmse']:.6e}")

            # Show baseline vs test at max error location
            idx = errors['max_abs_idx']
            print(f"    At max error (idx={idx}):")
            print(f"      Baseline: {baseline_data[idx]:.12e}")
            print(f"      Test:     {test_data[idx]:.12e}")
            print(f"      Diff:     {test_data[idx] - baseline_data[idx]:.12e}")
            print()

    print("=" * 80)
    print("SUMMARY: Precision Comparison")
    print("=" * 80)

    # Create summary table for x_output (most important)
    x_output_file = 'x_output.hex'
    baseline_x = read_hex_floats(baseline_dir / x_output_file)

    print(f"\n{'Precision':<12} {'Max Abs Err':<15} {'Mean Abs Err':<15} {'Mean Rel Err':<15} {'RMSE':<15}")
    print("-" * 80)

    for test_name in test_names:
        test_file = Path(result_dir) / test_name / x_output_file
        if test_file.exists():
            test_x = read_hex_floats(test_file)
            if len(test_x) == len(baseline_x):
                errors = calculate_errors(baseline_x, test_x)
                precision = test_name.replace('qdldl_test_', '')
                print(f"{precision:<12} {errors['max_abs_error']:<15.6e} {errors['mean_abs_error']:<15.6e} "
                      f"{errors['mean_rel_error']:<15.6e} {errors['rmse']:<15.6e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_precision_error.py <result_directory>")
        print("Example: python analyze_precision_error.py ../result")
        sys.exit(1)

    result_dir = sys.argv[1]
    analyze_precision(result_dir)
