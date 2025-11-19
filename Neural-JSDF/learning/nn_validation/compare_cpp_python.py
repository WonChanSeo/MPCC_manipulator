"""
Compare C++ and Python inference results
"""

import numpy as np
import matplotlib.pyplot as plt
import os

print("="*80)
print("COMPARING C++ vs PYTHON INFERENCE RESULTS")
print("="*80)

# ============================================================================
# 1. Load data
# ============================================================================

# Check if files exist
if not os.path.exists('python_validation_reference.npz'):
    print("Error: python_validation_reference.npz not found!")
    print("Run validate_cpp_inference.py first")
    exit(1)

if not os.path.exists('cpp_validation_output.txt'):
    print("Error: cpp_validation_output.txt not found!")
    print("Run the C++ test program first")
    exit(1)

# Load Python reference
print("\nLoading Python reference data...")
python_data = np.load('python_validation_reference.npz')
python_input = python_data['input']
python_output = python_data['output']

print(f"  Python input shape: {python_input.shape}")
print(f"  Python output shape: {python_output.shape}")

# Load C++ output
print("\nLoading C++ output...")
cpp_output = np.loadtxt('cpp_validation_output.txt')

print(f"  C++ output shape: {cpp_output.shape}")

# Transpose C++ output to match Python shape (samples × outputs)
if cpp_output.ndim == 2:
    cpp_output = cpp_output.T  # (9 × 100) -> (100 × 9)
    print(f"  C++ output (transposed): {cpp_output.shape}")

# ============================================================================
# 2. Check compatibility
# ============================================================================

print("\n" + "-"*80)
print("Checking data compatibility...")
print("-"*80)

# Now both should be (100 × 9) shape
if cpp_output.shape != python_output.shape:
    print(f"Warning: Output shape mismatch!")
    print(f"  C++: {cpp_output.shape}")
    print(f"  Python: {python_output.shape}")
else:
    print(f"✓ Output shapes match: {cpp_output.shape}")

# ============================================================================
# 3. Compare results
# ============================================================================

print("\n" + "="*80)
print("COMPARISON RESULTS")
print("="*80)

if cpp_output.shape == python_output.shape:
    # Direct comparison of full batch outputs
    diff = np.abs(cpp_output - python_output)

    print(f"\nAbsolute differences:")
    print(f"  Max:    {np.max(diff):.10e}")
    print(f"  Mean:   {np.mean(diff):.10e}")
    print(f"  Median: {np.median(diff):.10e}")
    print(f"  Min:    {np.min(diff):.10e}")

    # Relative error
    rel_error = diff / (np.abs(python_output) + 1e-10)
    print(f"\nRelative errors:")
    print(f"  Max:    {np.max(rel_error):.10e}")
    print(f"  Mean:   {np.mean(rel_error):.10e}")
    print(f"  Median: {np.median(rel_error):.10e}")

    # Per-sample comparison (first 10 samples, first 3 outputs)
    print(f"\nSample comparison (first 10 samples, first 3 outputs):")
    print(f"{'Sample':>7} {'Out':>4} {'Python':>15} {'C++':>15} {'Abs Diff':>15} {'Rel Error':>15}")
    print("-"*85)
    for i in range(min(10, cpp_output.shape[0])):
        for j in range(min(3, cpp_output.shape[1])):
            py_val = python_output[i, j]
            cpp_val = cpp_output[i, j]
            abs_diff = abs(py_val - cpp_val)
            rel_err = abs_diff / (abs(py_val) + 1e-10)
            print(f"{i:>7} {j:>4} {py_val:>15.10f} {cpp_val:>15.10f} {abs_diff:>15.10e} {rel_err:>15.10e}")

    # Tolerance check
    print("\n" + "-"*80)
    print("Tolerance checks:")
    print("-"*80)

    tolerances = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    max_diff = np.max(diff)
    for tol in tolerances:
        passed = max_diff < tol
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Max diff < {tol:.0e}: {status} (max diff: {max_diff:.10e})")

    # Visualization
    print("\n" + "-"*80)
    print("Creating visualization...")
    print("-"*80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('C++ vs Python Inference Comparison', fontsize=14, fontweight='bold')

    # 1. Histogram of differences
    ax1 = axes[0, 0]
    ax1.hist(diff.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Absolute Difference', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.set_title('Distribution of Absolute Differences', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axvline(np.mean(diff), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(diff):.2e}')
    ax1.legend()

    # 2. Heatmap of differences (per sample per output)
    ax2 = axes[0, 1]
    im = ax2.imshow(diff, aspect='auto', cmap='hot', interpolation='nearest')
    ax2.set_xlabel('Output Dimension', fontweight='bold')
    ax2.set_ylabel('Sample Index', fontweight='bold')
    ax2.set_title('Difference Heatmap', fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Absolute Difference')

    # 3. Scatter plot (all outputs)
    ax3 = axes[1, 0]
    ax3.scatter(python_output.flatten(), cpp_output.flatten(), alpha=0.3, s=10)

    # Add perfect match line
    min_val = min(np.min(python_output), np.min(cpp_output))
    max_val = max(np.max(python_output), np.max(cpp_output))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')

    ax3.set_xlabel('Python Output', fontweight='bold')
    ax3.set_ylabel('C++ Output', fontweight='bold')
    ax3.set_title('Scatter Plot (All Outputs)', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Per-output dimension statistics
    ax4 = axes[1, 1]
    diff_per_dim = np.max(diff, axis=0)  # Max diff per output dimension
    x = np.arange(diff.shape[1])
    ax4.bar(x, diff_per_dim, color='orange', alpha=0.7)
    ax4.set_xlabel('Output Dimension', fontweight='bold')
    ax4.set_ylabel('Max Absolute Difference', fontweight='bold')
    ax4.set_title('Max Difference per Output Dimension', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = 'cpp_vs_python_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    max_diff = np.max(diff)
    if max_diff < 1e-5:
        print("✓✓✓ EXCELLENT: C++ and Python outputs match within 1e-5")
    elif max_diff < 1e-4:
        print("✓✓ GOOD: C++ and Python outputs match within 1e-4")
    elif max_diff < 1e-3:
        print("✓ ACCEPTABLE: C++ and Python outputs match within 1e-3")
    else:
        print("✗ WARNING: C++ and Python outputs differ by more than 1e-3")
        print(f"  Max difference: {max_diff:.10e}")
        print("  This may indicate:")
        print("  - Different quantization settings (FlexFloat vs Python)")
        print("  - Different precision (bfloat16 vs float32)")
        print("  - Implementation differences")

    print("="*80)

else:
    print("Error: Cannot compare - dimension mismatch!")
    print(f"  C++ output has {cpp_output.shape[0]} values")
    print(f"  Python has {python_min_per_row.shape[0]} outputs")

print("\n✓ Comparison complete!")
