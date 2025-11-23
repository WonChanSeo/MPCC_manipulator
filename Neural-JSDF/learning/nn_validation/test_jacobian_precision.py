#!/usr/bin/env python3
"""
Test Jacobian precision with different FlexFloat configurations

Analyzes how Jacobian (gradient) accuracy degrades with reduced precision.
This is important for gradient-based optimization and control.
"""

import sys
import os
import numpy as np
from scipy.io import loadmat
import subprocess

# Add paths
sys.path.insert(0, '/home/mms-wonchan/git/MPCC_manipulator/cpp/build')


def build_cpp_with_precision(exp_bits, mant_bits, build_dir):
    """Build the C++ library with specified FlexFloat precision"""
    cmake_cmd = [
        'cmake', '..',
        '-DNN_USE_FLEXFLOAT=ON',
        f'-DNN_FF_EXPONENT_BITS={exp_bits}',
        f'-DNN_FF_MANTISSA_BITS={mant_bits}'
    ]

    print(f"\n{'='*60}")
    print(f"Building with E{exp_bits}M{mant_bits} precision...")
    print(f"{'='*60}")

    # Run cmake
    result = subprocess.run(cmake_cmd, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CMake failed: {result.stderr}")
        return False

    # Run make
    result = subprocess.run(['make', 'MPCC_WRAPPER', '-j8'], cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Make failed: {result.stderr}")
        return False

    print("Build successful!")
    return True


def run_jacobian_test_subprocess(exp_bits, mant_bits, n_samples, output_file):
    """Run Jacobian test in subprocess"""
    test_code = f'''
import sys
sys.path.insert(0, '/home/mms-wonchan/git/MPCC_manipulator/cpp/build')
import numpy as np
from scipy.io import loadmat
import time
import MPCC_WRAPPER as mw

# Load dataset
data = loadmat('/home/mms-wonchan/git/MPCC_manipulator/Neural-JSDF/learning/data-sampling/datasets/data_mesh_test.mat')
dataset = data['dataset']
inputs = dataset[:{n_samples}, :10]

# Create model
param_path = '/home/mms-wonchan/git/MPCC_manipulator/cpp/NNmodel/env/parameter/'
nn_model = mw.EnvCollNNmodel(param_path)

# Setup network
n_hidden = np.array([256, 256, 256, 256], dtype=np.float64)
nn_model.setNeuralNetwork(10, 9, n_hidden, True)

# Run inference and collect Jacobians
n_samples_actual = inputs.shape[0]
nn_pred = np.zeros((n_samples_actual, 9))
jacobians = np.zeros((n_samples_actual, 9, 10))  # (n_samples, n_output, n_input)

t0 = time.time()
for i in range(n_samples_actual):
    single_input = inputs[i:i+1, :].T.copy()
    result = nn_model.calculateMlpOutputBatch(single_input, False)
    nn_pred[i, :] = np.array(result[0])
    jacobians[i, :, :] = np.array(result[1])  # Get Jacobian matrix

    # Progress update every 100 samples
    if (i + 1) % 100 == 0 or i == 0:
        elapsed = time.time() - t0
        samples_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
        remaining_samples = n_samples_actual - (i + 1)
        eta_sec = remaining_samples / samples_per_sec if samples_per_sec > 0 else 0
        print(f"Progress: {{i+1}}/{{n_samples_actual}} samples ({{(i+1)/n_samples_actual*100:.1f}}%) - {{samples_per_sec:.2f}} samples/s - ETA: {{eta_sec/60:.1f}} min", flush=True)
t_inference = time.time() - t0

# Save results
np.savez('{output_file}',
         exponent_bits={exp_bits},
         mantissa_bits={mant_bits},
         nn_pred=nn_pred,
         jacobians=jacobians,
         inference_time=t_inference,
         n_samples=n_samples_actual)

print(f"Collected {{n_samples_actual}} Jacobians")
print(f"Jacobian shape: {{jacobians.shape}}")
print(f"Inference time: {{t_inference:.3f}}s")
'''

    result = subprocess.run(['python3', '-u', '-c', test_code])
    if result.returncode != 0:
        print(f"Test failed with return code: {result.returncode}")
        return False
    return True


def main():
    # Set working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Build directory
    build_dir = '/home/mms-wonchan/git/MPCC_manipulator/cpp/build'

    # Create output directory
    output_dir = os.path.join(script_dir, 'jacobian_precision_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Precision configurations to test
    configs = [
        (8, 23, "FP32"),
        (8, 7, "BF16"),
        (8, 6, "E8M6"),
        (8, 5, "E8M5"),
        (8, 4, "E8M4"),
    ]

    # Use full dataset for comprehensive Jacobian analysis
    n_samples = 9900

    print("\n" + "="*80)
    print("JACOBIAN PRECISION TEST")
    print(f"Testing {len(configs)} configurations with {n_samples} samples")
    print("="*80)

    # Collect Jacobians for each precision
    jacobian_data = {}

    for exp_bits, mant_bits, name in configs:
        # Build with this precision
        if not build_cpp_with_precision(exp_bits, mant_bits, build_dir):
            print(f"Skipping {name} due to build failure")
            continue

        # Run test
        output_file = os.path.join(output_dir, f"jacobian_E{exp_bits}M{mant_bits}.npz")
        print(f"\nTesting E{exp_bits}M{mant_bits} ({name}) with {n_samples} samples...")

        if run_jacobian_test_subprocess(exp_bits, mant_bits, n_samples, output_file):
            # Load results
            data = np.load(output_file)
            jacobian_data[name] = {
                'exp': exp_bits,
                'mant': mant_bits,
                'jacobians': data['jacobians'],
                'outputs': data['nn_pred']
            }
            print(f"  Saved results to: {output_file}")

    # Analyze Jacobian errors vs FP32 baseline
    if 'FP32' not in jacobian_data:
        print("\nError: FP32 baseline not available!")
        return

    print("\n" + "="*80)
    print("JACOBIAN ERROR ANALYSIS (vs FP32 Baseline)")
    print("="*80)

    fp32_jac = jacobian_data['FP32']['jacobians']

    results = []
    for name in ['BF16', 'E8M6', 'E8M5', 'E8M4']:
        if name not in jacobian_data:
            continue

        jac = jacobian_data[name]['jacobians']

        # Calculate errors
        jac_errors = np.abs(jac - fp32_jac)

        # Per-element statistics
        mean_error = jac_errors.mean()
        std_error = jac_errors.std()
        max_error = jac_errors.max()
        median_error = np.median(jac_errors)
        p95_error = np.percentile(jac_errors, 95)
        p99_error = np.percentile(jac_errors, 99)

        # Relative error (avoid division by zero)
        fp32_jac_nonzero = fp32_jac.copy()
        fp32_jac_nonzero[np.abs(fp32_jac_nonzero) < 1e-10] = 1e-10
        rel_errors = np.abs(jac_errors / fp32_jac_nonzero) * 100
        mean_rel_error = np.mean(rel_errors)

        # Frobenius norm error per sample
        frobenius_errors = np.linalg.norm(jac_errors, ord='fro', axis=(1, 2))
        mean_frob = frobenius_errors.mean()
        max_frob = frobenius_errors.max()

        print(f"\n{name} (E{jacobian_data[name]['exp']}M{jacobian_data[name]['mant']}):")
        print(f"  Absolute error:")
        print(f"    Mean:   {mean_error:.6e}")
        print(f"    Std:    {std_error:.6e}")
        print(f"    Median: {median_error:.6e}")
        print(f"    Max:    {max_error:.6e}")
        print(f"    95th percentile: {p95_error:.6e}")
        print(f"    99th percentile: {p99_error:.6e}")
        print(f"  Relative error (%):")
        print(f"    Mean:   {mean_rel_error:.2f}%")
        print(f"  Frobenius norm error:")
        print(f"    Mean:   {mean_frob:.6e}")
        print(f"    Max:    {max_frob:.6e}")

        results.append({
            'name': name,
            'exp': jacobian_data[name]['exp'],
            'mant': jacobian_data[name]['mant'],
            'mean_abs_error': mean_error,
            'std_abs_error': std_error,
            'max_abs_error': max_error,
            'median_abs_error': median_error,
            'p95_abs_error': p95_error,
            'p99_abs_error': p99_error,
            'mean_rel_error': mean_rel_error,
            'mean_frob_error': mean_frob,
            'max_frob_error': max_frob
        })

    # Save summary
    summary_file = os.path.join(output_dir, "jacobian_error_summary.npz")
    np.savez(summary_file,
             names=[r['name'] for r in results],
             exponent_bits=[r['exp'] for r in results],
             mantissa_bits=[r['mant'] for r in results],
             mean_abs_errors=[r['mean_abs_error'] for r in results],
             max_abs_errors=[r['max_abs_error'] for r in results],
             mean_rel_errors=[r['mean_rel_error'] for r in results],
             mean_frob_errors=[r['mean_frob_error'] for r in results],
             max_frob_errors=[r['max_frob_error'] for r in results])
    print(f"\nSummary saved to: {summary_file}")

    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Config':<8} {'Mean Abs Err':<15} {'Max Abs Err':<15} {'Mean Rel Err (%)':<18} {'Mean Frob Err':<15}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<8} {r['mean_abs_error']:<15.6e} {r['max_abs_error']:<15.6e} "
              f"{r['mean_rel_error']:<18.2f} {r['mean_frob_error']:<15.6e}")

    print("="*80)
    print("\nJacobian precision test completed!")


if __name__ == "__main__":
    main()
