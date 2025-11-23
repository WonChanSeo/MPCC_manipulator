#!/usr/bin/env python3
"""
Test C++ calculateMlpOutputBatch precision with Neural-JSDF dataset

Tests the C++ implementation across different FlexFloat precision configurations
and compares results with ground truth from data_mesh_test.mat
"""

import sys
import os
import subprocess
import numpy as np
from scipy.io import loadmat
import time

# Add paths
sys.path.insert(0, '/home/mms-wonchan/git/MPCC_manipulator/cpp/build')

def build_cpp_with_precision(exp_bits, mant_bits, build_dir):
    """
    Build the C++ library with specified FlexFloat precision

    Args:
        exp_bits: Number of exponent bits
        mant_bits: Number of mantissa bits
        build_dir: Build directory path

    Returns:
        True if build succeeded, False otherwise
    """
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
    result = subprocess.run(['make', '-j8'], cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Make failed: {result.stderr}")
        return False

    print("Build successful!")
    return True


def run_precision_test_subprocess(exp_bits, mant_bits, n_samples, output_file):
    """
    Run precision test in a subprocess to ensure fresh module loading

    Args:
        exp_bits: Number of exponent bits
        mant_bits: Number of mantissa bits
        n_samples: Number of samples to test
        output_file: Path to save results

    Returns:
        True if successful, False otherwise
    """
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
gt_distances = dataset[:{n_samples}, 10:]

# Create model
param_path = '/home/mms-wonchan/git/MPCC_manipulator/cpp/NNmodel/env/parameter/'
nn_model = mw.EnvCollNNmodel(param_path)

# Setup network
n_hidden = np.array([256, 256, 256, 256], dtype=np.float64)
nn_model.setNeuralNetwork(10, 9, n_hidden, True)

# Run inference
n_samples_actual = inputs.shape[0]
nn_pred = np.zeros((n_samples_actual, 9))

t0 = time.time()
for i in range(n_samples_actual):
    single_input = inputs[i:i+1, :].T.copy()
    result = nn_model.calculateMlpOutputBatch(single_input, False)
    nn_pred[i, :] = np.array(result[0])
t_inference = time.time() - t0

# Calculate errors in cm
nn_pred_cm = nn_pred
gt_cm = gt_distances * 100
errors = np.abs(nn_pred_cm - gt_cm)

# Save results
np.savez('{output_file}',
         exponent_bits={exp_bits},
         mantissa_bits={mant_bits},
         nn_pred=nn_pred_cm,
         gt_distances=gt_cm,
         errors=errors,
         inference_time=t_inference,
         n_samples=n_samples_actual,
         mean_error=errors.mean(),
         std_error=errors.std(),
         max_error=errors.max(),
         median_error=np.median(errors),
         p95_error=np.percentile(errors, 95),
         p99_error=np.percentile(errors, 99))

print(f"Mean error: {{errors.mean():.4f}} cm")
print(f"Std error: {{errors.std():.4f}} cm")
print(f"Max error: {{errors.max():.4f}} cm")
print(f"Inference time: {{t_inference:.3f}}s")
'''

    result = subprocess.run(['python3', '-c', test_code], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Test failed: {result.stderr}")
        return False
    print(result.stdout)
    return True


def run_precision_test(exp_bits, mant_bits, inputs, gt_distances, n_samples=None):
    """
    Run precision test with specified configuration

    Args:
        exp_bits: Number of exponent bits
        mant_bits: Number of mantissa bits
        inputs: Input data (N, 10) - 7 joint angles + 3 query point
        gt_distances: Ground truth distances (N, 9) - SDF values in meters
        n_samples: Number of samples to test (None = all)

    Returns:
        Dictionary with test results
    """
    # Reload the module to get the freshly built version
    if 'MPCC_WRAPPER' in sys.modules:
        del sys.modules['MPCC_WRAPPER']

    try:
        import MPCC_WRAPPER as mw
    except ImportError as e:
        print(f"Failed to import MPCC_WRAPPER: {e}")
        return None

    # Create EnvCollNNmodel
    # Use the cpp/NNmodel/env/parameter/ directory for weight files
    param_path = '/home/mms-wonchan/git/MPCC_manipulator/cpp/NNmodel/env/parameter/'
    nn_model = mw.EnvCollNNmodel(param_path)

    # Setup network parameters (matching sdf_256x5_mesh_50000.pt)
    # Network: 10 -> 256 -> 256 -> 256 -> 256 -> 9 (4 hidden layers)
    n_input = 10
    n_output = 9
    n_hidden = np.array([256, 256, 256, 256], dtype=np.float64)
    is_nerf = True
    nn_model.setNeuralNetwork(n_input, n_output, n_hidden, is_nerf)

    if n_samples is not None:
        inputs = inputs[:n_samples]
        gt_distances = gt_distances[:n_samples]

    n_samples_actual = inputs.shape[0]
    print(f"\nTesting E{exp_bits}M{mant_bits} with {n_samples_actual} samples...")

    # For C++ batch inference, we need to pass each sample individually
    # since the batch function expects multiple obstacle points for one joint config
    # But here we have multiple joint configs with one obstacle point each

    # Run individual inferences and collect results
    nn_pred = np.zeros((n_samples_actual, 9))

    t0 = time.time()
    for i in range(n_samples_actual):
        # Prepare single sample as column vector (10, 1)
        single_input = inputs[i:i+1, :].T.copy()
        result = nn_model.calculateMlpOutputBatch(single_input, False)
        outputs = np.array(result[0])
        nn_pred[i, :] = outputs
    t_inference = time.time() - t0

    # Unit conversion:
    # - NN output is in cm (network is trained with *100 scaling)
    # - Ground truth from mat file is in meters
    # So we convert NN output to meters by dividing by 100, then compute error in cm
    nn_pred_m = nn_pred / 100.0  # Convert NN output from cm to meters

    # For error reporting in cm (consistent with Python validation scripts)
    nn_pred_cm = nn_pred  # Already in cm
    gt_cm = gt_distances * 100  # Convert GT from meters to cm

    # Calculate errors in cm
    errors = np.abs(nn_pred_cm - gt_cm)

    results = {
        'exponent_bits': exp_bits,
        'mantissa_bits': mant_bits,
        'n_samples': n_samples_actual,
        'inference_time': t_inference,
        'samples_per_sec': n_samples_actual / t_inference,
        'mean_error': errors.mean(),
        'std_error': errors.std(),
        'max_error': errors.max(),
        'median_error': np.median(errors),
        'p95_error': np.percentile(errors.flatten(), 95),
        'p99_error': np.percentile(errors.flatten(), 99),
        'per_link_mean': [errors[:, i].mean() for i in range(9)],
        'per_link_max': [errors[:, i].max() for i in range(9)],
        'nn_pred': nn_pred_cm,
        'gt_distances': gt_cm,
        'errors': errors
    }

    # Print summary
    print(f"  Inference time: {t_inference:.3f}s ({results['samples_per_sec']:.1f} samples/s)")
    print(f"  Mean error: {results['mean_error']:.4f} cm")
    print(f"  Std error:  {results['std_error']:.4f} cm")
    print(f"  Max error:  {results['max_error']:.4f} cm")
    print(f"  Median:     {results['median_error']:.4f} cm")
    print(f"  95th pctl:  {results['p95_error']:.4f} cm")
    print(f"  99th pctl:  {results['p99_error']:.4f} cm")

    return results


def main():
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Load dataset
    dataset_path = '/home/mms-wonchan/git/MPCC_manipulator/Neural-JSDF/learning/data-sampling/datasets/data_mesh_test.mat'
    print(f"Loading dataset from: {dataset_path}")

    data = loadmat(dataset_path)
    dataset = data['dataset']

    print(f"Dataset shape: {dataset.shape}")
    print("Columns: [7 joints][3 query point][9 distances]")

    # Extract inputs and ground truth
    inputs = dataset[:, :10]  # 7 joints + 3 query point
    gt_distances = dataset[:, 10:]  # 9 link distances

    print(f"Inputs shape: {inputs.shape}")
    print(f"Ground truth shape: {gt_distances.shape}")

    # Build directory
    build_dir = '/home/mms-wonchan/git/MPCC_manipulator/cpp/build'

    # Create output directory
    output_dir = os.path.join(script_dir, 'cpp_precision_results')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Precision configurations to test
    # Format: (exponent_bits, mantissa_bits, name)
    configs = [
        (8, 23, "FP32"),       # Full precision (reference)
        (8, 7, "BF16"),        # Standard BF16 - current default
        (8, 6, "E8M6"),        # Reduced mantissa
        (8, 5, "E8M5"),        # Further reduced
        (8, 4, "E8M4"),        # Aggressive reduction
    ]

    # Number of samples to test (set to None for all samples)
    n_samples = 9900  # Use all 9900 samples

    print("\n" + "="*80)
    print("C++ calculateMlpOutputBatch PRECISION TEST")
    print(f"Testing {len(configs)} configurations with {n_samples or 'all'} samples")
    print("="*80)

    all_results = []

    for exp_bits, mant_bits, name in configs:
        # Build with this precision
        if not build_cpp_with_precision(exp_bits, mant_bits, build_dir):
            print(f"Skipping {name} due to build failure")
            continue

        # Run test in subprocess to ensure fresh module loading
        output_file = os.path.join(output_dir, f"cpp_precision_E{exp_bits}M{mant_bits}.npz")
        print(f"\nTesting E{exp_bits}M{mant_bits} ({name}) with {n_samples} samples...")

        if run_precision_test_subprocess(exp_bits, mant_bits, n_samples, output_file):
            # Load results from saved file
            data = np.load(output_file)
            results = {
                'name': name,
                'exponent_bits': int(data['exponent_bits']),
                'mantissa_bits': int(data['mantissa_bits']),
                'mean_error': float(data['mean_error']),
                'std_error': float(data['std_error']),
                'max_error': float(data['max_error']),
                'median_error': float(data['median_error']),
                'p95_error': float(data['p95_error']),
                'p99_error': float(data['p99_error']),
                'inference_time': float(data['inference_time']),
                'n_samples': int(data['n_samples']),
            }
            all_results.append(results)
            print(f"  Saved results to: {output_file}")

    # Print final comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(f"{'Config':<12} {'Exp':<5} {'Mant':<5} {'Mean (cm)':<12} {'Std (cm)':<12} {'Max (cm)':<12} {'Median (cm)':<12}")
    print("-"*80)

    for r in all_results:
        print(f"{r['name']:<12} {r['exponent_bits']:<5} {r['mantissa_bits']:<5} "
              f"{r['mean_error']:<12.4f} {r['std_error']:<12.4f} "
              f"{r['max_error']:<12.4f} {r['median_error']:<12.4f}")

    print("="*80)

    # Save summary
    summary_file = os.path.join(output_dir, "cpp_precision_summary.npz")
    np.savez(summary_file,
             names=[r['name'] for r in all_results],
             exponent_bits=[r['exponent_bits'] for r in all_results],
             mantissa_bits=[r['mantissa_bits'] for r in all_results],
             mean_errors=[r['mean_error'] for r in all_results],
             std_errors=[r['std_error'] for r in all_results],
             max_errors=[r['max_error'] for r in all_results],
             median_errors=[r['median_error'] for r in all_results],
             p95_errors=[r['p95_error'] for r in all_results],
             p99_errors=[r['p99_error'] for r in all_results],
             inference_times=[r['inference_time'] for r in all_results])
    print(f"\nSummary saved to: {summary_file}")

    print("\nTest completed!")
    return all_results


if __name__ == "__main__":
    main()
