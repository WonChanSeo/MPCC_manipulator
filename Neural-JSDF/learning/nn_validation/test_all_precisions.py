#!/usr/bin/env python3
"""
Test all precision combinations between C++ and Python.
This script builds C++ with different precision settings and compares with Python.
"""

import numpy as np
import torch
import os
import subprocess
import sys

# Precision configurations to test
PRECISIONS = [
    (8, 23, "FP32"),
    (8, 7, "BF16"),
    (8, 6, "E8M6"),
    (8, 5, "E8M5"),
    (8, 4, "E8M4"),
]

def quantize_float(tensor, exponent_bits=8, mantissa_bits=7):
    """
    Quantize tensor to specified exponent and mantissa bits
    IEEE 754 bit manipulation (from validation_fma_quant.py)
    """
    # Early return for FP32 (no quantization needed)
    if exponent_bits >= 8 and mantissa_bits >= 23:
        if isinstance(tensor, torch.Tensor):
            return tensor
        return tensor

    # Convert to torch tensor if numpy
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor.astype(np.float32))
    elif not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float32)

    # Convert to proper dtype first
    if tensor.dtype != torch.float32:
        tensor = tensor.float()

    # Handle scalar tensors by adding dimension
    original_shape = tensor.shape
    is_scalar = (tensor.ndim == 0)

    if is_scalar:
        tensor = tensor.unsqueeze(0)

    # Flatten for processing
    tensor_flat = tensor.flatten()

    # Convert to numpy for bit manipulation
    tensor_np = tensor_flat.detach().cpu().numpy().astype(np.float32)

    # View as uint32 for bit manipulation
    bits = tensor_np.view(np.uint32)

    # Quantize mantissa
    mantissa_bits_to_zero = 23 - mantissa_bits
    mantissa_mask = np.uint32(0x007FFFFF >> mantissa_bits_to_zero)
    mantissa_mask = mantissa_mask << mantissa_bits_to_zero

    # Quantize exponent
    exponent = (bits >> 23) & 0xFF
    exponent_bits_to_zero = 8 - exponent_bits
    exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero

    # Reconstruct quantized bits
    sign_bit = bits & 0x80000000
    exponent_field = (exponent_quantized & 0xFF) << 23
    mantissa_field = bits & mantissa_mask
    quantized_bits = sign_bit | exponent_field | mantissa_field

    # Convert back to float32
    quantized_np = quantized_bits.view(np.float32)

    # Convert to torch tensor
    quantized_tensor = torch.from_numpy(quantized_np).to(tensor.device)

    # Restore original shape
    if is_scalar:
        quantized_tensor = quantized_tensor.squeeze(0)
    else:
        quantized_tensor = quantized_tensor.reshape(original_shape)

    return quantized_tensor

def fma_float(a, b, c):
    """FMA using float64 for extended precision intermediate."""
    result = np.float64(a) * np.float64(b) + np.float64(c)
    return float(result)

def fma_matmul(x, weight, bias, exponent_bits, mantissa_bits):
    """Perform FMA-quantized matrix multiplication + bias."""
    out_features = weight.shape[0]
    in_features = weight.shape[1]

    x_q = quantize_float(x, exponent_bits, mantissa_bits)
    weight_q = quantize_float(weight, exponent_bits, mantissa_bits)
    bias_q = quantize_float(bias, exponent_bits, mantissa_bits)

    output = np.zeros(out_features, dtype=np.float32)

    for o in range(out_features):
        acc = 0.0
        for i in range(in_features):
            w_val = float(weight_q[o, i])
            x_val = float(x_q[i])
            acc = fma_float(w_val, x_val, acc)
            acc = float(quantize_float(np.array(acc, dtype=np.float32), exponent_bits, mantissa_bits))

        b_val = float(bias_q[o])
        acc = fma_float(b_val, 1.0, acc)
        acc = float(quantize_float(np.array(acc, dtype=np.float32), exponent_bits, mantissa_bits))
        output[o] = acc

    return output

def relu(x):
    return np.maximum(x, 0)

def load_weights():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = os.path.join(script_dir, '..', '..', '..', 'cpp', 'NNmodel', 'env', 'parameter')
    layer_sizes = [(256, 30), (256, 256), (256, 256), (256, 256), (9, 256)]

    weights = []
    biases = []

    for i, (out_size, in_size) in enumerate(layer_sizes):
        w_path = os.path.join(weight_dir, f'weight_{i}.txt')
        b_path = os.path.join(weight_dir, f'bias_{i}.txt')
        w_data = np.loadtxt(w_path)
        w = w_data.reshape(out_size, in_size).astype(np.float32)
        weights.append(w)
        b_data = np.loadtxt(b_path)
        b = b_data.astype(np.float32)
        biases.append(b)

    return weights, biases

def python_inference(inputs, weights, biases, exponent_bits, mantissa_bits):
    """Run Python inference with specified precision."""
    n_samples = inputs.shape[0]
    outputs = np.zeros((n_samples, 9), dtype=np.float32)

    for s in range(n_samples):
        x = inputs[s]
        x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])

        h = x_nerf
        for layer_idx in range(5):
            h = fma_matmul(h, weights[layer_idx], biases[layer_idx], exponent_bits, mantissa_bits)
            if layer_idx < 4:
                h = relu(h)

        outputs[s] = h

    return outputs

def build_cpp_with_precision(exponent_bits, mantissa_bits, name):
    """Build C++ with specified precision settings using build_with_options.sh."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(script_dir, '..', '..', '..', 'cpp')

    # Map precision to preset name
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

    print(f"  Building C++ with preset '{preset}' (E{exponent_bits}M{mantissa_bits})...")

    # Run build_with_options.sh
    result = subprocess.run(
        ['bash', 'build_with_options.sh', preset],
        cwd=cpp_dir,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Build failed!")
        # Show last 500 chars of error
        stderr = result.stderr
        if len(stderr) > 500:
            stderr = "..." + stderr[-500:]
        print(f"  {stderr}")
        return False

    return True

def run_cpp_inference():
    """Run C++ inference and load results."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_dir = os.path.join(script_dir, '..', '..', '..', 'cpp')
    build_dir = os.path.join(cpp_dir, 'build')

    # Run the test
    result = subprocess.run(['./test_validation_inference'], cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Run failed: {result.stderr}")
        return None

    # Load output
    output_file = os.path.join(build_dir, 'cpp_validation_output.txt')
    cpp_outputs = np.loadtxt(output_file)

    return cpp_outputs

def compare_results(python_outputs, cpp_outputs, precision_name):
    """Compare Python and C++ outputs."""
    n_samples = python_outputs.shape[0]
    n_outputs = python_outputs.shape[1]
    total = n_samples * n_outputs

    diffs = np.abs(python_outputs - cpp_outputs)
    exact_matches = np.sum(diffs == 0)
    max_diff = np.max(diffs)
    mean_diff = np.mean(diffs)

    # Find samples with differences
    sample_diffs = np.max(diffs, axis=1)
    samples_with_diff = np.sum(sample_diffs > 0)

    print(f"\n  {precision_name} Results:")
    print(f"    Exact matches: {exact_matches}/{total} ({100*exact_matches/total:.2f}%)")
    print(f"    Max difference: {max_diff:.6f}")
    print(f"    Mean difference: {mean_diff:.6f}")
    print(f"    Samples with differences: {samples_with_diff}/{n_samples}")

    # Show worst samples
    if samples_with_diff > 0:
        worst_samples = np.argsort(sample_diffs)[-3:][::-1]
        print(f"    Worst samples: {worst_samples.tolist()} (max diffs: {sample_diffs[worst_samples].tolist()})")

    return {
        'precision': precision_name,
        'exact_matches': exact_matches,
        'total': total,
        'match_rate': 100 * exact_matches / total,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'samples_with_diff': samples_with_diff
    }

def main():
    print("="*70)
    print("Multi-Precision C++ vs Python Comparison Test")
    print("="*70)

    # Load data
    data = np.load('python_validation_reference.npz')
    inputs = data['input']
    weights, biases = load_weights()

    print(f"\nLoaded {inputs.shape[0]} samples, {len(weights)} layers")

    results = []

    for exp_bits, mant_bits, name in PRECISIONS:
        print(f"\n{'='*70}")
        print(f"Testing {name} (E{exp_bits}M{mant_bits})")
        print("="*70)

        # Build C++ with this precision
        if not build_cpp_with_precision(exp_bits, mant_bits, name):
            print(f"  FAILED to build C++ for {name}")
            continue

        # Run C++ inference
        cpp_outputs = run_cpp_inference()
        if cpp_outputs is None:
            print(f"  FAILED to run C++ for {name}")
            continue

        # Run Python inference
        print(f"  Running Python inference...")
        python_outputs = python_inference(inputs, weights, biases, exp_bits, mant_bits)

        # Compare
        result = compare_results(python_outputs, cpp_outputs, name)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Precision':<12} {'Match Rate':>12} {'Max Diff':>12} {'Mean Diff':>12} {'Bad Samples':>12}")
    print("-"*60)

    for r in results:
        print(f"{r['precision']:<12} {r['match_rate']:>11.2f}% {r['max_diff']:>12.6f} {r['mean_diff']:>12.6f} {r['samples_with_diff']:>12}")

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)

if __name__ == '__main__':
    main()
