#!/usr/bin/env python3
"""
Test multiple Python precisions against current C++ output.
This allows testing without rebuilding C++.
"""

import numpy as np
import torch
import os

# Precision configurations to test
PRECISIONS = [
    (8, 23, "FP32"),
    (8, 7, "BF16"),
    (8, 6, "E8M6"),
    (8, 5, "E8M5"),
    (8, 4, "E8M4"),
]

def quantize_float(x, exponent_bits=8, mantissa_bits=7):
    """Quantize to custom float format using truncation."""
    if isinstance(x, torch.Tensor):
        x_np = x.cpu().numpy()
    else:
        x_np = np.array(x, dtype=np.float32)

    scalar_input = x_np.ndim == 0
    if scalar_input:
        x_np = x_np.reshape(1)

    # Calculate bits to truncate
    bits_to_truncate = 23 - mantissa_bits
    mask = 0xFFFFFFFF << bits_to_truncate

    x_bytes = x_np.view(np.uint32)
    x_truncated = (x_bytes & mask).view(np.float32)

    if scalar_input:
        x_truncated = x_truncated[0]

    if isinstance(x, torch.Tensor):
        return torch.from_numpy(np.array(x_truncated)).to(x.device)
    return x_truncated

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
    print("Python Multi-Precision Test Against C++ E8M7 Output")
    print("="*70)
    print("\nNote: C++ was built with E8M7 (BF16) precision.")
    print("Testing which Python precision best matches the C++ output.\n")

    # Load data
    data = np.load('python_validation_reference.npz')
    inputs = data['input']
    weights, biases = load_weights()

    print(f"Loaded {inputs.shape[0]} samples, {len(weights)} layers")

    # Load C++ output (built with E8M7)
    cpp_outputs = np.loadtxt('cpp_validation_output.txt')
    print(f"Loaded C++ output: shape {cpp_outputs.shape}")

    results = []

    for exp_bits, mant_bits, name in PRECISIONS:
        print(f"\n{'='*70}")
        print(f"Testing Python {name} (E{exp_bits}M{mant_bits}) vs C++ E8M7")
        print("="*70)

        # Run Python inference
        print(f"  Running Python inference with {name}...")
        python_outputs = python_inference(inputs, weights, biases, exp_bits, mant_bits)

        # Compare
        result = compare_results(python_outputs, cpp_outputs, name)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Python Precisions vs C++ E8M7")
    print("="*70)
    print(f"\n{'Precision':<12} {'Match Rate':>12} {'Max Diff':>12} {'Mean Diff':>12} {'Bad Samples':>12}")
    print("-"*60)

    for r in results:
        marker = " <-- Best" if r['match_rate'] == max(x['match_rate'] for x in results) else ""
        print(f"{r['precision']:<12} {r['match_rate']:>11.2f}% {r['max_diff']:>12.6f} {r['mean_diff']:>12.6f} {r['samples_with_diff']:>12}{marker}")

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)

if __name__ == '__main__':
    main()
