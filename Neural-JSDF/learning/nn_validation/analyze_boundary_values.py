#!/usr/bin/env python3
"""
Analyze ReLU boundary values for Sample 15.
The hypothesis is that values very close to 0 may be on different sides
of the ReLU boundary in Python vs C++ due to FMA precision differences.
"""

import numpy as np
import torch
import os

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

def fma_float(a, b, c):
    """FMA using float64 for extended precision intermediate."""
    result = np.float64(a) * np.float64(b) + np.float64(c)
    return float(result)

def fma_matmul(x, weight, bias):
    """Perform FMA-quantized matrix multiplication + bias."""
    out_features = weight.shape[0]
    in_features = weight.shape[1]

    x_q = quantize_float(x)
    weight_q = quantize_float(weight)
    bias_q = quantize_float(bias)

    output = np.zeros(out_features, dtype=np.float32)

    for o in range(out_features):
        acc = 0.0
        for i in range(in_features):
            w_val = float(weight_q[o, i])
            x_val = float(x_q[i])
            acc = fma_float(w_val, x_val, acc)
            acc = float(quantize_float(np.array(acc, dtype=np.float32)))

        b_val = float(bias_q[o])
        acc = fma_float(b_val, 1.0, acc)
        acc = float(quantize_float(np.array(acc, dtype=np.float32)))
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

def analyze_sample(inputs, weights, biases, sample_idx):
    """Analyze pre-activation values for a sample."""
    print(f"\n{'='*70}")
    print(f"Analyzing ReLU Boundary Values for Sample {sample_idx}")
    print(f"{'='*70}")

    x = inputs[sample_idx]
    x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])

    # Track pre-activations at each layer
    all_pre_activations = []

    h = x_nerf
    for layer_idx in range(4):  # Only hidden layers (0-3)
        pre_act = fma_matmul(h, weights[layer_idx], biases[layer_idx])
        all_pre_activations.append(pre_act)
        h = relu(pre_act)

        # Find values very close to 0
        small_vals = np.abs(pre_act) < 0.1
        near_zero_vals = pre_act[small_vals]

        if len(near_zero_vals) > 0:
            print(f"\nLayer {layer_idx}: {np.sum(small_vals)} values with |x| < 0.1")
            sorted_vals = sorted(near_zero_vals, key=abs)
            print(f"  Smallest magnitude values:")
            for i, v in enumerate(sorted_vals[:10]):
                # Check if this value is negative (would become 0 after ReLU)
                status = "→ 0" if v < 0 else "→ kept"
                print(f"    {v:12.6f} {status}")

    # Check the final layer
    final_out = fma_matmul(h, weights[4], biases[4])
    all_pre_activations.append(final_out)

    return all_pre_activations

def compare_samples(inputs, weights, biases):
    """Compare samples that match vs samples that don't match."""
    print("\n" + "="*70)
    print("Comparing Boundary Value Patterns")
    print("="*70)

    # Samples to compare
    samples = {
        'good': [0, 5, 10],  # These match perfectly
        'bad': [15]          # This has differences
    }

    for category, sample_list in samples.items():
        print(f"\n{category.upper()} SAMPLES:")
        print("-" * 40)

        for sample_idx in sample_list:
            x = inputs[sample_idx]
            x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])

            # Count values very close to 0 at each layer
            h = x_nerf
            for layer_idx in range(4):
                pre_act = fma_matmul(h, weights[layer_idx], biases[layer_idx])
                h = relu(pre_act)

                # Count values in different ranges
                v_small = np.sum(np.abs(pre_act) < 0.01)
                v_tiny = np.sum(np.abs(pre_act) < 0.001)

                if layer_idx == 0:
                    print(f"Sample {sample_idx}, Layer {layer_idx}: {v_small} vals with |x|<0.01, {v_tiny} with |x|<0.001")

def main():
    # Load data
    data = np.load('python_validation_reference.npz')
    inputs = data['input']
    weights, biases = load_weights()

    # Analyze sample 15 in detail
    pre_acts = analyze_sample(inputs, weights, biases, 15)

    # Compare with good samples
    compare_samples(inputs, weights, biases)

    # Also check the smallest non-zero values after ReLU in layer 3
    print("\n" + "="*70)
    print("Checking Layer 3 Post-ReLU Values for Sample 15")
    print("="*70)

    x = inputs[15]
    x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])
    h = x_nerf
    for layer_idx in range(4):
        pre_act = fma_matmul(h, weights[layer_idx], biases[layer_idx])
        h = relu(pre_act)

        if layer_idx == 3:
            # Get non-zero post-ReLU values
            non_zero_h = h[h > 0]
            sorted_h = np.sort(non_zero_h)
            print(f"\nSmallest positive values after ReLU:")
            for i, v in enumerate(sorted_h[:15]):
                print(f"  {v:.10f}")

if __name__ == '__main__':
    main()
