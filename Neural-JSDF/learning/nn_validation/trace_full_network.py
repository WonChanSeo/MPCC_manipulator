#!/usr/bin/env python3
"""
Full network trace for Sample 15 to compare with C++ output.
This traces through all 5 layers to find where the difference occurs.
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
    """
    Perform FMA-quantized matrix multiplication + bias.
    x: shape (in_features,)
    weight: shape (out_features, in_features)
    bias: shape (out_features,)
    Returns: shape (out_features,)
    """
    out_features = weight.shape[0]
    in_features = weight.shape[1]

    # Quantize all inputs
    x_q = quantize_float(x)
    weight_q = quantize_float(weight)
    bias_q = quantize_float(bias)

    output = np.zeros(out_features, dtype=np.float32)

    for o in range(out_features):
        acc = 0.0

        # FMA loop
        for i in range(in_features):
            w_val = float(weight_q[o, i])
            x_val = float(x_q[i])

            acc = fma_float(w_val, x_val, acc)

            # Quantize accumulator after each FMA
            acc = float(quantize_float(np.array(acc, dtype=np.float32)))

        # Add bias
        b_val = float(bias_q[o])
        acc = fma_float(b_val, 1.0, acc)
        acc = float(quantize_float(np.array(acc, dtype=np.float32)))

        output[o] = acc

    return output

def relu(x):
    """ReLU activation."""
    return np.maximum(x, 0)

def load_weights():
    """Load weights from text files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_dir = os.path.join(script_dir, '..', '..', '..', 'cpp', 'NNmodel', 'env', 'parameter')

    # Architecture: 30 -> 256 -> 256 -> 256 -> 256 -> 9
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

def load_cpp_outputs():
    """Load C++ output from the validation file."""
    cpp_output = np.loadtxt('cpp_validation_output.txt')
    return cpp_output

def forward_pass_verbose(inputs, weights, biases, sample_idx):
    """Run forward pass for a single sample with verbose output."""
    print(f"\n{'='*70}")
    print(f"Forward Pass for Sample {sample_idx}")
    print(f"{'='*70}")

    x = inputs[sample_idx]
    print(f"\nInput (raw): {x[:5]}... (shape: {x.shape})")

    # NERF encoding
    x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])
    print(f"NERF encoded: {x_nerf[:5]}... (shape: {x_nerf.shape})")

    layer_outputs = []

    for layer_idx in range(5):
        weight = weights[layer_idx]
        bias = biases[layer_idx]

        print(f"\n--- Layer {layer_idx} ---")
        print(f"Input shape: {x_nerf.shape if layer_idx == 0 else layer_outputs[-1].shape}")
        print(f"Weight shape: {weight.shape}, Bias shape: {bias.shape}")

        if layer_idx == 0:
            pre_act = fma_matmul(x_nerf, weight, bias)
        else:
            pre_act = fma_matmul(layer_outputs[-1], weight, bias)

        print(f"Pre-activation (first 5): {pre_act[:min(5, len(pre_act))]}")
        print(f"Pre-activation stats: min={pre_act.min():.4f}, max={pre_act.max():.4f}")

        if layer_idx < 4:  # Apply ReLU for all but last layer
            post_act = relu(pre_act)
            zeros = np.sum(post_act == 0)
            print(f"Post-ReLU zeros: {zeros}/{len(post_act)}")
            layer_outputs.append(post_act)
        else:
            layer_outputs.append(pre_act)
            print(f"Final output (all 9):")
            for i in range(9):
                print(f"  Output[{i}]: {pre_act[i]:.6f}")

    return layer_outputs[-1]

def main():
    print("="*70)
    print("Full Network Trace for Python vs C++ Comparison")
    print("="*70)

    # Load data
    data = np.load('python_validation_reference.npz')
    inputs = data['input']  # Shape: (100, 10)
    print(f"\nLoaded {inputs.shape[0]} input samples")

    weights, biases = load_weights()
    print(f"Loaded {len(weights)} layer weights")

    cpp_outputs = load_cpp_outputs()
    print(f"Loaded C++ outputs: shape {cpp_outputs.shape}")

    # Focus on Sample 15
    sample_idx = 15

    # Run forward pass
    python_output = forward_pass_verbose(inputs, weights, biases, sample_idx)

    # Compare with C++ output
    cpp_sample_output = cpp_outputs[sample_idx]  # Shape is already (100, 9)

    print(f"\n{'='*70}")
    print(f"Comparison: Python vs C++ for Sample {sample_idx}")
    print(f"{'='*70}")

    print(f"\n{'Output':>6} {'Python':>14} {'C++':>14} {'Diff':>14}")
    print("-" * 50)

    for i in range(9):
        py_val = python_output[i]
        cpp_val = cpp_sample_output[i]
        diff = abs(py_val - cpp_val)
        marker = " *" if diff > 0.001 else ""
        print(f"{i:>6} {py_val:>14.6f} {cpp_val:>14.6f} {diff:>14.6f}{marker}")

    # Also check a few other samples
    print(f"\n{'='*70}")
    print("Quick comparison for samples 0, 5, 10, 20, 50")
    print("="*70)

    for s_idx in [0, 5, 10, 20, 50]:
        x = inputs[s_idx]
        x_nerf = np.concatenate([x, np.sin(x), np.cos(x)])

        # Forward pass
        h = x_nerf
        for layer_idx in range(5):
            h = fma_matmul(h, weights[layer_idx], biases[layer_idx])
            if layer_idx < 4:
                h = relu(h)

        cpp_out = cpp_outputs[s_idx]

        diffs = np.abs(h - cpp_out)
        max_diff = np.max(diffs)
        print(f"Sample {s_idx:3d}: max_diff = {max_diff:.6f}")

if __name__ == '__main__':
    main()
