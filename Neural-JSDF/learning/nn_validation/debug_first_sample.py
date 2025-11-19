#!/usr/bin/env python3
"""
Debug script to trace first sample inference step-by-step
Compare with C++ debug output
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '../..')

from validation_fma_quant import FMAQuantValidation, quantize_float

print("="*80)
print("DEBUG: First Sample Inference (Python)")
print("="*80)

# Initialize model
validator = FMAQuantValidation(exponent_bits=8, mantissa_bits=7)

# Load input
ref = np.load('python_validation_reference.npz')
input_data = torch.from_numpy(ref['input']).float()

# Get first sample
first_sample = input_data[0:1, :]  # Shape: (1, 10)
print(f"\nFirst sample input shape: {first_sample.shape}")
print(f"First sample values:")
for i in range(10):
    print(f"  x[{i}] = {first_sample[0, i].item():.10f}")

# Normalize input
from sdf.network_macros_mod import scale_to_net
x_norm = scale_to_net(first_sample, validator.nn_model.norm_dict, 'x')

# NERF encoding
x_sin = torch.sin(x_norm)
x_cos = torch.cos(x_norm)
x_nerf = torch.cat((x_norm, x_sin, x_cos), dim=-1)

print(f"\nAfter NERF encoding shape: {x_nerf.shape}")
print(f"First 5 NERF values:")
for i in range(5):
    print(f"  x_nerf[{i}] = {x_nerf[0, i].item():.10f}")

# Quantize NERF encoded input
x_q = quantize_float(x_nerf, 8, 7)

print(f"\nAfter quantization (first 5):")
for i in range(5):
    print(f"  x_q[{i}] = {x_q[0, i].item():.10f}")

# Get first layer weights and bias
layer_0 = validator.layers[0]
weight = layer_0['weight']  # Shape: (256, 30)
bias = layer_0['bias']      # Shape: (256,)

print(f"\nFirst layer weight shape: {weight.shape}")
print(f"First layer bias shape: {bias.shape}")

# Quantize weights
weight_q = quantize_float(weight, 8, 7)

print(f"\nFirst 5 weights for output[0]:")
for i in range(5):
    print(f"  w[0, {i}] = {weight[0, i].item():.10f} -> {weight_q[0, i].item():.10f}")

# Manual FMA for output[0] of first sample
print(f"\n=== FMA Loop for Output[0], Sample[0] ===")
acc = torch.tensor(0.0)

for i in range(5):  # First 5 iterations
    mul_result = x_q[0, i] * weight_q[0, i]
    acc = acc + mul_result
    acc_q = quantize_float(acc, 8, 7)

    print(f"  FMA[{i}]: w={weight_q[0, i].item():.10f}, "
          f"x={x_q[0, i].item():.10f}, "
          f"w*x={mul_result.item():.10f}, "
          f"acc={acc_q.item():.10f}")

    acc = acc_q

# Continue for all inputs
for i in range(5, 30):
    mul_result = x_q[0, i] * weight_q[0, i]
    acc = acc + mul_result
    acc_q = quantize_float(acc, 8, 7)
    acc = acc_q

print(f"\nBefore bias: acc = {acc.item():.10f}")
print(f"Bias[0] = {bias[0].item():.10f}")

# Add bias
bias_q = quantize_float(bias[0], 8, 7)
acc = acc + bias_q
acc_q = quantize_float(acc, 8, 7)

print(f"After bias: acc = {acc_q.item():.10f}")

# ReLU
output = torch.relu(acc_q)
print(f"After ReLU: output = {output.item():.10f}")

# Run full inference for comparison
print(f"\n=== Full Inference (for verification) ===")
full_output = validator.calc_nn_pred(first_sample)
print(f"Full output shape: {full_output.shape}")
print(f"Output[0,0] = {full_output[0, 0].item():.10f}")

print("="*80)
