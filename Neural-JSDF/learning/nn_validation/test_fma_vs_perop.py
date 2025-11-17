"""
Test to compare FMA vs Per-Op with FP32 (E8M23)
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '/home/mms-wonchan/git/MPCC_manipulator/Neural-JSDF/learning')

from utils.config import Config
from models.mlp.mlp_model import NeuralMLP

# Import quantize functions
from validation_fma_quant import quantize_float, fma_matmul_output_stationary, FMAQuantValidation
from validation_per_op_quant import PerOpQuantValidation

def scale_to_net(x, norm_dict, key):
    mean = norm_dict[key]['mean']
    std = norm_dict[key]['std']
    return (x - mean) / std

def scale_to_base(x, norm_dict, key):
    mean = norm_dict[key]['mean']
    std = norm_dict[key]['std']
    return x * std + mean

print("="*80)
print("Comparing FP32 (E8M23) FMA vs Per-Op")
print("="*80)

# Create Per-Op validator
print("\nInitializing Per-Op validator...")
perop = PerOpQuantValidation('perop-quant', 8, 23, 'cpu')  # E8M23 (FP32)

# Create FMA validator
print("Initializing FMA validator...")
fma = FMAQuantValidation('fma-quant', 8, 23, 'cpu')  # E8M23 (FP32)

# Generate a single test input
torch.manual_seed(42)
q = torch.rand(7) * 2 - 1  # Random joint angles

# Run Per-Op
print("\nRunning Per-Op inference...")
perop_output = perop.calc_nn_pred(q.unsqueeze(0))
print(f"Per-Op output shape: {perop_output.shape}")
print(f"Per-Op output (first 5): {perop_output[0, :5]}")

# Run FMA
print("\nRunning FMA inference...")
fma_output = fma.calc_nn_pred(q.unsqueeze(0))
print(f"FMA output shape: {fma_output.shape}")
print(f"FMA output (first 5): {fma_output[0, :5]}")

# Compare
diff = torch.abs(perop_output - fma_output)
print("\n" + "="*80)
print("Comparison:")
print("="*80)
print(f"Max difference: {torch.max(diff).item():.10e}")
print(f"Mean difference: {torch.mean(diff).item():.10e}")
print(f"Are they close? {torch.allclose(perop_output, fma_output, atol=1e-5, rtol=1e-5)}")

# Show per-element differences
print("\nPer-element differences (first 9 outputs):")
for i in range(min(9, perop_output.shape[1])):
    print(f"  Element {i}: Per-Op={perop_output[0,i]:.10f}, FMA={fma_output[0,i]:.10f}, Diff={diff[0,i]:.10e}")

print("\n" + "="*80)
print("✓ Comparison complete")
print("="*80)
