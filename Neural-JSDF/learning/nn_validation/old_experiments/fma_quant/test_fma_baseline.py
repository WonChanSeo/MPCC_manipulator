"""
Test FMA implementation against native PyTorch model forward pass
This will determine if the 32.49cm error is due to a bug or is the model's baseline error
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation_fma_quant import FMAQuantValidation, quantize_float

print("="*80)
print("Testing FMA E8M23 (no quantization) vs Native Model Forward Pass")
print("="*80)

# Initialize FMA validator with E8M23 (no quantization)
print("\nInitializing FMA validator (E8M23)...")
fma = FMAQuantValidation(exponent_bits=8, mantissa_bits=23)

# Generate a small test set
print("Generating 10 test samples...")
device = torch.device('cpu')
params = {'device': device, 'dtype': torch.float32}
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params)
q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params)
q_span = q_max - q_min

torch.manual_seed(42)
input_data = q_min + q_span * torch.rand(10, 10, **params)

# Run FMA forward pass (manual implementation)
print("\nRunning FMA manual forward pass...")
fma_output = fma.calc_nn_pred(input_data)
print(f"FMA output shape: {fma_output.shape}")
print(f"FMA output (first sample, first 5): {fma_output[0, :5]}")

# Run native model forward pass (no quantization)
print("\nRunning native model forward pass...")
with torch.no_grad():
    native_output = fma.nn_model.model(input_data)
print(f"Native output shape: {native_output.shape}")
print(f"Native output (first sample, first 5): {native_output[0, :5]}")

# Compare
diff = torch.abs(fma_output - native_output)
print("\n" + "="*80)
print("Comparison:")
print("="*80)
print(f"Max difference: {torch.max(diff).item():.10e}")
print(f"Mean difference: {torch.mean(diff).item():.10e}")
print(f"Are they close (atol=1e-5)? {torch.allclose(fma_output, native_output, atol=1e-5, rtol=1e-5)}")
print(f"Are they close (atol=1e-4)? {torch.allclose(fma_output, native_output, atol=1e-4, rtol=1e-4)}")
print(f"Are they close (atol=1e-3)? {torch.allclose(fma_output, native_output, atol=1e-3, rtol=1e-3)}")

# Show first sample detailed comparison
print("\nFirst sample detailed comparison:")
for i in range(min(9, fma_output.shape[1])):
    print(f"  Output {i}: FMA={fma_output[0,i]:.10f}, Native={native_output[0,i]:.10f}, Diff={diff[0,i]:.10e}")

if torch.allclose(fma_output, native_output, atol=1e-3, rtol=1e-3):
    print("\n✓ FMA implementation matches native model!")
    print("  → The 32.49cm error is the model's baseline approximation error")
else:
    print("\n✗ BUG FOUND: FMA implementation does NOT match native model!")
    print(f"  → There is a bug in the FMA manual forward pass")
    print(f"  → Max difference: {torch.max(diff).item():.10e}")

print("\n" + "="*80)
