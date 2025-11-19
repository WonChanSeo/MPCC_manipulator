#!/usr/bin/env python3
"""
Export weights from .pt file to .txt files in FP32 precision
This ensures C++ uses the exact same weights as Python
"""

import torch
import numpy as np
import os

print("="*80)
print("EXPORTING WEIGHTS FROM .PT TO .TXT (FP32 PRECISION)")
print("="*80)

# Load the model
pt_file = '../../env/sdf_256x5_mesh_50000.pt'
output_dir = '../../../cpp/NNmodel/env/parameter'

print(f"\nLoading model from: {pt_file}")
model = torch.load(pt_file, map_location='cpu')

if 'model_state_dict' in model:
    state_dict = model['model_state_dict']
else:
    state_dict = model

print(f"✓ Model loaded")

# Extract weights and biases
weights = []
biases = []

for key in sorted(state_dict.keys()):
    if 'weight' in key:
        weights.append(state_dict[key].numpy())
        print(f"  {key}: {state_dict[key].shape}")
    elif 'bias' in key:
        biases.append(state_dict[key].numpy())

print(f"\n✓ Extracted {len(weights)} weight matrices and {len(biases)} bias vectors")

# Verify output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"\n✓ Created output directory: {output_dir}")
else:
    print(f"\n✓ Output directory exists: {output_dir}")

# Backup old files
print(f"\nBacking up old weight files...")
for i in range(len(weights)):
    old_w = os.path.join(output_dir, f'weight_{i}.txt')
    old_b = os.path.join(output_dir, f'bias_{i}.txt')

    if os.path.exists(old_w):
        backup_w = os.path.join(output_dir, f'weight_{i}.txt.bak')
        os.rename(old_w, backup_w)
        print(f"  Backed up: weight_{i}.txt -> weight_{i}.txt.bak")

    if os.path.exists(old_b):
        backup_b = os.path.join(output_dir, f'bias_{i}.txt.bak')
        os.rename(old_b, backup_b)
        print(f"  Backed up: bias_{i}.txt -> bias_{i}.txt.bak")

# Save weights and biases
print(f"\nSaving weights and biases in FP32 precision...")

for i, (w, b) in enumerate(zip(weights, biases)):
    # Weight file
    weight_file = os.path.join(output_dir, f'weight_{i}.txt')
    np.savetxt(weight_file, w, fmt='%.10f')
    print(f"  ✓ Saved weight_{i}.txt: shape {w.shape}")

    # Bias file
    bias_file = os.path.join(output_dir, f'bias_{i}.txt')
    np.savetxt(bias_file, b, fmt='%.10f')
    print(f"  ✓ Saved bias_{i}.txt: shape {b.shape}")

# Verify the first few values match
print(f"\n" + "="*80)
print("VERIFICATION")
print("="*80)

# Compare first weight file
print(f"\nComparing first 5 values of weight_0.txt:")
print(f"Original Python values:")
for i in range(5):
    print(f"  [{i}]: {weights[0][0, i]:.10f}")

# Load back and compare
weight_0_reloaded = np.loadtxt(os.path.join(output_dir, 'weight_0.txt'))
print(f"\nReloaded from .txt file:")
for i in range(5):
    print(f"  [{i}]: {weight_0_reloaded[0, i]:.10f}")

diff = np.abs(weights[0] - weight_0_reloaded)
print(f"\nMax difference: {np.max(diff):.2e}")

if np.allclose(weights[0], weight_0_reloaded):
    print("✓ Weights match perfectly!")
else:
    print("✗ Warning: Weights don't match!")

print(f"\n" + "="*80)
print("EXPORT COMPLETE")
print("="*80)
print(f"\nExported {len(weights)} weight files and {len(biases)} bias files")
print(f"Output directory: {output_dir}")
print(f"\nNext steps:")
print(f"  1. Rebuild C++: cd ../../../cpp && ./build_with_options.sh bf16")
print(f"  2. Run test: cd build && ./test_validation_inference")
print(f"  3. Compare: cd ../../Neural-JSDF/learning/nn_validation && python3 compare_cpp_python.py")
print("="*80)
