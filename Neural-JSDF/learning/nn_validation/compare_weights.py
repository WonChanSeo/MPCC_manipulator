"""
Compare weights between parameter/*.txt files and sdf_256x5_mesh_50000.pt
"""
import torch
import numpy as np
import sys

print("="*80)
print("WEIGHT COMPARISON: parameter/*.txt vs sdf_256x5_mesh_50000.pt")
print("="*80)
print()

# Load .pt file
pt_file = '../../env/sdf_256x5_mesh_50000.pt'
print(f"Loading {pt_file}...")
checkpoint = torch.load(pt_file, map_location='cpu')

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("✓ Found model_state_dict in checkpoint")
else:
    state_dict = checkpoint
    print("✓ Using checkpoint directly as state_dict")

print(f"✓ Loaded {len(state_dict)} parameters from .pt file")
print()

# Print .pt file structure
print("State dict keys:")
for key in state_dict.keys():
    shape = state_dict[key].shape
    print(f"  {key}: {shape}")
print()

# Load parameter/*.txt files
param_dir = '../../env/parameter/'
print(f"Loading weights from {param_dir}...")

txt_weights = {}
txt_biases = {}

for i in range(5):
    # Load weight
    weight_file = f'{param_dir}weight_{i}.txt'
    try:
        weight = np.loadtxt(weight_file)
        txt_weights[i] = weight
        print(f"✓ Loaded weight_{i}.txt: shape {weight.shape}")
    except Exception as e:
        print(f"✗ Failed to load weight_{i}.txt: {e}")

    # Load bias
    bias_file = f'{param_dir}bias_{i}.txt'
    try:
        bias = np.loadtxt(bias_file)
        txt_biases[i] = bias
        print(f"✓ Loaded bias_{i}.txt: shape {bias.shape}")
    except Exception as e:
        print(f"✗ Failed to load bias_{i}.txt: {e}")

print()
print("="*80)
print("COMPARISON")
print("="*80)
print()

# Compare each layer
layer_mapping = [
    ('layers.0.0.0.weight', 'weight_0', 0),
    ('layers.0.0.0.bias', 'bias_0', 0),
    ('layers.0.1.0.weight', 'weight_1', 1),
    ('layers.0.1.0.bias', 'bias_1', 1),
    ('layers.0.2.0.weight', 'weight_2', 2),
    ('layers.0.2.0.bias', 'bias_2', 2),
    ('layers.0.3.0.weight', 'weight_3', 3),
    ('layers.0.3.0.bias', 'bias_3', 3),
    ('layers.0.4.0.weight', 'weight_4', 4),
    ('layers.0.4.0.bias', 'bias_4', 4),
]

all_match = True

for pt_key, txt_name, idx in layer_mapping:
    if pt_key not in state_dict:
        print(f"✗ {pt_key} not found in .pt file")
        all_match = False
        continue

    pt_tensor = state_dict[pt_key].numpy()

    if 'weight' in txt_name:
        if idx not in txt_weights:
            print(f"✗ {txt_name}.txt not loaded")
            all_match = False
            continue
        txt_array = txt_weights[idx]
    else:  # bias
        if idx not in txt_biases:
            print(f"✗ {txt_name}.txt not loaded")
            all_match = False
            continue
        txt_array = txt_biases[idx]

    # Reshape txt array to match pt tensor if needed
    if pt_tensor.shape != txt_array.shape:
        # Try transposing or reshaping
        if pt_tensor.size == txt_array.size:
            if len(txt_array.shape) == 2:
                txt_array = txt_array.T
            txt_array = txt_array.reshape(pt_tensor.shape)
        else:
            print(f"✗ {pt_key} vs {txt_name}: shape mismatch")
            print(f"    .pt shape: {pt_tensor.shape}, .txt shape: {txt_array.shape}")
            all_match = False
            continue

    # Compare values
    diff = np.abs(pt_tensor - txt_array)
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff < 1e-6:
        status = "✓ MATCH"
    elif max_diff < 1e-3:
        status = "~ CLOSE"
    else:
        status = "✗ DIFFER"
        all_match = False

    print(f"{status:10} {pt_key:20} vs {txt_name:15} | max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")

print()
print("="*80)
if all_match:
    print("RESULT: ✓ All weights MATCH!")
    print("The parameter/*.txt files contain the SAME weights as sdf_256x5_mesh_50000.pt")
else:
    print("RESULT: ✗ Some weights DIFFER or structure mismatch")
    print("The parameter/*.txt files may be from a DIFFERENT model")
print("="*80)
