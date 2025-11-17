"""
Extract weights and biases from sdf_256x5_mesh_50000.pt in BF16 format
"""
import torch
import numpy as np
import os

# Load model
model_path = 'sdf_256x5_mesh_50000.pt'
checkpoint = torch.load(model_path, map_location='cpu')
state_dict = checkpoint['model_state_dict']

print(f"Loaded model from: {model_path}")
print(f"Epoch: {checkpoint['epoch']}")
print(f"Model state dict keys: {list(state_dict.keys())}")
print()

# Create output directory
output_dir = 'parameter_bf16'
os.makedirs(output_dir, exist_ok=True)

# Extract weights and biases
layer_mapping = {
    'layers.0.0.0.weight': ('weight_0', 256, 30),
    'layers.0.0.0.bias': ('bias_0', 256, 1),
    'layers.0.1.0.weight': ('weight_1', 256, 256),
    'layers.0.1.0.bias': ('bias_1', 256, 1),
    'layers.0.2.0.weight': ('weight_2', 256, 256),
    'layers.0.2.0.bias': ('bias_2', 256, 1),
    'layers.0.3.0.weight': ('weight_3', 256, 256),
    'layers.0.3.0.bias': ('bias_3', 256, 1),
    'layers.0.4.0.weight': ('weight_4', 9, 256),
    'layers.0.4.0.bias': ('bias_4', 9, 1),
}

for key, (name, rows, cols) in layer_mapping.items():
    if key in state_dict:
        # Convert to BF16
        tensor_fp32 = state_dict[key].cpu()
        tensor_bf16 = tensor_fp32.to(torch.bfloat16)

        # Convert back to FP32 for saving as text (numpy doesn't have bfloat16)
        # But we keep the precision of BF16
        tensor_np = tensor_bf16.float().numpy()

        # Save as text file
        output_path = os.path.join(output_dir, f'{name}.txt')

        if 'weight' in name:
            # Weights are saved row by row
            np.savetxt(output_path, tensor_np, fmt='%.8f', delimiter=' ')
        else:
            # Biases are saved as a single column
            np.savetxt(output_path, tensor_np, fmt='%.8f')

        print(f"Saved {name}: shape {tensor_np.shape} -> {output_path}")
        print(f"  Original (FP32) range: [{tensor_fp32.min():.6f}, {tensor_fp32.max():.6f}]")
        print(f"  BF16 quantized range: [{tensor_np.min():.6f}, {tensor_np.max():.6f}]")
        print(f"  Max absolute difference: {np.abs(tensor_fp32.numpy() - tensor_np).max():.6e}")
        print()
    else:
        print(f"Warning: {key} not found in state_dict")

print(f"\nAll weights and biases saved to: {output_dir}/")
print(f"Format: BF16 precision (saved as text in FP32 format)")
