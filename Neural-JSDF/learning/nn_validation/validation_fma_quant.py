"""
Neural-JSDF FMA-level Quantization Script
Simulates output-stationary dataflow with per-FMA quantization

Output Stationary Dataflow:
- For each output element:
  - Initialize accumulator to 0
  - For each input element:
    - multiply: result = weight * input
    - quantize multiply result
    - add: accumulator = accumulator + result
    - quantize accumulator (FMA quantization)
  - Store quantized accumulator as output

This is the most realistic simulation of extreme low-precision hardware.
"""

import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from sdf.network_macros_mod import scale_to_net, scale_to_base
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time


def quantize_float(tensor, exponent_bits, mantissa_bits):
    """
    Quantize tensor to specified exponent and mantissa bits
    IEEE 754 bit manipulation

    FIXED: Properly handle scalar tensors and avoid unnecessary conversions for FP32
    """
    # Early return for FP32 (no quantization needed)
    if exponent_bits >= 8 and mantissa_bits >= 23:
        return tensor

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
    # Use .astype to ensure proper dtype
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


def fma_matmul_output_stationary(x, weight, bias, exponent_bits, mantissa_bits):
    """
    Output-stationary matrix multiplication with per-FMA quantization

    Args:
        x: input (batch_size, in_features)
        weight: weight (out_features, in_features)
        bias: bias (out_features,)
        exponent_bits: number of exponent bits
        mantissa_bits: number of mantissa bits

    Returns:
        output: (batch_size, out_features) with per-FMA quantization
    """
    batch_size, in_features = x.shape
    out_features, _ = weight.shape

    # Quantize inputs and weights once
    x_q = quantize_float(x, exponent_bits, mantissa_bits)
    weight_q = quantize_float(weight, exponent_bits, mantissa_bits)

    # Output buffer
    output = torch.zeros(batch_size, out_features, device=x.device)

    # Output stationary: iterate over each output element
    for b in range(batch_size):
        for o in range(out_features):
            # Initialize accumulator for this output element
            acc = torch.tensor(0.0, device=x.device)

            # FMA loop: accumulate input_i * weight_{o,i}
            for i in range(in_features):
                # Multiply
                mul_result = x_q[b, i] * weight_q[o, i]

                # Add (FMA)
                acc = acc + mul_result

                # Quantize accumulator after each FMA
                acc = quantize_float(acc, exponent_bits, mantissa_bits)

            # Add bias
            if bias is not None:
                bias_q = quantize_float(bias[o], exponent_bits, mantissa_bits)
                acc = acc + bias_q
                acc = quantize_float(acc, exponent_bits, mantissa_bits)

            output[b, o] = acc

    return output


def fma_relu(x, exponent_bits, mantissa_bits):
    """
    ReLU with quantization
    """
    out = torch.relu(x)
    out_q = quantize_float(out, exponent_bits, mantissa_bits)
    return out_q


class FMAQuantValidation:
    """Validation with FMA-level quantization (output stationary)"""

    def __init__(self, exponent_bits=8, mantissa_bits=7):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

        device = torch.device('cpu', 0)
        self.dtype = torch.float32
        self.dtype_name = f"fma-quant (E{exponent_bits}M{mantissa_bits})"
        self.params = {'device': device, 'dtype': self.dtype}

        print(f"Initializing with FMA-level quantization: {self.dtype_name}")
        print(f"Output-stationary dataflow with per-FMA quantization")

        # Load nn model
        fname = 'sdf_256x5_mesh_50000.pt'
        print(f"Weight file: {fname}")

        # Create original model
        self.nn_model = RobotSdfCollisionNet.__new__(RobotSdfCollisionNet)
        self.nn_model.in_channels = 10
        self.nn_model.out_channels = 9

        from sdf.network_macros_mod import MLPRegression
        import torch.nn as nn
        act_fn = nn.ReLU
        mlp_layers = [256] * 4
        skips = []
        self.nn_model.model = MLPRegression(10, 9, mlp_layers, skips, act_fn=act_fn, nerf=True)
        self.nn_model.m = torch.zeros((500, 1)).to(device)
        self.nn_model.m[:, 0] = 1
        self.nn_model.order = list(range(9))

        # Load weights
        checkpoint = torch.load('../../env/' + fname, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        self.nn_model.model.load_state_dict(state_dict)

        # Load normalization parameters
        if 'norm' in checkpoint:
            self.nn_model.norm_dict = checkpoint['norm']
            for k in self.nn_model.norm_dict.keys():
                self.nn_model.norm_dict[k]['mean'] = self.nn_model.norm_dict[k]['mean'].to(device)
                self.nn_model.norm_dict[k]['std'] = self.nn_model.norm_dict[k]['std'].to(device)
            print("Normalization parameters loaded")

        self.nn_model.tensor_args = self.params
        self.nn_model.model.to(**self.params)

        # Extract layer weights for manual forward pass
        self.layers = []
        for name, module in self.nn_model.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                self.layers.append({
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone() if module.bias is not None else None,
                    'name': name
                })

        print(f"Extracted {len(self.layers)} linear layers")
        print(f"Model loaded with FMA quantization")

        # Load robot meshes
        data_mat = loadmat('../data-sampling/meshes/mesh_light_pts.mat')['mesh'][0]
        self.meshes = []
        self.v_int_pts = []
        for link in data_mat:
            vertices = torch.tensor(link[0][0][1]).to(**self.params)
            int_pts = torch.tensor(link[0][0][3]).to(**self.params)
            v_int_pts = torch.cat((vertices, int_pts), 0)
            self.v_int_pts.append(v_int_pts)

        # Robot parameters
        dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
        dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
        dh_alpha = torch.tensor([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
        self.dh = torch.vstack((dh_d, dh_a*0, dh_a, dh_alpha)).T.to(**self.params)

    def calc_nn_pred(self, input):
        """
        Calculate NN predictions with FMA-level quantization
        Manual forward pass through all layers with nerf encoding
        """
        # Normalize input (scale_to_net)
        x_norm = scale_to_net(input, self.nn_model.norm_dict, 'x')

        # Positional encoding (nerf): [x, sin(x), cos(x)]
        x_sin = torch.sin(x_norm)
        x_cos = torch.cos(x_norm)
        x_nerf = torch.cat((x_norm, x_sin, x_cos), dim=-1)

        # Quantize positional encoded input
        x = quantize_float(x_nerf, self.exponent_bits, self.mantissa_bits)

        # Forward pass through each layer
        for i, layer in enumerate(self.layers):
            print(f"  Layer {i}: {x.shape} -> ", end='', flush=True)

            # FMA-level quantized matmul
            x = fma_matmul_output_stationary(
                x,
                layer['weight'],
                layer['bias'],
                self.exponent_bits,
                self.mantissa_bits
            )

            print(f"{x.shape}", flush=True)

            # ReLU (except last layer)
            if i < len(self.layers) - 1:
                x = fma_relu(x, self.exponent_bits, self.mantissa_bits)

        # Denormalize output (scale_to_base)
        output = scale_to_base(x, self.nn_model.norm_dict, 'y')

        return output.float()

    def get_mesh_fk(self, q):
        """Get forward kinematics for mesh vertices"""
        fk = dh_fk(torch.cat((q, torch.tensor([0]).to(q.device)), 0), self.dh)
        v_vec = []
        for i, P in enumerate(fk):
            P_q = quantize_float(P, self.exponent_bits, self.mantissa_bits)
            v_vec.append(self.v_int_pts[i] @ P_q[:3,:3].T + P_q[:3,3:4].T)
        return v_vec

    def get_mindists(self, v_vec, y):
        """Get minimum distances"""
        mindists = []
        for v_link in v_vec:
            dist = torch.cdist(v_link.float(), y.unsqueeze(0).float(), p=2)
            mindist, _ = dist.min(dim=0, keepdim=False)
            mindists.append(mindist)
        return torch.cat(mindists, 0)

    def calc_mesh_mindist(self, input):
        """Calculate ground truth mesh distances"""
        q = input[:, :7]
        y = input[:, 7:]
        n_pts = y.shape[0]
        all_dists = torch.zeros(q.shape[0], 9, **self.params)
        for i in range(n_pts):
            v = self.get_mesh_fk(q[i])
            mindists = self.get_mindists(v, y[i])
            all_dists[i, :] = mindists.to(self.dtype)
        res = all_dists
        return (100*res.float())


def run_fma_test(exponent_bits, mantissa_bits, n_samples=5000):
    """
    Run validation test with FMA-level quantization

    WARNING: This is VERY slow due to Python loops in FMA implementation
    """
    print("\n" + "="*80)
    print(f"TESTING FMA-LEVEL QUANTIZATION: E{exponent_bits}M{mantissa_bits}")
    print(f"(Every FMA is quantized in output-stationary fashion)")
    print("="*80)

    # Create validator
    v = FMAQuantValidation(exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

    # Setup test range
    device = torch.device('cpu', 0)
    params = {'device': device, 'dtype': torch.float32}
    q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params)
    q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params)
    q_span = q_max - q_min

    # Generate test data
    print(f"Generating {n_samples} test samples...")
    input_data_fp32 = q_min + q_span * torch.rand(n_samples, 10, **params)

    # Quantize input to target precision
    print(f"Quantizing input to E{exponent_bits}M{mantissa_bits}...")
    input_data = quantize_float(input_data_fp32, exponent_bits, mantissa_bits)

    # Run inference
    print("Running FMA-level quantized inference...")
    print("WARNING: This will be VERY slow due to Python loops")
    t0 = time.time()
    nn_pred = v.calc_nn_pred(input_data)
    t_nn = time.time() - t0

    # Calculate ground truth
    print("Calculating ground truth...")
    t0 = time.time()
    mesh_dist = v.calc_mesh_mindist(input_data)
    t_mesh = time.time() - t0

    # Calculate errors
    errors = torch.abs(nn_pred - mesh_dist)

    # Collect statistics
    results = {
        'exponent_bits': exponent_bits,
        'mantissa_bits': mantissa_bits,
        'n_samples': n_samples,
        'nn_time': t_nn,
        'mesh_time': t_mesh,
        'mean_error': errors.mean().item(),
        'std_error': errors.std().item(),
        'max_error': errors.max().item(),
        'median_error': errors.median().item(),
        'p95_error': torch.quantile(errors.flatten(), 0.95).item(),
        'p99_error': torch.quantile(errors.flatten(), 0.99).item(),
        'input_data': input_data.numpy(),
        'nn_pred': nn_pred.numpy(),
        'mesh_dist': mesh_dist.numpy(),
        'errors': errors.numpy()
    }

    # Print summary
    print(f"\n{'-'*80}")
    print(f"NN inference time: {t_nn:.2f} s ({n_samples/t_nn:.1f} samples/s)")
    print(f"\nError Statistics (cm):")
    print(f"  Mean:   {results['mean_error']:>8.2f}")
    print(f"  Std:    {results['std_error']:>8.2f}")
    print(f"  Median: {results['median_error']:>8.2f}")
    print(f"  Max:    {results['max_error']:>8.2f}")
    print(f"  95th:   {results['p95_error']:>8.2f}")
    print(f"  99th:   {results['p99_error']:>8.2f}")
    print(f"{'-'*80}")

    return results


if __name__ == "__main__":
    # Test configurations
    # WARNING: Start with small sample size due to extreme slowness
    configs = [
        (8, 23, "FP32"),
        (8, 7, "BF16"),
        (8, 6, "E8M6"),
        (8, 5, "E8M5"),
        (8, 4, "E8M4"),
    ]

    # Use small sample size for testing
    n_samples = 100  # Start with 100 samples to test

    print("="*80)
    print("NEURAL-JSDF FMA-LEVEL QUANTIZATION TEST")
    print("Each FMA (Fused Multiply-Add) is immediately quantized")
    print("Output-stationary dataflow simulation")
    print(f"Testing {len(configs)} configurations with {n_samples} samples each")
    print("WARNING: This will be VERY slow due to Python loops!")
    print("="*80)

    all_results = []

    for exp_bits, mant_bits, name in configs:
        results = run_fma_test(exp_bits, mant_bits, n_samples=n_samples)
        all_results.append(results)

        # Save results
        output_file = f"data_fma_E{exp_bits}M{mant_bits}.npz"
        np.savez(output_file,
                 exponent_bits=exp_bits,
                 mantissa_bits=mant_bits,
                 input=results['input_data'],
                 nn_pred=results['nn_pred'],
                 mesh_dist=results['mesh_dist'],
                 errors=results['errors'])
        print(f"✓ Saved to: {output_file}")

    # Save summary
    summary_file = "fma_quant_summary.npz"
    np.savez(summary_file,
             config_names=[name for _, _, name in configs],
             exponent_bits=[r['exponent_bits'] for r in all_results],
             mantissa_bits=[r['mantissa_bits'] for r in all_results],
             mean_errors=[r['mean_error'] for r in all_results],
             std_errors=[r['std_error'] for r in all_results],
             max_errors=[r['max_error'] for r in all_results],
             median_errors=[r['median_error'] for r in all_results],
             p95_errors=[r['p95_error'] for r in all_results],
             p99_errors=[r['p99_error'] for r in all_results],
             nn_times=[r['nn_time'] for r in all_results])
    print(f"\n✓ Summary saved to: {summary_file}")

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Config':<10} {'Exp':<5} {'Mant':<5} {'Mean':<10} {'Max':<10} {'Time (s)':<10}")
    print("-"*80)
    for i, (exp, mant, name) in enumerate(configs):
        r = all_results[i]
        print(f"{name:<10} {exp:<5} {mant:<5} {r['mean_error']:<10.2f} {r['max_error']:<10.2f} {r['nn_time']:<10.1f}")
    print("="*80)

    print("\nTo run with full 5000 samples, edit n_samples in the script.")
    print("Expected time for 5000 samples: ~10-30 minutes per configuration")
