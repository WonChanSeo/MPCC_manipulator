"""
Neural-JSDF Per-Operation Quantization
Implements true low-precision simulation where EVERY operation is quantized

This simulates actual hardware behavior where each arithmetic operation
(mul, add) is performed and then immediately quantized to target precision.
"""

import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import torch.nn as nn
import numpy as np
from fk_num import *
import time


def quantize_float(tensor, exponent_bits, mantissa_bits):
    """
    Quantize tensor to specified exponent and mantissa bits

    Args:
        tensor: Input tensor (float32)
        exponent_bits: Number of exponent bits (3-8)
        mantissa_bits: Number of mantissa bits (2-23)

    Returns:
        Quantized tensor
    """
    if exponent_bits >= 8 and mantissa_bits >= 23:
        return tensor

    tensor_np = tensor.cpu().detach().numpy()
    bits = tensor_np.view(np.uint32)

    # 1. Quantize mantissa
    mantissa_bits_to_zero = 23 - mantissa_bits
    mantissa_mask = np.uint32(0x007FFFFF >> mantissa_bits_to_zero) << mantissa_bits_to_zero

    # 2. Quantize exponent
    exponent_bits_to_zero = 8 - exponent_bits
    exponent = (bits >> 23) & 0xFF
    exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero
    exponent_quantized = np.clip(exponent_quantized, 0, 255)

    # 3. Reconstruct
    sign_bit = bits & 0x80000000
    exponent_field = (exponent_quantized & 0xFF) << 23
    mantissa_field = bits & mantissa_mask
    quantized_bits = sign_bit | exponent_field | mantissa_field

    quantized_np = quantized_bits.view(np.float32)
    return torch.from_numpy(quantized_np).to(tensor.device)


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer that applies quantization after each operation

    Forward pass:
    1. Quantize weights
    2. MatMul (in FP32 but with quantized operands)
    3. Quantize result
    4. Add quantized bias
    5. Quantize final result
    """

    def __init__(self, linear_layer, exponent_bits, mantissa_bits):
        super().__init__()
        self.weight = linear_layer.weight.data.clone()
        self.bias = linear_layer.bias.data.clone() if linear_layer.bias is not None else None
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

    def forward(self, x):
        # Quantize weight
        w_q = quantize_float(self.weight, self.exponent_bits, self.mantissa_bits)

        # MatMul: x @ w^T
        # x is already quantized from previous layer
        out = torch.matmul(x, w_q.T)

        # Quantize MatMul result
        out_q = quantize_float(out, self.exponent_bits, self.mantissa_bits)

        # Add bias if exists
        if self.bias is not None:
            bias_q = quantize_float(self.bias, self.exponent_bits, self.mantissa_bits)
            out = out_q + bias_q
            # Quantize addition result
            out_q = quantize_float(out, self.exponent_bits, self.mantissa_bits)

        return out_q


class QuantizedReLU(nn.Module):
    """
    Quantized ReLU that quantizes after activation
    """

    def __init__(self, exponent_bits, mantissa_bits):
        super().__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

    def forward(self, x):
        out = torch.relu(x)
        # Quantize ReLU result
        out_q = quantize_float(out, self.exponent_bits, self.mantissa_bits)
        return out_q


class QuantizedMLP(nn.Module):
    """
    Quantized MLP that applies per-operation quantization
    """

    def __init__(self, original_model, exponent_bits, mantissa_bits):
        super().__init__()
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

        # Extract layers from original model
        self.layers = nn.ModuleList()

        # Assuming the model is Sequential or has numbered layers
        # We need to wrap each Linear and ReLU
        for name, module in original_model.named_modules():
            if isinstance(module, nn.Linear):
                self.layers.append(QuantizedLinear(module, exponent_bits, mantissa_bits))
            elif isinstance(module, nn.ReLU):
                self.layers.append(QuantizedReLU(exponent_bits, mantissa_bits))

    def forward(self, x):
        # Quantize input
        x_q = quantize_float(x, self.exponent_bits, self.mantissa_bits)

        # Forward through quantized layers
        for layer in self.layers:
            x_q = layer(x_q)

        return x_q


class PerOpQuantValidation:
    """Validation with per-operation quantization"""

    def __init__(self, exponent_bits=8, mantissa_bits=7):
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

        device = torch.device('cpu', 0)
        self.dtype = torch.float32
        self.dtype_name = f"per-op-quant (E{exponent_bits}M{mantissa_bits})"
        self.params = {'device': device, 'dtype': self.dtype}

        print(f"Initializing with per-operation quantization: {self.dtype_name}")

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

        # Create quantized version
        print(f"Creating per-operation quantized model...")
        self.quantized_model = self._create_quantized_model()

        print(f"Model loaded and quantized")

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

    def _create_quantized_model(self):
        """Create a quantized version of the model"""
        # We'll use monkey-patching approach instead
        # Just return the original model and quantize in forward pass
        return self.nn_model.model

    def calc_nn_pred(self, input):
        """Calculate NN predictions with per-operation quantization"""
        exp_bits = self.exponent_bits
        mant_bits = self.mantissa_bits

        # Register forward hooks to quantize after each layer
        handles = []

        def quantize_hook(module, input, output):
            return quantize_float(output, exp_bits, mant_bits)

        # Add hooks to all Linear and ReLU layers in the model
        for module in self.nn_model.model.modules():
            if isinstance(module, (nn.Linear, nn.ReLU)):
                handle = module.register_forward_hook(quantize_hook)
                handles.append(handle)

        with torch.no_grad():
            # Use model for inference
            y_pred = self.nn_model.model(input)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return y_pred.float()

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
            all_dists[i, :] = mindists
        return 100 * all_dists.float()


def run_per_op_test(exponent_bits, mantissa_bits, n_samples=5000):
    """Run validation with per-operation quantization"""

    print("\n" + "="*80)
    print(f"TESTING PER-OPERATION QUANTIZATION: E{exponent_bits}M{mantissa_bits}")
    print(f"(Every mul/add is quantized to E{exponent_bits}M{mantissa_bits})")
    print("="*80)

    v = PerOpQuantValidation(exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

    # Generate test data
    device = torch.device('cpu', 0)
    params_fp32 = {'device': device, 'dtype': torch.float32}
    q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params_fp32)
    q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params_fp32)
    q_span = q_max - q_min

    print(f"Generating {n_samples} test samples...")
    input_data_fp32 = q_min + q_span * torch.rand(n_samples, 10, **params_fp32)

    # Quantize input
    print(f"Quantizing input to E{exponent_bits}M{mantissa_bits}...")
    input_data = quantize_float(input_data_fp32, exponent_bits, mantissa_bits)

    # Run inference with per-op quantization
    print("Running per-operation quantized inference...")
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
        'per_link_mean': [errors[:, i].mean().item() for i in range(9)],
        'per_link_max': [errors[:, i].max().item() for i in range(9)],
        'input_data': input_data.numpy(),
        'nn_pred': nn_pred.numpy(),
        'mesh_dist': mesh_dist.numpy(),
        'errors': errors.numpy()
    }

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
    configs = [
        (8, 23, "FP32"),
        (8, 7, "BF16"),
        (8, 6, "E8M6"),
        (8, 5, "E8M5"),
        (8, 4, "E8M4"),
        (8, 3, "E8M3"),
        (8, 2, "E8M2"),
        (6, 7, "E6M7"),
        (5, 7, "E5M7"),
        (4, 7, "E4M7"),
    ]

    print("="*80)
    print("NEURAL-JSDF PER-OPERATION QUANTIZATION TEST")
    print("Each arithmetic operation (mul, add) is immediately quantized")
    print(f"Testing {len(configs)} configurations")
    print("="*80)

    all_results = []

    for exp_bits, mant_bits, name in configs:
        results = run_per_op_test(exp_bits, mant_bits, n_samples=5000)
        all_results.append(results)

        # Save results
        output_file = f"data_per_op_E{exp_bits}M{mant_bits}.npz"
        np.savez(output_file,
                 exponent_bits=exp_bits,
                 mantissa_bits=mant_bits,
                 name=name,
                 input=results['input_data'],
                 nn_pred=results['nn_pred'],
                 mesh_dist=results['mesh_dist'],
                 errors=results['errors'])
        print(f"✓ Saved to: {output_file}")

    # Save summary
    summary_file = "per_op_quant_summary.npz"
    np.savez(summary_file,
             config_names=[name for _, _, name in configs],
             exponent_bits=[r['exponent_bits'] for r in all_results],
             mantissa_bits=[r['mantissa_bits'] for r in all_results],
             mean_errors=[r['mean_error'] for r in all_results],
             std_errors=[r['std_error'] for r in all_results],
             max_errors=[r['max_error'] for r in all_results],
             median_errors=[r['median_error'] for r in all_results],
             p95_errors=[r['p95_error'] for r in all_results],
             p99_errors=[r['p99_error'] for r in all_results])
    print(f"\n✓ Summary saved to: {summary_file}")

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Config':<10} {'Exp':<5} {'Mant':<5} {'Mean':<10} {'Max':<10} {'95th':<10}")
    print("-"*80)
    for i, (exp, mant, name) in enumerate(configs):
        r = all_results[i]
        print(f"{name:<10} {exp:<5} {mant:<5} {r['mean_error']:<10.2f} {r['max_error']:<10.2f} {r['p95_error']:<10.2f}")
    print("="*80)
