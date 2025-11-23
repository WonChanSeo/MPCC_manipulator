"""
Neural-JSDF Exponent + Mantissa Precision Comparison Script
Tests network inference with configurable exponent AND mantissa bits

IEEE 754 Float32: sign(1) + exponent(8) + mantissa(23) = 32 bits
This script tests various combinations of exponent and mantissa bits
"""

import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time


class FlexiblePrecisionValidation:
    """Validation class with configurable exponent AND mantissa bits"""

    def __init__(self, exponent_bits=8, mantissa_bits=7):
        """
        Initialize validation with specified exponent and mantissa precision

        Args:
            exponent_bits: Number of exponent bits (3-8)
            mantissa_bits: Number of mantissa bits (2-23)
        """
        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits

        # torch parameters - always use float32 as base
        device = torch.device('cpu', 0)
        self.dtype = torch.float32
        self.dtype_name = f"custom (E{exponent_bits}M{mantissa_bits})"
        self.params = {'device': device, 'dtype': self.dtype}

        print(f"Initializing with precision: {self.dtype_name}")

        # Load nn model
        fname = 'sdf_256x5_mesh_50000.pt'
        print(f"Weight file: {fname} (identical to parameter/*.txt files)")

        # Create model and manually fix CUDA requirement
        self.nn_model = RobotSdfCollisionNet.__new__(RobotSdfCollisionNet)

        # Manually initialize without calling __init__ to avoid CUDA
        self.nn_model.in_channels = 10
        self.nn_model.out_channels = 9
        dropout_ratio = 0
        mlp_layers = [256] * 4
        skips = []

        # Import MLPRegression and ReLU activation
        from sdf.network_macros_mod import MLPRegression
        import torch.nn as nn
        act_fn = nn.ReLU
        self.nn_model.model = MLPRegression(10, 9, mlp_layers, skips, act_fn=act_fn, nerf=True)
        self.nn_model.m = torch.zeros((500, 1)).to(device)
        self.nn_model.m[:, 0] = 1
        self.nn_model.order = list(range(9))

        # Load weights manually
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

        # Set tensor_args for model
        self.nn_model.tensor_args = self.params

        # Convert model to float32
        self.nn_model.model.to(**self.params)
        self.nn_model.model_jit = self.nn_model.model
        self.nn_model.model_jit = torch.jit.script(self.nn_model.model_jit)
        self.nn_model.model_jit = torch.jit.optimize_for_inference(self.nn_model.model_jit)

        print(f"Model loaded from: {fname}")
        print(f"Model dtype: {next(self.nn_model.model.parameters()).dtype}")

        # load robot matlab meshes
        data_mat = loadmat('../data-sampling/meshes/mesh_light_pts.mat')['mesh'][0]
        self.meshes = []
        self.faces = []
        self.vertices = []
        self.link_names = []
        self.v_int_pts = []
        for link in data_mat:
            faces = torch.tensor(link[0][0][0].astype(int)).to(**self.params)
            vertices = torch.tensor(link[0][0][1]).to(**self.params)
            link_name = link[0][0][2][0]
            int_pts = torch.tensor(link[0][0][3]).to(**self.params)
            v_int_pts = torch.cat((vertices, int_pts), 0)
            self.meshes.append({'vertices': vertices, 'faces': faces, 'link_name': link_name})
            self.faces.append(faces)
            self.vertices.append(vertices)
            self.link_names.append(link_name)
            self.v_int_pts.append(v_int_pts)

        # specify robot parameters
        dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])
        dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])
        dh_alpha = torch.tensor([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
        self.dh = torch.vstack((dh_d, dh_a*0, dh_a, dh_alpha)).T.to(**self.params)

    def quantize_float(self, tensor, exponent_bits, mantissa_bits):
        """
        Quantize tensor to specified exponent and mantissa bits

        IEEE 754 Float32: sign(1) + exponent(8) + mantissa(23)

        Args:
            tensor: Input tensor (float32)
            exponent_bits: Number of exponent bits (3-8)
            mantissa_bits: Number of mantissa bits (2-23)

        Returns:
            Quantized tensor
        """
        if exponent_bits >= 8 and mantissa_bits >= 23:
            # Full FP32 precision
            return tensor

        tensor_np = tensor.cpu().numpy()
        bits = tensor_np.view(np.uint32)

        # IEEE 754 Float32 layout: [sign: 1 bit][exponent: 8 bits][mantissa: 23 bits]
        # Bit positions: 31 (sign), 30-23 (exponent), 22-0 (mantissa)

        # 1. Quantize mantissa
        mantissa_bits_to_zero = 23 - mantissa_bits
        mantissa_mask = np.uint32(0x007FFFFF >> mantissa_bits_to_zero) << mantissa_bits_to_zero

        # 2. Quantize exponent
        exponent_bits_to_zero = 8 - exponent_bits
        # Extract exponent (bits 30-23)
        exponent = (bits >> 23) & 0xFF
        # Quantize exponent by zeroing lower bits
        exponent_quantized = (exponent >> exponent_bits_to_zero) << exponent_bits_to_zero

        # Check for overflow/underflow
        # FP32 exponent bias is 127, valid range is 0-255
        # After quantization, we need to ensure it stays in valid range
        exponent_quantized = np.clip(exponent_quantized, 0, 255)

        # Reconstruct bits
        sign_bit = bits & 0x80000000  # Bit 31
        exponent_field = (exponent_quantized & 0xFF) << 23  # Bits 30-23
        mantissa_field = bits & mantissa_mask  # Bits 22-0

        quantized_bits = sign_bit | exponent_field | mantissa_field

        # Convert back to float32
        quantized_np = quantized_bits.view(np.float32)

        return torch.from_numpy(quantized_np).to(tensor.device)

    def calc_nn_pred(self, input):
        """Calculate NN predictions with specified precision"""
        # Quantize input to target precision
        input_quantized = self.quantize_float(input, self.exponent_bits, self.mantissa_bits)

        # Run inference
        y_pred = self.nn_model.model_jit(input_quantized)

        # Quantize output
        y_pred_quantized = self.quantize_float(y_pred, self.exponent_bits, self.mantissa_bits)

        # Convert back to FP32 for compatibility
        return y_pred_quantized.float()

    def get_mesh_fk(self, q):
        """Get forward kinematics for mesh vertices"""
        fk = dh_fk(torch.cat((q, torch.tensor([0]).to(q.device)), 0), self.dh)
        v_vec = []
        for i, P in enumerate(fk):
            P_quantized = self.quantize_float(P, self.exponent_bits, self.mantissa_bits)
            v_vec.append(self.v_int_pts[i] @ P_quantized[:3,:3].T + P_quantized[:3,3:4].T)
        return v_vec

    def get_mindists(self, v_vec, y):
        """Get minimum distances between mesh vertices and query point"""
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
        return(100*res.float())


def run_precision_test(exponent_bits, mantissa_bits, n_samples=5000):
    """
    Run validation test with specified exponent and mantissa precision

    Args:
        exponent_bits: Number of exponent bits (3-8)
        mantissa_bits: Number of mantissa bits (2-23)
        n_samples: Number of test samples

    Returns:
        Dictionary with test results
    """
    print("\n" + "="*80)
    print(f"TESTING PRECISION: E{exponent_bits}M{mantissa_bits}")
    print(f"(Exponent: {exponent_bits} bits, Mantissa: {mantissa_bits} bits)")
    print("="*80)

    # Create validator
    v = FlexiblePrecisionValidation(exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

    # Setup test range - use FULL FP32 for input generation
    device = torch.device('cpu', 0)
    params_fp32 = {'device': device, 'dtype': torch.float32}
    q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params_fp32)
    q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params_fp32)
    q_span = q_max - q_min

    # Generate test data in FP32
    print(f"Generating {n_samples} test samples...")
    input_data_fp32 = q_min + q_span * torch.rand(n_samples, 10, **params_fp32)

    # Quantize input data to target precision
    print(f"Quantizing input to E{exponent_bits}M{mantissa_bits} precision...")
    input_data = v.quantize_float(input_data_fp32, exponent_bits, mantissa_bits)

    # Run inference
    print("Running NN inference...")
    t0 = time.time()
    nn_pred = v.calc_nn_pred(input_data)
    t_nn = time.time() - t0

    # Calculate ground truth
    print("Calculating ground truth mesh distances...")
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
        'per_link_mean': [errors[:, i].mean().item() for i in range(9)],
        'per_link_max': [errors[:, i].max().item() for i in range(9)],
        'input_data': input_data.numpy(),
        'nn_pred': nn_pred.numpy(),
        'mesh_dist': mesh_dist.numpy(),
        'errors': errors.numpy()
    }

    # Print summary
    print(f"\n{'-'*80}")
    print(f"NN inference time: {t_nn:.2f} s ({n_samples/t_nn:.1f} samples/s)")
    print(f"Mesh calculation time: {t_mesh:.2f} s")
    print(f"\nError Statistics (cm):")
    print(f"  Mean:   {results['mean_error']:>8.2f}")
    print(f"  Std:    {results['std_error']:>8.2f}")
    print(f"  Median: {results['median_error']:>8.2f}")
    print(f"  Max:    {results['max_error']:>8.2f}")
    print(f"  95th percentile: {results['p95_error']:>8.2f}")
    print(f"  99th percentile: {results['p99_error']:>8.2f}")
    print(f"{'-'*80}")

    return results


if __name__ == "__main__":
    # Test multiple exponent and mantissa combinations
    # Format: (exponent_bits, mantissa_bits, name)
    configs = [
        (8, 23, "FP32"),       # Full precision
        (5, 10, "FP16"),       # Standard FP16
        (8, 7, "BF16"),        # Standard BF16
        (8, 16, "E8M16"),      # High mantissa
        (8, 12, "E8M12"),
        (8, 8, "E8M8"),
        (8, 6, "E8M6"),
        (8, 4, "E8M4"),
        (8, 2, "E8M2"),
        (6, 7, "E6M7"),        # Reduced exponent
        (5, 7, "E5M7"),
        (4, 7, "E4M7"),
        (6, 6, "E6M6"),        # Balanced reduction
        (5, 5, "E5M5"),
        (4, 4, "E4M4"),
    ]

    print("="*80)
    print("NEURAL-JSDF EXPONENT + MANTISSA PRECISION COMPARISON")
    print(f"Testing {len(configs)} precision configurations")
    print("="*80)

    all_results = []

    # Run test for each configuration
    for exp_bits, mant_bits, name in configs:
        results = run_precision_test(exp_bits, mant_bits, n_samples=5000)
        all_results.append(results)

        # Save individual results
        output_file = f"data_E{exp_bits}M{mant_bits}.npz"
        np.savez(output_file,
                 exponent_bits=exp_bits,
                 mantissa_bits=mant_bits,
                 name=name,
                 input=results['input_data'],
                 nn_pred=results['nn_pred'],
                 mesh_dist=results['mesh_dist'],
                 errors=results['errors'])
        print(f"✓ Saved results to: {output_file}")

    # Save summary statistics
    summary_file = "exp_mantissa_comparison_summary.npz"
    summary_data = {
        'exponent_bits': [r['exponent_bits'] for r in all_results],
        'mantissa_bits': [r['mantissa_bits'] for r in all_results],
        'mean_errors': [r['mean_error'] for r in all_results],
        'std_errors': [r['std_error'] for r in all_results],
        'max_errors': [r['max_error'] for r in all_results],
        'median_errors': [r['median_error'] for r in all_results],
        'p95_errors': [r['p95_error'] for r in all_results],
        'p99_errors': [r['p99_error'] for r in all_results],
        'nn_times': [r['nn_time'] for r in all_results],
    }
    np.savez(summary_file, **summary_data)
    print(f"\n✓ Saved summary to: {summary_file}")

    # Print final comparison table
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE")
    print("="*80)
    print(f"{'Config':<12} {'Exp':<5} {'Mant':<5} {'Mean':<10} {'Std':<10} {'Max':<10} {'Median':<10}")
    print(f"{'':12} {'Bits':<5} {'Bits':<5} {'(cm)':<10} {'(cm)':<10} {'(cm)':<10} {'(cm)':<10}")
    print("-"*80)

    for i, (exp, mant, name) in enumerate(configs):
        r = all_results[i]
        print(f"{name:<12} {exp:<5} {mant:<5} {r['mean_error']:<10.2f} {r['std_error']:<10.2f} "
              f"{r['max_error']:<10.2f} {r['median_error']:<10.2f}")

    print("="*80)
    print("Test completed!")
    print("="*80)
