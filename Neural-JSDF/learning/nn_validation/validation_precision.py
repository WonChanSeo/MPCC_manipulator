"""
Neural-JSDF Precision Comparison Script
Tests network inference with various mantissa bit configurations (lower than bf16)

BFloat16 reference: 8 exponent bits + 7 mantissa bits
This script tests: 7, 6, 5, 4, 3, 2 mantissa bits
"""

import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time


class PrecisionValidation:
    """Validation class that can use different mantissa bit configurations"""

    def __init__(self, mantissa_bits=7, use_custom_dtype=False):
        """
        Initialize validation with specified mantissa precision

        Args:
            mantissa_bits: Number of mantissa bits (7 for bf16, 6-2 for lower precision)
            use_custom_dtype: If True, emulate lower precision. If False, use standard torch dtype
        """
        self.mantissa_bits = mantissa_bits
        self.use_custom_dtype = use_custom_dtype

        # torch parameters
        device = torch.device('cpu', 0)

        # Select dtype based on mantissa bits
        if mantissa_bits == 23 and not use_custom_dtype:
            # FP32 full precision (23-bit mantissa)
            self.dtype = torch.float32
            self.dtype_name = f"float32 ({mantissa_bits}-bit mantissa)"
        elif mantissa_bits == 10 and not use_custom_dtype:
            # FP16 has 10-bit mantissa
            self.dtype = torch.float16
            self.dtype_name = f"float16 ({mantissa_bits}-bit mantissa)"
        elif mantissa_bits == 7 and not use_custom_dtype:
            # BF16 has 7-bit mantissa
            self.dtype = torch.bfloat16
            self.dtype_name = f"bfloat16 ({mantissa_bits}-bit mantissa)"
        else:
            # For other precision levels, use float32 and manually quantize
            self.dtype = torch.float32
            self.dtype_name = f"custom ({mantissa_bits}-bit mantissa)"
            self.use_custom_dtype = True

        self.params = {'device': device, 'dtype': self.dtype}

        print(f"Initializing with precision: {self.dtype_name}")

        # Load nn model
        # Using weight from env/ directory (same as parameter/*.txt files)
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

        # Convert model to specified dtype
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

    def quantize_mantissa(self, tensor, mantissa_bits):
        """
        Manually quantize tensor to specified mantissa bits
        This emulates lower-precision floating point arithmetic

        Args:
            tensor: Input tensor (float32)
            mantissa_bits: Number of mantissa bits to keep

        Returns:
            Quantized tensor
        """
        if not self.use_custom_dtype or mantissa_bits >= 23:
            return tensor

        # Get float32 bit representation
        # IEEE 754: sign(1) + exponent(8) + mantissa(23)
        tensor_np = tensor.cpu().numpy()

        # View as uint32
        bits = tensor_np.view(np.uint32)

        # Create mantissa mask (keep only mantissa_bits)
        # Float32 has 23 mantissa bits
        bits_to_zero = 23 - mantissa_bits
        mask = np.uint32(0xFFFFFFFF << bits_to_zero)

        # Apply mask
        quantized_bits = bits & mask

        # Convert back to float32
        quantized_np = quantized_bits.view(np.float32)

        return torch.from_numpy(quantized_np).to(tensor.device)

    def calc_nn_pred(self, input):
        """Calculate NN predictions with specified precision"""
        # Convert input to target dtype
        input_converted = input.to(self.dtype)

        # If using custom quantization, apply it
        if self.use_custom_dtype:
            input_converted = self.quantize_mantissa(input_converted, self.mantissa_bits)

        # Run inference
        y_pred = self.nn_model.model_jit(input_converted)

        # Quantize output if using custom dtype
        if self.use_custom_dtype:
            y_pred = self.quantize_mantissa(y_pred, self.mantissa_bits)

        # Convert back to FP32 for compatibility
        return y_pred.float()

    def get_mesh_fk(self, q):
        """Get forward kinematics for mesh vertices"""
        fk = dh_fk(torch.cat((q, torch.tensor([0]).to(q.device)), 0), self.dh)
        v_vec = []
        for i, P in enumerate(fk):
            P_converted = P.to(self.dtype)
            if self.use_custom_dtype:
                P_converted = self.quantize_mantissa(P_converted, self.mantissa_bits)
            v_vec.append(self.v_int_pts[i] @ P_converted[:3,:3].T + P_converted[:3,3:4].T)
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

    def calc_err(self, input):
        """Calculate prediction error"""
        q = input[:, :7]
        y = input[:, 7:].to(**self.params)
        res_meshes = self.calc_mesh_mindist(input)
        res_nn = self.calc_nn_pred(input)
        err = torch.abs(res_meshes - res_nn)
        max_err, _ = err.max(dim=-1)
        return max_err


def run_precision_test(mantissa_bits, n_samples=5000, use_custom=False):
    """
    Run validation test with specified mantissa precision

    Args:
        mantissa_bits: Number of mantissa bits (7 for bf16, lower for reduced precision)
        n_samples: Number of test samples
        use_custom: Whether to use custom quantization

    Returns:
        Dictionary with test results
    """
    print("\n" + "="*80)
    print(f"TESTING PRECISION: {mantissa_bits}-bit mantissa")
    print("="*80)

    # Create validator
    v = PrecisionValidation(mantissa_bits=mantissa_bits, use_custom_dtype=use_custom)

    # Setup test range
    device = torch.device('cpu', 0)
    params = {'device': device, 'dtype': torch.float32}
    q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params)
    q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params)
    q_span = q_max - q_min

    # Generate test data
    print(f"Generating {n_samples} test samples...")
    input_data = q_min + q_span * torch.rand(n_samples, 10, **params)

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
    # Test multiple precision levels
    # FP32 has 23 mantissa bits, BF16 has 7, test from 23 down to 2
    mantissa_configs = [23, 16, 12, 10, 8, 7, 6, 5, 4, 3, 2]

    print("="*80)
    print("NEURAL-JSDF MANTISSA PRECISION COMPARISON")
    print("Testing mantissa bit configurations: 23 (fp32), 16, 12, 10, 8, 7 (bf16), 6, 5, 4, 3, 2")
    print("="*80)

    all_results = []

    # Run test for each precision level
    for mantissa_bits in mantissa_configs:
        # Use native dtype for 23 (FP32), 10 (FP16), 7 (BF16)
        # Use custom quantization for others
        use_custom = mantissa_bits not in [23, 10, 7]
        results = run_precision_test(mantissa_bits, n_samples=5000, use_custom=use_custom)
        all_results.append(results)

        # Save individual results
        output_file = f"data_mantissa_{mantissa_bits}bit.npz"
        np.savez(output_file,
                 mantissa_bits=mantissa_bits,
                 input=results['input_data'],
                 nn_pred=results['nn_pred'],
                 mesh_dist=results['mesh_dist'],
                 errors=results['errors'])
        print(f"✓ Saved results to: {output_file}")

    # Save summary statistics
    summary_file = "precision_comparison_summary.npz"
    summary_data = {
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
    print(f"{'Mantissa':<10} {'Mean Err':<12} {'Std Err':<12} {'Max Err':<12} {'Median':<12} {'95th %ile':<12}")
    print(f"{'Bits':<10} {'(cm)':<12} {'(cm)':<12} {'(cm)':<12} {'(cm)':<12} {'(cm)':<12}")
    print("-"*80)

    for r in all_results:
        print(f"{r['mantissa_bits']:<10} {r['mean_error']:<12.2f} {r['std_error']:<12.2f} "
              f"{r['max_error']:<12.2f} {r['median_error']:<12.2f} {r['p95_error']:<12.2f}")

    print("="*80)
    print("Test completed! Run visualize_precision_comparison.py to see plots.")
    print("="*80)
