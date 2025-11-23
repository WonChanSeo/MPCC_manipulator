import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time
from validation_fp32 import Validation as ValidationFP32
from validation_bf16 import Validation as ValidationBF16

# torch parameters
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

print("=" * 80)
print("SAME INPUT COMPARISON - FP32 vs BF16")
print("=" * 80)
print()

# Load both models
print("Loading FP32 model...")
v_fp32 = ValidationFP32()

print("\nLoading BF16 model...")
v_bf16 = ValidationBF16()

# Generate test data with FIXED SEED for reproducibility
print("\n" + "=" * 80)
print("Generating test data with fixed random seed...")
print("=" * 80)

torch.manual_seed(42)
np.random.seed(42)

q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1, -1, -0.2]).to(**params)
q_max = torch.tensor([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1, 1, 1.3]).to(**params)
q_span = q_max - q_min

# Generate 10 batches of 1000 samples with same input
for i in range(10):
    print(f"\nBatch {i+1}/10...")

    # Generate random input (same for both models)
    input = q_min + q_span * torch.rand(1000, 10, **params)

    # FP32 predictions
    t0 = time.time()
    fp32_pred = v_fp32.calc_nn_pred(input)
    t_fp32 = time.time()

    # BF16 predictions
    bf16_pred = v_bf16.calc_nn_pred(input)
    t_bf16 = time.time()

    # Mesh ground truth (calculated once, same for both)
    mesh_dist = v_fp32.calc_mesh_mindist(input)
    t_mesh = time.time()

    print(f'  FP32 NN time: {t_fp32-t0:.3f} s')
    print(f'  BF16 NN time: {t_bf16-t_fp32:.3f} s')
    print(f'  Mesh time: {t_mesh-t_bf16:.3f} s')

    # Check prediction differences
    pred_diff = torch.abs(fp32_pred - bf16_pred)
    print(f'  Mean prediction difference: {pred_diff.mean():.4f} cm')
    print(f'  Max prediction difference: {pred_diff.max():.4f} cm')

    # Save data: input, FP32 pred, BF16 pred, mesh ground truth
    # Format: [q1-q7, x, y, z, fp32_link0-8, bf16_link0-8, mesh_link0-8]
    data = torch.cat((input, fp32_pred, bf16_pred, mesh_dist), 1)

    with open("data_same_input.csv", "ab") as f:
        np.savetxt(f, data.numpy(), delimiter=',', fmt='%4.3f')

print()
print("=" * 80)
print("Same input comparison data generated!")
print("Output saved to: data_same_input.csv")
print("Format: [input(10), fp32_pred(9), bf16_pred(9), mesh(9)] = 37 columns")
print("=" * 80)
