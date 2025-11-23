import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time
from validation import Validation

# torch parameters
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

print("=" * 60)
print("FP32 INFERENCE TEST (Early Checkpoint)")
print("Model: sdf_256x5_mesh_50000.pt")
print("Precision: Float32")
print("=" * 60)
print()

v = Validation()

# Sample 10000 queries
q_min = torch.tensor([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, -0.0698, -2.8973])
q_max = torch.tensor([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 3.6652, 2.8973])
q_span = q_max - q_min

data_all = []
for i in range(10):
    print(f"Batch {i+1}/10...")

    # Generate random samples
    input = q_min + q_span * torch.rand(1000, 7, **params)
    y = (torch.rand(1000, 3, **params) - 0.5)  # Sample position between -0.5 and 0.5
    input = torch.cat((input, y), 1)

    # NN inference
    start = time.time()
    a = v.calc_nn_pred(input)
    nn_time = time.time() - start
    print(f"  NN time (FP32): {nn_time:.2f} s")

    # Mesh ground truth
    start = time.time()
    b = v.calc_mesh_mindist(input)
    mesh_time = time.time() - start
    print(f"  Mesh time: {mesh_time:.2f} s")

    # Combine data
    data = torch.cat((input, a, b), 1)
    data_all.append(data)

# Save combined data
data_combined = torch.cat(data_all, 0)
np.savetxt("data_fp32_early.csv", data_combined.numpy(), delimiter=',',
           fmt='%.6f', header='q1,q2,q3,q4,q5,q6,q7,x,y,z,nn_d0,nn_d1,nn_d2,nn_d3,nn_d4,nn_d5,nn_d6,nn_d7,nn_d8,mesh_d0,mesh_d1,mesh_d2,mesh_d3,mesh_d4,mesh_d5,mesh_d6,mesh_d7,mesh_d8',
           comments='')

print()
print("=" * 60)
print("FP32 inference test completed!")
print("Output saved to: data_fp32_early.csv")
print("=" * 60)
