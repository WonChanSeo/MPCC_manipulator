import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time
from validation_fp32 import Validation

# torch parameters
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

print("=" * 60)
print("FP32 INFERENCE TEST")
print("Model: Loading from env/parameter/*.txt")
print("Precision: Float32")
print("=" * 60)
print()

v = Validation()
q_min = torch.tensor([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,	-2.8973, -1, -1, -0.2]).to(**params)
q_max = torch.tensor([2.8973,	1.7628,	2.8973,	-0.0698,  2.8973,	3.7525,	2.8973, 1, 1, 1.3]).to(**params)
q_span = q_max - q_min

# generate data
for i in range(10):
    print(f"Batch {i+1}/10...")
    input = q_min + q_span * torch.rand(1000, 10, **params)
    t0 = time.time()
    a = v.calc_nn_pred(input)
    t_nn = time.time()
    b = v.calc_mesh_mindist(input)
    t_mesh = time.time()
    print(f'  NN time (FP32): {t_nn-t0:.2f} s')
    print(f'  Mesh time: {t_mesh-t_nn:.2f} s')
    #print('Mean L1 error: %4.2f' % ((a-b).abs().mean()))
    #print('Max L1 error: %4.2f' % ((a-b).abs().max()))
    data = torch.cat((input, a, b), 1)
    with open("data_fp32.csv", "ab") as f:
        np.savetxt(f, data.numpy(), delimiter=',', fmt='%4.3f')

print()
print("=" * 60)
print("FP32 inference test completed!")
print("Output saved to: data_fp32.csv")
print("=" * 60)
