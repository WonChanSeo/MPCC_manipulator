import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time


# torch parameters - USE FP32
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.float32}

class Validation:
    def __init__(self):

        # Load nn model from FP32 parameters in env/parameter
        print("=" * 60)
        print("Loading FP32 model from env/parameter")
        print("=" * 60)

        # Create model structure
        from sdf.network_macros_mod import MLPRegression
        import torch.nn as nn

        self.nn_model = RobotSdfCollisionNet.__new__(RobotSdfCollisionNet)
        self.nn_model.in_channels = 10
        self.nn_model.out_channels = 9

        act_fn = nn.ReLU
        mlp_layers = [256] * 4
        skips = []

        self.nn_model.model = MLPRegression(10, 9, mlp_layers, skips, act_fn=act_fn, nerf=True)
        self.nn_model.m = torch.zeros((500, 1)).to(device)
        self.nn_model.m[:, 0] = 1
        self.nn_model.order = list(range(9))

        # Load FP32 parameters from env/parameter/*.txt files
        param_dir = '../../env/parameter/'

        # Load weights and biases manually
        # Model structure: layers[0] contains 5 Sequential blocks
        # Each block has Linear layer at [i][0]
        with torch.no_grad():
            # Load weight_0 (input layer: 30 -> 256, positional encoding makes 10 -> 30)
            weight_0 = np.loadtxt(param_dir + 'weight_0.txt')
            weight_0 = torch.tensor(weight_0, dtype=torch.float32)  # Already (256, 30)
            self.nn_model.model.layers[0][0][0].weight.copy_(weight_0)

            bias_0 = np.loadtxt(param_dir + 'bias_0.txt')
            bias_0 = torch.tensor(bias_0, dtype=torch.float32)
            self.nn_model.model.layers[0][0][0].bias.copy_(bias_0)

            # Load weight_1 (256 -> 256)
            weight_1 = np.loadtxt(param_dir + 'weight_1.txt')
            weight_1 = torch.tensor(weight_1, dtype=torch.float32).reshape(256, 256)
            self.nn_model.model.layers[0][1][0].weight.copy_(weight_1)

            bias_1 = np.loadtxt(param_dir + 'bias_1.txt')
            bias_1 = torch.tensor(bias_1, dtype=torch.float32)
            self.nn_model.model.layers[0][1][0].bias.copy_(bias_1)

            # Load weight_2 (256 -> 256)
            weight_2 = np.loadtxt(param_dir + 'weight_2.txt')
            weight_2 = torch.tensor(weight_2, dtype=torch.float32).reshape(256, 256)
            self.nn_model.model.layers[0][2][0].weight.copy_(weight_2)

            bias_2 = np.loadtxt(param_dir + 'bias_2.txt')
            bias_2 = torch.tensor(bias_2, dtype=torch.float32)
            self.nn_model.model.layers[0][2][0].bias.copy_(bias_2)

            # Load weight_3 (256 -> 256)
            weight_3 = np.loadtxt(param_dir + 'weight_3.txt')
            weight_3 = torch.tensor(weight_3, dtype=torch.float32).reshape(256, 256)
            self.nn_model.model.layers[0][3][0].weight.copy_(weight_3)

            bias_3 = np.loadtxt(param_dir + 'bias_3.txt')
            bias_3 = torch.tensor(bias_3, dtype=torch.float32)
            self.nn_model.model.layers[0][3][0].bias.copy_(bias_3)

            # Load weight_4 (output layer: 256 -> 9)
            weight_4 = np.loadtxt(param_dir + 'weight_4.txt')
            weight_4 = torch.tensor(weight_4, dtype=torch.float32).reshape(9, 256)
            self.nn_model.model.layers[0][4][0].weight.copy_(weight_4)

            bias_4 = np.loadtxt(param_dir + 'bias_4.txt')
            bias_4 = torch.tensor(bias_4, dtype=torch.float32)
            self.nn_model.model.layers[0][4][0].bias.copy_(bias_4)

        print("✓ FP32 parameters loaded from env/parameter/")
        print(f"  - weight_0: {weight_0.shape}")
        print(f"  - weight_1: {weight_1.shape}")
        print(f"  - weight_2: {weight_2.shape}")
        print(f"  - weight_3: {weight_3.shape}")
        print(f"  - weight_4: {weight_4.shape}")

        # Set tensor_args for model
        self.nn_model.tensor_args = params

        # Convert model to FP32 and optimize
        self.nn_model.model.to(**params)
        self.nn_model.model_jit = self.nn_model.model
        self.nn_model.model_jit = torch.jit.script(self.nn_model.model_jit)
        self.nn_model.model_jit = torch.jit.optimize_for_inference(self.nn_model.model_jit)

        print(f"✓ Model compiled with FP32 precision")
        print(f"  Model dtype: {next(self.nn_model.model.parameters()).dtype}")
        print("=" * 60)
        print()

        # load robot matlab meshes
        data_mat = loadmat('../data-sampling/meshes/mesh_light_pts.mat')['mesh'][0]
        self.meshes = []
        self.faces = []
        self.vertices = []
        self.link_names = []
        self.v_int_pts = []
        for link in data_mat:
            faces = torch.tensor(link[0][0][0].astype(int)).to(**params)
            vertices = torch.tensor(link[0][0][1]).to(**params)
            link_name = link[0][0][2][0]
            int_pts = torch.tensor(link[0][0][3]).to(**params)
            v_int_pts = torch.cat((vertices, int_pts), 0)
            self.meshes.append({'vertices': vertices, 'faces': faces, 'link_name': link_name})
            self.faces.append(faces)
            self.vertices.append(vertices)
            self.link_names.append(link_name)
            self.v_int_pts.append(v_int_pts)

        # specify robot parameters
        dh_a = torch.tensor([0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0])        # "r" in matlab
        dh_d = torch.tensor([0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107])       # "d" in matlab
        dh_alpha = torch.tensor([0, -np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])  # "alpha" in matlab
        self.dh = torch.vstack((dh_d, dh_a*0, dh_a, dh_alpha)).T.to(**params)          # (d, theta, a (or r), alpha)

    def calc_nn_pred(self, input):
        # FP32 inference
        y_pred = self.nn_model.model_jit(input)
        return y_pred

    def get_mesh_fk(self, q):
        fk = dh_fk(torch.cat((q, torch.tensor([0]).to(q.device)), 0), self.dh)
        v_vec = []
        for i, P in enumerate(fk):
            v_vec.append(self.v_int_pts[i] @ P[:3,:3].T + P[:3,3:4].T)
        return v_vec

    def get_mindists(self, v_vec, y):
        mindists = []
        for v_link in v_vec:
            dist = torch.cdist(v_link, y.unsqueeze(0), p=2)
            mindist, _ = dist.min(dim=0, keepdim=False)
            mindists.append(mindist)
        return torch.cat(mindists, 0)

    def calc_mesh_mindist(self, input):
        q = input[:, :7]
        y = input[:, 7:]
        n_pts = y.shape[0]
        all_dists = torch.zeros(q.shape[0], 9, **params)
        for i in range(n_pts):
            v = self.get_mesh_fk(q[i])
            mindists = self.get_mindists(v, y[i])
            all_dists[i, :] = mindists
        res = all_dists
        return(100*res)

    def calc_err(self, input):
        q = input[:, :7]
        y = input[:, 7:].to(**params)
        res_meshes = self.calc_mesh_mindist(input)
        res_nn = self.calc_nn_pred(input)
        err = torch.abs(res_meshes - res_nn)
        max_err, _ = err.max(dim=-1)
        return max_err

    def calc_err_thr(self, input, link=None):
        q = input[:, :7]
        y = input[:, 7:].to(**params)
        res_meshes = self.calc_mesh_mindist(input)
        res_nn = self.calc_nn_pred(input)
        err = torch.abs(res_meshes - res_nn)
        THR = 2
        err[(res_meshes < THR) & (res_nn < THR)] = 0
        if link is not None:
            err = err[:, link].reshape(-1, 1)
        max_err, _ = err.max(dim=-1)
        return max_err

    def fitness(self, input, link=None):
        res = self.calc_err_thr(torch.tensor(input).to(**params), link).cpu().numpy()
        return -1*res
