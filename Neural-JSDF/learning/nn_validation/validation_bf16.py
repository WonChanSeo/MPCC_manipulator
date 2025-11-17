import sys
sys.path.append('../nn-learning/')
from sdf.robot_sdf import RobotSdfCollisionNet
from scipy.io import loadmat
import torch
import numpy as np
from fk_num import *
import time


# torch parameters - USE BF16
device = torch.device('cpu', 0)
params = {'device': device, 'dtype': torch.bfloat16}  # Changed to BF16

class Validation:
    def __init__(self):

        # Load nn model
        fname = 'sdf_256x5_mesh_50000.pt'

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
        self.nn_model.m = torch.zeros((500, 1)).to(device)  # Use CPU instead of CUDA
        self.nn_model.m[:, 0] = 1
        self.nn_model.order = list(range(9))

        # Load weights manually
        checkpoint = torch.load('../../env/' + fname, map_location='cpu')
        # Extract model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Load weights into model
        self.nn_model.model.load_state_dict(state_dict)

        # Load normalization parameters (CRITICAL for correct outputs!)
        if 'norm' in checkpoint:
            self.nn_model.norm_dict = checkpoint['norm']
            for k in self.nn_model.norm_dict.keys():
                self.nn_model.norm_dict[k]['mean'] = self.nn_model.norm_dict[k]['mean'].to(device)
                self.nn_model.norm_dict[k]['std'] = self.nn_model.norm_dict[k]['std'].to(device)
            print("Normalization parameters loaded")

        # Set tensor_args for model
        self.nn_model.tensor_args = params

        # Convert model to BF16
        self.nn_model.model.to(**params)
        self.nn_model.model_jit = self.nn_model.model
        self.nn_model.model_jit = torch.jit.script(self.nn_model.model_jit)
        self.nn_model.model_jit = torch.jit.optimize_for_inference(self.nn_model.model_jit)

        print(f"Model loaded in BF16 precision from: {fname}")
        print(f"Model dtype: {next(self.nn_model.model.parameters()).dtype}")

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
        # Convert input to BF16
        input_bf16 = input.to(torch.bfloat16)
        y_pred = self.nn_model.model_jit(input_bf16)
        # Convert back to FP32 for compatibility
        return y_pred.float()

    def get_mesh_fk(self, q):
        fk = dh_fk(torch.cat((q, torch.tensor([0]).to(q.device)), 0), self.dh)
        v_vec = []
        for i, P in enumerate(fk):
            # Convert FK transform to BF16 to match mesh data
            P_bf16 = P.to(torch.bfloat16)
            v_vec.append(self.v_int_pts[i] @ P_bf16[:3,:3].T + P_bf16[:3,3:4].T)
        return v_vec

    def get_mindists(self, v_vec, y):
        mindists = []
        for v_link in v_vec:
            dist = torch.cdist(v_link.float(), y.unsqueeze(0).float(), p=2)
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
            all_dists[i, :] = mindists.to(torch.bfloat16)
        #res, _ = all_dists.min(dim=1, keepdim=True)
        res = all_dists
        return(100*res.float())

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
