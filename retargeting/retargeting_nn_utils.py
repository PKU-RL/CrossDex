import numpy as np
import torch
import torch.nn as nn
import os, sys
import pickle
from dex_retargeting.constants import RobotName, ROBOT_NAME_MAP
from pathlib import Path
import yaml

# load eigengrasps
def load_pca_data(fn):
    with open(fn, 'rb') as f:
        d = pickle.load(f)
    principal_vectors = d['eigen_vectors']
    min_values = d['min_values']
    max_values = d['max_values']
    D_mean = d['D_mean']
    D_std = d['D_std']
    return principal_vectors, min_values, max_values, D_mean, D_std

def load_position_baseline_data(fn):
    with open(fn, 'rb') as f:
        d = pickle.load(f)
    min_values = d['min_values']
    max_values = d['max_values']
    return min_values, max_values

class EigenRetargetModel:
    def __init__(self, dataset="grab", robot_name=RobotName.leap, add_random_dataset=False, retargeting_type="dexpilot", 
                 position_baseline=False, device="cuda", n_eigengrasps=9):
        current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        if isinstance(robot_name, RobotName):
            robot_name = ROBOT_NAME_MAP[robot_name]
        assert isinstance(robot_name, str)
        if not position_baseline:
            pca_fn = current_dir.parent / 'results/pca_{}_{}.pkl'.format(n_eigengrasps, dataset)
        else:
            pca_fn = current_dir.parent / 'results/stat_position_baseline.pkl'
        if position_baseline:
            config_fn = current_dir / "models/position_baseline_retargeting_nn_{}_{}_{}.yaml".format(robot_name, dataset, retargeting_type)
            nn_fn = current_dir / "models/position_baseline_retargeting_nn_{}_{}_{}.pth".format(robot_name, dataset, retargeting_type)
        elif add_random_dataset:
            config_fn = current_dir / "models/retargeting_nn_{}_{}_{}_random.yaml".format(robot_name, dataset, retargeting_type)
            nn_fn = current_dir / "models/retargeting_nn_{}_{}_{}_random.pth".format(robot_name, dataset, retargeting_type)
        else:
            config_fn = current_dir / "models/retargeting_nn_{}_{}_{}.yaml".format(robot_name, dataset, retargeting_type)
            nn_fn = current_dir / "models/retargeting_nn_{}_{}_{}.pth".format(robot_name, dataset, retargeting_type)
        print("Load pre-trained retargeting model: PCA: {}; Config: {}; NN: {}".format(pca_fn, config_fn, nn_fn))
        with open(config_fn, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.position_baseline = position_baseline
        self.robot_dim = config['robot_dim']
        self.robot_joint_names = config['robot_joint_names']
        self.robot_name = config['robot_name']
        self.robot_urdf_path = config['robot_urdf_path']
        self.device = torch.device(device)
        self.retargeting_nn = RetargetingNN(robot_dim=self.robot_dim, mano_dim=45 if not position_baseline else 15).to(self.device)
        self.retargeting_nn.load_state_dict(torch.load(nn_fn))
        self.retargeting_nn.eval()
        if not position_baseline:
            self.principal_vectors, self.min_values, self.max_values, self.D_mean, self.D_std = load_pca_data(pca_fn)
            self.principal_vectors = torch.as_tensor(self.principal_vectors).to(self.device)
            self.min_values = torch.as_tensor(self.min_values).to(self.device)
            self.max_values = torch.as_tensor(self.max_values).to(self.device)
            self.D_mean = torch.as_tensor(self.D_mean).to(self.device)
            self.D_std = torch.as_tensor(self.D_std).to(self.device)
        else:
            self.min_values, self.max_values = load_position_baseline_data(pca_fn)
            self.min_values = torch.as_tensor(self.min_values).to(self.device)
            self.max_values = torch.as_tensor(self.max_values).to(self.device)
            #self.principal_vectors, self.D_mean, self.D_std = None, None, None
        #print(self.min_values, self.max_values, self.D_mean, self.D_std)
        #sys.exit(0)

    # recon 45-dim pose from 9-dim eigengrasps
    # input (N,9) output (N,45)
    def eigengrasp_to_mano_pose45(self, x):
        x_clipped = torch.max(torch.min(x, self.max_values), self.min_values)
        standardized_pose = torch.matmul(x_clipped, self.principal_vectors)
        original_pose = standardized_pose * self.D_std + self.D_mean
        return original_pose

    # map 9-dim eigengrasps to robot joint pos
    # input torch (N,9) output torch (N,robot_dim)
    @torch.no_grad
    def retarget(self, x):
        if not self.position_baseline:
            pose45 = self.eigengrasp_to_mano_pose45(x)
            robot_pos = self.retargeting_nn(pose45)
        else:
            robot_pos = self.retargeting_nn(x.float())
        return robot_pos

    def retarget_np(self, x):
        x_ = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        robot_pos = self.retarget(x_).cpu().numpy()
        return robot_pos

    # map 45-dim mano pose to robot joint pos
    # input torch (N,45) output torch (N,robot_dim)
    @torch.no_grad
    def retarget_from_pose45(self, x):
        robot_pos = self.retargeting_nn(x)
        return robot_pos

    def retarget_np_from_pose45(self, x):
        x_ = torch.as_tensor(x, dtype=torch.float32).to(self.device)
        robot_pos = self.retarget_from_pose45(x_).cpu().numpy()
        return robot_pos



# retargeting model: mapping 45-dim mano pose to x-dim robot joint pos
class RetargetingNN(nn.Module):
    def __init__(self, robot_dim, mano_dim=45, hidden_dim=512):
        super(RetargetingNN, self).__init__()
        activation_fn = nn.ReLU 
        self.model = nn.Sequential(
                nn.Linear(mano_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, robot_dim)
            )

    def forward(self, x):
        return self.model(x)


def load_dataset(dataset="grab", add_random_dataset=False, robot_name=RobotName.leap, retargeting_type="dexpilot", position_baseline=False):
    def _check_datasets(d1, d2):
        assert d1['robot_name']==d2['robot_name']
        assert d1['urdf_path']==d2['urdf_path']
        for n1, n2 in zip(d1['joint_names'], d2['joint_names']):
            assert n1==n2

    if not position_baseline:
        data_path = 'dataset/retargeting_{}_{}_{}.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
    else:
        data_path = 'dataset/position_baseline_retargeting_{}_{}_{}.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print("dataset loaded:", data_path)
    if add_random_dataset:
        data_path = 'dataset/retargeting_{}_{}_{}_random.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, retargeting_type)
        with open(data_path, 'rb') as f:
            data_random = pickle.load(f)
        #keys = list(data_random.keys())
        _check_datasets(data, data_random)
        data['mano_pose45'] = np.concatenate([data['mano_pose45'], data_random['mano_pose45']], axis=0)
        data['robot_joint_pos'] = np.concatenate([data['robot_joint_pos'], data_random['robot_joint_pos']], axis=0)
    return data





if __name__=="__main__":
    dataset = load_dataset(add_random_dataset=False)
    print(dataset['mano_pose45'].shape)
