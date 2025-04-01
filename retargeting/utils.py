import numpy as np
import torch
import os, sys, importlib
from copy import copy
from manopth import demo
import pickle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import dex_retargeting
from dex_retargeting.constants import RobotName, RetargetingType, HandType, get_default_config_path, ROBOT_NAME_MAP, RETARGETING_TYPE_MAP
from dex_retargeting.retargeting_config import RetargetingConfig
from pathlib import Path
import sapien
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

# recon 45-dim pose from 9-dim pca coordinates
def reconstruct_mano_pose45(eigengrasp_vectors, D_mean, D_std, coordinates):
    """
    根据9维坐标重构45维姿态
    :param eigengrasp_vectors: PCA计算得到的特征向量，形状为(9, 45)
    :param D_mean: 数据集的均值
    :param D_std: 数据集的标准差
    :param coordinates: 9维坐标，形状为(9,)
    :return: 重构的45维姿态，形状为(45,)
    """
    # 使用PCA对象进行逆变换得到标准化的45维姿态
    standardized_pose = np.dot(coordinates, eigengrasp_vectors)
    # 反标准化得到原始尺度的45维姿态
    original_pose = standardized_pose * D_std + D_mean

    return original_pose

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


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)

    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    # 隐藏坐标系和网格线
    ax.set_axis_off()  # 关闭坐标轴
    ax.grid(False)  # 关闭背景网格
    ax.set_xticks([])  # 关闭X轴刻度
    ax.set_yticks([])  # 关闭Y轴刻度
    ax.set_zticks([])  # 关闭Z轴刻度
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()

def cam_equal_aspect_3d(ax, verts):
    """
    设置3D绘图区域，使得各个坐标轴的比例相同
    """
    extents = np.stack([verts.min(axis=0), verts.max(axis=0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_' + dim + 'lim')((ctr - r, ctr + r))

# visualize 9-dim eigengrasps with interactive sliding bars
class SliderApp:
    def __init__(self, root, L_values, R_values, principal_vectors, D_mean, D_std, mano_layer, queue=None, position_baseline=False):
        self.root = root
        self.root.title("Slider Application")

        self.principal_vectors=principal_vectors
        self.D_std = D_std
        self.D_mean = D_mean
        self.mano_layer=mano_layer
        self.queue = queue # to put the mano 21*3 position data
        self.position_baseline = position_baseline
        
        # 创建变量来存储拖动条的值
        self.slider_values = [tk.DoubleVar() for _ in range(9 if not self.position_baseline else 15)]
        
        # 创建拖动条及其最小值和最大值标签
        for i in range(9 if not self.position_baseline else 15):
            frame = ttk.Frame(root)
            frame.pack(fill="x", padx=10, pady=5)
            
            min_label = ttk.Label(frame, text=f"{L_values[i]}")
            min_label.pack(side="left")
            
            slider = ttk.Scale(frame, from_=L_values[i], to=R_values[i], orient="horizontal", variable=self.slider_values[i])
            slider.pack(side="left", fill="x", expand=True, padx=10)
            slider.set(0)  # 初始值设置为0
            slider.bind("<Motion>", self.update_hand)
            slider.bind("<ButtonRelease-1>", self.update_hand)
            
            max_label = ttk.Label(frame, text=f"{R_values[i]}")
            max_label.pack(side="right")

        # 创建一个图像区域
        self.fig, self.ax = plt.subplots(figsize=(4, 3), subplot_kw={'projection': '3d'})
        self.ax.set_title("Hand Visualization")
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        # 初始化图像
        self.update_hand()

    def update_hand(self, *args):
        # 获取当前拖动条的值
        pose = torch.zeros([1,48])
        #pose[0,0] = -1.57
        #pose[0,0:3] = torch.tensor([-1.57,0,0])
        params = np.array([var.get() for var in self.slider_values[:9]])

        if not self.position_baseline:
            pose45 = reconstruct_mano_pose45(self.principal_vectors, self.D_mean, self.D_std, params)
            pose[0,3:] = torch.tensor(pose45)
            #beta = torch.zeros((1,10))
            #torch.tensor([[0.69939941,-0.16909726,-0.89550918,-0.0976461,0.07754239,0.33628672,-0.05547792,0.52487278, -0.38668063, -0.00133091]])
            hand_verts, hand_joints = self.mano_layer(pose) #, beta)
            hand_info = {'verts': hand_verts, 'joints': hand_joints}
        
            #print(hand_joints.shape)
            if self.queue is not None:
                self.queue.put(np.array(hand_joints[0]/1000.))

            # 清除当前图像
            self.ax.clear()
            # 显示新的手部图像
            display_hand(hand_info, mano_faces=self.mano_layer.th_faces, ax=self.ax, show=False)
            self.canvas.draw()
        else:
            fingertip_pos = params
            if self.queue is not None:
                self.queue.put(np.array(fingertip_pos))
            self.ax.clear()
            self.canvas.draw()

# dex retargeter
def load_retargeting(robot_name: RobotName=RobotName.leap, retargeting_type: RetargetingType=RetargetingType.dexpilot, 
    hand_type: HandType=HandType.right, add_dummy_free_joint=True):
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    print(config_path)
    robot_dir = Path(importlib.util.find_spec('dex_retargeting').origin).absolute().parent.parent / "assets" / "robots" / "hands"
    #print(robot_dir)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    override = dict(add_dummy_free_joint=add_dummy_free_joint)
    config = RetargetingConfig.load_from_file(config_path, override=override)
    retargeting = config.build()
    return retargeting, config

# setup sapien hand viewer
def create_sapien_hands_viewer(retargeting_joint_names, urdf_paths):
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])
    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2]))
    scene.add_area_light_for_ray_tracing(sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5)
    # Camera
    cam = scene.add_camera(name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10)
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())
    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True

    retargetings_to_sapien, robots = [], []

    num_robots = len(urdf_paths)
    for i, urdf_path in enumerate(urdf_paths):
        filepath = Path(urdf_path)
        robot_name = filepath.stem
        '''
        if "ability" in robot_name:
            loader.scale = 1.5
        elif "dclaw" in robot_name:
            loader.scale = 1.25
        elif "allegro" in robot_name:
            loader.scale = 1.4
        elif "shadow" in robot_name:
            loader.scale = 0.9
        elif "bhand" in robot_name:
            loader.scale = 1.5
        elif "leap" in robot_name:
            loader.scale = 1.4
        elif "svh" in robot_name:
            loader.scale = 1.5
        elif "inspire" in robot_name:
            loader.scale = 1.5
        '''
        if "glb" not in robot_name:
            filepath = str(filepath).replace(".urdf", "_glb.urdf")
        else:
            filepath = str(filepath)
        robot = loader.load(filepath)

        pose_offset = 0.25*(i+1-num_robots/2)

        if "ability" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.15]))
        elif "shadow" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.2]))
        elif "dclaw" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.15]))
        elif "allegro" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.05]))
        elif "bhand" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.2]))
        elif "leap" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.15]))
        elif "svh" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.13]))
        elif "inspire" in robot_name:
            robot.set_pose(sapien.Pose([0, pose_offset, -0.13]))

        # Different robot loader may have different orders for joints
        sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
        #print(sapien_joint_names, retargeting_joint_names)
        retargeting_to_sapien = np.array([retargeting_joint_names[i].index(name) for name in sapien_joint_names]).astype(int)
        retargetings_to_sapien.append(retargeting_to_sapien)
        robots.append(robot)

    return viewer, retargetings_to_sapien, robots


### Load hand pose datasets
import json
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

def load_handpose_dataset(name):
    if name=="grab":
        dataset = []
        pths = ['../GRAB/hand_dataset/s{}.pkl'.format(i) for i in range(1,11)]
        for p in pths:
            with open(p, 'rb') as f:
                data = pickle.load(f)   
                data = np.concatenate(data, axis=0)
                dataset.append(data)
        dataset = np.concatenate(dataset, axis=0)
        return dataset
    elif name=="freihand":
        mano_path = "../FreiHAND/FreiHAND_pub_v2/training_mano.json"
        mano_list = json_load(mano_path)
        dataset = np.array(mano_list).reshape(-1,61)
        poses = dataset[:, 3:48]
        return poses
    else:
        raise NotImplementedError


if __name__=="__main__":
    retargetings, configs = [], []
    for robot_name in [RobotName.inspire, RobotName.leap, RobotName.shadow, RobotName.allegro, RobotName.svh]:
        retargeting, config = load_retargeting(robot_name)
        retargetings.append(retargeting)
        configs.append(config)
    viewer, retargetings_to_sapien, robots = create_sapien_hands_viewer([r.joint_names for r in retargetings], 
        [c.urdf_path for c in configs])
    for _ in range(1000):
        viewer.render()
