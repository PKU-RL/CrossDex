import numpy as np
import json
import os, glob
from copy import copy
from manopth.manolayer import ManoLayer
from manopth import demo
import torch
import pickle
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def reconstruct_pose(eigengrasp_vectors, D_mean, D_std, coordinates):
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


def load_data(fn):
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


class SliderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Slider Application")
        
        # 创建变量来存储拖动条的值
        self.slider_values = [tk.DoubleVar() for _ in range(9)]
        
        # 创建拖动条及其最小值和最大值标签
        for i in range(9):
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
        pose[0,0] = -1.57
        params = np.array([var.get() for var in self.slider_values[:9]])
        pose45 = reconstruct_pose(principal_vectors, D_mean, D_std, params)
        pose[0,3:] = torch.tensor(pose45)
        hand_verts, hand_joints = mano_layer(pose)
        hand_info = {'verts': hand_verts, 'joints': hand_joints}

        # 清除当前图像
        self.ax.clear()
        # 显示新的手部图像
        display_hand(hand_info, mano_faces=mano_layer.th_faces, ax=self.ax, show=False)
        self.canvas.draw()
    


if __name__=='__main__':
    fn = 'pca_9_grab.pkl'
    principal_vectors, min_values, max_values, D_mean, D_std = load_data(fn)

    # 定义拖动条的取值范围
    L_values = min_values
    R_values = max_values
    mano_layer = ManoLayer(mano_root='../mano-models', 
        use_pca=False, ncomps=45, flat_hand_mean='grab' in fn)

    # 创建主窗口
    root = tk.Tk()
    # 创建应用实例
    app = SliderApp(root)
    # 运行主循环
    root.mainloop()
