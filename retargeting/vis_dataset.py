'''
dataset format: 
 - 'robot_name': str
 - 'joint_names': list[str]
 - 'mano_pose45': np array (N, 45)
 - 'robot_joint_pos': np array (N, n_dof)
'''
import numpy as np
import os
from copy import copy
from manopth.manolayer import ManoLayer
from manopth import demo
import torch
import pickle
import tyro
import multiprocessing
from queue import Empty
from typing import Optional, Tuple, List
from tqdm import trange
from utils import *
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.clear()
    
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



def main(dataset:str="grab", random_dataset:bool=False, robot_name:RobotName=RobotName.leap, 
    retargeting_type: RetargetingType=RetargetingType.dexpilot, hand_type: HandType=HandType.right):

    device = torch.device('cuda')
    mano_layer = ManoLayer(mano_root='../mano-models', 
        use_pca=False, ncomps=45, flat_hand_mean='grab' in dataset).to(device)
    
    if random_dataset:
        data_path = 'dataset/retargeting_{}_{}_{}_random.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, RETARGETING_TYPE_MAP[retargeting_type])
    else:
        data_path = 'dataset/retargeting_{}_{}_{}.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, RETARGETING_TYPE_MAP[retargeting_type])
    print(data_path)
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    joint_names, mano_pose45, robot_joint_pos, urdf_path = \
        data['joint_names'], data['mano_pose45'], data['robot_joint_pos'], data['urdf_path']
    viewer, retargetings_to_sapien, robots = create_sapien_hands_viewer([joint_names], [urdf_path])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #print(robot_joint_pos.shape, retargetings_to_sapien)
    for i in trange(0,len(robot_joint_pos),10):
        robots[0].set_qpos(robot_joint_pos[i][retargetings_to_sapien[0]])
        viewer.render()
        pose = torch.zeros([1, 48])
        pose[0, 1] = 1.57
        pose[0, 3:] = torch.tensor(mano_pose45[i])
        hand_verts, hand_joints = mano_layer(pose.to(device))
        hand_info = {'verts': hand_verts.cpu(), 'joints': hand_joints.cpu()}
        display_hand(hand_info, ax=ax, show=False)
        #if random_dataset:
        #    plt.pause(1)
        plt.pause(0.02)

if __name__=='__main__':
    tyro.cli(main)
