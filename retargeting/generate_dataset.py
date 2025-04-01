'''
generate retargeting datasets, save pickle
dataset format: 
 - 'robot_name': str
 - 'joint_names': list[str]
 - 'mano_pose45': np array (N, 45)
 - 'robot_joint_pos': np array (N, n_dof)
 - 'urdf_path': str
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
from tqdm import tqdm, trange
from utils import *


def generate_from_dataset(dataset, mano_layer, robot_name, retargeting_type, hand_type, save_path):
    retargeting, config = load_retargeting(robot_name, retargeting_type, hand_type, add_dummy_free_joint=True)
    print(retargeting.joint_names[6:], config.urdf_path)
    
    mano_pose45, robot_joint_pos = [], []
    for data in tqdm(dataset):
        pose = torch.zeros([1,48])
        pose[0,3:] = torch.tensor(data)
        hand_verts, hand_joints = mano_layer(pose)
        joint_pos = np.array(hand_joints[0]/1000.)
        indices = retargeting.optimizer.target_link_human_indices
        if retargeting_type == RetargetingType.position: #"POSITION":
            indices = indices
            ref_value = joint_pos[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
        qpos = retargeting.retarget(ref_value)[6:]
        mano_pose45.append(np.asarray(data,dtype=np.float32))
        robot_joint_pos.append(np.asarray(qpos,dtype=np.float32))

    ret = {
        'robot_name': ROBOT_NAME_MAP[robot_name],
        'joint_names': retargeting.joint_names[6:],
        'mano_pose45': np.asarray(mano_pose45),
        'robot_joint_pos': np.asarray(robot_joint_pos),
        'urdf_path': config.urdf_path,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(ret, f)


def generate_with_random_eigengrasp(n_data, principal_vectors, min_values, max_values, D_mean, D_std,
    mano_layer, robot_name, retargeting_type, hand_type, save_path):
    retargeting, config = load_retargeting(robot_name, retargeting_type, hand_type, add_dummy_free_joint=True)
    mano_pose45, robot_joint_pos = [], []
    last_act = np.zeros(9)
    n_generated = 0
    max_step_size = 0.05
    while n_generated < n_data:
        ## because we use sequential retargeting, action sequence should be smooth for better results.
        next_act = np.random.uniform(min_values, max_values)
        distance = np.linalg.norm(next_act-last_act)
        num_steps = int(np.ceil(distance/max_step_size))
        interpolation_acts = [last_act + (next_act-last_act)*t/num_steps for t in range(1,num_steps+1)]
        #print(rand_act, '\n', interpolation_acts, len(interpolation_acts))
        
        for act in interpolation_acts:
            data = reconstruct_mano_pose45(principal_vectors, D_mean, D_std, act)
            pose = torch.zeros([1,48])
            pose[0,3:] = torch.tensor(data)
            hand_verts, hand_joints = mano_layer(pose)
            joint_pos = np.array(hand_joints[0]/1000.)
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == RetargetingType.position: #"POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)[6:]
            mano_pose45.append(np.asarray(data,dtype=np.float32))
            robot_joint_pos.append(np.asarray(qpos,dtype=np.float32))

            n_generated+=1
            if n_generated>=n_data:
                break
        last_act = next_act
        print(n_generated)

    ret = {
        'robot_name': ROBOT_NAME_MAP[robot_name],
        'joint_names': retargeting.joint_names[6:],
        'mano_pose45': np.asarray(mano_pose45),
        'robot_joint_pos': np.asarray(robot_joint_pos),
        'urdf_path': config.urdf_path,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(ret, f)

def main(dataset:str="grab", generate_random_dataset:bool=False, robot_name:RobotName=RobotName.leap, 
    retargeting_type: RetargetingType=RetargetingType.dexpilot, hand_type: HandType=HandType.right):

    mano_layer = ManoLayer(mano_root='../mano-models', 
        use_pca=False, ncomps=45, flat_hand_mean='grab' in dataset)
    
    # randomly generate data in the eigengrasp space
    if generate_random_dataset:
        save_path = 'dataset/retargeting_{}_{}_{}_random.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, RETARGETING_TYPE_MAP[retargeting_type])
        print(save_path)
        n_data = 1000000
        pca_fn = '../results/pca_9_{}.pkl'.format(dataset)
        principal_vectors, min_values, max_values, D_mean, D_std = load_pca_data(pca_fn)
        generate_with_random_eigengrasp(n_data, principal_vectors, min_values, max_values, D_mean, D_std,
            mano_layer, robot_name, retargeting_type, hand_type, save_path)

    # generate data from hand pose dataset
    else:
        save_path = 'dataset/retargeting_{}_{}_{}.pkl'.format(ROBOT_NAME_MAP[robot_name], dataset, RETARGETING_TYPE_MAP[retargeting_type])
        print(save_path)
        data = load_handpose_dataset(dataset)
        print(data.shape)
        generate_from_dataset(data, mano_layer, robot_name, retargeting_type, hand_type, save_path)

if __name__=='__main__':
    tyro.cli(main)
