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
from utils import *


def retargeting(robots, retargeting_type, hand_type, queue: multiprocessing.Queue):
    # create retargeter
    retargetings, configs = [], []
    for robot_name in robots:
        retargeting, config = load_retargeting(robot_name, retargeting_type=retargeting_type)
        retargetings.append(retargeting)
        configs.append(config)
        print("robot:{}, joint names:{}, limits:{}".format(
            robot_name, retargeting.optimizer.target_joint_names, 
            retargeting.joint_limits))
    viewer, retargetings_to_sapien, robots = create_sapien_hands_viewer([r.joint_names for r in retargetings], 
        [c.urdf_path for c in configs])

    while True:
        viewer.render()
        try:
            joint_pos = queue.get(timeout=0)
        except Empty:
            continue
        #print(joint_pos)
        for retargeting, robot, retargeting_to_sapien in zip(retargetings, robots, retargetings_to_sapien):
            indices = retargeting.optimizer.target_link_human_indices
            print(indices, joint_pos)
            if retargeting_type == RetargetingType.position: #"POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
                print(ref_value)
                print(retargeting_type)
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            robot.set_qpos(qpos[retargeting_to_sapien])

'''
def slider_app(L_values, R_values, principal_vectors, D_mean, D_std, mano_layer, queue: multiprocessing.Queue):
    # create interactive eigengrasp viewer
    root = tk.Tk()
    app = SliderApp(root, L_values, R_values, principal_vectors, D_mean, D_std, mano_layer)
    root.mainloop()
'''

def main(dataset:str="grab", robots:List[RobotName]=[RobotName.inspire,RobotName.leap,RobotName.shadow,
    RobotName.allegro,RobotName.ability,RobotName.svh], 
    retargeting_type: RetargetingType=RetargetingType.dexpilot, hand_type: HandType=HandType.right):
    pca_fn = '../results/pca_9_{}.pkl'.format(dataset)
    principal_vectors, min_values, max_values, D_mean, D_std = load_pca_data(pca_fn)
    #print(np.linalg.norm(principal_vectors, axis=1))

    L_values = min_values
    R_values = max_values
    mano_layer = ManoLayer(mano_root='../mano-models', 
        use_pca=False, ncomps=45, flat_hand_mean='grab' in pca_fn)


    queue = multiprocessing.Queue(maxsize=1000)
    consumer_process = multiprocessing.Process(target=retargeting, args=(robots, retargeting_type, hand_type, queue))
    consumer_process.start()
    
    # create interactive eigengrasp viewer
    root = tk.Tk()
    app = SliderApp(root, L_values, R_values, principal_vectors, D_mean, D_std, mano_layer, queue)
    root.mainloop()

    #producer_process = multiprocessing.Process(target=slider_app, args=(L_values, R_values, 
    #    principal_vectors, D_mean, D_std, mano_layer, queue))
    #producer_process.start()
    #producer_process.join()
    consumer_process.join()
    print("done")


if __name__=='__main__':
    tyro.cli(main)
