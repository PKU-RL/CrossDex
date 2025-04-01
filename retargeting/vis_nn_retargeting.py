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
from retargeting_nn_utils import *


def retargeting(retargeting_models, queue: multiprocessing.Queue):
    viewer, retargetings_to_sapien, robots = create_sapien_hands_viewer(
        [m.robot_joint_names for m in retargeting_models], 
        [m.robot_urdf_path for m in retargeting_models])

    while True:
        viewer.render()
        try:
            eigengrasp = queue.get(timeout=0)
        except Empty:
            continue
        
        for model, robot, retargeting_to_sapien in zip(retargeting_models, robots, retargetings_to_sapien):
            if model.position_baseline:
                qpos = model.retarget_np(eigengrasp.reshape(1,15))[0]
            else:
                qpos = model.retarget_np(eigengrasp.reshape(1,9))[0]
            robot.set_qpos(qpos[retargeting_to_sapien])

class SliderAppNew(SliderApp):
    def update_hand(self, *args):
        pose = torch.zeros([1,48])
        params = np.array([var.get() for var in self.slider_values[:9 if not self.position_baseline else 15]])
        
        if not self.position_baseline:
            pose45 = reconstruct_mano_pose45(self.principal_vectors, self.D_mean, self.D_std, params)
            pose[0,1]=1.57
            pose[0,3:] = torch.tensor(pose45)

        hand_verts, hand_joints = self.mano_layer(pose) #, beta)
        hand_info = {'verts': hand_verts, 'joints': hand_joints}
        
        if self.queue is not None:
            self.queue.put(params)

        self.ax.clear()
        display_hand(hand_info, mano_faces=self.mano_layer.th_faces, ax=self.ax, show=False)
        self.canvas.draw()


def main(dataset:str="grab", add_random_dataset:bool=False, 
         robots:List[RobotName]=[RobotName.shadow, RobotName.allegro, RobotName.svh,
                                 RobotName.ability, RobotName.leap, RobotName.inspire],
         retargeting_type:str="dexpilot", position_baseline:bool=False):
    retargeting_models = [EigenRetargetModel(dataset, r, add_random_dataset, retargeting_type, position_baseline) for r in robots]
    mano_layer = ManoLayer(mano_root='../mano-models', 
        use_pca=False, ncomps=45, flat_hand_mean='grab' in dataset)
    
    if not position_baseline:
        pca_fn = '../results/pca_9_{}.pkl'.format(dataset)
        principal_vectors, min_values, max_values, D_mean, D_std = load_pca_data(pca_fn)
    else:
        fn = "../results/stat_position_baseline.pkl"
        min_values, max_values = load_position_baseline_data(fn)
        principal_vectors, D_mean, D_std = None, None, None
    L_values = min_values
    R_values = max_values
    print(L_values, R_values)

    multiprocessing.set_start_method('spawn', force=True)
    queue = multiprocessing.Queue(maxsize=1000)
    consumer_process = multiprocessing.Process(target=retargeting, args=(retargeting_models, queue))
    consumer_process.start()
    
    # create interactive eigengrasp viewer
    root = tk.Tk()
    app = SliderAppNew(root, L_values, R_values, principal_vectors, D_mean, D_std, mano_layer, queue, position_baseline)
    root.mainloop()

    consumer_process.join()
    print("done")


if __name__=='__main__':
    tyro.cli(main)
