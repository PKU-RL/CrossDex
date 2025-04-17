import numpy as np
import os, sys
from utils import get_pointcloud_from_mesh

if __name__ =="__main__":
    mesh_dir = '../../assets/ycb_assets/meshes'
    save_dir = '../../assets/ycb_assets/pointclouds'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for fn in os.listdir(mesh_dir):
        pc = get_pointcloud_from_mesh(mesh_dir, fn)
        np.save(os.path.join(save_dir, fn.split('.')[0]+'.npy'), pc)