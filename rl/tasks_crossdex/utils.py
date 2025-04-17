import json, os
import numpy as np
import torch
from isaacgym.torch_utils import *

def load_robot_randomization_dict(json_path):
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
    #print(data)
    ret = {}
    for name, itms in data.items():
        kys = list(itms.keys())
        #print(kys)
        ret[name] = [itms[k] for k in kys]
    return ret


def load_object_point_clouds(object_files, asset_root):
    ret = []
    for fn in object_files:
        substrs = fn.split('/')
        pc_fn = os.path.join(substrs[0], 'pointclouds', substrs[-1].replace('.urdf','.npy'))
        print("object file: {} -> pcl file: {}".format(fn, pc_fn))
        pc = np.load(os.path.join(asset_root, pc_fn))
        #pc = np.load("vision/real_pcl.npy")
        ret.append(pc)
    return ret


########## torch jit functions ###########
@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:,:,:3]
    elif len(pt_new.size()) == 2:
        return pt_new[:,:3]


if __name__=="__main__":
    ret = load_robot_randomization_dict("../../robot_randomization/urdf_rand_xyz/results.json")
    print(ret)