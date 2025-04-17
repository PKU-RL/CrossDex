from urdfpy import URDF
import numpy as np
import os, sys
import trimesh
import open3d as o3d
import queue
import time
import torch
from queue import Empty

def get_pointcloud_from_mesh(mesh_dir, filename, num_sample=4096):
    all_points = []
    mesh = trimesh.load_mesh(os.path.join(mesh_dir, filename))
    points = mesh.sample(num_sample)
    return points

# visualize static point cloud
def vis_pointcloud(pc, add_coordinate_frame=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pc)
    vis.add_geometry(pointcloud)
    if add_coordinate_frame:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)
    while True:
        vis.poll_events()
        vis.update_renderer()

# realtime visualize point cloud with threading
# pc_queue: queue.Queue(1)
def vis_pointcloud_realtime(pc_queue, add_coordinate_frame=True, zoom=1, coord_len=0.5):
    def vis_pointcloud_realtime_thread(q, add_coordinate_frame=True):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600)
        pointcloud = o3d.geometry.PointCloud()
        vis.add_geometry(pointcloud)
        # draw coordinate frame
        if add_coordinate_frame:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=coord_len, origin=[0, 0, 0])
            vis.add_geometry(coordinate_frame)
        # adjust view point
        view_control = vis.get_view_control()
        view_control.set_lookat([0, 0, 0])  # 观察点设置为世界坐标原点
        view_control.set_front([1, 0.2, 0.4])  # 前视方向
        view_control.set_up([0, 0, 1])  # 上方向设置为Z轴方向
        view_control.set_zoom(zoom)  # 设置缩放比例以适应场景
        while True:
            try:
                pcd = q.get()
                pointcloud.points = o3d.utility.Vector3dVector(pcd)
                vis.update_geometry(pointcloud)
                vis.poll_events()
                vis.update_renderer()
            except Empty:
                continue

    import threading
    thread = threading.Thread(target=vis_pointcloud_realtime_thread, args=(pc_queue,))
    thread.daemon = True
    thread.start()


########## torch point clouds processing ##########

def farthest_point_sample(xyz, npoint, device, init=None):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if init is not None:
        farthest = torch.tensor(init).long().reshape(B).to(device)
    else:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx, device):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.size()[0]
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points



if __name__=="__main__":
    #pc = get_pointcloud_from_mesh('../../assets/obj/meshes', 'cup.obj')
    #print(pc)

    #vis_pointcloud(pc)
    #pc = np.load('real_pcl.npy')  #('../../assets/ycb_assets/pointclouds/014_lemon.npy')
    pc = np.load('../../assets/obj/pointclouds/cup.npy')
    print(pc.shape, pc.dtype)

    # pc = torch.tensor(pc, dtype=torch.float32).to('cuda').unsqueeze(0).expand(512,-1,-1)
    # print(pc.shape)
    # from tqdm import trange
    # for i in trange(1):
    #     idx = farthest_point_sample(pc, 512, 'cuda')
    #     pc_sample = index_points(pc, idx, 'cuda')
    #     print(pc_sample.shape)

    
    pc_th = torch.tensor(pc, dtype=torch.float32).to('cuda').unsqueeze(0)
    Q = queue.Queue(1)
    vis_pointcloud_realtime(Q, coord_len=0.1)
    for t in range(1000):
        indices = farthest_point_sample(pc_th, 512, 'cuda')
        indices = indices[0].cpu().numpy()
        p = pc[indices]
        #if Q.full():
        #    Q.get()
        Q.put(p)
        #time.sleep(0.01)
    
