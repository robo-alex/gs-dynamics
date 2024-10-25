import numpy as np
import torch
from dgl.geometry import farthest_point_sampler


def fps(pcd, particle_num, init_idx=-1):
    # pcd: (n, 3) numpy array
    # pcd_fps: (self.particle_num, 3) numpy array
    pcd_tensor = torch.from_numpy(pcd).float()[None, ...]
    if init_idx == -1:
        # init_idx = findClosestPoint(pcd, pcd.mean(axis=0))
        pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num)[0]
    else:
        pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num, init_idx)[0]
    pcd_fps_tensor = pcd_tensor[0, pcd_fps_idx_tensor]
    pcd_fps = pcd_fps_tensor.numpy()
    dist = np.linalg.norm(pcd[:, None] - pcd_fps[None, :], axis=-1)
    dist = dist.min(axis=1)
    return pcd_fps, dist.max()

def fps_rad(pcd, radius):
    # pcd: (n, 3) numpy array
    # pcd_fps: (-1, 3) numpy array
    # radius: float
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps

def fps_rad_idx(pcd, radius):
    # pcd: (n, 3) numpy array
    # pcd_fps: (-1, 3) numpy array
    # radius: float
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    idx_lst = [rand_idx]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        idx_lst.append(dist.argmax())
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    idx_lst = np.stack(idx_lst, axis=0)
    return pcd_fps, idx_lst

def fps_rad_idx_torch(pcd, radius):
    # pcd: (n, 3) torch tensor
    # pcd_fps: (-1, 3) torch tensor
    # radius: float
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    idx_lst = [rand_idx]
    dist = torch.norm(pcd - pcd_fps_lst[0], dim=1)
    while dist.max() > radius:
        argmax = dist.argmax().item()
        pcd_fps_lst.append(pcd[argmax])
        idx_lst.append(argmax)
        dist = torch.minimum(dist, torch.norm(pcd - pcd_fps_lst[-1], dim=1))
    pcd_fps = torch.stack(pcd_fps_lst, dim=0)
    idx_lst = torch.tensor(idx_lst).to(pcd_fps.device)
    return pcd_fps, idx_lst


def opengl2cam(pcd, cam_extrinsic, global_scale=1):
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    opencv_T_world = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    # print('opencv_T_world inverse', np.linalg.inv(opencv_T_world))
    cam = np.matmul(np.linalg.inv(opencv_T_world), np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).T).T[:, :3] / global_scale
    # print('cam', cam)
    # print()
    return cam

def cam2opengl(pcd, cam_extrinsic, global_scale=1):
    opencv_T_opengl = np.array([[1, 0, 0, 0],
                                [0, -1, 0, 0],
                                [0, 0, -1, 0],
                                [0, 0, 0, 1]])
    opencv_T_world = np.matmul(np.linalg.inv(cam_extrinsic), opencv_T_opengl)
    opengl = np.matmul(opencv_T_world, np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1).T).T[:, :3] * global_scale
    return opengl


def rgba_to_rgb(im, bg=None):
    if bg == None:
        bg = np.zeros((im.shape[0], im.shape[1], 3))
    alpha = im[:, :, 3:4].copy() / 255.0
    im_new = im[:, :, :3].copy() * alpha + bg * (1 - alpha)
    return im_new

def rgb_colormap(repeat=1):
    base = np.asarray([
        [0, 0, 255],
        [0, 255, 0],
        [255, 0, 0],
    ])
    return np.repeat(base, repeat, axis=0)

def label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
    Returns:
    A colormap for visualizing segmentation results.
    """
    return np.asarray([
        # [0, 0, 0],
        # [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        # [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])
