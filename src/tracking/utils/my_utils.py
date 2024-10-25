import numpy as np
import open3d as o3d

def depth2fgpcd(depth, mask, cam_params):
    # depth: (h, w)
    # fgpcd: (n, 3)
    # mask: (h, w)
    h, w = depth.shape
    mask = np.logical_and(mask, depth > 0)
    fgpcd = np.zeros((mask.sum(), 3))
    fx, fy, cx, cy = cam_params
    pos_x, pos_y = np.meshgrid(np.arange(w), np.arange(h))
    pos_x = pos_x[mask]
    pos_y = pos_y[mask]
    fgpcd[:, 0] = (pos_x - cx) * depth[mask] / fx
    fgpcd[:, 1] = (pos_y - cy) * depth[mask] / fy
    fgpcd[:, 2] = depth[mask]
    return fgpcd



def np2o3d(pcd, color=None, seg=None):
    # pcd: (n, 3)
    # color: (n, 3)
    pcd_dicts = {}
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)

    for i, pos in enumerate(pcd_o3d.points):
        pcd_dicts[tuple(pos)] = {
            'color': pcd_o3d.colors[i],
            'seg': seg[i]
        }
    return pcd_o3d, pcd_dicts


def depth2normal(d_im, K):
    # :param d_im: (H, W) depth image in meters
    # :param K: (3, 3) camera intrinsics
    # :return (H, W, 3) normal image
    
    H, W = d_im.shape
    cx, cy, fx, fy = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
    
    pcd = np.zeros((H * W, 3))
    xy_grid = np.mgrid[0:W, 0:H].T.reshape(-1, 2)
    pcd[:, 0] = (xy_grid[:, 0] - cx) * d_im.reshape(-1) / fx
    pcd[:, 1] = (xy_grid[:, 1] - cy) * d_im.reshape(-1) / fy
    pcd[:, 2] = d_im.reshape(-1)
    
    pcd = pcd.reshape(H, W, 3)
    
    window = 10
    
    pcd = np.pad(pcd, ((0, window), (0, window), (0, 0)), mode='edge') # shape (H+1, W+1, 3)
    
    pcd_h_diff = pcd[window:, :W, :] - pcd[:-window, :W, :]
    pcd_v_diff = pcd[:H, window:, :] - pcd[:H, :-window, :]
    pcd_normals = np.cross(pcd_h_diff, pcd_v_diff) # shape (H, W, 3)
    pcd_normals = pcd_normals / (np.linalg.norm(pcd_normals, axis=2, keepdims=True) + 1e-6) # shape (H, W, 3)
    
    return pcd_normals