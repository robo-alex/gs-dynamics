import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_camera(w, h, k, w2c, near=0.01, far=100, bg=[0, 0, 0]):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor(bg, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def params2rendervar_wt(params, t):
    print(params['unnorm_rotations'][t])
    rendervar = {
        'means3D': params['means3D'][t],
        'colors_precomp': params['rgb_colors'][t],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
        'opacities': torch.sigmoid(params['logit_opacities'][t]),
        'scales': torch.exp(params['log_scales'][t]),
        'means2D': torch.zeros_like(params['means3D'][t], requires_grad=True, device="cuda") + 0
    }
    return rendervar

def params2rendervar_consistent_rgb(params, variables):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': variables['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    # breakpoint()
    # print(pts.shape)
    pts_cont = np.ascontiguousarray(pts, np.float64)
    # print(pts_cont.shape)
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    if len(pcd.points) == 0:
        print("Point cloud is empty!")
    else:
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)

def o3d_knn_tensor(pts_tensor, num_knn):
    if pts_tensor.numel() == 0:
        print("Point cloud is empty!")
        return None, None

    pts_np = pts_tensor.detach().cpu().numpy() if pts_tensor.is_cuda else pts_tensor.numpy()
    pts_np_cont = np.ascontiguousarray(pts_np, dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np_cont)

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    indices = []
    sq_dists = []

    for p in pts_np_cont:
        [_, idx, dist] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(idx[1:])  # Skip the first index since it's the point itself
        sq_dists.append(dist[1:])

    return torch.tensor(sq_dists, dtype=pts_tensor.dtype, device=pts_tensor.device), torch.tensor(indices, dtype=torch.long, device=pts_tensor.device)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params(output_params, seq, exp):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params", **to_save)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
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

def quat2mat(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.shape[0], 3, 3)).to(device)
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot

def rot2quat(rot):
    # Preallocate quaternion tensor
    q = torch.zeros((rot.shape[0], 4)).to(rot.device)
    
    # Compute quaternion components
    q[:, 0] = 0.5 * torch.sqrt(1 + rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2])
    q[:, 1] = (rot[:, 2, 1] - rot[:, 1, 2]) / (4 * q[:, 0])
    q[:, 2] = (rot[:, 0, 2] - rot[:, 2, 0]) / (4 * q[:, 0])
    q[:, 3] = (rot[:, 1, 0] - rot[:, 0, 1]) / (4 * q[:, 0])
    
    return q