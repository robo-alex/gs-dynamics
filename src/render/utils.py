import torch
import numpy as np
import cv2


def project(points, intr, extr):
    # extr: (4, 4)
    # intr: (3, 3)
    # points: (n_points, 3)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = points @ extr.T  # (n_points, 4)
    points = points[:, :3] / points[:, 2:3]  # (n_points, 3)
    points = points @ intr.T
    points = points[:, :2] / points[:, 2:3]  # (n_points, 2)
    return points


class Visualizer:
    def __init__(self):
        import matplotlib.pyplot as plt
        self.history = 40
        self.linewidth = 3
        self.color_map = plt.get_cmap("viridis")
        self.kps = []
        self.kps_orig = []
        self.obj_kps = []
        self.z_min = -0.02
        self.radius = 10

    def draw_keypoints(self, im, keypoints, keypoints_orig, obj_kp):
        self.kps.append(keypoints)
        self.kps_orig.append(keypoints_orig)
        if len(self.kps) > self.history:
            self.kps.pop(0)
            self.kps_orig.pop(0)
        for k in range(len(self.kps) - 1):
            kp = self.kps[k]
            kp_next = self.kps[k + 1]
            color = np.array(self.color_map(k / (len(self.kps) - 1 + 1e-4)))[:3][::-1] * 255
            color = np.concatenate([color, [255]])

            z = self.kps_orig[k][0, 2]
            cv2.line(im, 
                (int(kp[0, 0]), int(kp[0, 1])), 
                (int(kp_next[0, 0]), int(kp_next[0, 1])), 
                color, self.radius
            )
        return im


def quat2mat(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.shape[0], 3, 3)).to(q.device)
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


def mat2quat(rot):
    t = torch.clamp(rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2], min=-1)
    q = torch.zeros((rot.shape[0], 4)).to(rot.device)

    mask_0 = t > -1
    t_0 = torch.sqrt(t[mask_0] + 1)
    q[mask_0, 0] = 0.5 * t_0
    t_0 = 0.5 / t_0
    q[mask_0, 1] = (rot[mask_0, 2, 1] - rot[mask_0, 1, 2]) * t_0
    q[mask_0, 2] = (rot[mask_0, 0, 2] - rot[mask_0, 2, 0]) * t_0
    q[mask_0, 3] = (rot[mask_0, 1, 0] - rot[mask_0, 0, 1]) * t_0

    # i = 0, j = 1, k = 2
    mask_1 = ~mask_0 & (rot[:, 0, 0] >= rot[:, 1, 1]) & (rot[:, 0, 0] >= rot[:, 2, 2])
    t_1 = torch.sqrt(1 + rot[mask_1, 0, 0] - rot[mask_1, 1, 1] - rot[mask_1, 2, 2])
    t_1 = 0.5 / t_1
    q[mask_1, 0] = (rot[mask_1, 2, 1] - rot[mask_1, 1, 2]) * t_1
    q[mask_1, 1] = 0.5 * t_1
    q[mask_1, 2] = (rot[mask_1, 1, 0] + rot[mask_1, 0, 1]) * t_1
    q[mask_1, 3] = (rot[mask_1, 2, 0] + rot[mask_1, 0, 2]) * t_1

    # i = 1, j = 2, k = 0
    mask_2 = ~mask_0 & (rot[:, 1, 1] >= rot[:, 2, 2]) & (rot[:, 1, 1] > rot[:, 0, 0])
    t_2 = torch.sqrt(1 + rot[mask_2, 1, 1] - rot[mask_2, 0, 0] - rot[mask_2, 2, 2])
    t_2 = 0.5 / t_2
    q[mask_2, 0] = (rot[mask_2, 0, 2] - rot[mask_2, 2, 0]) * t_2
    q[mask_2, 1] = (rot[mask_2, 2, 1] + rot[mask_2, 1, 2]) * t_2
    q[mask_2, 2] = 0.5 * t_2
    q[mask_2, 3] = (rot[mask_2, 0, 1] + rot[mask_2, 1, 0]) * t_2

    # i = 2, j = 0, k = 1
    mask_3 = ~mask_0 & (rot[:, 2, 2] > rot[:, 0, 0]) & (rot[:, 2, 2] > rot[:, 1, 1])
    t_3 = torch.sqrt(1 + rot[mask_3, 2, 2] - rot[mask_3, 0, 0] - rot[mask_3, 1, 1])
    t_3 = 0.5 / t_3
    q[mask_3, 0] = (rot[mask_3, 1, 0] - rot[mask_3, 0, 1]) * t_3
    q[mask_3, 1] = (rot[mask_3, 0, 2] + rot[mask_3, 2, 0]) * t_3
    q[mask_3, 2] = (rot[mask_3, 1, 2] + rot[mask_3, 2, 1]) * t_3
    q[mask_3, 3] = 0.5 * t_3

    assert torch.allclose(mask_1 + mask_2 + mask_3 + mask_0, torch.ones_like(mask_0))
    return q


def convert_opencv_to_opengl(w2c_opencv):
    """
    Convert extrinsics from OpenCV format to OpenGL format.
    """
    # Convert from OpenCV to OpenGL format
    R = w2c_opencv[:3, :3]
    t = w2c_opencv[:3, 3].reshape(3, 1)
    R_opengl = R.T
    t_opengl = -R.T @ t
    w2c_opengl = np.hstack((R_opengl, t_opengl))
    w2c_opengl = np.vstack((w2c_opengl, np.array([0, 0, 0, 1])))

    return w2c_opengl


def relations_to_matrix(Rr, Rs):
    relations = torch.zeros((Rr.shape[-1], Rs.shape[-1]), dtype=int, device=Rr.device)
    for j in range(Rr.shape[1]):
        assert Rr[0, j].sum() == 1
        assert Rs[0, j].sum() == 1
        relations[Rr[0, j].argmax().item(), Rs[0, j].argmax().item()] = 1
    return relations


def interpolate_motions(bones, motions, relations, xyz, rot=None, quat=None, weights=None, device='cuda'):
    # bones: (n_bones, 3)
    # motions: (n_bones, 3)
    # relations: (n_bones, n_bones)
    # xyz: (n_particles, 3)
    # rot: (n_particles, 3, 3)
    # quat: (n_particles, 4)
    # weights: (n_particles, n_bones)

    n_bones, _ = bones.shape
    n_particles, _ = xyz.shape

    # Compute the bone transformations
    bone_transforms = torch.zeros(n_bones, 4, 4).to(device)
    for i in range(n_bones):
        # find adjacent bones
        adjacency = relations[i].nonzero().squeeze(1)
        n_adj = len(adjacency)
        if n_adj == 0:
            bone_transforms[i, :3, :3] = torch.eye(3).to(device)
            continue
        adj_bones = bones[adjacency] - bones[i]  # (n_adj, 3)
        adj_bones_new = (bones[adjacency] + motions[adjacency]) - (bones[i] + motions[i])

        # add weights to the adj_bones
        W = torch.eye(n_adj).to(device)

        # fit a transformation
        F = adj_bones_new.T @ W @ adj_bones
        cov_rank = torch.linalg.matrix_rank(F)
        if cov_rank == 1:
            U, S, V = torch.svd(F)
            assert torch.allclose(S[1:], torch.zeros_like(S[1:]))
            x = torch.tensor([1., 0., 0.]).to(device)
            axis = U[:, 0]
            perp_axis = torch.linalg.cross(axis, x)
            if torch.norm(perp_axis) < 1e-6:
                R = torch.eye(3).to(device)
            else:
                perp_axis = perp_axis / torch.norm(perp_axis)
                third_axis = torch.cross(x, perp_axis)
                assert (torch.norm(third_axis) - 1) < 1e-6
                third_axis_after = torch.cross(axis, perp_axis)
                X = torch.stack([x, perp_axis, third_axis], dim=1)
                Y = torch.stack([axis, perp_axis, third_axis_after], dim=1)
                R = Y @ X.T
        else:
            try:
                U, S, V = torch.svd(F)
                S = torch.eye(3).to(torch.float32).to(device)
                if torch.linalg.det(F) < 0:
                    S[cov_rank, cov_rank] = -1
                R = U @ S @ V.T
            except:
                # svd failed, use the identity matrix
                R = torch.eye(3).to(device)
            
            if torch.abs(torch.linalg.det(R) - 1) > 1e-3:
                if torch.abs(torch.linalg.det(R) + 1) < 1e-3:
                    S[cov_rank, cov_rank] *= -1
                    R = U @ S @ V.T
                else:
                    print('det != 1')
                    print(R)
                    import ipdb; ipdb.set_trace()

        bone_transforms[i, :3, :3] = R

    bone_transforms[:, :3, 3] = motions

    # Compute the weights
    if weights is None:
        weights = torch.ones(n_particles, n_bones).to(device)

        dist = torch.cdist(xyz[None], bones[None])[0]  # (n_particles, n_bones)
        dist = torch.clamp(dist, min=1e-4)
        weights = 1 / dist
        weights = weights / weights.sum(dim=1, keepdim=True)  # (n_particles, n_bones)
    
    # Compute the transformed particles
    xyz_transformed = torch.zeros(n_particles, n_bones, 3).to(device)
    for i in range(n_bones):
        xyz_transformed[:, i] = (xyz - bones[i]) @ bone_transforms[i, :3, :3].T + bone_transforms[i, :3, 3] + bones[i]
    xyz_transformed = (xyz_transformed * weights[:, :, None]).sum(dim=1)  # (n_particles, 3)

    def quaternion_multiply(q1, q2):
        # q1: bsz x 4
        # q2: bsz x 4
        q = torch.zeros(q1.shape).to(q1.device)
        q[:, 0] = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
        q[:, 1] = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
        q[:, 2] = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
        q[:, 3] = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]
        return q

    if quat is not None:
        base_quats = mat2quat(bone_transforms[:, :3, :3])  # (n_bones, 4)
        base_quats = torch.nn.functional.normalize(base_quats, dim=-1)  # (n_particles, 4)
        quats = (base_quats[None] * weights[:, :, None]).sum(dim=1)  # (n_particles, 4)
        quats = torch.nn.functional.normalize(quats, dim=-1)
        rot = quaternion_multiply(quats, quat)

    # xyz_transformed: (n_particles, 3)
    # rot: (n_particles, 3, 3) / (n_particles, 4)
    # weights: (n_particles, n_bones)
    return xyz_transformed, rot, weights