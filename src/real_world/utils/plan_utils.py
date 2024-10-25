import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import time
import cv2
import math
import os
import scipy
from dgl.geometry import farthest_point_sampler

from real_world.utils.pcd_utils import get_tabletop_points
from data.utils import fps_rad_idx
from data.dataset import construct_edges_from_states


def chamfer(x, y):  # x: (B, N, D), y: (B, M, D)
    x = x[:, None].repeat(1, y.shape[1], 1, 1)  # (B, M, N, D)
    y = y[:, :, None].repeat(1, 1, x.shape[2], 1)  # (B, M, N, D)
    dis = torch.norm(x - y, 2, dim=-1)  # (B, M, N)
    dis_xy = torch.mean(dis.min(dim=2).values, dim=1)  # dis_xy: mean over N
    dis_yx = torch.mean(dis.min(dim=1).values, dim=1)  # dis_yx: mean over M
    return dis_xy + dis_yx


def em_distance(x, y):
    # x: [B, N, D]
    # y: [B, M, D]
    x_ = x[:, :, None, :].repeat(1, 1, y.size(1), 1)  # x: [B, N, M, D]
    y_ = y[:, None, :, :].repeat(1, x.size(1), 1, 1)  # y: [B, N, M, D]
    dis = torch.norm(torch.add(x_, -y_), 2, dim=3)  # dis: [B, N, M]
    x_list = []
    y_list = []
    for i in range(dis.shape[0]):
        cost_matrix = dis[i].detach().cpu().numpy()
        try:
            ind1, ind2 = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=False)
        except:
            print("Error in linear sum assignment!")
        x_list.append(x[i, ind1])
        y_list.append(y[i, ind2])
    new_x = torch.stack(x_list)
    new_y = torch.stack(y_list)
    emd = torch.mean(torch.norm(torch.add(new_x, -new_y), 2, dim=2), dim=1)
    return emd


def sample_action_seq(act_seq, action_lower_lim, action_upper_lim, n_sample, device, iter_index=0, noise_level=0.3, push_length=0.01):
    if iter_index == 0:
        # resample completely
        act_seqs = torch.rand((n_sample, act_seq.shape[0], act_seq.shape[1]), device=device) * \
            (action_upper_lim - action_lower_lim) + action_lower_lim
    else:
        # beta_filter = 0.7
        n_look_ahead = act_seq.shape[0]
        
        assert act_seq.shape[-1] == 4  # (x, y, theta, length)
        act_seqs = torch.stack([act_seq.clone()] * n_sample)
        xs = act_seqs[:, :, 0]
        ys = act_seqs[:, :, 1]
        thetas = act_seqs[:, :, 2]
        lengths = act_seqs[:, :, 3]
        
        x_ends = xs - lengths * push_length * torch.cos(thetas)
        y_ends = ys - lengths * push_length * torch.sin(thetas)

        # act_residual = torch.zeros((n_sample, 4), dtype=act_seqs.dtype, device=device)
        for i in range(n_look_ahead):
            # noise_sample = torch.normal(0, noise_level, (n_sample, 2), device=device)
            noise_sample = torch.normal(0, noise_level, (n_sample, 4), device=device)
            beta = 0.1 * (10 ** i)
            act_residual = beta * noise_sample
            # act_residual = beta_filter * noise_sample + act_residual * (1. - beta_filter)
            
            # xs_i = xs[:, i] + act_residual[:, 0]
            # ys_i = ys[:, i] + act_residual[:, 1]
            # xs_i = xs[:, i]
            # ys_i = ys[:, i]
            # x_ends_i = x_ends[:, i] + act_residual[:, 0]
            # y_ends_i = y_ends[:, i] + act_residual[:, 1]

            xs_i = xs[:, i] + act_residual[:, 0]
            ys_i = ys[:, i] + act_residual[:, 1]
            x_ends_i = x_ends[:, i] + act_residual[:, 2]
            y_ends_i = y_ends[:, i] + act_residual[:, 3]


            thetas_i = torch.atan2(ys_i - y_ends_i, xs_i - x_ends_i)
            lengths_i = torch.norm(torch.stack([x_ends_i - xs_i, y_ends_i - ys_i], dim=-1), dim=-1).clone() / push_length


            act_seq_i = torch.stack([xs_i, ys_i, thetas_i, lengths_i], dim=-1)
            # act_seq_i = torch.stack([thetas_i, lengths_i], dim=-1)
            act_seq_i = clip_actions(act_seq_i, action_lower_lim, action_upper_lim)
            act_seqs[1:, i] = act_seq_i[1:].clone()

    return act_seqs  # (n_sample, n_look_ahead, action_dim)


def clip_actions(action, action_lower_lim, action_upper_lim):
    action_new = action.clone()
    action_new[..., 0] = angle_normalize(action[..., 0])
    action_new.data.clamp_(action_lower_lim, action_upper_lim)
    return action_new


def optimize_action_mppi(act_seqs, reward_seqs, reward_weight=100.0, action_lower_lim=None, action_upper_lim=None, push_length=0.01):
    weight_seqs = F.softmax(reward_seqs * reward_weight, dim=0).unsqueeze(-1)

    assert act_seqs.shape[-1] == 4  # (x, y, theta, length)
    xs = act_seqs[:, :, 0]
    ys = act_seqs[:, :, 1]
    thetas = act_seqs[:, :, 2]
    lengths = act_seqs[:, :, 3]

    x_ends = xs - lengths * push_length * torch.cos(thetas)
    y_ends = ys - lengths * push_length * torch.sin(thetas)

    x = torch.sum(weight_seqs * xs, dim=0)  # (n_look_ahead,)
    y = torch.sum(weight_seqs * ys, dim=0)  # (n_look_ahead,)

    x_end = torch.sum(weight_seqs * x_ends, dim=0)  # (n_look_ahead,)
    y_end = torch.sum(weight_seqs * y_ends, dim=0)  # (n_look_ahead,)

    theta = torch.atan2(y - y_end, x - x_end)  # (n_look_ahead,)
    length = torch.norm(torch.stack([x_end - x, y_end - y], dim=-1), dim=-1) / push_length  # (n_look_ahead,)

    act_seq = torch.stack([x, y, theta, length], dim=-1)  # (n_look_ahead, action_dim)

    act_seq = clip_actions(act_seq, action_lower_lim, action_upper_lim)
    return act_seq


def decode_action(action, push_length=0.01):
    x_start = action[:, :, 0]
    y_start = action[:, :, 1]
    theta = action[:, :, 2]
    length = action[:, :, 3].detach()
    action_repeat = length.to(torch.int32)
    x_end = x_start - push_length * torch.cos(theta)
    y_end = y_start - push_length * torch.sin(theta)
    decoded_action = torch.stack([x_start, y_start, x_end, y_end], dim=-1)
    return decoded_action, action_repeat


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


def box_loss(state, target):
    # state: (B, N, 3)
    # target: (2, 2)
    xmin, xmax, ymin, ymax = target[0, 0], target[0, 1], target[1, 0], target[1, 1]
    x_diff = torch.maximum(xmin - state[:, :, 0], torch.zeros_like(state[:, :, 0])) + \
        torch.maximum(state[:, :, 0] - xmax, torch.zeros_like(state[:, :, 0]))
    y_diff = torch.maximum(ymin - state[:, :, 2], torch.zeros_like(state[:, :, 2])) + \
        torch.maximum(state[:, :, 2] - ymax, torch.zeros_like(state[:, :, 2]))
    r_diff = (x_diff ** 2 + y_diff ** 2) ** 0.5  # (B, N)
    return r_diff.mean(dim=1)  # (B,)


def visualize_img(state_init, res, rgb_vis, obj_pcd, intr, extr, adj_thresh, 
        target_state=None, target_box=None, state_after=None, save_dir=None, postfix=None, topk=None, connect_all=False, push_length=0.01):
    # state_init: (n_points, 3)
    # state: (n_look_forward, n_points, 3)
    # target_state: (n_points_raw, 3)
    # rgb_vis: np.ndarray (H, W, 3)
    # obj_pcd: np.ndarray (n_points, 3)

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

    # best result
    action_best = res['act_seq']  # (n_look_forward, action_dim)
    state_best = res['best_model_output']['state_seqs'][0]  # (n_look_forward, max_nobj, 3)

    # plot
    action, repeat = decode_action(action_best.unsqueeze(0), push_length=push_length)  # (1, n_look_forward, action_dim)
    action = action[0]  # (n_look_forward, action_dim)
    repeat = repeat[0, 0].item()

    state_init_vis = state_init.detach().cpu().numpy()  # (n_points, 3)
    state_vis = state_best[0].detach().cpu().numpy()  # (n_points, 3)
    if target_state is not None:
        target_state_vis = target_state.detach().cpu().numpy()  # (n_target_points, 3)
    action_vis = action[0].detach().cpu().numpy()  # (action_dim,)

    Rr, Rs = construct_edges_from_states(torch.from_numpy(state_init_vis), adj_thresh, 
                mask=torch.ones(state_init_vis.shape[0], dtype=bool),
                tool_mask=torch.zeros(state_init_vis.shape[0], dtype=bool), topk=topk, connect_all=connect_all)
    Rr = Rr.numpy()  # (n_rel, n_points)
    Rs = Rs.numpy()  # (n_rel, n_points)

    Rr_best, Rs_best = construct_edges_from_states(torch.from_numpy(state_vis), adj_thresh,
                mask=torch.ones(state_vis.shape[0], dtype=bool),
                tool_mask=torch.zeros(state_vis.shape[0], dtype=bool), topk=topk, connect_all=connect_all)
    Rr_best = Rr_best.numpy()  # (n_rel, n_points)
    Rs_best = Rs_best.numpy()  # (n_rel, n_points)

    if state_after is not None:
        state_after_vis = state_after.detach().cpu().numpy()  # (n_points, 3)

        # construct relations
        Rr_after, Rs_after = construct_edges_from_states(torch.from_numpy(state_after_vis), adj_thresh,
                    mask=torch.ones(state_after_vis.shape[0], dtype=bool),
                    tool_mask=torch.zeros(state_after_vis.shape[0], dtype=bool), topk=topk, connect_all=connect_all)
        Rr_after = Rr_after.numpy()
        Rs_after = Rs_after.numpy()

        Rr = Rr_after.copy()
        Rs = Rs_after.copy()

        state_init_vis = state_after_vis.copy()

    # plot state_init_vis, Rr, Rs, action_vis, state_vis, target_state_vis on rgb_vis
    # preparation
    state_init_proj = project(state_init_vis, intr, extr)
    state_proj = project(state_vis, intr, extr)
    if target_state is not None:
        target_state_proj = project(target_state_vis, intr, extr)

    # visualize
    rgb_orig = rgb_vis.copy()

    color_start = (202, 63, 41)
    color_action = (27, 74, 242)
    color_pred = (237, 158, 49)
    color_target = (26, 130, 81)
    color_target_corr = (0, 255, 255)

    # starting state
    point_size = 5
    for k in range(state_init_proj.shape[0]):
        cv2.circle(rgb_vis, (int(state_init_proj[k, 0]), int(state_init_proj[k, 1])), point_size, 
            color_start, -1)

    # starting edges
    edge_size = 2
    for k in range(Rr.shape[0]):
        if Rr[k].sum() == 0: continue
        receiver = Rr[k].argmax()
        sender = Rs[k].argmax()
        cv2.line(rgb_vis, 
            (int(state_init_proj[receiver, 0]), int(state_init_proj[receiver, 1])), 
            (int(state_init_proj[sender, 0]), int(state_init_proj[sender, 1])), 
            color_start, edge_size)
    
    # action arrow
    x_start = action_vis[0]
    y_start = action_vis[1]
    x_end = action_vis[2]
    y_end = action_vis[3]
    x_delta = x_end - x_start
    y_delta = y_end - y_start

    z = state_init[:, 2].mean().item()

    arrow_size = 3
    tip_length = 0.5
    for i in range(repeat):
        action_start_point = np.array([x_start + i * x_delta, y_start + i * y_delta, z])
        action_end_point = np.array([x_end + i * x_delta, y_end + i * y_delta, z])
        action_start_point_proj = project(action_start_point[None], intr, extr)[0]
        action_end_point_proj = project(action_end_point[None], intr, extr)[0]
        cv2.arrowedLine(rgb_vis,
            (int(action_start_point_proj[0]), int(action_start_point_proj[1])),
            (int(action_end_point_proj[0]), int(action_end_point_proj[1])),
            color_action, arrow_size, tipLength=tip_length)

    rgb_overlay = rgb_vis.copy()

    # target point cloud
    if target_state is not None:
        for k in range(target_state_proj.shape[0]):
            cv2.circle(rgb_vis, (int(target_state_proj[k, 0]), int(target_state_proj[k, 1])), point_size, 
                color_target, -1)
    
    if target_box is not None:
        x_min, x_max, y_min, y_max = target_box[0, 0].item(), target_box[0, 1].item(), target_box[1, 0].item(), target_box[1, 1].item()
        edge = 0.003
        rect_1 = np.array([[x_min - edge, y_min - edge, 0], [x_min + edge, y_min - edge, 0], [x_min + edge, y_max + edge, 0], [x_min - edge, y_max + edge, 0]])
        rect_2 = np.array([[x_max - edge, y_min - edge, 0], [x_max + edge, y_min - edge, 0], [x_max + edge, y_max + edge, 0], [x_max - edge, y_max + edge, 0]])
        rect_3 = np.array([[x_min + edge, y_min - edge, 0], [x_max - edge, y_min - edge, 0], [x_max - edge, y_min + edge, 0], [x_min + edge, y_min + edge, 0]])
        rect_4 = np.array([[x_min + edge, y_max - edge, 0], [x_max - edge, y_max - edge, 0], [x_max - edge, y_max + edge, 0], [x_min + edge, y_max + edge, 0]])

        rect_1_proj = project(rect_1, intr, extr)
        rect_2_proj = project(rect_2, intr, extr)
        rect_3_proj = project(rect_3, intr, extr)
        rect_4_proj = project(rect_4, intr, extr)

        cv2.fillConvexPoly(rgb_vis, rect_1_proj.astype(np.int32), color_target)
        cv2.fillConvexPoly(rgb_vis, rect_2_proj.astype(np.int32), color_target)
        cv2.fillConvexPoly(rgb_vis, rect_3_proj.astype(np.int32), color_target)
        cv2.fillConvexPoly(rgb_vis, rect_4_proj.astype(np.int32), color_target)

    # predicted state
    for k in range(state_proj.shape[0]):
        cv2.circle(rgb_vis, (int(state_proj[k, 0]), int(state_proj[k, 1])), point_size, 
            color_pred, -1)
    
    # predicted edges
    for k in range(Rr_best.shape[0]):
        if Rr_best[k].sum() == 0: continue
        receiver = Rr_best[k].argmax()
        sender = Rs_best[k].argmax()
        cv2.line(rgb_vis, 
            (int(state_proj[receiver, 0]), int(state_proj[receiver, 1])), 
            (int(state_proj[sender, 0]), int(state_proj[sender, 1])), 
            color_pred, edge_size)
    
    rgb_vis = cv2.addWeighted(rgb_overlay, 0.5, rgb_vis, 0.5, 0)
    
    if save_dir is not None:
        cv2.imwrite(os.path.join(save_dir, f'rgb_vis_{postfix}.png'), rgb_vis)
        cv2.imwrite(os.path.join(save_dir, f'rgb_orig_{postfix}.png'), rgb_orig)


def construct_state(obj_kps, fps_radius, eef_kps=None, max_nobj=100, max_neef=1):

    if eef_kps is None:
        eef_kps = np.zeros((0, 3))

    particle_tensor = torch.from_numpy(obj_kps).float()[None, ...]
    fps_idx_tensor = farthest_point_sampler(particle_tensor, max_nobj, start_idx=np.random.randint(0, obj_kps.shape[0]))[0]
    fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)
    downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
    _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
    fps_idx_2 = fps_idx_2.astype(int)
    fps_idx = fps_idx_1[fps_idx_2]

    # downsample to get current obj kp
    obj_kps = obj_kps[fps_idx]
    obj_kp_num = obj_kps.shape[0]
    eef_kp_num = eef_kps.shape[0]

    # load masks
    state_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    state_mask[max_nobj : max_nobj + eef_kp_num] = True
    state_mask[:obj_kp_num] = True

    eef_mask = np.zeros((max_nobj + max_neef), dtype=bool)
    eef_mask[max_nobj : max_nobj + eef_kp_num] = True

    state = np.zeros((max_nobj + max_neef, 3))
    state[:obj_kp_num] = obj_kps
    state[max_nobj : max_nobj + eef_kp_num] = eef_kps

    state_graph = {
        "obj_state": obj_kps,  # (N, state_dim)
        "obj_state_raw": obj_kps[:obj_kp_num],
        "eef_state": eef_kps,  # (M, state_dim)
        "state": state,  # (N+M, state_dim)
    }
    return state_graph


def get_state_cur(env, device, fps_radius=0.02, max_nobj=100, max_neef=1, visualize_img=False):
    obs = env.get_obs(get_color=True, get_depth=True)
    intr_list = env.get_intrinsics()
    R_list, t_list = env.get_extrinsics()
    bbox = env.get_bbox()

    rgb_list = []
    depth_list = []
    for i in range(4):
        rgb = obs[f'color_{i}'][-1]
        depth = obs[f'depth_{i}'][-1]
        rgb_list.append(rgb)
        depth_list.append(depth)

    pcd = get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox=bbox)
    obj_kps = np.array(pcd.points).astype(np.float32)

    state_graph = construct_state(obj_kps, fps_radius=fps_radius, max_nobj=max_nobj, max_neef=max_neef)
    state_cur = state_graph['obj_state_raw']  # (N, state_dim)
    state_cur = torch.tensor(state_cur, dtype=torch.float32, device=device)

    if not visualize_img:
        return state_cur

    rgb_vis = obs['color_0'][-1]
    intr = env.get_intrinsics()[0]
    R, t = env.get_extrinsics()[0][0], env.get_extrinsics()[1][0]
    extr = np.eye(4)
    extr[:3, :3] = R.T
    extr[:3, 3] = -R.T @ t
    return state_cur, obj_kps, rgb_vis, intr, extr

