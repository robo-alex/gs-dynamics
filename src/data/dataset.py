import glob
import json
import os
import pickle as pkl
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from dgl.geometry import farthest_point_sampler

from data.utils import fps_rad_idx
from real_world.utils.pcd_utils import rpy_to_rotation_matrix


def load_pairs(pairs_path, episode_idx):
    pair_lists = []
    frame_pairs = np.loadtxt(os.path.join(pairs_path, f'{episode_idx}.txt'))
    episodes = np.ones((frame_pairs.shape[0], 1)) * episode_idx
    pairs = np.concatenate([episodes, frame_pairs], axis=1)
    pair_lists.extend(pairs)
    pair_lists = np.array(pair_lists).astype(int)
    return pair_lists

def load_eef_pos(data_dir, output_dir):
    with open(os.path.join(output_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
    fn = np.array(meta['fn'])  # n_frames, 4

    frame_idx_lists = []
    for i in range(len(fn)):
        frame_idx = int(fn[i][0].split('/')[-1].split('_')[1].split('.')[0])
        frame_idx_lists.append(frame_idx)
    frame_idx_lists = np.array(frame_idx_lists)
    num_frames = len(frame_idx_lists)

    with open(os.path.join(data_dir, 'actions.txt'), 'r') as f:
        json_data = f.read()
    json_data = json_data.rstrip('\n').split('\n')  # a list of length len(fn)

    if len(json_data) != num_frames:
        # assert len(json_data) < num_frames
        json_data = [json_data[0]] * (max(frame_idx_lists) + 1 - len(json_data)) + json_data
    
    if len(json_data) - num_frames > 10:
        json_data = json_data[:num_frames]

    joint_angles = []
    poses = []
    for idx in range(len(frame_idx_lists)):
        try:
            actions = json.loads(json_data[frame_idx_lists[idx]])
        except:
            actions = json.loads(json_data[-1])
        joint_angles.append(actions['joint_angles'])
        poses.append(actions['pose'])
    joint_angles = np.array(joint_angles)
    poses = np.array(poses)

    with open(os.path.join(data_dir, "calibration_handeye_result.pkl"), "rb") as f:
        calib = pkl.load(f)

    eef_xyz = poses[:, :3]
    eef_rpy = poses[:, 3:]

    def get_eef_points(xyz, rpy, calib):
        R_gripper2base = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
        t_gripper2base = np.array(xyz) / 1000

        gripper_point = np.array([[0.0, 0.0, 0.17]])  # gripper

        R_base2world = calib['R_base2world']
        t_base2world = calib['t_base2world']
        R_gripper2world = R_base2world @ R_gripper2base
        t_gripper2world = R_base2world @ t_gripper2base + t_base2world
        gripper_points_in_world = R_gripper2world @ gripper_point.T + t_gripper2world[:, np.newaxis]
        gripper_points_in_world = gripper_points_in_world.T

        return gripper_points_in_world  # (1, 3)

    eef_particles = []
    for curr_frame in range(num_frames):
        eef_particles_curr = get_eef_points(eef_xyz[curr_frame], eef_rpy[curr_frame], calib)
        eef_particles.append(eef_particles_curr)
    eef_particles = np.array(eef_particles)
    return eef_particles, frame_idx_lists

def construct_edges_from_states(states, adj_thresh, mask, tool_mask, topk=10, connect_all=False):
    # :param states: (N, state_dim) torch tensor
    # :param adj_thresh: float
    # :param mask: (N) torch tensor, true when index is a valid particle
    # :param tool_mask: (N) torch tensor, true when index is a valid tool particle
    # :param pushing_direction: (state_dim) torch tensor, pushing direction
    # :return:
    # - Rr: (n_rel, N) torch tensor
    # - Rs: (n_rel, N) torch tensor

    N, state_dim = states.shape
    s_receiv = states[:, None, :].repeat(1, N, 1)
    s_sender = states[None, :, :].repeat(N, 1, 1)

    # dis: particle_num x particle_num
    # adj_matrix: particle_num x particle_num
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, None].repeat(1, N)
    mask_2 = mask[None, :].repeat(N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    tool_mask_1 = tool_mask[:, None].repeat(1, N)
    tool_mask_2 = tool_mask[None, :].repeat(N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations
    adj_matrix = ((dis - threshold) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)

    n_tool = tool_mask.sum().long().item()
    if n_tool > 0:
        dis_obj = dis[:-n_tool, :-n_tool]
    else:
        dis_obj = dis
    topk_idx = torch.topk(dis_obj, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(dis_obj)
    topk_matrix.scatter_(-1, topk_idx, 1)
    if n_tool > 0:
        adj_matrix[:-n_tool, :-n_tool] = adj_matrix[:-n_tool, :-n_tool] * topk_matrix
    else:
        adj_matrix = adj_matrix * topk_matrix

    if connect_all:
        obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
        obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender
        adj_matrix[obj_tool_mask_1] = 1.  # 0. if do not want obj sender, tool receiver relations
        adj_matrix[obj_tool_mask_2] = 1.
        adj_matrix[tool_mask_12] = 0.  # avoid tool to tool relations

    n_rels = adj_matrix.sum().long().item()
    rels_idx = torch.arange(n_rels).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rr[rels_idx, rels[:, 0]] = 1
    Rs[rels_idx, rels[:, 1]] = 1
    return Rr, Rs

def construct_edges_from_states_batch(states, adj_thresh, mask, tool_mask, topk=10, connect_all=False):  # helper function for construct_graph
    # :param states: (B, N, state_dim) torch tensor
    # :param adj_thresh: (B, ) torch tensor
    # :param mask: (B, N) torch tensor, true when index is a valid particle
    # :param tool_mask: (B, N) torch tensor, true when index is a valid tool particle
    # :param pushing_direction: (B, state_dim) torch tensor, pushing direction
    # :return:
    # - Rr: (B, n_rel, N) torch tensor
    # - Rs: (B, n_rel, N) torch tensor

    B, N, state_dim = states.shape
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)

    # dis: B x particle_num x particle_num
    # adj_matrix: B x particle_num x particle_num
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # (N, N, state_dim)
    dis = torch.sum(s_diff ** 2, -1)
    mask_1 = mask[:, :, None].repeat(1, 1, N)
    mask_2 = mask[:, None, :].repeat(1, N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10  # avoid invalid particles to particles relations
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # avoid tool to tool relations

    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # add topk constraints
    topk = min(dis.shape[-1], topk)

    n_tool = tool_mask.sum(dim=-1).long()
    assert n_tool.max().item() == n_tool.min().item(), 'only support fixed number of tool particles'
    n_tool = n_tool.max().item()
    if n_tool > 0:
        dis_obj = dis[:, :-n_tool, :-n_tool]
    else:
        dis_obj = dis
    topk_idx = torch.topk(dis_obj, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(dis_obj)
    topk_matrix.scatter_(-1, topk_idx, 1)
    if n_tool > 0:
        adj_matrix[:, :-n_tool, :-n_tool] = adj_matrix[:, :-n_tool, :-n_tool] * topk_matrix
    else:
        adj_matrix = adj_matrix * topk_matrix
    
    if connect_all:
        obj_tool_mask_1 = tool_mask_1 * mask_2  # particle sender, tool receiver
        obj_tool_mask_2 = tool_mask_2 * mask_1  # particle receiver, tool sender
        adj_matrix[obj_tool_mask_1] = 1.  # 0. if do not want obj sender, tool receiver relations
        adj_matrix[obj_tool_mask_2] = 1.
        adj_matrix[tool_mask_12] = 0.  # avoid tool to tool relations
    
    n_rels = adj_matrix.sum(dim=(1,2))
    n_rel = n_rels.max().long().item()
    rels_idx = []
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1
    return Rr, Rs

def pad(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = np.zeros((max_dim, x.shape[1]), dtype=np.float32)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = np.zeros((x.shape[0], max_dim, x.shape[2]), dtype=np.float32)
        x_pad[:, :x_dim] = x
    return x_pad

def pad_torch(x, max_dim, dim=0):
    if dim == 0:
        x_dim = x.shape[0]
        x_pad = torch.zeros((max_dim, x.shape[1]), dtype=x.dtype, device=x.device)
        x_pad[:x_dim] = x
    elif dim == 1:
        x_dim = x.shape[1]
        x_pad = torch.zeros((x.shape[0], max_dim, x.shape[2]), dtype=x.dtype, device=x.device)
        x_pad[:, :x_dim] = x
    return x_pad

class DynDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        phase='train',
    ):
        assert phase in ['train', 'valid']
        print(f'Loading {phase} dataset...')
        self.phase = phase

        self.dataset_config = dataset_config

        self.pair_lists = []
        self.obj_kypts = []
        self.tool_kypts = []

        # loop over datasets
        for i, dataset in enumerate(dataset_config['datasets']):
            assert i == 0  # only support single dataset
            print(f'Setting up dataset {dataset["name"]}')

            base_dir = Path(dataset['base_dir'])
            data_dir = base_dir / 'data'  # initial data such as rgbd, etc.
            output_dir = base_dir / 'ckpts' # 3DGS model output
            preprocess_dir = base_dir / 'preprocessed'  # to save preprocessed data

            name = dataset['name']
            exp_data_dir = data_dir / f'{name}'
            exp_output_dir = output_dir / f'exp_{name}'
            exp_preprocess_dir = preprocess_dir / f'exp_{name}'

            episodes = sorted(glob.glob(str(exp_preprocess_dir / 'episode_*')))
            episode_idxs = [int(epi.split('_')[-1]) for epi in episodes]

            # decide training and valid portion
            ratio = 0.8
            if phase == 'train':
                episodes = episodes[:int(len(episodes) * ratio)]
                episode_idxs = episode_idxs[:int(len(episode_idxs) * ratio)]
            else:
                episodes = episodes[int(len(episodes) * ratio):]
                episode_idxs = episode_idxs[int(len(episode_idxs) * ratio):]

            particle_pos_dict = {}
            eef_pos_dict = {}
            pair_lists_list = []

            # loop over episodes
            for episode_idx in episode_idxs:
                data_dir = os.path.join(exp_data_dir, f'episode_{episode_idx:02d}')
                output_dir = os.path.join(exp_output_dir, f'episode_{episode_idx:02d}', name, f'episode_{episode_idx:02d}')
                preprocess_dir = os.path.join(exp_preprocess_dir, f'episode_{episode_idx:02d}')

                # check if data dir exists
                print(f'Loading {phase} dataset from {output_dir}...')
                if not os.path.exists(output_dir):
                    raise ValueError(f'Data dir {output_dir} not found')

                # load particle pos (downsampled)
                params_dir = os.path.join(output_dir, 'param_downsampled.npy')
                if not os.path.exists(params_dir):
                    raise ValueError(f'Params dir {params_dir} not found')
                xyz = np.load(params_dir)

                particle_pos_dict[episode_idx] = xyz

                # load eef pos
                eef_xyz, _ = load_eef_pos(data_dir, output_dir)
                eef_pos_dict[episode_idx] = eef_xyz

                pairs_dir = os.path.join(preprocess_dir, 'frame_pairs')
                if not os.path.exists(pairs_dir):
                    raise ValueError(f'Pairs dir {pairs_dir} not found')

                pair_lists = load_pairs(pairs_dir, episode_idx)
                pair_mask = pair_lists[:, 1:].max(1) < len(xyz)  # filter out-of-range invalid pair lists
                pair_lists = pair_lists[pair_mask]

                pair_lists_list.append(pair_lists)

            pair_lists = np.concatenate(pair_lists_list, axis=0)
            print('Length of dataset is', len(pair_lists))

            self.pair_lists.extend(pair_lists)
            self.obj_kypts.append(particle_pos_dict)
            self.tool_kypts.append(eef_pos_dict)

        self.pair_lists = np.array(self.pair_lists)

    def __len__(self):
        return len(self.pair_lists)

    def __getitem__(self, i):
        dataset_idx = 0
        episode_order_idx = self.pair_lists[i][0].astype(int)
        pair = self.pair_lists[i][1:].astype(int)
        
        n_his = self.dataset_config['n_his']
        n_future = self.dataset_config['n_future']

        dataset_config = self.dataset_config['datasets'][dataset_idx]
        max_n = dataset_config['max_n']
        max_tool = dataset_config['max_tool']
        max_nobj = dataset_config['max_nobj']
        max_nR = dataset_config['max_nR']
        fps_radius_range = dataset_config['fps_radius_range']
        adj_radius_range = dataset_config['adj_radius_range']
        state_noise = dataset_config['state_noise'][self.phase]
        topk = dataset_config['topk']
        connect_all = dataset_config['connect_all']

        assert max_tool == 1, 'only support single tool'

        # get history keypoints
        obj_kps = []
        tool_kps = []
        for i in range(len(pair)):
            frame_idx = pair[i]
            try:
                obj_kp = self.obj_kypts[dataset_idx][episode_order_idx][frame_idx]
            except:
                print(f'Error loading episode {episode_order_idx}, frame {frame_idx}')
                raise Exception
            tool_kp = self.tool_kypts[dataset_idx][episode_order_idx][frame_idx]
            obj_kps.append([obj_kp])  # single object
            tool_kps.append(tool_kp)

        obj_kp_start = obj_kps[n_his-1]
        instance_num = len(obj_kp_start)
        assert instance_num == 1, 'only support single object'

        fps_idx_list = []
        ## sampling using raw particles
        for j in range(len(obj_kp_start)):
            # farthest point sampling
            particle_tensor = torch.from_numpy(obj_kp_start[j]).float()[None, ...]
            fps_idx_tensor = farthest_point_sampler(particle_tensor, min(max_nobj, particle_tensor.shape[1]), 
                                start_idx=np.random.randint(0, obj_kp_start[j].shape[0]))[0]
            fps_idx_1 = fps_idx_tensor.numpy().astype(np.int32)

            # downsample to uniform radius
            downsample_particle = particle_tensor[0, fps_idx_1, :].numpy()
            fps_radius = np.random.uniform(fps_radius_range[0], fps_radius_range[1])
            _, fps_idx_2 = fps_rad_idx(downsample_particle, fps_radius)
            fps_idx_2 = fps_idx_2.astype(int)
            fps_idx = fps_idx_1[fps_idx_2]
            fps_idx_list.append(fps_idx)

        # downsample to get current obj kp
        obj_kp_start = [obj_kp_start[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
        obj_kp_start = np.concatenate(obj_kp_start, axis=0) # (N, 3)
        obj_kp_num = obj_kp_start.shape[0]

        # get current state delta
        tool_kp = np.stack(tool_kps[n_his-1:n_his+1], axis=0)  # (2, num_tool_points, 3)
        tool_kp_num = tool_kp.shape[1]
        states_delta = np.zeros((max_nobj + max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        states_delta[max_nobj : max_nobj + tool_kp_num] = tool_kp[1] - tool_kp[0]

        # load future states
        obj_kp_future = np.zeros((n_future, max_nobj, obj_kp_start.shape[-1]), dtype=np.float32)
        # obj_future_mask = np.ones(n_future).astype(bool)  # (n_future,)
        for fi in range(n_future):
            obj_kp_fu = obj_kps[n_his+fi]
            obj_kp_fu = [obj_kp_fu[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_fu = np.concatenate(obj_kp_fu, axis=0) # (N, 3)
            obj_kp_fu = pad(obj_kp_fu, max_nobj)
            obj_kp_future[fi] = obj_kp_fu

        # load future tool keypoints
        tool_future = np.zeros((n_future - 1, max_nobj + max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        states_delta_future = np.zeros((n_future - 1, max_nobj + max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        for fi in range(n_future - 1):
            tool_kp_future = tool_kps[n_his+fi:n_his+fi+2]
            tool_kp_future = np.stack(tool_kp_future, axis=0)  # (2, 1, 3)
            tool_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[0]
            states_delta_future[fi, max_nobj : max_nobj + tool_kp_num] = tool_kp_future[1] - tool_kp_future[0]
        
        # load history states
        state_history = np.zeros((n_his, max_nobj + max_tool, obj_kp_start.shape[-1]), dtype=np.float32)
        for fi in range(n_his):
            obj_kp_his = obj_kps[fi]
            obj_kp_his = [obj_kp_his[j][fps_idx] for j, fps_idx in enumerate(fps_idx_list)]
            obj_kp_his = np.concatenate(obj_kp_his, axis=0)
            obj_kp_his = pad(obj_kp_his, max_nobj)
            state_history[fi, :max_nobj] = obj_kp_his

            tool_kp_his = tool_kps[fi]
            tool_kp_his = pad(tool_kp_his, max_tool)
            state_history[fi, max_nobj:] = tool_kp_his

        # load masks
        state_mask = np.zeros((max_nobj + max_tool), dtype=bool)
        state_mask[max_nobj : max_nobj + tool_kp_num] = True
        state_mask[:obj_kp_num] = True

        tool_mask = np.zeros((max_nobj + max_tool), dtype=bool)
        tool_mask[max_nobj : max_nobj + tool_kp_num] = True

        obj_mask = np.zeros((max_nobj,), dtype=bool)
        obj_mask[:obj_kp_num] = True

        # construct instance information
        # p_rigid = np.zeros(max_n, dtype=np.float32)  # clothes are nonrigid
        assert max_n == 1
        p_instance = np.zeros((max_nobj, max_n), dtype=np.float32)
        j_perm = np.random.permutation(instance_num)
        ptcl_cnt = 0
        # sanity check
        assert sum([len(fps_idx_list[j]) for j in range(len(fps_idx_list))]) == obj_kp_num
        # fill in p_instance
        for j in range(instance_num):
            p_instance[ptcl_cnt:ptcl_cnt + len(fps_idx_list[j_perm[j]]), j_perm[j]] = 1
            ptcl_cnt += len(fps_idx_list[j_perm[j]])

        # construct attributes
        attr_dim = 2
        attrs = np.zeros((max_nobj + max_tool, attr_dim), dtype=np.float32)
        attrs[:obj_kp_num, 0] = 1.
        attrs[max_nobj : max_nobj + tool_kp_num, 1] = 1.

        # add randomness
        # state randomness
        state_history += np.random.uniform(-state_noise, state_noise, size=state_history.shape)
        # rotation randomness (already translation-invariant)
        random_rot = np.random.uniform(-np.pi, np.pi)
        rot_mat = np.array([[np.cos(random_rot), -np.sin(random_rot), 0],
                            [np.sin(random_rot), np.cos(random_rot), 0],
                            [0, 0, 1]], dtype=state_history.dtype)  # 2D rotation matrix in xy plane
        state_history = state_history @ rot_mat[None]
        states_delta = states_delta @ rot_mat
        tool_future = tool_future @ rot_mat[None]
        states_delta_future = states_delta_future @ rot_mat[None]
        obj_kp_future = obj_kp_future @ rot_mat[None]

        # numpy to torch
        state_history = torch.from_numpy(state_history).float()
        states_delta = torch.from_numpy(states_delta).float()
        tool_future = torch.from_numpy(tool_future).float()
        states_delta_future = torch.from_numpy(states_delta_future).float()
        obj_kp_future = torch.from_numpy(obj_kp_future).float()
        attrs = torch.from_numpy(attrs).float()
        p_instance = torch.from_numpy(p_instance).float()
        state_mask = torch.from_numpy(state_mask)
        tool_mask = torch.from_numpy(tool_mask)
        obj_mask = torch.from_numpy(obj_mask)

        # construct edges
        adj_thresh = np.random.uniform(*adj_radius_range)
        Rr, Rs = construct_edges_from_states(state_history[-1], adj_thresh, 
                    mask=state_mask, tool_mask=tool_mask, topk=topk, connect_all=connect_all)
        Rr = pad_torch(Rr, max_nR)
        Rs = pad_torch(Rs, max_nR)

        # save graph
        graph = {
            # input information
            "state": state_history,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)

            # future information
            "tool_future": tool_future,  # (n_future-1, N+M, state_dim)
            "action_future": states_delta_future,  # (n_future-1, N+M, state_dim)

            # ground truth information
            "state_future": obj_kp_future,  # (n_future, N, state_dim)
            # "state_future_mask": obj_future_mask,  # (n_future,)

            # attr information
            "attrs": attrs,  # (N+M, attr_dim)
            "p_instance": p_instance,  # (N, n_instance)
            "obj_mask": obj_mask,  # (N,)

            "Rr": Rr,  # (max_nR, N)
            "Rs": Rs,  # (max_nR, N)
        }
        return graph
