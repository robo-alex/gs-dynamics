import argparse
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import time
import cv2
import math
import os
import yaml
import glob
from functools import partial
import copy

from real_world.utils.real_env import RealEnv
from real_world.utils.planner import Planner
from real_world.utils.plan_utils import sample_action_seq, clip_actions, optimize_action_mppi, \
                                        decode_action, chamfer, visualize_img, get_state_cur
from data.dataset import construct_edges_from_states_batch, pad_torch
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed


@torch.no_grad()
def dynamics(state, perturbed_action, model, device, adj_thresh, push_length, max_n=1, max_nR=500, n_his=3):
    time0 = time.time()

    bsz = perturbed_action.shape[0]
    n_look_forward = perturbed_action.shape[1]

    decoded_action, action_repeat = decode_action(perturbed_action, push_length=push_length)

    obj_kp = state[None, None].repeat(bsz, n_his, 1, 1)
    obj_kp_num = obj_kp.shape[2]
    eef_kp_num = 1
    max_nobj = obj_kp_num
    max_neef = eef_kp_num

    pred_state_seq = torch.zeros((bsz, n_look_forward, max_nobj, 3), device=device)

    for li in range(n_look_forward):
        print(f"look forward iter {li}")
        
        if li > 0:
            obj_kp = pred_state_seq[:, li-1:li].detach().clone().repeat(1, n_his, 1, 1)

        z = (obj_kp[:, -1, :, 2]).min(dim=1).values  # (bsz,)

        eef_kp = torch.zeros((bsz, 1, 3))
        eef_kp[:, 0, 0] = decoded_action[:, li, 0]  # TODO batch
        eef_kp[:, 0, 1] = decoded_action[:, li, 1]
        eef_kp[:, 0, 2] = z
        eef_kp_delta = torch.zeros((bsz, 1, 3))
        eef_kp_delta[:, 0, 0] = decoded_action[:, li, 2] - decoded_action[:, li, 0]
        eef_kp_delta[:, 0, 1] = decoded_action[:, li, 3] - decoded_action[:, li, 1]
        eef_kp_delta[:, 0, 2] = 0

        states = torch.zeros((bsz, n_his, max_nobj + max_neef, 3), device=device)
        states[:, :, :obj_kp_num] = obj_kp
        states[:, :, max_nobj : max_nobj + eef_kp_num] = eef_kp[:, None]

        states_delta = torch.zeros((bsz, max_nobj + max_neef, 3), device=device)
        states_delta[:, max_nobj : max_nobj + eef_kp_num] = eef_kp_delta

        attr_dim = 2
        attrs = torch.zeros((bsz, max_nobj + max_neef, attr_dim), dtype=torch.float32, device=device)
        attrs[:, :obj_kp_num, 0] = 1.
        attrs[:, max_nobj : max_nobj + eef_kp_num, 1] = 1.

        p_instance = torch.zeros((bsz, max_nobj, max_n), dtype=torch.float32, device=device)
        instance_num = 1
        instance_kp_nums = [obj_kp_num]
        for i in range(bsz):
            ptcl_cnt = 0
            j_perm = np.random.permutation(instance_num)
            for j in range(instance_num):
                p_instance[i, ptcl_cnt:ptcl_cnt + instance_kp_nums[j], j_perm[j]] = 1
                ptcl_cnt += instance_kp_nums[j]

        state_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
        state_mask[:, max_nobj : max_nobj + eef_kp_num] = True
        state_mask[:, :obj_kp_num] = True

        eef_mask = torch.zeros((bsz, max_nobj + max_neef), dtype=bool, device=device)
        eef_mask[:, max_nobj : max_nobj + eef_kp_num] = True

        obj_mask = torch.zeros((bsz, max_nobj,), dtype=bool, device=device)
        obj_mask[:, :obj_kp_num] = True

        Rr, Rs = construct_edges_from_states_batch(states[:, -1], adj_thresh, 
                    mask=state_mask, tool_mask=eef_mask, no_self_edge=True, topk=5)
        Rr = pad_torch(Rr, max_nR, dim=1)
        Rs = pad_torch(Rs, max_nR, dim=1)

        graph = {
            # input information
            "state": states,  # (n_his, N+M, state_dim)
            "action": states_delta,  # (N+M, state_dim)

            # attr information
            "attrs": attrs,  # (N+M, attr_dim)
            "p_instance": p_instance,  # (N, n_instance)
            "obj_mask": obj_mask,  # (N,)
            "state_mask": state_mask,  # (N+M,)
            "eef_mask": eef_mask,  # (N+M,)

            "Rr": Rr,  # (bsz, max_nR, N)
            "Rs": Rs,  # (bsz, max_nR, N)
        }

        # rollout
        for ai in range(1, 1 + action_repeat[:, li].max().item()):
            # print(f"rollout iter {i}")
            pred_state, pred_motion = model(**graph)

            repeat_mask = (action_repeat[:, li] == ai)
            pred_state_seq[repeat_mask, li] = pred_state[repeat_mask, :, :].clone()

            z_cur = pred_state[:, :, 2].min(dim=1).values
            eef_kp_cur = graph['state'][:, -1, max_nobj : max_nobj + eef_kp_num] + graph['action'][:, max_nobj : max_nobj + eef_kp_num]

            eef_kp_cur[:, 0, 2] = z_cur

            states_cur = torch.cat([pred_state, eef_kp_cur], dim=1)
            Rr, Rs = construct_edges_from_states_batch(states_cur, adj_thresh, 
                        mask=graph['state_mask'], tool_mask=graph['eef_mask'], no_self_edge=True, topk=5)
            Rr = pad_torch(Rr, max_nR, dim=1)
            Rs = pad_torch(Rs, max_nR, dim=1)

            state_history = torch.cat([graph['state'][:, 1:], states_cur[:, None]], dim=1)

            new_graph = {
                "state": state_history,  # (bsz, n_his, N+M, state_dim)
                "action": graph["action"],  # (bsz, N+M, state_dim)
                
                "Rr": Rr,  # (bsz, n_rel, N+M)
                "Rs": Rs,  # (bsz, n_rel, N+M)
                
                "attrs": graph["attrs"],  # (bsz, N+M, attr_dim)
                "p_instance": graph["p_instance"],  # (bsz, N, n_instance)
                "obj_mask": graph["obj_mask"],  # (bsz, N)
                "eef_mask": graph["eef_mask"],  # (bsz, N+M)
                "state_mask": graph["state_mask"],  # (bsz, N+M)
            }

            graph = new_graph

    out = {
        "state_seqs": pred_state_seq,  # (bsz, n_look_forward, max_nobj, 3)
        "action_seqs": decoded_action,  # (bsz, n_look_forward, action_dim)
    }
    time1 = time.time()
    print(f"dynamics time {time1 - time0}")
    return out


def running_cost(state, action, state_cur, target_state=None, bounding_box=None):  # tabletop coordinates
    # chamfer distance
    # state: (bsz, n_look_forward, max_nobj, 3)
    # action: (bsz, n_look_forward, action_dim)
    # target_state: numpy.ndarray (n_target, 3)
    # state_cur: (max_nobj, 3)
    bsz = state.shape[0]
    n_look_forward = state.shape[1]

    state_flat = state.reshape(bsz * n_look_forward, state.shape[2], state.shape[3])

    target_state = target_state[None].repeat(bsz * n_look_forward, 1, 1).to(state.device)  # (bsz * n_look_forward, n_target, 3)
    chamfer_distance = chamfer(state_flat, target_state).reshape(bsz, n_look_forward)

    x_start = action[:, :, 0]
    y_start = action[:, :, 1]
    action_point_2d = torch.stack([x_start, y_start], dim=-1)  # (bsz, n_look_forward, 2)
    state_2d = torch.cat([state_cur[:, :2][None, None].repeat(bsz, 1, 1, 1),
                            state[:, :-1, :, :2]], dim=1)  # (bsz, n_look_forward, max_nobj, 2)
    action_state_distance = torch.norm(action_point_2d[:, :, None] - state_2d, dim=-1).min(dim=-1).values  # (bsz, n_look_forward)
    pusher_size = 0.01  # hard code: 1cm
    action_state_distance = torch.maximum(action_state_distance - pusher_size, torch.zeros_like(action_state_distance))  # (bsz, n_look_forward)
    collision_penalty = torch.exp(-action_state_distance * 100.)  # (bsz, n_look_forward) 

    bbox = bounding_box[:, :2]  # (2, 2) only take x-y plane
    xmax = state.max(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    xmin = state.min(dim=2).values[:, :, 0]  # (bsz, n_look_forward)
    ymax = state.max(dim=2).values[:, :, 1]  # (bsz, n_look_forward)
    ymin = state.min(dim=2).values[:, :, 1]  # (bsz, n_look_forward)

    box_penalty = torch.stack([
        torch.maximum(xmin - bbox[0, 0], torch.zeros_like(xmin)),
        torch.maximum(bbox[0, 1] - xmax, torch.zeros_like(xmax)),
        torch.maximum(ymin - bbox[1, 0], torch.zeros_like(ymin)),
        torch.maximum(bbox[1, 1] - ymax, torch.zeros_like(ymax)),
    ], dim=-1)  # (bsz, n_look_forward, 4)
    box_penalty = torch.exp(-box_penalty * 100.).max(dim=-1).values  # (bsz, n_look_forward)

    reward = -chamfer_distance[:, -1] - 5. * collision_penalty.mean(dim=1) - 5. * box_penalty.mean(dim=1)  # (bsz,)

    print(f'min chamfer distance {chamfer_distance[:, -1].min().item()}, max reward {reward.max().item()}')
    out = {
        "reward_seqs": reward,
    }
    return out


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--resume', action='store_true')
    arg_parser.add_argument('--seed', type=int, default=43)
    arg_parser.add_argument('--config', type=str, default='config/dog_0522.yaml')
    arg_parser.add_argument('--epoch', type=str, default=50)
    args = arg_parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    use_robot = True
    use_gripper = False

    exposure_time = 5
    env = RealEnv(
        use_camera=True,
        WH=[1280, 720],
        obs_fps=5,
        n_obs_steps=1,
        use_robot=use_robot,
        speed=50,
        gripper_enable=use_gripper,
    )

    env.start(exposure_time=exposure_time)
    if use_robot:
        env.reset_robot()
    print('env started')
    time.sleep(exposure_time)
    print('start recording')

    env.calibrate(re_calibrate=False)

    bbox = env.get_bbox()
    action_lower_lim = [
        bbox[0, 0], # x min
        bbox[1, 0],  # y min
        -math.pi,  # theta min
        10,  # length min
    ]
    action_upper_lim = [
        bbox[0, 1], # x max
        bbox[1, 1],  # y max
        math.pi,  # theta max
        20,  # length max
    ]
    action_lower_lim = torch.tensor(action_lower_lim, dtype=torch.float32, device=device)
    action_upper_lim = torch.tensor(action_upper_lim, dtype=torch.float32, device=device)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)

    epoch = args.epoch
    train_config = config['train_config']
    model_config = config['model_config']
    dataset_config = config['dataset_config']['datasets'][0]
    fps_radius = (dataset_config['fps_radius_range'][0] + dataset_config['fps_radius_range'][1]) / 2
    adj_thresh = (dataset_config['adj_radius_range'][0] + dataset_config['adj_radius_range'][1]) / 2

    set_seed(args.seed)

    run_name = train_config['out_dir'].split('/')[-1]
    save_dir = f"vis/planning-{run_name}-model_{epoch}"
    if not args.resume and os.path.exists(save_dir) and len(glob.glob(os.path.join(save_dir, '*.npz'))) > 0:
        print('save dir already exists')
        env.stop()
        print('env stopped')
        return
    os.makedirs(save_dir, exist_ok=True)
    if args.resume:
        print('resume')
        n_resume = len(glob.glob(os.path.join(save_dir, 'interaction_*.npz')))
    else:
        n_resume = 0
    print('starting from iteration {}'.format(n_resume))
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(train_config['out_dir'], 'checkpoints', 'model_{}.pth'.format(epoch))

    model_config['n_his'] = train_config['n_his']

    model = DynamicsPredictor(model_config, device)

    model.to(device)

    model.eval()
    model.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))

    # construct target
    pcd = o3d.io.read_point_cloud("vis_real_world/target.pcd")
    target_state = np.array(pcd.points)
    target_state = torch.tensor(target_state, dtype=torch.float32, device=device)  # (N, 3)
    running_cost_func = partial(running_cost, target_state=target_state, bounding_box=env.get_bbox())

    n_actions = 10  # total action steps
    n_look_ahead = 1  # look forward horizon size
    n_sample = 10000
    n_sample_chunk = 1000

    n_chunk = np.ceil(n_sample / n_sample_chunk).astype(int)

    noise_level = 1.0
    reward_weight = 500.0
    planner_config = {
        'action_dim': len(action_lower_lim),
        'model_rollout_fn': partial(dynamics, model=model, device=device, adj_thresh=adj_thresh,
                                    push_length=env.push_length,
                                    max_n=dataset_config['datasets'][0]['max_nobj'],
                                    max_nR=dataset_config['datasets'][0]['max_nR'],
                                    n_his=train_config['n_his']),
        'evaluate_traj_fn': running_cost_func,
        'sampling_action_seq_fn': partial(sample_action_seq, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim, 
                                        n_sample=min(n_sample, n_sample_chunk), device=device, noise_level=noise_level, push_length=env.push_length),
        'clip_action_seq_fn': partial(clip_actions, action_lower_lim=action_lower_lim, action_upper_lim=action_upper_lim),
        'optimize_action_mppi_fn': partial(optimize_action_mppi, reward_weight=reward_weight, action_lower_lim=action_lower_lim, 
                                        action_upper_lim=action_upper_lim, push_length=env.push_length),
        'n_sample': min(n_sample, n_sample_chunk),
        'n_look_ahead': n_look_ahead,
        'n_update_iter': 1,
        'reward_weight': reward_weight,
        'action_lower_lim': action_lower_lim,
        'action_upper_lim': action_upper_lim,
        'planner_type': 'MPPI',
        'device': device,
        'verbose': False,
        'noise_level': noise_level,
        'rollout_best': True,
    }
    planner = Planner(planner_config)
    planner.total_chunks = n_chunk

    act_seq = torch.rand((planner_config['n_look_ahead'], action_upper_lim.shape[0]), device=device) * \
                (action_upper_lim - action_lower_lim) + action_lower_lim

    res_act_seq = torch.zeros((n_actions, action_upper_lim.shape[0]), device=device)

    if n_resume > 0:
        interaction_list = sorted(glob.glob(os.path.join(save_dir, 'interaction_*.npz')))
        for i in range(n_resume):
            res = np.load(interaction_list[i])
            act_save = res['act']
            state_init_save = res['state_init']
            state_pred_save = res['state_pred']
            state_real_save = res['state_real']
            res_act_seq[i] = torch.tensor(act_save, dtype=torch.float32, device=device)

    chamfer_distance_seq = []
    n_steps = n_actions
    for i in range(n_resume, n_actions):
        time1 = time.time()
        # get state
        state_cur, obj_pcd, rgb_vis, intr, extr = get_state_cur(env, device, fps_radius=fps_radius, 
                                                                max_nobj=dataset_config['datasets'][0]['max_nobj'],
                                                                max_neef=dataset_config['datasets'][0]['max_neef'], 
                                                                visualize_img=True)

        if i == 0:
            chamfer_dist = chamfer(torch.from_numpy(obj_pcd)[None].to(device), target_state[None]).item()
            print('chamfer distance', chamfer_dist)
            chamfer_distance_seq.append(chamfer_dist)

        env.update_state(state_cur.detach().cpu().numpy())
        # get action
        res_all = []
        for ci in range(n_chunk):
            planner.chunk_id = ci
            res = planner.trajectory_optimization(state_cur, act_seq)
            for k, v in res.items():
                res[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
            res_all.append(res)
        res = planner.merge_res(res_all)

        # vis
        visualize_img(state_cur, res, rgb_vis, obj_pcd, intr, extr, adj_thresh,
                target_state=target_state, state_after=None, save_dir=save_dir, postfix=f'{i}_0', 
                topk=dataset_config['datasets'][0]['topk'], 
                connect_all=dataset_config['datasets'][0]['connect_all'], 
                push_length=env.push_length)

        # step state
        if use_gripper:
            env.step_gripper(res['act_seq'][0].detach().cpu().numpy())
        else:
            env.step(res['act_seq'][0].detach().cpu().numpy())

        # update action
        res_act_seq[i] = res['act_seq'][0].detach().clone()
        act_seq = torch.cat(
            [
                res['act_seq'][1:],
                torch.rand((1, action_upper_lim.shape[0]), device=device) * (action_upper_lim - action_lower_lim) + action_lower_lim
            ], 
            dim=0
        )
        n_look_ahead = min(n_actions - i, planner_config['n_look_ahead'])
        act_seq = act_seq[:n_look_ahead]  # sliding window
        planner.n_look_ahead = n_look_ahead

        # save
        save = True
        if save:
            act_save = res['act_seq'][0].detach().cpu().numpy()  # (action_dim,)
            act_all_save = res['act_seq'].detach().cpu().numpy()  # (n_look_forward, action_dim)
            state_init_save = state_cur.detach().cpu().numpy()  # (max_nobj, 3)
            state_pred_save = res['best_model_output']['state_seqs'][0, 0].detach().cpu().numpy()  # (max_nobj, 3)
            state_pred_all_save = res['best_model_output']['state_seqs'][0].detach().cpu().numpy()  # (n_look_forward, max_nobj, 3)
            state_real, pcd_real, rgb_vis, _, _ = get_state_cur(env, device, fps_radius=fps_radius, 
                                                                max_nobj=dataset_config['datasets'][0]['max_nobj'],
                                                                max_neef=dataset_config['datasets'][0]['max_neef'], 
                                                                visualize_img=True)
            state_real_save = state_real.detach().cpu().numpy()
            assert act_all_save.shape[0] == n_look_ahead
            assert state_pred_all_save.shape[0] == n_look_ahead
            np.savez(
                os.path.join(save_dir, f'interaction_{i}.npz'),
                act=act_save,
                act_all=act_all_save,
                state_pred=state_pred_save,
                state_pred_all=state_pred_all_save,
                pcd_real=pcd_real,
                state_real=state_real_save,
                state_init=state_init_save,
            )

            chamfer_dist = chamfer(torch.from_numpy(pcd_real)[None].to(device), target_state[None]).item()
            print('chamfer distance', chamfer_dist)
            chamfer_distance_seq.append(chamfer_dist)

            # vis
            visualize_img(state_cur, res, rgb_vis, obj_pcd, intr, extr, adj_thresh,
                    target_state=target_state, state_after=state_real, save_dir=save_dir, postfix=f'{i}_1',
                    topk=dataset_config['datasets'][0]['topk'], 
                    connect_all=dataset_config['datasets'][0]['connect_all'], 
                    push_length=env.push_length)

            
            time2 = time.time()
            print(f"step {i} time {time2 - time1}")

    print(f"final action sequence {res_act_seq}")
    print(f"final chamfer distance sequence {chamfer_distance_seq}")

    with open(os.path.join(save_dir, 'stats.txt'), 'w') as f:
        f.write(f"final action sequence {res_act_seq}\n")
        f.write(f"final chamfer distance sequence {chamfer_distance_seq}\n")

    env.stop()
    print('env stopped')

    # make video with cv2
    result = cv2.VideoWriter(
        os.path.join(save_dir, 'rgb_vis.mp4'), 
        cv2.VideoWriter_fourcc(*'mp4v'), 1, (1280, 720))

    for i in range(n_steps):
        rgb_vis = cv2.imread(os.path.join(save_dir, f'rgb_vis_{i}_0.png'))
        result.write(rgb_vis)
        rgb_vis = cv2.imread(os.path.join(save_dir, f'rgb_vis_{i}_1.png'))
        result.write(rgb_vis)

    result.release()
    print('video saved')


if __name__ == '__main__':
    with torch.no_grad():
        main()
