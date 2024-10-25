import os
from pathlib import Path
import torch
import numpy as np
import glob
from PIL import Image
import cv2
import pickle as pkl
import json
import yaml
import argparse
import open3d as o3d
from dgl.geometry import farthest_point_sampler

from data.utils import label_colormap, opengl2cam
from real_world.utils.pcd_utils import depth2fgpcd, rpy_to_rotation_matrix


def get_eef_points(xyz, rpy, calib):
    R_gripper2base = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
    t_gripper2base = np.array(xyz) / 1000

    gripper_point = np.array([[0.0, 0.0, 0.18]])  # gripper

    R_base2world = calib['R_base2world']
    t_base2world = calib['t_base2world']
    R_gripper2world = R_base2world @ R_gripper2base
    t_gripper2world = R_base2world @ t_gripper2base + t_base2world
    gripper_points_in_world = R_gripper2world @ gripper_point.T + t_gripper2world[:, np.newaxis]
    gripper_points_in_world = gripper_points_in_world.T

    return gripper_points_in_world[0]  # only one point


def test_validity(data_dir, output_dir):  # test if the camera recording is problematic
    params_dir = os.path.join(output_dir, 'params.npz')
    if not os.path.exists(params_dir):
        raise ValueError(f'Params dir {params_dir} not found')

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

    if len(json_data) - num_frames < -10:
        print(f'warning: in data_dir {data_dir}, json_data length {len(json_data)}, num_frames {num_frames}')
        return False
    return True

def extract_pushes(data_dir, output_dir, save_dir, dist_thresh, n_his, n_future, episode_idx=0):
    # use overlapping samples
    # provide canonical frame info
    # compatible to other data layouts (make a general episode list)
    frame_idx_dir = os.path.join(save_dir, 'frame_pairs')
    os.makedirs(frame_idx_dir, exist_ok=True)

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
        # print(f'warning: json_data length {len(json_data)}, num_frames {num_frames}')
        # deal with camera frame recording mismatch
        json_data = [json_data[0]] * (max(frame_idx_lists) + 1 - len(json_data)) + json_data
    
    if len(json_data) - num_frames > 10:
        json_data = json_data[:num_frames]

    joint_angles = []
    poses = []
    for frame_idx in range(len(frame_idx_lists)):
        try:
            actions = json.loads(json_data[frame_idx_lists[frame_idx]])
        except:
            # import ipdb; ipdb.set_trace()
            actions = json.loads(json_data[-1])
        joint_angles.append(actions['joint_angles'])
        poses.append(actions['pose'])
    joint_angles = np.array(joint_angles)
    poses = np.array(poses)

    with open(os.path.join(data_dir, "calibration_handeye_result.pkl"), "rb") as f:
        calib = pkl.load(f)

    eef_xyz = poses[:, :3]
    eef_xyz[:, 1] -= 1.0  # slightly adjust calibration error
    eef_rpy = poses[:, 3:]

    # generate pairs
    frame_idxs = []
    dists = []

    # get start-end pairs
    cnt = 0
    for curr_frame in range(num_frames):

        first_frame = 0
        end_frame = num_frames

        eef_particles_curr = get_eef_points(eef_xyz[curr_frame], eef_rpy[curr_frame], calib)

        frame_traj = [curr_frame]
        dist_traj = []

        # search backward
        fi = curr_frame
        while fi >= first_frame:
            eef_particles_fi = get_eef_points(eef_xyz[fi], eef_rpy[fi], calib)
            x_curr = eef_particles_curr[0]
            y_curr = eef_particles_curr[1]
            z_curr = eef_particles_curr[2]
            x_fi = eef_particles_fi[0]
            y_fi = eef_particles_fi[1]
            z_fi = eef_particles_fi[2]
            dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (y_curr - y_fi) ** 2 + (z_curr - z_fi) ** 2)
            if dist_curr >= dist_thresh:
                frame_traj.append(fi)
                dist_traj.append(dist_curr)
                eef_particles_curr = eef_particles_fi
            fi -= 1
            if len(frame_traj) == n_his:
                break
        else:
            # pad to n_his
            curr_len = len(frame_traj)
            frame_traj = frame_traj + [frame_traj[-1]] * (n_his - curr_len)
            dist_traj = dist_traj + [0] * (n_his - curr_len)

        frame_traj = frame_traj[::-1]
        dist_traj = dist_traj[::-1]
        fi = curr_frame
        eef_particles_curr = get_eef_points(eef_xyz[curr_frame], eef_rpy[curr_frame], calib)
        
        # search forward
        while fi < end_frame:
            eef_particles_fi = get_eef_points(eef_xyz[fi], eef_rpy[fi], calib)
            x_curr = eef_particles_curr[0]
            y_curr = eef_particles_curr[1]
            z_curr = eef_particles_curr[2]
            x_fi = eef_particles_fi[0]
            y_fi = eef_particles_fi[1]
            z_fi = eef_particles_fi[2]
            dist_curr = np.sqrt((x_curr - x_fi) ** 2 + (y_curr - y_fi) ** 2 + (z_curr - z_fi) ** 2)

            if dist_curr >= dist_thresh or (fi == end_frame - 1 and dist_curr >= 0.75 * dist_thresh):
                frame_traj.append(fi)
                dist_traj.append(dist_curr)
                eef_particles_curr = eef_particles_fi

            fi += 1
            if len(frame_traj) == n_his + n_future:
                break
        else:
            # When assuming quasi-static, we can pad to n_his + n_future
            curr_len = len(frame_traj)
            frame_traj = frame_traj + [frame_traj[-1]] * (n_his + n_future - curr_len)
            dist_traj = dist_traj + [0] * (n_his + n_future - curr_len)
        
        cnt += 1

        frame_idxs.append(frame_traj)
        dists.append(dist_traj)

        # push_centered
        if curr_frame == end_frame - 1:
            frame_idxs = np.array(frame_idxs)
            np.savetxt(os.path.join(frame_idx_dir, f'{episode_idx}.txt'), frame_idxs, fmt='%d')
            print(f'episode {episode_idx} has {cnt} unit pushes')
            frame_idxs = []
            dists = np.array(dists)
            np.savetxt(os.path.join(frame_idx_dir, f'dist_{episode_idx}.txt'), dists, fmt='%f')
            dists = []


def downsample(output_dir):
    n_downsample = 1000

    params_dir = os.path.join(output_dir, 'params.npz')
    if not os.path.exists(params_dir):
        raise ValueError(f'Params dir {params_dir} not found')
    params = np.load(params_dir)
    xyz = params['means3D']  # n_frames, n_particles, 3

    opacity_mask = (params['logit_opacities'] > 0)[:, 0]
    xyz = xyz[:, opacity_mask]

    xyz_motion = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    xyz_motion_sum = np.sum(xyz_motion, axis=0)
    
    def detect_outliers(data, m):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d / mdev if mdev else np.zeros(len(d))
        return s < m
    
    valid_mask = detect_outliers(xyz_motion_sum, m=3)
    xyz = xyz[:, valid_mask]

    xyz_tensor = torch.from_numpy(xyz).float()
    fps_idx = farthest_point_sampler(xyz_tensor[0:1], n_downsample, start_idx=0)[0]
    xyz_tensor = xyz_tensor[:, fps_idx]
    xyz = xyz_tensor.numpy()

    # trajectory smoothing
    for _ in range(10):
        xyz[1:-1] = (xyz[:-2] + xyz[1:-1] + xyz[2:]) / 3

    np.save(os.path.join(output_dir, f"param_downsampled.npy"), xyz)
    print(f"Downsampled {xyz.shape[1]} points")


def preprocess(config):
    base_dir = Path(config['dataset_config']['datasets'][0]['base_dir'])
    data_dir = base_dir / 'data'  # initial data such as rgbd, etc.
    output_dir = base_dir / 'ckpts'  # 3DGS model output
    preprocess_dir = base_dir / 'preprocessed'  # to save preprocessed data

    name = config['dataset_config']['datasets'][0]['name']  # 'episodes_giraffe_0521'
    exp_data_dir = data_dir / f'{name}'
    exp_output_dir = output_dir / f'exp_{name}'
    exp_preprocess_dir = preprocess_dir / f'exp_{name}'

    dist_thresh = config['train_config']['dist_thresh']
    n_his = config['train_config']['n_his']
    n_future = config['train_config']['n_future']
    
    print('dist_thresh', dist_thresh)
    
    episodes = sorted(glob.glob(str(exp_output_dir / 'episode_*')))
    episode_idxs = [int(epi.split('_')[-1]) for epi in episodes]
    
    episodes = [epi for epi, idx in zip(episodes, episode_idxs) \
            if os.path.exists(os.path.join(epi, name, f'episode_{idx:02d}', 'params.npz'))]
    episode_idxs = [int(epi.split('_')[-1]) for epi in episodes]

    n_episodes = len(episode_idxs)
    print(f'Processing {n_episodes} episodes')

    if n_episodes == 0:
        no_episodes_ver = True
        episode_idxs = [0]
        n_episodes = 1
        exp_output_dir = output_dir / f'exp_{name}'
    else:
        no_episodes_ver = False

    for episode_idx in episode_idxs:
        print(f'Processing episode {episode_idx}')

        if no_episodes_ver:
            assert n_episodes == 1
            epi_data_dir = exp_data_dir
            epi_output_dir = exp_output_dir / name
            epi_preprocess_dir = exp_preprocess_dir
        else:
            epi_data_dir = exp_data_dir / f'episode_{episode_idx:02d}'
            epi_output_dir = exp_output_dir / f'episode_{episode_idx:02d}' / name / f'episode_{episode_idx:02d}'
            epi_preprocess_dir = exp_preprocess_dir / f'episode_{episode_idx:02d}'

        if not test_validity(epi_data_dir, epi_output_dir):
            print(f'Episode {episode_idx} is invalid')
            continue

        os.makedirs(epi_preprocess_dir, exist_ok=True)

        extract_pushes(epi_data_dir, epi_output_dir, epi_preprocess_dir, 
                dist_thresh=dist_thresh, n_his=n_his, n_future=n_future, episode_idx=episode_idx)
        try:
            downsample(epi_output_dir)
        except:
            print(f'Failed to downsample episode {episode_idx}')
            os.system(f'rm -r {epi_preprocess_dir}')
            continue

        # save metadata
        os.makedirs(epi_preprocess_dir, exist_ok=True)
        with open(os.path.join(epi_preprocess_dir, 'metadata.txt'), 'w') as f:
            f.write(f'{dist_thresh},{n_future},{n_his}')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/debug.yaml')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    
    preprocess(config)
