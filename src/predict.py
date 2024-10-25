import torch
import os
import subprocess
import copy
import numpy as np
import cv2
import glob
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

from render.renderer import Renderer
from render.dynamics_module import DynamicsModule
from render.utils import Visualizer, project, convert_opencv_to_opengl
from data.utils import rgba_to_rgb


def load_valid_paths(dataset):
    print(f'Setting up dataset {dataset["name"]}')

    base_dir = Path(dataset['base_dir'])
    data_dir = base_dir / 'data'  # initial data such as rgbd, etc.
    output_dir = base_dir / 'ckpts' # dataset['output_dir']  # 3DGS model output
    preprocess_dir = base_dir / 'preprocessed'  # to save preprocessed data

    name = dataset['name']
    exp_data_dir = data_dir / f'{name}'
    exp_output_dir = output_dir / f'exp_{name}'
    exp_preprocess_dir = preprocess_dir / f'exp_{name}'

    episodes = sorted(glob.glob(str(exp_output_dir / 'episode_*')))
    episode_idxs = [int(epi.split('_')[-1]) for epi in episodes]
    
    episodes = [epi for epi, idx in zip(episodes, episode_idxs) \
            if os.path.exists(os.path.join(epi, name, f'episode_{idx:02d}', 'params.npz'))]
    episode_idxs = [int(epi.split('_')[-1]) for epi in episodes]

    n_episodes = len(episode_idxs)
    print(f'Found {n_episodes} episodes')

    if n_episodes == 0:
        no_episodes_ver = True
        episode_idxs = [0]
        n_episodes = 1
        print('Using entire dataset as one episode')
    else:
        no_episodes_ver = False
        # using valid portion
        episodes = episodes[int(len(episodes) * 0.8):]
        episode_idxs = episode_idxs[int(len(episode_idxs) * 0.8):]
        n_episodes = len(episode_idxs)
        print(f'Using episodes: {episode_idxs}')

    data_dirs = {}
    output_dirs = {}
    preprocess_dirs = {}
    for episode_idx in episode_idxs:

        if no_episodes_ver:
            assert n_episodes == 1
            data_dirs[episode_idx] = exp_data_dir
            output_dirs[episode_idx] = os.path.join(exp_output_dir, name)
            preprocess_dirs[episode_idx] = exp_preprocess_dir
        else:
            data_dirs[episode_idx] = os.path.join(exp_data_dir, f'episode_{episode_idx:02d}')
            output_dirs[episode_idx] = os.path.join(exp_output_dir, f'episode_{episode_idx:02d}', 
                                                    name, f'episode_{episode_idx:02d}')
            preprocess_dirs[episode_idx] = os.path.join(exp_preprocess_dir, f'episode_{episode_idx:02d}')

    return data_dirs, output_dirs


@torch.no_grad()
def predict(config, epoch):
    train_config = config['train_config']
    dataset_config = config['dataset_config']['datasets'][0]

    run_name = train_config['out_dir'].split('/')[-1]
    save_dir = f"output/render-{run_name}-model_{epoch}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dynamics_module = DynamicsModule(config, epoch, device)
    renderer = Renderer(device)

    data_paths, output_paths = load_valid_paths(dataset_config)

    for episode_idx in data_paths.keys():

        data_path = data_paths[episode_idx]
        output_path = output_paths[episode_idx]

        print(f'Collecting scene data for episode {episode_idx}')
        scene_data, vis_data = dynamics_module.collect_scene_data(data_path, output_path)
        print(f'Collected {len(scene_data)} frames')

        save_dir_episode = os.path.join(save_dir, f'episode_{episode_idx:02d}')
        os.makedirs(save_dir_episode, exist_ok=True)

        for num_cam in range(4):
            print(f'Rendering episode {episode_idx}, camera {num_cam}')
            cam_params = np.load(os.path.join(data_path, f'camera_{num_cam}/camera_params.npy'))
            w2c = np.load(os.path.join(data_path, f'camera_{num_cam}/camera_extrinsics.npy'))
            save_dir_cam = os.path.join(save_dir_episode, f'camera_{num_cam}')
            os.makedirs(save_dir_cam, exist_ok=True)

            w2c = np.linalg.inv(w2c)
            w2c = convert_opencv_to_opengl(w2c)

            fx, fy, cx, cy = cam_params[0], cam_params[1], cam_params[2], cam_params[3]
            k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

            visualizer = Visualizer()

            for i in tqdm(range(len(scene_data)), dynamic_ncols=True):
                im, depth = renderer.render(w2c, k, scene_data[i], bg=[0.0, 0.0, 0.0])
                im = im.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255
                im = im.copy().astype(np.uint8)

                scene_mask_data_i = copy.deepcopy(scene_data[i])
                scene_mask_data_i['colors_precomp'] = torch.ones_like(scene_data[i]['colors_precomp'])
                mask, _ = renderer.render(w2c, k, scene_mask_data_i, bg=[0.0, 0.0, 0.0])
                mask = mask.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]

                im = cv2.cvtColor(im / (mask + 1e-4), cv2.COLOR_RGB2RGBA)
                # Then assign the mask to the last channel of the image
                im[:, :, 3] = mask.mean(axis=2) * 255

                # if provide background image, specify here
                # im_bg = cv2.imread('render/bg.png')
                # im = im * mask + im_bg * (1 - mask)

                cv2.imwrite(os.path.join(save_dir_cam, f"im_{i:06}.png"), rgba_to_rgb(im))
                
                kp_vis = vis_data[i]['kp']  # (n, 3)
                kp_vis = kp_vis[kp_vis.sum(axis=1) != 0]
                kp_proj = project(kp_vis, intr=k, extr=w2c)

                tool_kp_vis = vis_data[i]['tool_kp']  # (1, 3)
                tool_kp_proj = project(tool_kp_vis, intr=k, extr=w2c)

                im = visualizer.draw_keypoints(im, tool_kp_proj, tool_kp_vis, kp_proj)
                cv2.imwrite(os.path.join(save_dir_cam, f"im_traj_{i:06}.png"), rgba_to_rgb(im))

            frame_rate = 15
            width, height = 1280, 720

            traj_video_name = os.path.join(save_dir_episode, f'render_traj_{num_cam}.mp4')
            subprocess.run([
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel', 'error',
                '-framerate', str(frame_rate),
                '-i', f'{save_dir_cam}/im_traj_*.png',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                traj_video_name
            ])
            print(f'Saved video to {traj_video_name}')

        del scene_data
        del vis_data
        torch.cuda.empty_cache()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/debug.yaml')
    arg_parser.add_argument('--epoch', type=str, default=100)
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    
    if 'ablation' in args.config.split('/'):
        with open('ablation.yaml', 'r') as f:
            ablation_memo = yaml.load(f, Loader=yaml.CLoader)
        assert os.path.abspath(args.config) in ablation_memo['config_list']
        index = ablation_memo['config_list'].index(os.path.abspath(args.config))
        config['train_config']['out_dir'] = ablation_memo['log_list'][index]

        ckpts = os.path.join(config['train_config']['out_dir'], 'checkpoints')
        epochs = [int(ckpt.split('_')[-1].split('.')[0]) for ckpt in os.listdir(ckpts) if 'model' in ckpt]
        args.epoch = np.max(epochs)

    elif 'dynamics' in args.config.split('/'):
        out_dir = config['train_config']['out_dir'].split('/')
        for i, d in enumerate(out_dir):
            if d == 'log':
                out_dir = '/'.join(['..', 'dynamics', *out_dir[i:]])
                break
        config['train_config']['out_dir'] = out_dir

    predict(config, args.epoch)
