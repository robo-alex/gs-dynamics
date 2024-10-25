import argparse
import numpy as np
import torch
import time
import os
import yaml

from utils.real_env import RealEnv
from utils.perception_module import PerceptionModule
from utils.pcd_utils import visualize_o3d
from gs.trainer import GSTrainer
from render.dynamics_module import DynamicsModule

def init_env():
    use_robot = True
    use_gripper = False
    exposure_time = 5
    env = RealEnv(
        use_camera=True,
        WH=[640, 480],
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
    env.calibrate(re_calibrate=False, visualize=False)
    return env

def main(args, config, gs_config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_config = config['train_config']
    epoch = args.epoch
    run_name = train_config['out_dir'].split('/')[-1]
    save_dir = f"output/gs-{run_name}-model_{epoch}"
    os.makedirs(save_dir, exist_ok=True)
    vis_path = os.path.join(save_dir, 'env_vis')
    os.makedirs(vis_path, exist_ok=True)
    gs_vis_path = os.path.join(save_dir, 'gs_vis')
    os.makedirs(gs_vis_path, exist_ok=True)

    visualize = True
    
    try:
        env = init_env()
        pm = PerceptionModule(vis_path,device)
        dm = DynamicsModule(config, args.epoch, device)
        gs_trainer = GSTrainer(gs_config, device)
        
        for trial in range(10):
            vis_dir_train = os.path.join(gs_vis_path, f'trial_{trial}_train')
            os.makedirs(vis_dir_train, exist_ok=True)
            vis_dir_rollout = os.path.join(gs_vis_path, f'trial_{trial}_rollout')
            os.makedirs(vis_dir_rollout, exist_ok=True)
            pcd, imgs, masks = pm.get_tabletop_points_env(env, obj_names=['rope'], return_imgs=True)
            if visualize:
                visualize_o3d([pcd])
            
            print('training Gaussian Splatting ...')
            gs_trainer.clear()
            gs_trainer.update_state_env(pcd, env, imgs, masks)
            gs_trainer.train(vis_dir=vis_dir_train)
            
            print('executing action ...')
            act = torch.tensor([[0.2, 0.0, 0.0], [0.4, 0.0, 0.0]]).to(device)
            gs_trainer.rollout_and_render(dm, act, vis_dir=vis_dir_rollout)
            env.step(act.detach().cpu().numpy(), decoded=True)

    finally:
        env.stop()
        print('env stopped')

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--resume', action='store_true')
    arg_parser.add_argument('--seed', type=int, default=42)
    arg_parser.add_argument('--config', type=str, default='config/rope.yaml')
    arg_parser.add_argument('--gs_config', type=str, default='config/gs/default.yaml')
    arg_parser.add_argument('--epoch', type=str, default='latest')
    args = arg_parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    with open(args.gs_config, 'r') as f:
        gs_config = yaml.load(f, Loader=yaml.CLoader)
    main(args, config, gs_config)
