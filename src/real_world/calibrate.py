import cv2
import argparse
import time
import numpy as np
import torch
import open3d as o3d
from PIL import Image

from utils.real_env import RealEnv
from utils.pcd_utils import get_tabletop_points, visualize_o3d

def construct_goal_from_perception():
    use_robot = True

    exposure_time = 10
    env = RealEnv(
        WH=[1280, 720],
        obs_fps=15,
        n_obs_steps=2,
        use_robot=use_robot,
        speed=100,
    )

    try:
        env.start(exposure_time=exposure_time)
        if use_robot:
            env.reset_robot()
        print('env started')
        time.sleep(exposure_time)
        print('start recording')

        env.calibrate(re_calibrate=False)

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

        pcd = get_tabletop_points(rgb_list, depth_list, R_list, t_list, intr_list, bbox)

        visualize_o3d([pcd])
        o3d.io.write_point_cloud("vis_real_world/target.pcd", pcd)

    finally:
        env.stop()
        print('env stopped')


def calibrate(use_robot=True, reset_robot=True, wrist=None):
    exposure_time = 5
    env = RealEnv(
        use_camera=True,
        WH=[1280, 720],
        obs_fps=5,
        n_obs_steps=2,
        use_robot=use_robot,
        speed=100,
        wrist=wrist,
    )

    try:
        env.start(exposure_time=exposure_time)
        if use_robot and reset_robot:
            env.reset_robot()
        print('env started')
        time.sleep(exposure_time)
        print('start recording')

        env.calibrate(re_calibrate=True)
    
    finally:
        env.stop()
        print('env stopped')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calibrate_fixed", action="store_true")
    parser.add_argument("--construct_goal", action="store_true")
    parser.add_argument("--examine_points", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        calibrate(reset_robot=False)
    elif args.calibrate_fixed:  # only calibrate fixed cameras
        calibrate(use_robot=False)
    elif args.construct_goal:
        construct_goal_from_perception()
    else:
        print("No arguments provided")
