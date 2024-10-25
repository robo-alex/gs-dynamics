import argparse
import numpy as np
import gradio as gr
import torch
import torch.nn.functional as F
import open3d as o3d
import time
import cv2
import math
import os
import yaml
import shutil
import sys
import glob
import subprocess
from functools import partial
import copy
from PIL import Image

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


from utils.real_env import RealEnv
from utils.perception_module import PerceptionModule
from utils.pcd_utils import visualize_o3d
from utils.gradio_utils import (draw_mask_on_image, draw_points_on_image,
                          draw_raw_points_on_image,
                          get_latest_points_pair, get_valid_mask,
                          on_change_single_global_state)
from gs.trainer import GSTrainer, make_video
from gs.helpers import setup_camera
from gs.convert import save_to_splat
from render.dynamics_module import DynamicsModule


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

def reproject(depth, intr, extr):
    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    xx = xx.flatten()
    yy = yy.flatten()
    points = np.stack([xx, yy, depth.flatten()], axis=1)
    
    mask = depth.flatten() > 0
    mask = np.logical_and(mask, depth.flatten() < 2)
    points = points[mask]

    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    points[:, 0] = (points[:, 0] - cx) / fx * points[:, 2]
    points[:, 1] = (points[:, 1] - cy) / fy * points[:, 2]
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    inv_extr = np.linalg.inv(extr)
    points = points @ inv_extr.T
    return points[:, :3]

def click_to_xyz(click_x, click_y, intr, extr, z=-0.01):

    inv_extr = np.linalg.inv(extr)
    test_point_1 = np.array([0.0, 0.0, 0.0, 1.0]) @ inv_extr.T

    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    click_x = (click_x - cx) / fx * 1.0
    click_y = (click_y - cy) / fy * 1.0
    test_point_2 = np.array([click_x, click_y, 1.0, 1.0]) @ inv_extr.T

    test_z_1 = test_point_1[2]
    test_z_2 = test_point_2[2]

    depth_ratio = (z - test_z_1) / (test_z_2 - test_z_1)
    
    point = test_point_1 + depth_ratio * (test_point_2 - test_point_1)
    return point[:3]

def interpolate_actions(start, target, length=0.01, pad=0):
    diff = target - start
    n = int(np.linalg.norm(diff) / length)
    waypoints = np.linspace(start, target, n)
    waypoints = np.concatenate([waypoints, target[None].repeat(pad, axis=0)], axis=0)
    return waypoints


class DynamicsVisualizer:

    def __init__(self, args, config, gs_config):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        train_config = config['train_config']
        epoch = args.epoch
        run_name = train_config['out_dir'].split('/')[-1]
        save_dir = f"output/gsgradio-{run_name}-model_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        vis_path = os.path.join(save_dir, 'env_vis')
        os.makedirs(vis_path, exist_ok=True)
        gs_vis_path = os.path.join(save_dir, 'gs_vis')
        os.makedirs(gs_vis_path, exist_ok=True)

        self.save_dir = save_dir
        self.vis_path = vis_path
        self.gs_vis_path = gs_vis_path

        self.visualize_3d = False
        self.vis_cam_id = 0

        self.pm = PerceptionModule(vis_path,device)
        self.dm = DynamicsModule(config, args.epoch, device)
        self.gs_trainer = GSTrainer(gs_config, device)

        self.use_robot = True
        self.use_gripper = False
        self.exposure_time = 5
        self.env = RealEnv(
            use_camera=True,
            WH=[640, 480],
            obs_fps=5,
            n_obs_steps=1,
            use_robot=self.use_robot,
            speed=100,
            gripper_enable=self.use_gripper,
        )

        self.save_for_demo = False

    def __enter__(self):
        self.env.start(exposure_time=self.exposure_time)
        if self.use_robot:
            self.env.reset_robot()
        print('env started')
        time.sleep(self.exposure_time)
        print('start recording')
        self.env.calibrate(re_calibrate=False, visualize=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.stop()
        print('env stopped')

    def reset(self, train_gs=True):
        pcd, imgs, masks = self.pm.get_tabletop_points_env(self.env, obj_names=['rope'], return_imgs=True)
        self.imgs = imgs
        self.masks = masks
        if self.visualize_3d:
            visualize_o3d([pcd])
        
        self.gs_trainer.clear(clear_params=train_gs)
        self.gs_trainer.update_state_env(pcd, self.env, imgs, masks)
        
        vis_dir_train = os.path.join(self.gs_vis_path, f'sim_train')
        if os.path.exists(vis_dir_train):
            shutil.rmtree(vis_dir_train)
        os.makedirs(vis_dir_train, exist_ok=True)

        if train_gs:
            print('training Gaussian Splatting ...')
            self.gs_trainer.train(vis_dir=vis_dir_train)
            self.particle_pos = self.gs_trainer.params['means3D'].clone().detach().cpu().numpy()
            self.mean_z = self.particle_pos[:, 2].mean()
            self.actions = None

            if self.save_for_demo:
                obj_name = f'obj_{time.time()}'
                self.obj_name = obj_name
                curr_save_dir = os.path.join(self.save_dir, obj_name)
                os.makedirs(curr_save_dir, exist_ok=True)
                for v in range(4):
                    img_save = Image.fromarray((imgs[v]).astype(np.uint8))
                    img_save.save(os.path.join(curr_save_dir, f'img_{v}.png'))
                    save_to_splat(
                        pts=self.gs_trainer.params['means3D'].detach().cpu().numpy(),
                        colors=self.gs_trainer.params['rgb_colors'].detach().cpu().numpy(),
                        scales=torch.exp(self.gs_trainer.params['log_scales']).detach().cpu().numpy(),
                        quats=torch.nn.functional.normalize(self.gs_trainer.params['unnorm_rotations'], dim=-1).detach().cpu().numpy(),
                        opacities=torch.sigmoid(self.gs_trainer.params['logit_opacities']).detach().cpu().numpy(),
                        output_file=os.path.join(curr_save_dir, 'gs_orig.splat'),
                    )

    @torch.no_grad()
    def step_sim(self, action):  # (2, 3)
        print('executing action in sim ...')
        vis_dir_rollout = os.path.join(self.gs_vis_path, f'sim_rollout')
        if os.path.exists(vis_dir_rollout):
            shutil.rmtree(vis_dir_rollout)
        os.makedirs(vis_dir_rollout, exist_ok=True)
        self.params_prev = copy.deepcopy(self.gs_trainer.params)
        rendervar_list, visvar_list = self.gs_trainer.rollout_and_render(self.dm, action, vis_dir=vis_dir_rollout)

        if self.save_for_demo:
            for v in range(4):
                im_list = self.render(rendervar_list, visvar_list, cam_id=v)
                vis_dir_rollout_temp = os.path.join(self.gs_vis_path, f'sim_rollout_temp_{v}')
                if os.path.exists(vis_dir_rollout_temp):
                    shutil.rmtree(vis_dir_rollout_temp)
                os.makedirs(vis_dir_rollout_temp, exist_ok=True)
                for i, im in enumerate(im_list):
                    cv2.imwrite(os.path.join(vis_dir_rollout_temp, f'{i:04d}.png'), im)
                assert self.action_name is not None and self.obj_name is not None
                curr_save_dir = os.path.join(self.save_dir, self.obj_name, self.action_name)
                video_dir = os.path.join(curr_save_dir, f"video_{v}.mp4")
                make_video(vis_dir_rollout_temp, video_dir, '%04d.png', 10)

        im_list = self.render(rendervar_list, visvar_list)

        # save images and video
        for i, im in enumerate(im_list):
            cv2.imwrite(os.path.join(vis_dir_rollout, f'{i:04d}.png'), im)
        self.video_dir = os.path.join(os.path.dirname(vis_dir_rollout), f"{vis_dir_rollout.split('/')[-1]}.mp4")
        make_video(vis_dir_rollout, self.video_dir, '%04d.png', 10)

        # update particle pos
        self.particle_pos = rendervar_list[-1]['means3D'].clone().detach().cpu().numpy()
    
    def step_real(self, action):  # (2, 3)
        print('executing action in real ...')
        action = torch.tensor([action[0, 0], action[0, 1], action[1, 0], action[1, 1]]).to(torch.float32).to(self.device)
        self.env.step(action.detach().cpu().numpy(), decoded=True)

    def render(self, rendervar_list, visvar_list, cam_id=None):
        im_list = []
        for rendervar, visvar in zip(rendervar_list, visvar_list):
            rendervar = {k: v.to(self.device) for k, v in rendervar.items()}
            if cam_id is not None:
                k = self.gs_trainer.metadata['k'][cam_id]
                w2c = self.gs_trainer.metadata['w2c'][cam_id]
                w, h = self.gs_trainer.metadata['w'], self.gs_trainer.metadata['h']
                bg = self.bg_imgs[cam_id]
                # setup camera
                render_cam = setup_camera(w, h, k, w2c, self.gs_trainer.config['near'], self.gs_trainer.config['far'], [0.0, 0.0, 0.0])
            else:
                render_cam = self.cam
                bg = self.bg
            im, _, depth, = GaussianRasterizer(raster_settings=render_cam)(**rendervar)
            im = im.permute(1, 2, 0)
            im = im.detach().cpu().numpy()[:, :, ::-1] * 255

            if bg is not None:
                # calculate mask
                rendervar_mask = copy.deepcopy(rendervar)
                rendervar_mask['colors_precomp'] = torch.ones_like(rendervar_mask['colors_precomp'])
                mask, _, _, = GaussianRasterizer(raster_settings=render_cam)(**rendervar_mask)
                mask = mask.detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]

                # rgb to rgba
                # im = cv2.cvtColor(im / (mask + 1e-4), cv2.COLOR_RGB2RGBA)
                # im[:, :, 3] = mask.mean(axis=2) * 255

                # overlay on bg
                bg_im = bg[:, :, ::-1].copy()
                bg_im = bg_im * 0.7 + np.ones_like(bg_im) * 255 * 0.3  # make bg brighter
                bg_im = bg_im.astype(np.uint8)
                im = (im * mask + bg_im * (1 - mask)).astype(np.uint8)

            # generate eef pos
            radius_scale = 0.006
            rad_draw = int(im.shape[1] * radius_scale)
            rad_draw_outer = rad_draw + 2
            eef_pos = visvar['eef']
            if cam_id is not None:
                intr = self.gs_trainer.metadata['k'][cam_id]
                extr = self.gs_trainer.metadata['w2c'][cam_id]
            else:
                intr = self.gs_trainer.metadata['k'][self.vis_cam_id]
                extr = self.gs_trainer.metadata['w2c'][self.vis_cam_id]
            eef_pos_proj = project(eef_pos, intr, extr)[0]
            eef_pos_proj = eef_pos_proj.astype(int)
            im = im.copy().astype(np.uint8)
            im = cv2.circle(im, tuple(eef_pos_proj), rad_draw_outer, (20, 20, 255), -1)
            im = cv2.circle(im, tuple(eef_pos_proj), rad_draw, (100, 100, 255), -1)
            im_list.append(im)
        return im_list

    # visualizer functions

    def clear_state(self, global_state, target=None):
        """Clear target history state from global_state
        If target is not defined, points and mask will be both removed.
        1. set global_state['points'] as empty dict
        2. set global_state['mask'] as full-one mask.
        """
        if isinstance(global_state, gr.State):
            state = global_state.value
        else:
            state = global_state
        if target is None:
            target = ['point', 'mask']
        if not isinstance(target, list):
            target = [target]
        if 'point' in target:
            state['points'] = dict()
            print('Clear Points State!')
        if 'mask' in target:
            image_raw = state["images"]["image_raw"]
            state['mask'] = np.ones((image_raw.size[1], image_raw.size[0]),
                                        dtype=np.uint8)
            print('Clear mask State!')

        return global_state
    
    def draw_particles(self, global_state, draw_particles=True):
        intr = global_state['intr']
        extr = global_state['extr']
        particle_pos = self.particle_pos
        particle_pos = project(particle_pos, intr, extr)
        particle_pos = particle_pos.astype(int)

        image = global_state['images']['image_orig']
        particle_pos = particle_pos.clip(0, np.array(image.size) - 1)
        if draw_particles:
            image = draw_raw_points_on_image(image, particle_pos)
        global_state['images']['image_raw'] = image
        global_state['images']['image_show'] = image
        return global_state

    def init_images(self, global_state, draw_particles=True, update_bg=True):
        """This function is called only ones with Gradio App is started.
        0. pre-process global_state, unpack value from global_state of need
        1. Re-init renderer
        2. run `renderer._render_drag_impl` with `is_drag=False` to generate
        new image
        3. Assign images to global state and re-generate mask
        """
        if isinstance(global_state, gr.State):
            state = global_state.value
        else:
            state = global_state

        k = self.gs_trainer.metadata['k'][self.vis_cam_id]
        w2c = self.gs_trainer.metadata['w2c'][self.vis_cam_id]
        state['extr'] = w2c
        state['intr'] = k

        # show self.particle_pos on image

        w, h = self.gs_trainer.metadata['w'], self.gs_trainer.metadata['h']
        self.width = w
        self.height = h

        init_image = Image.fromarray(self.imgs[self.vis_cam_id])
        assert init_image.size[0] == self.width and init_image.size[1] == self.height
        if update_bg:
            self.bg_imgs = copy.deepcopy(self.imgs)
        self.bg = self.bg_imgs[self.vis_cam_id]
        assert self.bg.shape[0] == self.height and self.bg.shape[1] == self.width
        state['images']['image_orig'] = init_image

        state = self.draw_particles(state, draw_particles=draw_particles)
        state['mask'] = np.ones((init_image.size[1], init_image.size[0]), dtype=np.uint8)

        # setup camera
        self.cam = setup_camera(w, h, k, w2c, self.gs_trainer.config['near'], self.gs_trainer.config['far'], [0.0, 0.0, 0.0])

        return global_state
    
    def update_bg(self):
        assert self.imgs is not None
        self.bg_imgs = copy.deepcopy(self.imgs)
        self.bg = self.bg_imgs[self.vis_cam_id]
        assert self.bg.shape[0] == self.height and self.bg.shape[1] == self.width
    
    def update_mean_z(self):
        self.particle_pos = self.gs_trainer.params['means3D'].clone().detach().cpu().numpy()
        self.mean_z = self.particle_pos[:, 2].mean()

    def preprocess_mask_info(self, global_state, image):
        """Function to handle mask information.
        1. last_mask is None: Do not need to change mask, return mask
        2. last_mask is not None:
            2.1 global_state is remove_mask:
            2.2 global_state is add_mask:
        """
        if isinstance(image, dict):
            last_mask = get_valid_mask(image['mask'])
        else:
            last_mask = None
        mask = global_state['mask']

        # mask in global state is a placeholder with all 1.
        if (mask == 1).all():
            mask = last_mask

        # last_mask = global_state['last_mask']
        editing_mode = global_state['editing_state']

        if last_mask is None:
            return global_state

        if editing_mode == 'remove_mask':
            updated_mask = np.clip(mask - last_mask, 0, 1)
            print(f'Last editing_state is {editing_mode}, do remove.')
        elif editing_mode == 'add_mask':
            updated_mask = np.clip(mask + last_mask, 0, 1)
            print(f'Last editing_state is {editing_mode}, do add.')
        else:
            updated_mask = mask
            print(f'Last editing_state is {editing_mode}, '
                'do nothing to mask.')

        global_state['mask'] = updated_mask
        # global_state['last_mask'] = None  # clear buffer
        return global_state
    
    def update_image_draw_unpaired(self, image, points, mask, show_mask, intr, extr, global_state=None):
        intr_orig = global_state['intr']
        extr_orig = global_state['extr']
        image_draw = draw_points_on_image(image, points, intr, extr, z=self.mean_z, intr_orig=intr_orig, extr_orig=extr_orig)
        if show_mask and mask is not None and not (mask == 0).all() and not (
                mask == 1).all():
            image_draw = draw_mask_on_image(image_draw, mask)

        image_draw = Image.fromarray(np.array(image_draw))  # add_watermark_np(np.array(image_draw)))
        if global_state is not None:
            global_state['images']['image_show'] = image_draw
        return image_draw

    def update_image_draw(self, image, points, mask, show_mask, global_state=None):
        intr = global_state['intr']
        extr = global_state['extr']
        image_draw = draw_points_on_image(image, points, intr, extr, z=self.mean_z)
        if show_mask and mask is not None and not (mask == 0).all() and not (
                mask == 1).all():
            image_draw = draw_mask_on_image(image_draw, mask)

        image_draw = Image.fromarray(np.array(image_draw))  # add_watermark_np(np.array(image_draw)))
        if global_state is not None:
            global_state['images']['image_show'] = image_draw
        return image_draw

    def on_click_add_point(self, global_state, image: dict):
        """Function switch from add mask mode to add points mode.
        1. Updaste mask buffer if need
        2. Change global_state['editing_state'] to 'add_points'
        3. Set current image with mask
        """
        global_state = self.preprocess_mask_info(global_state, image)
        global_state['editing_state'] = 'add_points'
        mask = global_state['mask']
        image_raw = global_state['images']['image_raw']
        image_draw = self.update_image_draw(image_raw, global_state['points'], mask,
                                        global_state['show_mask'], global_state)

        return (global_state, image_draw) # gr.Image(value=image_draw, width=self.width, height=self.height))

    def on_click_clear_points(self, global_state):
        """Function to handle clear all control points
        1. clear global_state['points'] (clear_state)
        2. re-init network
        2. re-draw image
        """
        self.clear_state(global_state, target='point')

        # renderer: Renderer = global_state["renderer"]
        # renderer.feat_refs = None

        image_raw = global_state['images']['image_raw']
        image_draw = self.update_image_draw(image_raw, {}, global_state['mask'],
                                    global_state['show_mask'], global_state)
        return global_state, image_draw

    def on_click_reset_image(self, global_state, image: dict):
        """Function to handle reset image
        1. clear global_state['points'] and global_state['mask']
        2. re-init network
        3. re-draw image
        """
        image = Image.fromarray(np.array(image))
        global_state['images']['image_orig'] = image

        particle_pos = self.particle_pos
        particle_pos = project(particle_pos, global_state['intr'], global_state['extr'])
        particle_pos = particle_pos.astype(int)
        particle_pos = particle_pos.clip(0, np.array(image.size) - 1)
        image = draw_raw_points_on_image(image, particle_pos)

        global_state['images']['image_raw'] = image
        global_state['images']['image_show'] = image # Image.fromarray(np.array(image))
        global_state['mask'] = np.ones((image.size[1], image.size[0]), dtype=np.uint8)
        return global_state

    def on_click_image(self, global_state, evt: gr.SelectData):
        """This function only support click for point selection
        """
        xy = evt.index
        if global_state['editing_state'] != 'add_points':
            print(f'In {global_state["editing_state"]} state. '
                'Do not add points.')

            return global_state, global_state['images']['image_show']

        points = global_state["points"]

        point_idx = get_latest_points_pair(points)
        if point_idx is None:
            points[0] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')
        elif points[point_idx].get('target', None) is None:
            points[point_idx]['target'] = xy
            print(f'Click Image - Target - {xy}')
        else:
            points[point_idx + 1] = {'start': xy, 'target': None}
            print(f'Click Image - Start - {xy}')

        image_raw = global_state['images']['image_raw']
        image_draw = self.update_image_draw(
            image_raw,
            global_state['points'],
            global_state['mask'],
            global_state['show_mask'],
            global_state,
        )
        return global_state, image_draw

    def on_click_reset(self, global_state):
        self.reset()
        self.clear_state(global_state, target='point')
        self.init_images(global_state)
        reset_video = gr.Video(
            value=None,
            width=self.width,
            height=self.height,
        )
        return (global_state, global_state['images']['image_show'], reset_video)

    def on_click_run_sim(self, global_state):
        image_orig = global_state['images']['image_orig']
        points = global_state['points']

        for points_id, point in points.items():
            assert points_id == 0, 'Only support one pair of points'

        start = points[0]['start']
        target = points[0]['target']
        assert target is not None

        start_im_x = start[0]
        start_im_y = start[1]
        target_im_x = target[0]
        target_im_y = target[1]

        start_im = np.array([start_im_x, start_im_y])

        if self.save_for_demo:
            obj_name = self.obj_name
            action_name = f'action_{time.time()}'
            self.action_name = action_name
            curr_save_dir = os.path.join(self.save_dir, obj_name, action_name)
            os.makedirs(curr_save_dir, exist_ok=True)
            # save image_show
            for v in range(4):
                image_draw = self.update_image_draw_unpaired(Image.fromarray(self.imgs[v]), points, global_state['mask'], global_state['show_mask'], 
                    self.gs_trainer.metadata['k'][v], self.gs_trainer.metadata['w2c'][v], global_state)
                image_draw.save(os.path.join(curr_save_dir, f'img_{v}.png'))

        self.update_mean_z()
        start_world = click_to_xyz(start_im_x, start_im_y, global_state['intr'], global_state['extr'], z=self.mean_z)
        target_world = click_to_xyz(target_im_x, target_im_y, global_state['intr'], global_state['extr'], z=self.mean_z)

        print('start_world', start_world)
        print('target_world', target_world)

        actions = torch.stack(
            [torch.from_numpy(start_world), torch.from_numpy(target_world)], dim=0).to(torch.float32).to(self.device)
        self.actions = actions
        self.update_bg()
        self.step_sim(actions)

        if self.save_for_demo:
            save_to_splat(
                pts=self.gs_trainer.params['means3D'].detach().cpu().numpy(),
                colors=self.gs_trainer.params['rgb_colors'].detach().cpu().numpy(),
                scales=torch.exp(self.gs_trainer.params['log_scales']).detach().cpu().numpy(),
                quats=torch.nn.functional.normalize(self.gs_trainer.params['unnorm_rotations'], dim=-1).detach().cpu().numpy(),
                opacities=torch.sigmoid(self.gs_trainer.params['logit_opacities']).detach().cpu().numpy(),
                output_file=os.path.join(curr_save_dir, 'gs_pred.splat'),
            )

        global_state = self.draw_particles(global_state)

        # draw previous push
        image_draw = self.update_image_draw(global_state['images']['image_raw'], 
                global_state['points'], global_state['mask'], global_state['show_mask'], global_state)

        # clear state
        self.clear_state(global_state, target='point')
        
        video_draw = gr.Video(
            value=self.video_dir,
            width=self.gs_trainer.metadata['w'],
            height=self.gs_trainer.metadata['h'],
        )
        return (global_state, image_draw, video_draw)
    
    def on_click_run_real(self, global_state):
        if self.actions is None:
            return global_state, global_state['images']['image_show']
        self.step_real(self.actions)

        self.reset(train_gs=False)  # load new images to self.img
        self.init_images(global_state, draw_particles=False, update_bg=False)  # change to new images

        image_draw = self.update_image_draw(global_state['images']['image_raw'], 
                global_state['points'], global_state['mask'], global_state['show_mask'], global_state)

        return (global_state, image_draw)
    
    def on_click_switch_view(self, global_state):
        self.vis_cam_id = (self.vis_cam_id + 1) % len(self.imgs)

        self.reset(train_gs=False)
        self.init_images(global_state, draw_particles=True, update_bg=False)  # change to new images
        
        if self.actions is None:
            if isinstance(global_state, gr.State):
                state = global_state.value
            else:
                state = global_state
            state['points'] = dict()
            return global_state, global_state['images']['image_show'], gr.Video(value=None, width=self.width, height=self.height)
        
        self.gs_trainer.params = self.params_prev
        self.step_sim(self.actions)

        global_state = self.draw_particles(global_state, draw_particles=False)

        # draw previous push
        image_draw = self.update_image_draw(global_state['images']['image_raw'], 
                global_state['points'], global_state['mask'], global_state['show_mask'], global_state)

        # clear state
        self.clear_state(global_state, target='point')
        
        video_draw = gr.Video(
            value=self.video_dir,
            width=self.gs_trainer.metadata['w'],
            height=self.gs_trainer.metadata['h'],
        )
        return (global_state, image_draw, video_draw)

    
    def on_click_vis_gs(self, global_state):
        self.gs_dir = os.path.join(self.gs_vis_path, 'pred.splat')
        save_to_splat(
            pts=self.gs_trainer.params['means3D'].detach().cpu().numpy(),
            colors=self.gs_trainer.params['rgb_colors'].detach().cpu().numpy(),
            scales=torch.exp(self.gs_trainer.params['log_scales']).detach().cpu().numpy(),
            quats=torch.nn.functional.normalize(self.gs_trainer.params['unnorm_rotations'], dim=-1).detach().cpu().numpy(),
            opacities=torch.sigmoid(self.gs_trainer.params['logit_opacities']).detach().cpu().numpy(),
            output_file=self.gs_dir,
        )
        form_3dgs = gr.Model3D(
            value=self.gs_dir,
            clear_color=[1.0, 1.0, 1.0, 0.0],
            label="3D Model",
        )
        return (global_state, form_3dgs)

    def launch(self, share=False):
        with gr.Blocks() as app:
            global_state = gr.State({
                "images": {
                    # image_orig: the original image, change with seed/model is changed
                    # image_raw: image with mask and points, change durning optimization
                    # image_show: image showed on screen
                },
                'mask': None,  # mask for visualization, 1 for editing and 0 for unchange
                'last_mask': None,  # last edited mask
                'show_mask': True,  # add button
                'extr': None,
                'intr': None,
                "params": {
                    "seed": 0,
                    "motion_lambda": 20,
                    "r1_in_pixels": 3,
                    "r2_in_pixels": 12,
                    "magnitude_direction_in_pixels": 1.0,
                    "latent_space": "w+",
                    "trunc_psi": 0.7,
                    "trunc_cutoff": None,
                    "lr": 0.001,
                },
                # "device": visualizer.device,
                "draw_interval": 1,
                # "renderer": Renderer(disable_timing=True),
                "points": {},
                "curr_point": None,
                "curr_type_point": "start",
                'editing_state': 'add_points',
            })
            self.reset()
            self.clear_state(global_state, target='point')
            global_state = self.init_images(global_state)

            with gr.Row():
                with gr.Column(scale=1, min_width=20):
                    with gr.Row():
                        reset = gr.Button('Reset')

                    with gr.Row():
                        run_sim = gr.Button('Run sim')
                    
                    with gr.Row():
                        run_real = gr.Button('Run real')
                    
                    with gr.Row():
                        switch_view = gr.Button('Switch view')
            
                with gr.Column(scale=4):
                    form_image = gr.Image(
                        value=global_state.value['images']['image_show'],
                        width=self.width,
                        height=self.height,
                    )
                
                with gr.Column(scale=4):
                    form_video = gr.Video(
                        value=None,
                        width=self.width,
                        height=self.height,
                    )

            with gr.Row():
                with gr.Column(scale=1, min_width=20):
                    with gr.Row():
                        vis_gs = gr.Button('Visualize GS')
                
                with gr.Column(scale=8):
                    form_3dgs = gr.Model3D(
                        value=None,
                    )
            
            form_image.select(
                self.on_click_image,
                inputs=[global_state],
                outputs=[global_state, form_image],
            )
            
            reset.click(self.on_click_reset,
                    inputs=[global_state],
                    outputs=[global_state, form_image, form_video])

            run_sim.click(self.on_click_run_sim,
                    inputs=[global_state],
                    outputs=[global_state, form_image, form_video])
            
            run_real.click(self.on_click_run_real,
                    inputs=[global_state],
                    outputs=[global_state, form_image])
            
            switch_view.click(self.on_click_switch_view, 
                    inputs=[global_state],
                    outputs=[global_state, form_image, form_video])
            
            vis_gs.click(self.on_click_vis_gs,
                    inputs=[global_state],
                    outputs=[global_state, form_3dgs])

        app.launch(share=share)


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
    with DynamicsVisualizer(args, config, gs_config) as visualizer:
        visualizer.launch(share=True)
