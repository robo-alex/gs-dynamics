import torch
import numpy as np
import cv2
import os
import subprocess
import open3d as o3d
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from real_world.gs.helpers import setup_camera
from real_world.gs.external import densify
from real_world.gs.train_utils import get_custom_dataset, initialize_params, initialize_optimizer, get_loss, report_progress, get_batch

def Rt_to_w2c(R, t):
    w2c = np.concatenate([np.concatenate([R, t.reshape(3, 1)], axis=1), np.array([[0, 0, 0, 1]])], axis=0)
    w2c = np.linalg.inv(w2c)
    return w2c

def make_video(
        image_root: str,
        video_path: str,
        image_pattern: str = '%04d.png',
        frame_rate: int = 10):
    subprocess.run([
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-loglevel', 'error',
        '-framerate', str(frame_rate),
        '-i', os.path.join(image_root, image_pattern),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_path
    ])

class GSTrainer:

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.clear()
    
    def clear(self, clear_params=True):
        # training data
        self.init_pt_cld = None
        self.metadata = None
        self.img_list = None
        self.seg_list = None
        # training results
        if clear_params:
            self.params = None
    
    @torch.no_grad
    def render(self, render_data, cam_id, bg=[0.7, 0.7, 0.7]):
        render_data = {k: v.to(self.device) for k, v in render_data.items()}
        w, h = self.metadata['w'], self.metadata['h']
        k = self.metadata['k'][cam_id]
        w2c = self.metadata['w2c'][cam_id]
        cam = setup_camera(w, h, k, w2c, self.config['near'], self.config['far'], bg)
        im, _, depth, = GaussianRasterizer(raster_settings=cam)(**render_data)
        return im, depth
    
    def update_state_env(self, pcd, env, imgs, masks):  # RGB
        R_list, t_list = env.get_extrinsics()
        intr_list = env.get_intrinsics()
        rgb_list = []
        seg_list = []
        for c in range(env.n_fixed_cameras):
            rgb_list.append(imgs[c] * masks[c][:, :, None])
            seg_list.append(masks[c] * 1.0)
        self.update_state(pcd, rgb_list, seg_list, R_list, t_list, intr_list)

    def update_state_no_env(self, pcd, imgs, masks, R_list, t_list, intr_list, n_cameras=4):  # RGB
        rgb_list = []
        seg_list = []
        for c in range(n_cameras):
            rgb_list.append(imgs[c] * masks[c][:, :, None])
            seg_list.append(masks[c] * 1.0)
        self.update_state(pcd, rgb_list, seg_list, R_list, t_list, intr_list)

    def update_state(self, pcd, img_list, seg_list, R_list, t_list, intr_list):
        pts = np.array(pcd.points).astype(np.float32)
        colors = np.array(pcd.colors).astype(np.float32)
        seg = np.ones_like(pts[:, 0:1])
        self.init_pt_cld = np.concatenate([pts, colors, seg], axis=1)
        w, h = img_list[0].shape[1], img_list[0].shape[0]
        assert np.all([img.shape[1] == w and img.shape[0] == h for img in img_list])
        self.metadata = {
            'w': w,
            'h': h,
            'k': [intr for intr in intr_list],
            'w2c': [Rt_to_w2c(R, t) for R, t in zip(R_list, t_list)]
        }
        self.img_list = img_list
        self.seg_list = seg_list
    
    def train(self, vis_dir):
        params, variables = initialize_params(self.init_pt_cld, self.metadata)
        optimizer = initialize_optimizer(params, variables)
        dataset = get_custom_dataset(self.img_list, self.seg_list, self.metadata)
        todo_dataset = []
        num_iters = self.config['num_iters']
        loss_weights = {'im': self.config['weight_im'], 'seg': self.config['weight_seg']}
        densify_params = {
            'grad_thresh': self.config['grad_thresh'],
            'remove_thresh': self.config['remove_threshold'], 
            'remove_thresh_5k': self.config['remove_thresh_5k'], 
            'scale_scene_radius': self.config['scale_scene_radius']
        }
        progress_bar = tqdm(range(num_iters), dynamic_ncols=True)
        for i in range(num_iters): 
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, loss_weights)
            loss.backward()
            with torch.no_grad():
                params, variables, num_pts = densify(params, variables, optimizer, i, **densify_params)
                report_progress(params, dataset[0], i, progress_bar, num_pts, vis_dir=vis_dir)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        params = {k: v.detach() for k, v in params.items()}
        self.params = params

    def rollout_and_render(self, dm, action, vis_dir=None, save_images=True, overwrite_params=True, remove_black=False):
        assert vis_dir is not None
        assert self.params is not None

        xyz_0 = self.params['means3D']
        rgb_0 = self.params['rgb_colors']
        quat_0 = torch.nn.functional.normalize(self.params['unnorm_rotations'])
        opa_0 = torch.sigmoid(self.params['logit_opacities'])
        scales_0 = torch.exp(self.params['log_scales'])

        low_opa_idx = opa_0[:, 0] < 0.1
        xyz_0 = xyz_0[~low_opa_idx]
        rgb_0 = rgb_0[~low_opa_idx]
        quat_0 = quat_0[~low_opa_idx]
        opa_0 = opa_0[~low_opa_idx]
        scales_0 = scales_0[~low_opa_idx]

        if remove_black:
            low_color_idx = rgb_0.sum(dim=-1) < 0.5
            xyz_0 = xyz_0[~low_color_idx]
            rgb_0 = rgb_0[~low_color_idx]
            quat_0 = quat_0[~low_color_idx]
            opa_0 = opa_0[~low_color_idx]
            scales_0 = scales_0[~low_color_idx]

        eef_xyz_start = action[0]
        eef_xyz_end = action[1]

        dist_thresh = 0.005  # 5mm
        n_steps = int((eef_xyz_end - eef_xyz_start).norm().item() / dist_thresh)
        eef_xyz = torch.lerp(eef_xyz_start, eef_xyz_end, torch.linspace(0, 1, n_steps).to(self.device)[:, None])
        eef_xyz_pad = torch.cat([eef_xyz, eef_xyz_end[None].repeat(dm.n_his, 1)], dim=0)
        eef_xyz_pad = eef_xyz_pad[:, None]  # (n_steps, 1, 3)
        n_steps = eef_xyz_pad.shape[0]

        inlier_idx_all = np.arange(len(xyz_0))  # no outlier removal

        xyz, rgb, quat, opa, xyz_bones, eef = dm.rollout(
                xyz_0, rgb_0, quat_0, opa_0, eef_xyz_pad, n_steps, inlier_idx_all)

        # interpolate smoothly
        change_points = (xyz - torch.concatenate([xyz[0:1], xyz[:-1]], dim=0)).norm(dim=-1).sum(dim=-1).nonzero().squeeze(1)
        change_points = torch.cat([torch.tensor([0]), change_points])
        for i in range(1, len(change_points)):
            start = change_points[i - 1]
            end = change_points[i]
            if end - start < 2:  # 0 or 1
                continue
            xyz[start:end] = torch.lerp(xyz[start][None], xyz[end][None], torch.linspace(0, 1, end - start + 1).to(xyz.device)[:, None, None])[:-1]
            rgb[start:end] = torch.lerp(rgb[start][None], rgb[end][None], torch.linspace(0, 1, end - start + 1).to(rgb.device)[:, None, None])[:-1]
            quat[start:end] = torch.lerp(quat[start][None], quat[end][None], torch.linspace(0, 1, end - start + 1).to(quat.device)[:, None, None])[:-1]
            opa[start:end] = torch.lerp(opa[start][None], opa[end][None], torch.linspace(0, 1, end - start + 1).to(opa.device)[:, None, None])[:-1]
            xyz_bones[start:end] = torch.lerp(xyz_bones[start][None], xyz_bones[end][None], torch.linspace(0, 1, end - start + 1).to(xyz_bones.device)[:, None, None])[:-1]
            eef[start:end] = torch.lerp(eef[start][None], eef[end][None], torch.linspace(0, 1, end - start + 1).to(eef.device)[:, None, None])[:-1]
        
        for _ in range(3):
            xyz[1:-1] = (xyz[:-2] + 2 * xyz[1:-1] + xyz[2:]) / 4
            # quat[1:-1] = (quat[:-2] + 2 * quat[1:-1] + quat[2:]) / 4

        quat = torch.nn.functional.normalize(quat, dim=-1)

        rendervar_list = []
        visvar_list = []
        # im_list = []
        for t in range(n_steps):
            rendervar = {
                'means3D': xyz[t],  
                'colors_precomp': rgb[t],
                'rotations': quat[t],
                'opacities': opa[t],
                'scales': scales_0,
                'means2D': torch.zeros_like(xyz[t]),
            }
            rendervar_list.append(rendervar)

            visvar = {
                'xyz_bones': xyz_bones[t].numpy(), # params['means3D'][t][fps_idx].detach().cpu().numpy(),
                'eef': eef[t].numpy(), # eef_xyz[t].detach().cpu().numpy(),
            }
            visvar_list.append(visvar)
            
            if save_images:
                im, _ = self.render(rendervar, 0, bg=[0, 0, 0])
                im = im.cpu().numpy().transpose(1, 2, 0)
                im = (im * 255).astype(np.uint8)
                # im_list.append(im)
                cv2.imwrite(os.path.join(vis_dir, f'{t:04d}.png'), im[:, :, ::-1].copy())
        
        if save_images:
            make_video(vis_dir, os.path.join(os.path.dirname(vis_dir), f"{vis_dir.split('/')[-1]}.mp4"), '%04d.png', 5)
        
        if overwrite_params:
            self.params['means3D'] = xyz[-1].to(self.device)
            self.params['rgb_colors'] = rgb[-1].to(self.device)
            self.params['unnorm_rotations'] = quat[-1].to(self.device)
            self.params['logit_opacities'] = torch.logit(opa[-1]).to(self.device)
            self.params['log_scales'] = torch.log(scales_0).to(self.device)
            self.params['means2D'] = torch.zeros_like(xyz[-1]).to(self.device)

        return rendervar_list, visvar_list
