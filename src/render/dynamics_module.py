import torch
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm
from dgl.geometry import farthest_point_sampler

from gnn.model import DynamicsPredictor
from data.dataset import construct_edges_from_states, load_eef_pos
from data.utils import fps_rad_idx_torch
from render.utils import interpolate_motions, relations_to_matrix



class DynamicsModule:
    
    def __init__(self, config, epoch, device):
        self.device = device
        train_config = config['train_config']
        model_config = config['model_config']
        if epoch == 'latest':
            checkpoint_dir = os.path.join(train_config['out_dir'], 'checkpoints', 'latest.pth')
        else:
            checkpoint_dir = os.path.join(train_config['out_dir'], 'checkpoints', 'model_{}.pth'.format(epoch))
        self.model = self.load_model(train_config, model_config, checkpoint_dir, self.device)
        self.n_his = train_config['n_his']
        self.dist_thresh = train_config['dist_thresh']

        dataset_config = config['dataset_config']['datasets'][0]
        self.max_nobj = dataset_config['max_nobj']
        self.adj_thresh = (dataset_config['adj_radius_range'][0] + dataset_config['adj_radius_range'][1]) / 2
        self.fps_radius = (dataset_config['fps_radius_range'][0] + dataset_config['fps_radius_range'][1]) / 2
        self.topk = dataset_config['topk']
        self.connect_all = dataset_config['connect_all']

    def load_model(self, train_config, model_config, checkpoint_dir, device):
        model_config['n_his'] = train_config['n_his']
        model = DynamicsPredictor(model_config, device)
        model.to(device)
        model.eval()
        model.load_state_dict(torch.load(checkpoint_dir))
        return model
    
    def downsample_vertices(self, xyz):  # (n, 3)
        particle_tensor = xyz[None, ...].detach().cpu()
        fps_idx_1 = farthest_point_sampler(particle_tensor, self.max_nobj, start_idx=0)[0]
        downsampled_particle = particle_tensor[0, fps_idx_1, :]
        _, fps_idx_2 = fps_rad_idx_torch(downsampled_particle, self.fps_radius)
        fps_idx = fps_idx_1[fps_idx_2]
        xyz = xyz[fps_idx]
        return xyz, fps_idx

    @torch.no_grad
    def rollout(self, xyz_0, rgb_0, quat_0, opa_0, eef_xyz, n_steps, inlier_idx_all):
        # xyz_0: (n_particles, 3)
        # rgb_0: (n_particles, 3)
        # quat_0: (n_particles, 4)
        # opa_0: (n_particles, 1)
        # n_step: including the initial state and the final state (n_step - 1 steps in between)

        model = self.model
        device = self.device

        all_pos = xyz_0
        fps_all_idx = farthest_point_sampler(xyz_0.cpu()[inlier_idx_all][None], 1000, start_idx=0)[0]
        fps_all_pos = all_pos[inlier_idx_all][fps_all_idx]
        fps_all_pos_history = fps_all_pos[None].repeat(model.model_config['n_his'], 1, 1)  # (n_his, n_particles, 3)

        eef_pos_history = eef_xyz[0][None].repeat(model.model_config['n_his'], 1, 1)  # (n_his, 1, 3)
        eef_pos = eef_xyz[0]  # (1, 3)

        particle_pos_0, _ = self.downsample_vertices(fps_all_pos)

        # results to store
        quat = quat_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_particles, 4)
        xyz = xyz_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_particles, 3)
        rgb = rgb_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_particles, 3)
        opa = opa_0.cpu()[None].repeat(n_steps, 1, 1)  # (n_steps, n_particles, 1)
        xyz_bones = torch.zeros(n_steps, self.max_nobj, 3)  # (n_steps, n_bones, 3)
        eef = eef_xyz.cpu()[0][None].repeat(n_steps, 1, 1)  # (n_steps, 1, 3)

        xyz_bones[0, :particle_pos_0.shape[0]] = particle_pos_0.cpu()

        eef_delta = torch.zeros(1, 3).to(device)
        for i in tqdm(range(1, n_steps), dynamic_ncols=True):
            assert torch.allclose(fps_all_pos, fps_all_pos_history[-1])
            assert torch.allclose(eef_pos, eef_pos_history[-1])

            if torch.norm(eef_xyz[i] - eef_pos) < self.dist_thresh:
                # rot[i] = rot[i - 1].clone()
                quat[i] = quat[i - 1].clone()
                xyz[i] = xyz[i - 1].clone()
                rgb[i] = rgb[i - 1].clone()
                opa[i] = opa[i - 1].clone()
                xyz_bones[i] = xyz_bones[i - 1].clone()
                eef[i] = eef[i - 1].clone()
                continue

            eef_pos_this_step = eef_xyz[i]
            eef_delta = eef_pos_this_step - eef_pos

            particle_pos, fps_idx = self.downsample_vertices(fps_all_pos)
            particle_pos_history = fps_all_pos_history[:, fps_idx]
            nobj = particle_pos.shape[0]

            states = torch.zeros((1, self.n_his, nobj + 1, 3), device=device)
            states[:, :, :nobj] = particle_pos_history
            states[:, :, nobj:] = eef_pos_history

            states_delta = torch.zeros((1, nobj + 1, 3), device=device)
            states_delta[:, nobj:] = eef_delta

            attrs = torch.zeros((1, nobj + 1, 2), dtype=torch.float32, device=device)
            attrs[:, :nobj, 0] = 1.
            attrs[:, nobj:, 1] = 1.

            p_instance = torch.ones((1, nobj, 1), dtype=torch.float32, device=device)

            state_mask = torch.ones((1, nobj + 1), dtype=bool, device=device)

            eef_mask = torch.zeros((1, nobj + 1), dtype=bool, device=device)
            eef_mask[:, nobj] = 1

            obj_mask = torch.zeros((1, nobj + 1), dtype=bool, device=device)
            obj_mask[:, :nobj] = 1

            Rr, Rs = construct_edges_from_states(states[0, -1], self.adj_thresh, 
                            mask=state_mask[0], tool_mask=eef_mask[0], topk=self.topk, connect_all=self.connect_all)
            Rr = Rr[None]
            Rs = Rs[None]

            graph = {
                # input information
                "state": states,  # (n_his, N+M, state_dim)
                "action": states_delta,  # (N+M, state_dim)

                # attr information
                "attrs": attrs,  # (N+M, attr_dim)
                # "p_rigid": p_rigid,  # (n_instance,)
                "p_instance": p_instance,  # (N, n_instance)
                "obj_mask": obj_mask,  # (N,)
                "state_mask": state_mask,  # (N+M,)
                "eef_mask": eef_mask,  # (N+M,)

                "Rr": Rr,  # (bsz, max_nR, N)
                "Rs": Rs,  # (bsz, max_nR, N)
            }

            pred_state, _ = model(**graph)  # (1, nobj, 3)

            eef_pos_history = torch.cat([eef_pos_history[1:], eef_pos_this_step[None]], dim=0)
            eef_pos = eef_pos_this_step

            # interpolate all_pos and particle_pos
            all_pos, all_rot, _ = interpolate_motions(
                bones=particle_pos,
                motions=pred_state[0] - particle_pos,
                relations=relations_to_matrix(Rr, Rs)[:nobj, :nobj],
                xyz=all_pos,
                quat=quat[i - 1].to(device),
            )
            fps_all_pos = all_pos[inlier_idx_all][fps_all_idx]
            fps_all_pos_history = torch.cat([fps_all_pos_history[1:], fps_all_pos[None]], dim=0)

            quat[i] = all_rot.cpu()
            xyz[i] = all_pos.cpu()
            rgb[i] = rgb[i - 1].clone()
            opa[i] = opa[i - 1].clone()
            xyz_bones[i, :nobj] = pred_state[0].cpu()
            eef[i] = eef_pos.cpu()

        return xyz, rgb, quat, opa, xyz_bones, eef

    def collect_scene_data(self, data_path, output_path):
        output_path_list = output_path.split('/')
        render_ckpts_path = '/'.join(output_path_list)
        params = dict(np.load(os.path.join(render_ckpts_path, "params.npz")))
        params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}

        xyz_0 = params['means3D'][0]
        rgb_0 = params['rgb_colors'][0]
        quat_0 = torch.nn.functional.normalize(params['unnorm_rotations'][0])
        opa_0 = torch.sigmoid(params['logit_opacities'])
        scales_0 = torch.exp(params['log_scales'])

        low_opa_idx = opa_0[:, 0] < 0.1
        xyz_0 = xyz_0[~low_opa_idx]
        rgb_0 = rgb_0[~low_opa_idx]
        quat_0 = quat_0[~low_opa_idx]
        opa_0 = opa_0[~low_opa_idx]
        scales_0 = scales_0[~low_opa_idx]

        outliers = None
        new_outlier = None
        rm_iter = 0
        inlier_idx_all = np.arange(len(xyz_0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_0.detach().cpu().numpy())
        while new_outlier is None or len(new_outlier.points) > 0:
            _, inlier_idx = pcd.remove_statistical_outlier(
                nb_neighbors = 50, std_ratio = 2.0 + rm_iter * 0.5
            )
            inlier_idx_all = inlier_idx_all[inlier_idx]
            new_pcd = pcd.select_by_index(inlier_idx)
            new_outlier = pcd.select_by_index(inlier_idx, invert=True)
            if outliers is None:
                outliers = new_outlier
            else:
                outliers += new_outlier
            pcd = new_pcd
            rm_iter += 1

        eef_xyz, frame_idxs = load_eef_pos(data_path, output_path)
        eef_xyz = torch.from_numpy(eef_xyz).float().to(self.device)

        n_steps = min(len(eef_xyz), 1000)

        xyz, rgb, quat, opa, xyz_bones, eef = self.rollout(
                xyz_0, rgb_0, quat_0, opa_0, eef_xyz, n_steps, inlier_idx_all)

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
        quat = torch.nn.functional.normalize(quat, dim=-1)


        scene_data = []  # Initialize a list to store rendering variables for each item
        vis_data = []
        for t in range(n_steps):
            rendervar = {
                'means3D': xyz[t],  
                'colors_precomp': rgb[t],
                'rotations': quat[t],
                'opacities': opa[t],
                'scales': scales_0,
                'means2D': torch.zeros_like(xyz[t]),
            }

            visvar = {
                'kp': xyz_bones[t].numpy(), # params['means3D'][t][fps_idx].detach().cpu().numpy(),
                'tool_kp': eef[t].numpy(), # eef_xyz[t].detach().cpu().numpy(),
            }
            scene_data.append(rendervar)
            vis_data.append(visvar)

        return scene_data, vis_data
