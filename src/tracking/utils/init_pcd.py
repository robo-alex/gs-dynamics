import os
import cv2
import numpy as np
import open3d as o3d
import argparse
from my_utils import depth2fgpcd, np2o3d

t = 1000
num_cam = 4
step = 0.004

def aggr_point_cloud_from_data(colors, depths, segs, Ks, poses, downsample=False, masks=None, boundaries=None):
    # colors: [N, H, W, 3] numpy array in uint8
    # depths: [N, H, W] numpy array in meters
    # segs: [N, H, W, 3] numpy array in uint8
    # Ks: [N, 3, 3] numpy array
    # poses: [N, 4, 4] numpy array
    # masks: [N, H, W] numpy array in bool
    N, H, W, _ = colors.shape
    colors = colors / 255.
    segs = np.squeeze(segs)

    start = 0
    end = N
    step = 1
    pcds = []
    pcds_all = []
    for i in range(start, end, step):
        depth = depths[i]
        color = colors[i]
        seg = segs[i]
        K = Ks[i]
        cam_param = [K[0,0], K[1,1], K[0,2], K[1,2]] # fx, fy, cx, cy
        if masks is None:
            mask = (depth > 0) & (depth < 100)
        else:
            mask = masks[i] & (depth > 0)
        pcd = depth2fgpcd(depth, mask, cam_param)
        
        pose = poses[i]
        pose = np.linalg.inv(pose)

        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T
        
        if boundaries is not None:
            x_lower = boundaries['x_lower']
            x_upper = boundaries['x_upper']
            y_lower = boundaries['y_lower']
            y_upper = boundaries['y_upper']
            z_lower = boundaries['z_lower']
            z_upper = boundaries['z_upper']
            
            trans_pcd_mask = (trans_pcd[:, 0] > x_lower) &\
                (trans_pcd[:, 0] < x_upper) &\
                    (trans_pcd[:, 1] > y_lower) &\
                        (trans_pcd[:, 1] < y_upper) &\
                            (trans_pcd[:, 2] > z_lower) &\
                                (trans_pcd[:, 2] < z_upper)
            
            pcd_o3d, pcd_dicts = np2o3d(trans_pcd[trans_pcd_mask], color[mask][trans_pcd_mask], seg[mask][trans_pcd_mask])
        else:
            pcd_o3d, pcd_dicts = np2o3d(trans_pcd, color[mask], seg[mask])
        # downsample
        if downsample:
            radius = 0.01
            pcd_o3d = pcd_o3d.voxel_down_sample(radius)
            idx = pcd_o3d.volume_down_sample_and_trace(radius)
            pcd_dicts = pcd_dicts[idx]
        pcds.append(pcd_o3d)
        pcds_all.append(pcd_dicts)
    aggr_pcd = o3d.geometry.PointCloud()
    aggr_pcd_dicts = []
    for pcd in pcds:
        aggr_pcd += pcd
    for pcd_dicts in pcds_all:
        aggr_pcd_dicts.append(pcd_dicts)
    return aggr_pcd, aggr_pcd_dicts

def read_camera_data(data_path, num_cam, t):
    colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', f'{t:06}.jpg')) for i in range(num_cam)], axis=0) 
    depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', f'{t:06}_depth.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000.
    segs = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'seg', f'seg_{t:06}.png')) for i in range(num_cam)], axis=0)
    return colors, depths, segs

def load_camera_parameters(data_path, num_cam):
    extrinsics = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy')) for i in range(num_cam)])
    cam_param = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy')) for i in range(num_cam)])
    intrinsics = np.zeros((num_cam, 3, 3))
    intrinsics[:, 0, 0] = cam_param[:, 0]
    intrinsics[:, 1, 1] = cam_param[:, 1]
    intrinsics[:, 0, 2] = cam_param[:, 2]
    intrinsics[:, 1, 2] = cam_param[:, 3]
    intrinsics[:, 2, 2] = 1
    return extrinsics, intrinsics

def process_point_cloud(colors, depths, segs, intrinsics, extrinsics, boundaries):
    # Assuming aggr_point_cloud_from_data is a pre-defined function
    pcd, aggr_pcd_dicts = aggr_point_cloud_from_data(colors[..., ::-1], depths, segs, intrinsics, extrinsics, downsample=False, boundaries=boundaries)
    # pcd.remove_statistical_outlier(nb_neighbors=600, std_ratio=0.2)
    pcd.remove_radius_outlier(nb_points=200, radius=0.01)
    return pcd, aggr_pcd_dicts

def initialize_point_cloud_struct(aggr_pcd_dicts):
    len_of_data = sum(len(aggr_pcds) for aggr_pcds in aggr_pcd_dicts)
    init_pt_cld = np.zeros((len_of_data, 7))
    init_pcd = o3d.geometry.PointCloud()
    return init_pt_cld, init_pcd

def update_point_cloud(aggr_pcd_dicts, init_pt_cld, init_pcd):
    current_index = 0
    for aggr_pcds in aggr_pcd_dicts:
        for point, attributes in aggr_pcds.items():
            init_pcd.points.append(point)
            init_pt_cld[current_index, :3] = np.asarray(point)
            color = attributes['color']
            init_pcd.colors.append(color)
            init_pt_cld[current_index, 3:6] = np.asarray(color)
            seg_value = 0 if attributes['seg'].all() == 0 else 1
            init_pt_cld[current_index, 6] = seg_value
            current_index += 1
    return init_pt_cld, init_pcd

def save_point_clouds(data_path, point_clouds, i):
    for name, pcd in point_clouds.items():
        o3d.io.write_point_cloud(os.path.join(data_path, f'{name}_{i}.ply'), pcd)
        print(f"{name}.ply saved!")

def save_npz_file(data_path, file_name, data):
    np.savez(os.path.join(data_path, file_name), data=data)
    print(f"{file_name} saved!")

def add_colors_to_point_cloud(point_cloud, colors):
    point_cloud.colors = o3d.utility.Vector3dVector(colors)


def main(data_path, num_cam, t, boundaries=None, seg_flag=False):
    colors, depths, segs = read_camera_data(data_path, num_cam, t)
    extrinsics, intrinsics = load_camera_parameters(data_path, num_cam)
    pcd, aggr_pcd_dicts = process_point_cloud(colors, depths, segs, intrinsics, extrinsics, boundaries)
    init_pt_cld, init_pcd = initialize_point_cloud_struct(aggr_pcd_dicts)
    init_pt_cld, init_pcd = update_point_cloud(aggr_pcd_dicts, init_pt_cld, init_pcd)

    segmented_points = init_pt_cld[init_pt_cld[:, 6] == 1]
    if segmented_points.shape[0] == 0:
        raise ValueError("No points found with seg_value of 1")

    if seg_flag:
        convert_pcd = o3d.geometry.PointCloud()
        convert_pcd.points = o3d.utility.Vector3dVector(segmented_points[:, :3])
        convert_pcd.colors = o3d.utility.Vector3dVector(segmented_points[:, 3:6])
    else:
        convert_pcd = o3d.geometry.PointCloud()
        convert_pcd.points = o3d.utility.Vector3dVector(init_pt_cld[:, :3])

    convert_pcd.remove_radius_outlier(nb_points=600, radius=0.01)

    convert_pcd_seg = o3d.geometry.PointCloud()
    convert_pcd_seg.points = o3d.utility.Vector3dVector(segmented_points[:, :3])
    convert_pcd_seg.colors = o3d.utility.Vector3dVector(segmented_points[:, 3:6])

    point_clouds = {
        'pcd': convert_pcd,
        'pcd_seg': convert_pcd_seg,
    }
    save_point_clouds(data_path, point_clouds, t)

    if seg_flag:
        save_npz_file(data_path, f'init_pt_cld_{t:04}.npz', segmented_points)
    else:
        save_npz_file(data_path, 'init_pt_cld.npz', init_pt_cld)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--data_path', type=str, required=True)
    # argparser.add_argument('--num_cam', type=int, default=4)
    # argparser.add_argument('--t', type=int, required=True, help='timestamp')
    # argparser.add_argument('--seg_flag', default=False, help='whether to use segmentation')
    args = argparser.parse_args()
    data_path = args.data_path
    num_cam = 4
    seg_flag = True
    # boundaries = {'x_lower': x_lower,
    #             'x_upper': x_upper,
    #             'y_lower': y_lower,
    #             'y_upper': y_upper,
    #             'z_lower': z_lower,
    #             'z_upper': z_upper,}

    boundaries = {
        'x_lower': -1,
        'x_upper': 1,
        'y_lower': -1,
        'y_upper': 1,
        'z_lower': -1,
        'z_upper': -0.01,
        }

    main(data_path, num_cam, 0, boundaries=boundaries, seg_flag=seg_flag)
