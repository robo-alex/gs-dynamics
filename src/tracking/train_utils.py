import numpy as np
import torch
import copy
from PIL import Image
from random import randint
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, o3d_knn, params2rendervar
from external import calc_ssim, calc_psnr, build_rotation, update_params_and_optimizer

def map_to_segmentation_path(img_path):
    # Split the path into directory and filename
    directory, filename = img_path.rsplit('/', 1)
    directory = directory.rsplit('/', 1)[0]
    number = int(filename.split('_')[-1].split('.')[0])

    seg_filename = f'seg_{number:06}.png'
    seg_path = f'{directory}/seg/{seg_filename}'

    return seg_path

def map_to_depth_path(img_path):
    # Split the path into directory and filename
    directory, filename = img_path.rsplit('/', 1)
    number = int(filename.split('_')[1].split('.')[0])

    depth_filename = f'depth_{number:06}.png'
    depth_path = f'{directory}/depth/{depth_filename}'

    return depth_path


def get_custom_dataset(t, md, seq):
    """
    Generates a dataset from given metadata and sequence.
    
    Parameters:
    - t: presumably an index or type, used to access specific entries in the metadata
    - md: a dictionary containing metadata about the dataset
    - seq: sequence name, presumably a string used to build the path to the data

    Returns:
    - dataset: a list of dictionaries where each dictionary corresponds to an image 
               and its associated segmentation along with other related data.
    """
    
    dataset = []
    
    # Loop over filenames corresponding to 't' in the metadata
    for c in range(len(md['fn'][t])):

        # print(f"Processing image {c} of {len(md['fn'][t])}")
        
        # Extract parameters from the metadata
        w, h = md['w'], md['h']                # Width and height of the images
        k = md['k'][t][c]                      # Camera parameter for the current image
        w2c = md['w2c'][t][c]                  # Another camera parameter for the current image
        
        # Set up a camera using extracted parameters and some default values
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        
        # Get the filename of the current image and open it
        fn = md['fn'][t][c]

        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/{fn}")))
        
        # Convert the image to a PyTorch tensor, move to GPU and normalize values to [0, 1]
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        
        # Open the corresponding segmentation image and convert it to a tensor
        seg_path = map_to_segmentation_path(fn)
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/{seg_path}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        
        # Create a color segmentation tensor. It seems to treat the segmentation as binary (object/background)
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))

        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})
        
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md, init_pt_cld_path):
    """
    Initializes parameters and variables required for a 3D point cloud based on provided data.

    Args:
    - seq (str): Identifier for the current data sequence.
    - md (dict): Contains metadata, including world-to-camera transformation matrices.

    Returns:
    - tuple: A tuple containing two dictionaries:
        1. params: Parameters related to the 3D point cloud.
        2. variables: Other associated variables.
    """

    # Load the initial point cloud data from the given path.
    init_pt_cld = np.load(f"./data/{seq}/{init_pt_cld_path}")["data"]  # for custom dataset

    # Extract the segmentation data.
    seg = init_pt_cld[:, 6]

    # Define a constant for the maximum number of cameras.
    max_cams = 50

    # Compute the squared distance for the K-Nearest Neighbors of each point in the 3D point cloud.
    sq_dist, indices = o3d_knn(init_pt_cld[:, :3], 3)

    # Calculate the mean squared distance for the 3 closest points and clip its minimum value.
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)

    # Initialize various parameters related to the 3D point cloud.
    params = {
        'means3D': init_pt_cld[:, :3],                          # 3D coordinates of the points.
        'rgb_colors': init_pt_cld[:, 3:6],                      # RGB color values for the points.
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),  # Segmentation colors.
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),  # Default rotations for each point.
        'logit_opacities': np.zeros((seg.shape[0], 1)),         # Initial opacity values for the points.
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),  # Scale factors for the points.
        'cam_m': np.zeros((max_cams, 3)),                       # Placeholder for camera motion.
        'cam_c': np.zeros((max_cams, 3)),                       # Placeholder for camera center.
    }

    # Convert the params to PyTorch tensors and move them to the GPU.
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    params['rgb_colors'].requires_grad = False

    # Calculate the camera centers from the world-to-camera transformation matrices.
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]

    # Calculate the scene radius based on the camera centers.
    scene_radius = 1.1 * np.max(np.linalg.norm(cam_centers - np.mean(cam_centers, 0)[None], axis=-1))

    # Initialize other associated variables.
    variables = {
        'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(), # Maximum 2D radius.
        'scene_radius': scene_radius,                                          # Scene radius.
        'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(), # Means2D gradient accumulator.
        'denom': torch.zeros(params['means3D'].shape[0]).cuda().float()        # Denominator.
    }

    return params, variables


def initialize_optimizer(params, variables):
    lrs = {
        'means3D': 0.00016 * variables['scene_radius'],
        'rgb_colors': 0.0,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, variables, is_initial_timestep, weight_soft_col_cons,
             weight_im, weight_seg, weight_rigid, weight_bg, weight_iso, weight_rot):

    # Initialize dictionary to store various loss components
    losses = {}

    # Convert parameters to rendering variables and retain gradient for 'means2D'
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    # Perform rendering to obtain image, radius, and other outputs
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    # Apply camera parameters to modify the rendered image
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    # Calculate image loss using L1 loss and ssim
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    # Prepare variables for segment rendering
    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']

    # Perform segment rendering
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    # Calculate segmentation loss
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    # Calculate additional losses for non-initial timesteps
    if not is_initial_timestep:
        # Calculate foreground related losses
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        # Compute relative rotation and apply to current and neighbor points
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

        # Calculate rigid, rotational, and isotropic losses
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])
        
        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])
        
        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        # Calculate loss to maintain points above a 'floor' level
        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        # Calculate losses for background points and rotations
        # print('is_fg', is_fg)
        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        # Calculate loss for soft color consistency
        # losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])
        losses['soft_col_cons'] = 0.0


    # Define weights for each loss component
    loss_weights = {'im': weight_im, 'seg': weight_seg, 'rigid': weight_rigid, 'iso': weight_iso, 'rot': weight_rot,
                    'floor': 2.0, 'bg': weight_bg, 'soft_col_cons': weight_soft_col_cons}

    # Calculate total loss as weighted sum of individual losses
    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    # Update variables related to rendering radius and seen areas
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables



def get_loss_post(params, curr_data, variables, is_initial_timestep, weight_soft_col_cons,
             weight_im, weight_seg, weight_rigid, weight_bg, weight_iso, weight_rot):

    # Initialize dictionary to store various loss components
    losses = {}

    # Convert parameters to rendering variables and retain gradient for 'means2D'
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    # Perform rendering to obtain image, radius, and other outputs
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    # Apply camera parameters to modify the rendered image
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    # Calculate image loss using L1 loss and ssim
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification
    # Prepare variables for segment rendering
    # segrendervar = params2rendervar(params)
    # segrendervar['colors_precomp'] = params['seg_colors']

    # Perform segment rendering
    # seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    # # Calculate segmentation loss
    # losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    # Calculate additional losses for non-initial timesteps
    if not is_initial_timestep:
        # Calculate foreground related losses
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        # Compute relative rotation and apply to current and neighbor points
        rel_rot = quat_mult(fg_rot, variables["prev_inv_rot_fg"])
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[variables["neighbor_indices"]]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)

        # Calculate rigid, rotational, and isotropic losses
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, variables["prev_offset"],
                                              variables["neighbor_weight"])
        
        losses['rot'] = weighted_l2_loss_v2(rel_rot[variables["neighbor_indices"]], rel_rot[:, None],
                                            variables["neighbor_weight"])
        
        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, variables["neighbor_dist"], variables["neighbor_weight"])

        # Calculate loss to maintain points above a 'floor' level
        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        # Calculate losses for background points and rotations
        # print('is_fg', is_fg)
        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, variables["init_bg_pts"]) + l1_loss_v2(bg_rot, variables["init_bg_rot"])

        # Calculate loss for soft color consistency
        # losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], variables["prev_col"])
        losses['soft_col_cons'] = 0.0


    # Define weights for each loss component
    loss_weights = {'im': weight_im, 'rigid': weight_rigid, 'iso': weight_iso, 'rot': weight_rot,
                    'floor': 2.0, 'bg': weight_bg, 'soft_col_cons': weight_soft_col_cons}

    # Calculate total loss as weighted sum of individual losses
    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    # Update variables related to rendering radius and seen areas
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables

def initialize_per_timestep(params, variables, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - variables["prev_pts"])
    new_rot = torch.nn.functional.normalize(rot + (rot - variables["prev_rot"]))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[variables["neighbor_indices"]] - fg_pts[:, None]
    variables['prev_inv_rot_fg'] = prev_inv_rot_fg.detach()
    variables['prev_offset'] = prev_offset.detach()
    variables["prev_col"] = params['rgb_colors'].detach()
    variables["prev_pts"] = pts.detach()
    variables["prev_rot"] = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, variables


def initialize_post_first_timestep(params, variables, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    variables["neighbor_indices"] = torch.tensor(neighbor_indices).cuda().long().contiguous()
    variables["neighbor_weight"] = torch.tensor(neighbor_weight).cuda().float().contiguous()
    variables["neighbor_dist"] = torch.tensor(neighbor_dist).cuda().float().contiguous()

    variables["init_bg_pts"] = init_bg_pts.detach()
    variables["init_bg_rot"] = init_bg_rot.detach()
    variables["prev_pts"] = params['means3D'].detach()
    variables["prev_rot"] = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c', 'rgb_colors']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return variables


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)