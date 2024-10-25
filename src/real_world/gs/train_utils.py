import torch
import numpy as np
import random
from PIL import Image

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from real_world.gs.helpers import setup_camera, l1_loss_v1, o3d_knn, params2rendervar
from real_world.gs.external import calc_ssim, calc_psnr


def get_custom_dataset(img_list, seg_list, metadata):
    """
    Generates a dataset from given metadata and sequence.
    """
    dataset = []
    
    # Loop over filenames corresponding to 't' in the metadata
    for c in range(len(img_list)):
        
        # Extract parameters from the metadata
        w, h = metadata['w'], metadata['h']
        k = metadata['k'][c]
        w2c = metadata['w2c'][c]
        
        # Set up a camera using extracted parameters and some default values
        cam = setup_camera(w, h, k, w2c, near=0.01, far=100)
        
        # Get the filename of the current image and open it
        if isinstance(img_list[c], str):
            im = np.array(Image.open(img_list[c]))
            im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        else:
            im = torch.tensor(img_list[c]).permute(2, 0, 1).float().cuda()
            if im.max() > 2.0: 
                im = im / 255
        
        # Open the corresponding segmentation image and convert it to a tensor
        if isinstance(seg_list[c], str):
            seg = np.array(Image.open(seg_list[c])).astype(np.float32)
        else:
            seg = seg_list[c].astype(np.float32)
        seg = torch.tensor(seg).float().cuda()

        # Create a color segmentation tensor. It seems to treat the segmentation as binary (object/background)
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        
        # Add the data to the dataset
        dataset.append({'cam': cam, 'im': im, 'seg': seg_col, 'id': c})

    return dataset


def initialize_params(init_pt_cld, metadata):
    """
    Initializes parameters and variables required for a 3D point cloud based on provided data.
    """

    # Extract the segmentation data.
    seg = init_pt_cld[:, 6]

    # Define a constant for the maximum number of cameras.
    max_cams = 4

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
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in params.items()}
    # params['rgb_colors'].requires_grad = False

    # Calculate the scene radius based on the camera centers.
    cam_centers = np.linalg.inv(metadata['w2c'])[:, :3, 3]
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


def get_loss(params, curr_data, variables, loss_weights):

    # Initialize dictionary to store various loss components
    losses = {}

    # Convert parameters to rendering variables and retain gradient for 'means2D'
    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()

    # Perform rendering to obtain image, radius, and other outputs
    im, radius, _ = Renderer(raster_settings=curr_data['cam'])(**rendervar)

    # Apply camera parameters to modify the rendered image
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]

    # Calculate image loss using L1 loss and ssim
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)

    # Calculate segmentation loss
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    # Calculate total loss as weighted sum of individual losses
    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    # Update variables related to rendering radius and seen areas
    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    return loss, variables


def report_progress(params, data, i, progress_bar, num_pts, every_i=100, vis_dir=None):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        if vis_dir:
            Image.fromarray((im.cpu().numpy().clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)).save(f"{vis_dir}/{i:06d}.png")
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"img 0 PSNR": f"{psnr:.{7}f}, number of points: {num_pts}"})
        progress_bar.update(every_i)


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(random.randint(0, len(todo_dataset) - 1))
    return curr_data
