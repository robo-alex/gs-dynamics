
"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

import torch
import torch.nn.functional as func
from torch.autograd import Variable
from math import exp


def build_rotation(q):
    norm = torch.sqrt(q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3])
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device='cuda')
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def calc_mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def calc_psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def gaussian(window_size, sigma):
    """
    Generate a 1D Gaussian kernel.

    Parameters:
    - window_size: The size (length) of the output Gaussian kernel.
    - sigma: The standard deviation of the Gaussian distribution.

    Returns:
    - A 1D tensor representing the Gaussian kernel normalized to have a sum of 1.
    """

    # For each position in the desired window size, calculate the Gaussian value. 
    # The middle of the window corresponds to the peak of the Gaussian.
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    
    # Normalize the Gaussian kernel to have a sum of 1 and return it.
    return gauss / gauss.sum()


def create_window(window_size, channel):
    """
    Generate a 2D Gaussian kernel window.

    Parameters:
    - window_size: The size (width and height) of the output 2D Gaussian kernel.
    - channel: Number of channels for which the window will be replicated.

    Returns:
    - A 4D tensor representing the Gaussian window for the specified number of channels.
    """

    # Create a 1D Gaussian kernel of size 'window_size' with standard deviation 1.5.
    # The unsqueeze operation adds an extra dimension, making it a 2D tensor.
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)

    # Compute the outer product of the 1D Gaussian kernel with itself to get a 2D Gaussian kernel.
    # This results in a symmetric 2D Gaussian kernel.
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    # Expand the 2D window to have the desired number of channels.
    # The expand operation replicates the 2D window for each channel.
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    # print('img1', img1.device)
    # window = window.to(device=img1.device)
    # assert torch.isfinite(img1).all(), "img1 contains NaN or Inf"
    # assert torch.isfinite(window).all(), "window contains NaN or Inf"
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean2d_gradient(variables):
    variables['means2D_gradient_accum'][variables['seen']] += torch.norm(
        variables['means2D'].grad[variables['seen'], :2], dim=-1)
    variables['denom'][variables['seen']] += 1
    return variables


def update_params_and_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group['params'][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group['params'][0]] = stored_state
        params[k] = group["params"][0]
    return params


def cat_params_to_optimizer(new_params, params, optimizer):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        stored_state = optimizer.state.get(group['params'][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(v)), dim=0)
            stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0)
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(torch.cat((group["params"][0], v), dim=0).requires_grad_(True))
            params[k] = group["params"][0]
    return params


def remove_points(to_remove, params, variables, optimizer):
    """

    Parameters:
    - to_remove: A boolean tensor where 'True' indicates the points to remove.
    - params: A dictionary containing parameters.
    - variables: A dictionary containing various variables.
    - optimizer: An optimizer object containing optimization information.

    Returns:
    - Updated params and variables dictionaries after removal.
    """
    
    # Find the points that we want to keep (the opposite of `to_remove`).
    to_keep = ~to_remove
    
    # Extract the keys from `params` except for 'cam_m' and 'cam_c'.
    keys = [k for k in params.keys() if k not in ['cam_m', 'cam_c']]
    
    for k in keys:
        # Find the parameter group associated with the current key in the optimizer.
        group = [g for g in optimizer.param_groups if g['name'] == k][0]
        
        # Try to get the state of this group from the optimizer (this contains momentum information, etc. for optimizers like Adam).
        stored_state = optimizer.state.get(group['params'][0], None)
        
        if stored_state is not None:
            # Update the stored state by keeping only the desired entries.
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            
            # Delete the old state and set a new parameter tensor, keeping only the desired entries and ensuring gradients can be computed.
            del optimizer.state[group['params'][0]]
            group["params"][0] = torch.nn.Parameter((group["params"][0][to_keep].requires_grad_(True)))
            optimizer.state[group['params'][0]] = stored_state
            params[k] = group["params"][0]
        else:
            # If there's no stored state, just update the parameter tensor.
            group["params"][0] = torch.nn.Parameter(group["params"][0][to_keep].requires_grad_(True))
            params[k] = group["params"][0]

    variables['means2D_gradient_accum'] = variables['means2D_gradient_accum'][to_keep]
    variables['denom'] = variables['denom'][to_keep]
    variables['max_2D_radius'] = variables['max_2D_radius'][to_keep]
    
    return params, variables


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def densify(params, variables, optimizer, i, grad_thresh, remove_thresh, remove_thresh_5k, scale_scene_radius):
    """
    Adjusts the density of points based on various conditions and thresholds.

    Parameters:
    - params: A dictionary containing parameters.
    - variables: A dictionary containing various variables.
    - optimizer: An optimizer object containing optimization information.
    - i: An iteration or step count.
    - remove_thresh: A threshold for removing points.

    Returns:
    - Updated params and variables dictionaries after adjustment.
    """
    if i <= 5000:
        variables = accumulate_mean2d_gradient(variables)
        if (i >= 500) and (i % 100 == 0):
            # Calculate the gradient of the means2D values and handle NaNs.
            grads = variables['means2D_gradient_accum'] / variables['denom']
            grads[grads.isnan()] = 0.0
            # Define points that should be cloned based on gradient thresholds and scales of the points.
            to_clone = torch.logical_and(grads >= grad_thresh, (
                        torch.max(torch.exp(params['log_scales']), dim=1).values <= scale_scene_radius * variables['scene_radius']))
            # Extract parameters for points that need cloning.
            new_params = {k: v[to_clone] for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            padded_grad = torch.zeros(num_pts, device="cuda")
            padded_grad[:grads.shape[0]] = grads
            to_split = torch.logical_and(padded_grad >= grad_thresh,
                                         torch.max(torch.exp(params['log_scales']), dim=1).values > scale_scene_radius * variables[
                                             'scene_radius'])
            n = 2  # number to split into
            new_params = {k: v[to_split].repeat(n, 1) for k, v in params.items() if k not in ['cam_m', 'cam_c']}
            stds = torch.exp(params['log_scales'])[to_split].repeat(n, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(params['unnorm_rotations'][to_split]).repeat(n, 1, 1)
            new_params['means3D'] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params['log_scales'] = torch.log(torch.exp(new_params['log_scales']) / (0.8 * n))
            params = cat_params_to_optimizer(new_params, params, optimizer)
            num_pts = params['means3D'].shape[0]

            variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda")
            variables['denom'] = torch.zeros(num_pts, device="cuda")
            variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat((to_split, torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda")))
            params, variables = remove_points(to_remove, params, variables, optimizer)

            remove_threshold = remove_thresh_5k if i == 5000 else remove_thresh
            to_remove = (torch.sigmoid(params['logit_opacities']) < remove_threshold).squeeze()
            # print("num of to remove: ", to_remove.sum())
            if i >= 3000:
                big_points_ws = torch.exp(params['log_scales']).max(dim=1).values > 0.1 * variables['scene_radius']
                # print("num of big points: ", big_points_ws.sum())
                to_remove = torch.logical_or(to_remove, big_points_ws)
            params, variables = remove_points(to_remove, params, variables, optimizer)
            
            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            new_params = {'logit_opacities': inverse_sigmoid(torch.ones_like(params['logit_opacities']) * 0.01)}
            params = update_params_and_optimizer(new_params, params, optimizer)

    num_pts = params['means3D'].shape[0]

    return params, variables, num_pts
