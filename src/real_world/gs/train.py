import argparse
import torch
import numpy as np
from tqdm import tqdm

from real_world.gs.external import densify
from real_world.gs.train_utils import get_custom_dataset, initialize_params, initialize_optimizer, get_loss, report_progress, get_batch


def train(img_list, seg_list, init_pt_cld, metadata, loss_weights, densify_params, num_iters):
    params, variables = initialize_params(init_pt_cld, metadata)
    optimizer = initialize_optimizer(params, variables)
    dataset = get_custom_dataset(img_list, seg_list, metadata)
    todo_dataset = []
    progress_bar = tqdm(range(num_iters))
    for i in range(num_iters):
        curr_data = get_batch(todo_dataset, dataset)
        loss, variables = get_loss(params, curr_data, variables, loss_weights)
        loss.backward()
        with torch.no_grad():
            report_progress(params, dataset[0], i, progress_bar)
            params, variables, num_pts = densify(params, variables, optimizer, i, **densify_params)
            print(f"Number of points: {num_pts}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    progress_bar.close()
    params = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_im', type=float, default=1.0)
    parser.add_argument('--weight_seg', type=float, default=3.0)
    parser.add_argument('--grad_thresh', type=float, default=0.0002)
    parser.add_argument('--remove_threshold', type=float, default=0.005)
    parser.add_argument('--remove_thresh_5k', type=float, default=0.25)
    parser.add_argument('--scale_scene_radius', type=float, default=0.01)
    parser.add_argument('--num_iters', type=int, default=10000)
    args = parser.parse_args()

    loss_weights = {'im': args.weight_im, 'seg': args.weight_seg}
    densify_params = {'grad_thresh': args.grad_thresh, 'remove_thresh': args.remove_threshold, 
        'remove_thresh_5k': args.remove_thresh_5k, 'scale_scene_radius': args.scale_scene_radius}
    img_list = None
    seg_list = None
    init_pt_cld = None
    metadata = None
    train(img_list, seg_list, init_pt_cld, metadata, loss_weights, densify_params, args.num_iters)
