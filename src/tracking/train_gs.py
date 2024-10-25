import torch
import json
from tqdm import tqdm
import argparse
from helpers import params2cpu, save_params
from external import densify
from train_utils import initialize_params, initialize_optimizer, initialize_per_timestep, initialize_post_first_timestep, get_batch, get_loss, report_progress, densify, get_custom_dataset


def train(seq, exp, remove_threshold, remove_thresh_5k, weight_soft_col_cons,
            weight_im, weight_seg, weight_rigid, weight_bg, weight_iso, weight_rot, num_knn, scale_scene_radius,
            metadata_path, init_pt_cld_path):
    
    md = json.load(open(f"./data/{seq}/{metadata_path}", 'r'))  # metadata for custom dataset
    num_timesteps = len(md['fn'])
    params, variables = initialize_params(seq, md, init_pt_cld_path)
    optimizer = initialize_optimizer(params, variables)
    output_params = []
    for t in range(num_timesteps):
        dataset = get_custom_dataset(t, md, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)
        if not is_initial_timestep:
            params, variables = initialize_per_timestep(params, variables, optimizer)
        num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, variables = get_loss(params, curr_data, variables, is_initial_timestep, weight_soft_col_cons,
                                        weight_im, weight_seg, weight_rigid, weight_bg, weight_iso, weight_rot)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, variables, num_pts = densify(params, variables, optimizer, i, remove_threshold, remove_thresh_5k, scale_scene_radius)
                    with open(f"./output/{exp}/{seq}/num_pts.txt", 'w') as f:
                        f.write(f"Number of points: {num_pts}\n")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            variables = initialize_post_first_timestep(params, variables, optimizer, num_knn)
        if (t % 5 == 0 and t > 0) or t == num_timesteps - 1:
            save_params(output_params, seq, exp)
            print(f"Saved ckpts at timestep {t}")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Run training with given sequence and experiment name.')
    parser.add_argument('--exp_name', type=str, required=True, help='The experiment name.')
    parser.add_argument('--sequence', type=str, required=True, help='The sequence to train on.')
    parser.add_argument('--remove_threshold', type=float, default=0.005, help='The threshold for removing points.')
    parser.add_argument('--remove_thresh_5k', type=float, default=0.25, help='The threshold for removing points at 5k iterations.')
    parser.add_argument('--weight_soft_col_cons', type=float, default=0.01, help='The weight for soft color consistency loss.')
    parser.add_argument('--weight_im', type=float, default=50.0, help='The weight for image loss.')
    parser.add_argument('--weight_seg', type=float, default=200.0, help='The weight for segmentation loss.')
    parser.add_argument('--weight_rigid', type=float, default=200.0, help='The weight for rigid loss.')
    parser.add_argument('--weight_bg', type=float, default=200.0, help='The weight for background loss.')
    parser.add_argument('--weight_iso', type=float, default=1000.0, help='The weight for isotropic loss.')
    parser.add_argument('--weight_rot', type=float, default=4.0, help='The weight for rotational loss.')
    parser.add_argument('--num_knn', type=int, default=20, help='The number of nearest neighbors to use for loss calculation.')
    parser.add_argument('--scale_scene_radius', type=float, default=0.05, help='The scale factor for the scene radius.')
    parser.add_argument('--metadata_path', type=str, required=True, help='The path to the metadata file.')
    parser.add_argument('--init_pt_cld_path', type=str, required=True, help='The path to the initial point cloud file.')

    # Parse the arguments
    args = parser.parse_args()

    weight_params = {
        'soft_col_cons': args.weight_soft_col_cons,
        'im': args.weight_im,
        'seg': args.weight_seg,
        'rigid': args.weight_rigid,
        'bg': args.weight_bg,
        'iso': args.weight_iso,
        'rot': args.weight_rot
    }
    
    train(args.sequence, args.exp_name, args.remove_threshold, args.remove_thresh_5k, 
          weight_params, args.num_knn, args.scale_scene_radius, 
          args.metadata_path, args.init_pt_cld_path)
    torch.cuda.empty_cache()
        