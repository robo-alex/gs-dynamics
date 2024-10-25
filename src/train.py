import glob
import numpy as np
import argparse
import yaml
import os
import time
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gnn.model import DynamicsPredictor
from gnn.utils import set_seed, umeyama_algorithm
from data.dataset import DynDataset

import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def dataloader_wrapper(dataloader, name):
    cnt = 0
    while True:
        print(f'[{name}] epoch {cnt}')
        cnt += 1
        for data in dataloader:
            yield data

def rigid_loss(pred, gt, **kwargs):
    pred_pos = pred
    orig_pos = kwargs['state'][:, 0, :pred_pos.shape[1]]
    obj_mask = kwargs['obj_mask']
    _, R_pred, t_pred = umeyama_algorithm(orig_pos, pred_pos, obj_mask, fixed_scale=True)
    pred_pos_ume = orig_pos.bmm(R_pred.transpose(1, 2)) + t_pred
    pred_pos_ume = pred_pos_ume.detach()
    loss = F.mse_loss(pred_pos[obj_mask], pred_pos_ume[obj_mask])
    return loss

def grad_manager(phase):
    if phase == 'train':
        return torch.enable_grad()
    else:
        return torch.no_grad()

def truncate_graph(data):
    Rr = data['Rr']
    Rs = data['Rs']
    Rr_nonzero = torch.sum(Rr, dim=-1) > 0
    Rs_nonzero = torch.sum(Rs, dim=-1) > 0
    n_Rr = torch.max(Rr_nonzero.sum(1), dim=0)[0].item()
    n_Rs = torch.max(Rs_nonzero.sum(1), dim=0)[0].item()
    max_n = max(n_Rr, n_Rs)
    data['Rr'] = data['Rr'][:, :max_n, :]
    data['Rs'] = data['Rs'][:, :max_n, :]
    return data

def mse_loss(pred, gt, **kwargs):
    return F.mse_loss(pred, gt)

def l1_loss(pred, gt, **kwargs):
    return F.l1_loss(pred, gt)

def length_loss(pred, gt, **kwargs):
    # kwargs = truncate_graph(kwargs)
    pos = kwargs['state'][:, 0, :pred.shape[1]].detach()
    Rr = kwargs['Rr'][:, :, :pred.shape[1]]
    Rs = kwargs['Rs'][:, :, :pred.shape[1]]
    
    pos_r = Rr.bmm(pos)
    pos_s = Rs.bmm(pos)
    pos_diff = pos_r - pos_s

    pred_r = Rr.bmm(pred)
    pred_s = Rs.bmm(pred)
    pred_diff = pred_r - pred_s

    pos_diff_len = torch.norm(pos_diff, dim=-1)
    pred_diff_len = torch.norm(pred_diff, dim=-1)

    return F.mse_loss(pred_diff_len, pos_diff_len)

def local_rigid_loss(pred, gt, **kwargs):
    pos = kwargs['state'][:, 0, :pred.shape[1]].detach()
    Rr = kwargs['Rr'][:, :, :pred.shape[1]]
    Rs = kwargs['Rs'][:, :, :pred.shape[1]]

    pos_r = Rr.bmm(pos)
    pos_s = Rs.bmm(pos)

    pred_r = Rr.bmm(pred)
    pred_s = Rs.bmm(pred)

    diff_r = pred_r - pos_r
    diff_s = pred_s - pos_s

    diff_r = torch.norm(diff_r, dim=-1)
    diff_s = torch.norm(diff_s, dim=-1)

    return F.mse_loss(diff_r, diff_s)


def train(config):
    train_config = config['train_config']
    model_config = config['model_config']
    dataset_config = config['dataset_config']

    torch.autograd.set_detect_anomaly(True)
    set_seed(train_config['random_seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(train_config['out_dir'], exist_ok=True)
    os.makedirs(os.path.join(train_config['out_dir'], 'checkpoints'), exist_ok=True)
    # if prep_save_dir is None:
    #     prep_save_dir = os.path.join(train_config['out_dir'], 'preprocess')
    #     os.makedirs(prep_save_dir, exist_ok=True)
    
    phases = train_config['phases']
    dataset_config['n_his'] = train_config['n_his']
    dataset_config['n_future'] = train_config['n_future']
    datasets = {phase: DynDataset(
        dataset_config=dataset_config, phase=phase
    ) for phase in phases}
    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=train_config['batch_size'],
        shuffle=(phase == 'train'),
        num_workers=8,
    ) for phase in phases}
    dataloaders = {phase: dataloader_wrapper(dataloaders[phase], phase) for phase in phases}

    model_config['n_his'] = train_config['n_his']
    
    model = DynamicsPredictor(model_config, device)

    model.to(device)

    if 'rigid_loss' in train_config.keys() and train_config['rigid_loss']:
        loss_funcs = [(mse_loss, 1.), (length_loss, 0.05), (rigid_loss, 0.05)]
    else:
        # loss_funcs = [(mse_loss, 1.)]
        # loss_funcs = [(mse_loss, 1.), (length_loss, 0.01)]
        # loss_funcs = [(l1_loss, 1.)]
        loss_funcs = []
        if 'mse_loss' in train_config.keys() and train_config['mse_loss'] > 0:
            loss_funcs.append((mse_loss, train_config['mse_loss']))
        else:
            loss_funcs.append((mse_loss, 1.))
        if 'length_loss' in train_config.keys() and train_config['length_loss'] > 0:
            loss_funcs.append((length_loss, train_config['length_loss']))
        else:
            loss_funcs.append((length_loss, 0.01))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_plot_list_train = []
    loss_plot_list_valid = [] 
    for epoch in range(train_config['n_epochs']):
        time1 = time.time()
        for phase in phases:
            with grad_manager(phase):
                if phase == 'train': 
                    model.train()
                else: 
                    model.eval()
                state_noise = dataset_config['datasets'][0]['state_noise'][phase]
                loss_sum_list = []
                n_iters = train_config['n_iters_per_epoch'][phase] \
                        if train_config['n_iters_per_epoch'][phase] != -1 else len(datasets[phase])
                for i in range(n_iters):
                    # t1 = time.time()
                    data = next(dataloaders[phase])
                    if phase == 'train':
                        optimizer.zero_grad()
                    data = {key: data[key].to(device) for key in data.keys()}
                    # data = truncate_graph(data)
                    loss_sum = 0
                    loss_item_sum = [0 for _ in loss_funcs]

                    future_state = data['state_future']  # (B, n_future, n_p, 3)
                    # future_mask = data['state_future_mask']  # (B, n_future)
                    future_tool = data['tool_future']  # (B, n_future-1, n_p+n_s, 3)
                    future_action = data['action_future']  # (B, n_future-1, n_p+n_s, 3)

                    for fi in range(train_config['n_future']):
                        gt_state = future_state[:, fi].clone()  # (B, n_p, 3)
                        # gt_mask = future_mask[:, fi].clone()  # (B,)

                        pred_state, pred_motion = model(**data)

                        pred_state_p = pred_state[:, :gt_state.shape[1], :3].clone()
                        # loss = [weight * func(pred_state_p[gt_mask], gt_state[gt_mask]) for func, weight in loss_funcs]
                        loss = [weight * func(pred_state_p, gt_state, **data) for func, weight in loss_funcs]
                        # print([l.item() for l in loss])

                        loss_sum += sum(loss)
                        loss_item_sum = [l + loss_item.item() for l, loss_item in zip(loss_item_sum, loss)]

                        if fi < train_config['n_future'] - 1:
                            # build next graph
                            next_tool = future_tool[:, fi].clone()  # (B, n_p+n_s, 3)
                            next_action = future_action[:, fi].clone()  # (B, n_p+n_s, 3)
                            next_state = next_tool.unsqueeze(1)  # (B, 1, n_p+n_s, 3)
                            next_state[:, -1, :pred_state_p.shape[1]] = pred_state_p 

                            # TODO
                            # next_state[:, -1] += state_noise * 2 * torch.rand(next_state[:, -1].shape, device=next_state.device) - state_noise

                            next_state = torch.cat([data['state'][:, 1:], next_state], dim=1)  # (B, n_his, n_p+n_s, 3)
                            data["state"] = next_state  # (B, n_his, N+M, state_dim)
                            data["action"] = next_action  # (B, N+M, state_dim) 

                    if phase == 'train':
                        # tt1 = time.time()
                        loss_sum.backward()
                        # tt2 = time.time()
                        # print(f'backward time: {tt2 - tt1}')
                        optimizer.step()
                        # tt3 = time.time()
                        # print(f'optimizer step time: {tt3 - tt2}')
                        if i % train_config['log_interval'] == 0:
                            print(f'Epoch {epoch}, iter {i}, loss {loss_sum.item()}, loss components {[l for l in loss_item_sum]}')
                            loss_sum_list.append(loss_sum.item())
                    if phase == 'valid':
                        loss_sum_list.append(loss_sum.item())
                        # if i % train_config['log_interval'] == 0:
                        #     print(f'[Valid] Epoch {epoch}, iter {i}, loss {loss_sum.item()}')
                        #     loss_sum_list.append(loss_sum.item())
                    # t2 = time.time()
                    # print(f'iter time: {t2 - t1}')
                if phase == 'valid':
                    print(f'\nEpoch {epoch}, valid loss {np.mean(loss_sum_list)}, loss components {[l for l in loss_item_sum]}')

                if phase == 'train':
                    loss_plot_list_train.append(np.mean(loss_sum_list))
                if phase == 'valid':
                    loss_plot_list_valid.append(np.mean(loss_sum_list))
        
        if ((epoch + 1) < 10) or ((epoch + 1) < 100 and (epoch + 1) % 10 == 0) or (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'model_{(epoch + 1)}.pth'))
        torch.save(model.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'latest.pth'))
        torch.save(optimizer.state_dict(), os.path.join(train_config['out_dir'], 'checkpoints', f'latest_optim.pth'))

        # plot figures
        plt.figure(figsize=(20, 5))
        plt.plot(loss_plot_list_train, label='train')
        plt.plot(loss_plot_list_valid, label='valid')
        # cut off figure
        ax = plt.gca()
        # y_min = min(min(loss_plot_list_train), min(loss_plot_list_valid))
        # y_min = min(loss_plot_list_valid)
        # y_max = min(3 * y_min, max(max(loss_plot_list_train), max(loss_plot_list_valid)))
        # ax.set_ylim([0, y_max])
        # save
        plt.legend()
        plt.savefig(os.path.join(train_config['out_dir'], 'loss.png'), dpi=300)
        plt.close()

        time2 = time.time()
        print(f'Epoch {epoch} time: {time2 - time1}\n')


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', type=str, default='config/debug.yaml')
    args = arg_parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    train(config)

