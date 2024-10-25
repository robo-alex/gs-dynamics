import numpy as np
import random
import torch


## umeyama algorithm
def umeyama_algorithm(X, Y, mask, fixed_scale=True):
    """
    input:
        X : B, N, 3
        Y : B, N, 3
        mask : B, N
    output:
        c : B
        R : B, 3, 3
        t : B, 1, 3
    """
    mu_x = (X * mask[:, :, None]).sum(dim=1, keepdims=True) / mask.sum(dim=1)[:, None, None]  # (B, 1, 3)
    mu_y = (Y * mask[:, :, None]).sum(dim=1, keepdims=True) / mask.sum(dim=1)[:, None, None]  # (B, 1, 3)
    var_x = ((X - mu_x) ** 2).sum(dim=2)  # (B, N)
    var_x = (var_x * mask).sum(dim=1) / mask.sum(dim=1)  # (B)

    X_centered = X - mu_x  # (B, N, 3)
    Y_centered = Y - mu_y  # (B, N, 3)

    X_centered_masked = X_centered * mask[:, :, None]  # (B, N, 3)
    Y_centered_masked = Y_centered * mask[:, :, None]  # (B, N, 3)
    cov_xy = torch.bmm(Y_centered_masked.transpose(1, 2), X_centered_masked) / mask.sum(dim=1)[:, None, None]  # (B, 3, 3)
    
    U, D, Vh = torch.linalg.svd(cov_xy)
    revert_mask = torch.linalg.det(U) * torch.linalg.det(Vh) < 0
    D[revert_mask, -1] *= -1
    U[revert_mask, :, -1] *= -1
    if fixed_scale:
        c = torch.ones_like(var_x)
    else:
        c = D.sum(dim=1) / var_x
    R = torch.bmm(U, Vh)
    t = mu_y - c[:, None, None] * mu_x.bmm(R.transpose(1, 2))
    return c, R, t


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
