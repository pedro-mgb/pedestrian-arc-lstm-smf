import math

import numpy as np
import torch


def phi_2d(seq, n):
    """
    computes bi-linear interpolation coefficients of a single 2D point
    :param seq: Tensor of shape (traj_len, batch, 2); the sequence of positions (should be normalized in [0,1] interval)
    :param n: number of nodes (in the case of a square grid, its dimension squared)
    :return: Tensor of shape (n, traj_len, batch), where n is the number of nodes (param n). Interpolated data
    """

    # the dimensions (assumed 2D square grid)
    n1 = int(math.sqrt(n))
    n2 = int(math.sqrt(n))
    # tensor of shape (n1, n1, traj_len, batch)
    phi_aux = torch.zeros(n1, n2, seq.shape[0], seq.shape[1], dtype=torch.double, device=seq.device)
    # x,y,i,j,dy,dx - tensors of shape (traj_len, batch)
    y = seq[:, :, 1] * (n1 - 1)
    x = seq[:, :, 0] * (n2 - 1)
    i = torch.round(torch.minimum(torch.ones_like(y) * (n1 - 1), torch.floor(y) + 1)).to(torch.long).to(seq.device)
    j = torch.round(torch.minimum(torch.ones_like(x) * (n2 - 1), torch.floor(x) + 1)).to(torch.long).to(seq.device)
    dy = y - i + 1
    dx = x - j + 1
    # (<class 'IndexError'>, IndexError('index 25 is out of bounds for dimension 0 with size 25'),
    # maybe be careful with indexes, and subtract one to each cause this was matlab?
    for seq_idx in range(i.shape[0]):
        for batch_idx in range(i.shape[1]):
            i_aux, j_aux, dx_aux, dy_aux = i[seq_idx, batch_idx], j[seq_idx, batch_idx], dx[seq_idx, batch_idx], \
                                           dy[seq_idx, batch_idx]
            phi_aux[i_aux - 1, j_aux - 1, seq_idx, batch_idx] = (1 - dx_aux) * (1 - dy_aux)
            phi_aux[i_aux, j_aux - 1, seq_idx, batch_idx] = (1 - dx_aux) * dy_aux
            phi_aux[i_aux - 1, j_aux, seq_idx, batch_idx] = dx_aux * (1 - dy_aux)
            phi_aux[i_aux, j_aux, seq_idx, batch_idx] = dx_aux * dy_aux
    """
    phi_aux[i - 1, j - 1, :, :] = (1 - dx) * (1 - dy)
    phi_aux[i, j - 1, :, :] = (1 - dx) * dy
    phi_aux[i - 1, j, :, :] = dx * (1 - dy)
    phi_aux[i, j, :, :] = dx * dy
    """

    # phi_t = torch.transpose(phi_aux, 0, 1)
    phi = torch.reshape(phi_aux, (n, phi_aux.shape[2], phi_aux.shape[3]))
    checksum = torch.tensor([seq.shape[0] * seq.shape[1]], device=seq.device)
    assert (torch.round(torch.sum(phi)).to(torch.long) == checksum), \
        'There are interpolated values whose sum does not equal to 1. Either the input sequence is not normalized in' \
        '[0, 1] interval, or there may be a problem with this interpolation function (e.g. rounding of torch.float to' \
        'torch.long)'
    return phi
