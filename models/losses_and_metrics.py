"""
Created on March 7th, 2021
Contains some common loss functions and error metrics used to train / evaluate models.

CREDITS: Some of these were taken/adapted from https://github.com/agrimgupta92/sgan, and also from
https://github.com/abduallahmohamed/Social-STGCNN; https://github.com/quancore/social-lstm;
https://github.com/vita-epfl/trajnetplusplustools; https://github.com/StanfordASL/Trajectron
"""
import os

import numpy as np
import torch
from scipy.stats import gaussian_kde

from models.utils.utils import trajs_with_inside_points

# Factor used for numerical stability
epsilon = float(np.finfo(np.float32).eps)


def nll(v_pred, v_gt, loss_mask, mode=None):
    """
    Computes the negative log-likelihood loss, assuming the data has a Bi-variate Gaussian distribution.
    :param v_pred: Tensor of shape (traj_len, batch, 5). Predicted velocity (or relative displacement)
    :param v_gt: Tensor of shape (traj_len, batch, 2). Ground truth velocity (or relative displacement)
    :param loss_mask: Tensor of shape (batch, traj_len). Could be used to decide which trajectories or parts of it would
    contribute to the loss, but it's not used here (assumption that it's all ones). It is however provided for sake of
    compatibility with other loss functions.
    :param mode: the mode in which to compute the loss (will influence the shape and dimensions of the tensor returned)
    :return: the NLL value
    """
    return __gaussian_2d_loss__(v_pred, v_gt, loss_mask, mode='log_average' if mode is None else mode)


def gaussian_likelihood_loss(v_pred, v_gt, loss_mask):
    """
    Computes the likelihood loss, assuming the data has a Bi-variate Gaussian distribution.
    :param v_pred: Tensor of shape (traj_len, batch, 5). Predicted velocity (or relative displacement)
    :param v_gt: Tensor of shape (traj_len, batch, 2). Ground truth velocity (or relative displacement)
    :param loss_mask: Tensor of shape (batch, traj_len). Could be used to decide which trajectories or parts of it would
    contribute to the loss, but it's not used here (assumption that it's all ones). It is however provided for sake of
    compatibility with other loss functions.
    :return: the loss value
    """
    return __gaussian_2d_loss__(v_pred, v_gt, loss_mask, mode='average')


def __gaussian_2d_loss__(v_pred, v_gt, loss_mask, mode='average'):
    """
    Computes the negative log-likelihood loss, assuming the data has a Bi-variate Gaussian distribution.
    :param v_pred: Tensor of shape (traj_len, batch, 5). Predicted velocity (or relative displacement)
    :param v_gt: Tensor of shape (traj_len, batch, 2). Ground truth velocity (or relative displacement)
    :param loss_mask: Tensor of shape (batch, traj_len). Could be used to decide which trajectories or parts of it would
    contribute to the loss, but it's not used here (assumption that it's all ones). It is however provided for sake of
    compatibility with other loss functions.
    :param mode: can be one of the following average (mean), sum, raw; or log_average (log_mean), log_sum, log_raw
    :return: the NLL value
    """
    # factor to multiply the loss by - if log isn't used
    multiply_factor = 100

    norm_x = v_gt[:, :, 0] - v_pred[:, :, 0]  # (mean) difference in x
    norm_y = v_gt[:, :, 1] - v_pred[:, :, 1]  # (mean) difference in y
    sx = torch.exp(v_pred[:, :, 2])  # standard deviation in x
    sy = torch.exp(v_pred[:, :, 3])  # standard deviation in y
    corr = torch.tanh(v_pred[:, :, 4])  # correlation factor
    sx_sy = sx * sy

    z = (norm_x / sx) ** 2 + (norm_y / sy) ** 2 - 2 * ((corr * norm_x * norm_y) / sx_sy)
    neg_rho = 1 - corr ** 2

    # Numerator
    result = torch.exp(-z / (2 * neg_rho))
    # Normalization factor
    denominator = 2 * np.pi * (sx_sy * torch.sqrt(neg_rho))
    # Final PDF calculation
    result = result / denominator

    mode = mode.lower()
    if mode == 'average' or mode == 'mean':
        return torch.mean(result * multiply_factor)
    elif mode == 'sum':
        return torch.sum(result * multiply_factor)
    elif 'log' in mode:
        return loss_to_log_loss(result, mode=mode)
    # else - return the raw loss
    return result * multiply_factor


def loss_to_log_loss(loss, mode='raw'):
    """
    convert a loss to a logarithmic loss
    :param loss: the loss tensor, in non-log format
    :param mode: the mode to apply, can contain 'sum', 'average' (or 'mean'), or 'raw'
    :return:
    """
    loss = -torch.log(torch.clamp(loss, min=epsilon))
    if 'sum' in mode:
        return torch.sum(loss)
    elif 'average' in mode or 'mean' in mode:
        return torch.mean(loss)
    # raw loss
    return loss


def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Computes standard L2 norm loss between predicted and ground truth trajectories. This involves squaring the
    differences between coordinates, and summing over all length (pred_traj_len)
    :param pred_traj: Tensor of shape (pred_traj_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (pred_traj_len, batch, 2). Ground truth trajectory to compare with prediction
    :param loss_mask: Tensor of shape (batch, traj_len). Applies a mask to the loss values (in case one doesn't want to
    consider some trajectories for the sake of loss computation).
    :param random: not used
    :param mode: Can be one of sum, average, raw
    :return: l2 loss depending on mode (tensor can have different dimensions)
    """
    traj_len, batch, _ = pred_traj.size()
    # switch to shape (batch, traj_len, 2)
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss[:, :, 0].data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)
    else:
        # different kind of mode, returning per each instant
        return loss.sum(dim=2).permute(1, 0)  # shape (traj_len, batch)


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Computes the euclidean displacement error between trajectories.
    :param pred_traj: Tensor of shape (traj_len, batch, 2). Predicted trajectory.
    :param pred_traj_gt: Tensor of shape (traj_len, batch, 2). Ground truth trajectory.
    :param consider_ped: Tensor of shape (batch) -> which pedestrians to consider (1 to consider, 0 otherwise; or
    possibly decimal values if we want to give more contribution to some pedestrians)
    :param mode: Can be one of sum, raw
    :return: the Euclidean displacement error
    """
    traj_len, _, _ = pred_traj.size()
    # switch to shape (batch, traj_len, 2)
    loss = pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    Computes the euclidean displacement error between two positions, assumed the be the final positions of trajectories.
    :param pred_pos: Tensor of shape (batch, 2). Predicted last position
    :param pred_pos_gt: Tensor of shape (batch, 2). Ground truth last position
    :param consider_ped: Tensor of shape (batch) -> which pedestrians to consider (1 to consider, 0 otherwise; or
    possibly decimal values if we want to give more contribution to some pedestrians)
    :param mode: Can be one of sum, raw
    :return: the Euclidean displacement error for this last position
    """
    loss = pred_pos_gt - pred_pos
    loss = loss ** 2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)


def num_collisions(trajectories, col_thresh=0.1, inter_points=2, mode='sum'):
    """
    Compute the number of pedestrians that have collided with at least one other pedestrian. Two trajectories are said
    to collide if for each two consecutive instant, the points of a line segment that unite those positions come to a
    distance smaller than a certain threshold.
    Each pedestrian will either have collided, or not collided. A pedestrian colliding multiple times, and at several
    instants, will only count as "one" collision.
    CREDITS: This was adapted from https://github.com/vita-epfl/trajnetplusplustools/
    :param trajectories: Tensor of shape (traj_len, num_pedestrians, 2). Set of trajectories of several pedestrians
    :param col_thresh: The distance threshold for which below this, a collision is said to occur.
    :param inter_points: when building the line segments that unite the two points of consecutive instants, how many
    intermediate points will be included (this excludes start and end of the segment). The higher this number, the more
    accurate the values, but the computation will take more time
    :param mode: Can be one of sum, raw, other
    :return: tensor, which can be of two types:
    - if mode is raw: Tensor of shape (num_pedestrians), where each entry has 1 or 0, indicating if has collided or not
    - if mode is sum: Tensor of shape () - single value between 0 and num_pedestrians, indicating how many pedestrians
    have collided.
    """
    return num_collisions_between_two(trajectories, trajectories, col_thresh, inter_points, mode)


def num_collisions_between_two(trajectories1, trajectories2, col_thresh=0.1, inter_points=2, mode='sum'):
    """
    Compute the number of pedestrians, apart of a first trajectory set, that have collided with at least one other
    pedestrian, apart of a second trajectory set.
    Only consider collisions for trajectories 1 (two collisions - between trajectory x of trajectories1 and trajectory y
    of trajectories2 and between trajectory x of trajectories1 and trajectory z of trajectories2 - counts as just one)
    Each pedestrian will either have collided, or not collided. A pedestrian colliding multiple times, and at several
    instants, will only count as "one" collision.
    CREDITS: This was adapted from https://github.com/vita-epfl/trajnetplusplustools/
    :param trajectories1: Tensor of shape (traj_len, num_pedestrians, 2). First et of trajectories of several
    pedestrians.
    :param trajectories2: Tensor of shape (traj_len, num_pedestrians, 2). Second Set of trajectories of several
    pedestrians, to compare with the first set.
    :param col_thresh: The distance threshold for which below this, a collision is said to occur.
    A list of thresholds may also be supplied, and the output tensor will have a new first dim, equal to the number of
    different tensors
    :param inter_points: when building the line segments that unite the two points of consecutive instants, how many
    intermediate points will be included (this excludes start and end of the segment). The higher this number, the more
    accurate the values, but the computation will take more time
    :param mode: Can be one of sum, raw, other
    :return: tensor, which can be of two types:
    - if mode is raw: Tensor of shape (num_pedestrians), where each entry has 1 or 0, indicating if each pedestrian with
    trajectory in trajectories1 has collided or not
    - if mode is sum: Tensor of shape () - single value between 0 and num_pedestrians, indicating how many pedestrians
    with trajectory in trajectories1 have collided.
    """
    assert trajectories1.shape == trajectories2.shape, 'The supplied sets of trajectories have different dimensions, ' \
                                                       f'{trajectories1.shape} and {trajectories2.shape}'
    num_peds = trajectories1.shape[1]
    if trajectories1.shape[1] < 2:  # not enough pedestrians to compute number of collisions
        collisions = torch.zeros(num_peds, device=trajectories1.device)
        return collisions if mode == 'raw' else torch.sum(collisions)
    # each is a tensor of shape [(seq_len-1)*(1+inter_points), num_pedestrians, 2]
    trajectories_intermediate_segments1 = trajs_with_inside_points(trajectories1, inter_points)
    trajectories_intermediate_segments2 = trajs_with_inside_points(trajectories2, inter_points)
    # euclidean distances between pedestrians - shape [(seq_len-1)*(1+inter_points), num_pedestrians, num_pedestrians]
    distances = torch.cdist(trajectories_intermediate_segments1, trajectories_intermediate_segments2, p=2)
    if isinstance(col_thresh, list):
        return __collisions_several_thresh__(distances, col_thresh, mode)
    # remove collisions with each pedestrian with respect to itself
    collisions = torch.where(distances < col_thresh, 1, 0) * \
                 (torch.ones_like(distances) -
                  torch.eye(distances.shape[1], distances.shape[1]).unsqueeze(0).repeat(distances.shape[0], 1, 1))
    # Summing along all dimensions except one - tensor of shape [num_pedestrians]
    # each pedestrian can collide with multiple, but if he collides several times with the same, only one counts
    # same metric as used by the Trajnet++ standard
    collisions = torch.sum(torch.clamp(torch.sum(collisions, dim=0), min=0, max=1), dim=1)

    # ALTERNATIVE #1 - maximum one collision per pedestrian. A pedestrian can collide once with e.g. 3 pedestrians, will
    # only count as one collision. Should yield smaller numbers than original metric
    # collisions = torch.sum(torch.sum(collisions, dim=0), dim=1)
    # collisions = torch.clamp(collisions, min=0, max=1)  # maximum one collision per pedestrian
    # ALTERNATIVE #2 - no limit on the number of collisions between pedestrians. Should yield larger numbers than
    # original metric. Note that results are displayed in percentages, which doesn't make complete sense for this
    # collisions = torch.sum(torch.sum(collisions, dim=0), dim=1)

    return collisions if mode == 'raw' else torch.sum(collisions)


def __collisions_several_thresh__(distances, person_radius_list, mode='sum'):
    """

    :param distances:
    :param person_radius_list:
    :param mode:
    :return:
    """
    num_radius = len(person_radius_list)
    distances = distances.unsqueeze(0).repeat(num_radius, 1, 1, 1)
    person_radius = torch.tensor(person_radius_list, device=distances.device).unsqueeze(
        1).unsqueeze(2).unsqueeze(3).repeat(1, distances.shape[1], distances.shape[2], distances.shape[3])
    collisions = torch.where(distances < person_radius, 1, 0) * \
                 (torch.ones_like(distances) - torch.eye(distances.shape[2], distances.shape[2]).unsqueeze(
                     0).unsqueeze(0).repeat(distances.shape[0], distances.shape[1], 1, 1))
    collisions_per_ped = torch.sum(torch.clamp(torch.sum(collisions, dim=1), min=0, max=1), dim=2)
    # collisions_per_ped = torch.sum(torch.sum(collisions, dim=1), dim=2)
    return collisions_per_ped if mode == 'raw' else torch.sum(collisions_per_ped, dim=1)


def compute_kde_nll(pred, gt, log_pdf_lower_bound=-20, mode='raw', ignore_if_fail=False):
    """
    Credits go to: https://github.com/vita-epfl/trajnetplusplustools and https://github.com/StanfordASL/Trajectron.
    :param pred: Tensor of shape (pred_traj_len, num_samples, num_ped, 2). Predicted trajectory samples
    :param gt: Tensor of shape (pred_traj_len, num_ped, 2). Ground truth or target trajectory, in absolute coordinates
    :param log_pdf_lower_bound: Minimum to clip the logarithm of the pdf (anything below will be clipped at this value)
    :param mode: Can be one of sum, raw
    :param ignore_if_fail: Do not stop if kde_nll computation fails for one or more trajectories
    :return: Single-value Tensor or of shape (num_ped), depending on mode. The KDE-NLL for each pedestrian.
    """
    pred_len = gt.shape[0]
    nll_all_ped_list = []
    for p in range(gt.shape[1]):
        ll = 0.0
        same_pred = 0
        # all predictions are the same, which can happen for constant velocity with 0 speed
        for timestep in range(gt.shape[0]):
            curr_gt, curr_pred = gt[timestep, p], pred[timestep, :, p]
            if torch.all(curr_pred[1:] == curr_pred[:-1]):
                same_pred += 1
                continue  # Identical prediction at particular time-step, skip
            try:
                scipy_kde = gaussian_kde(curr_pred.T)
                # We need [0] because it's a (1,)-shaped tensor
                log_pdf = np.clip(scipy_kde.logpdf(curr_gt.T), a_min=log_pdf_lower_bound, a_max=None)[0]
                if np.isnan(log_pdf) or np.isinf(log_pdf) or log_pdf > 100:
                    same_pred += 1  # Difficulties in computing Gaussian_KDE
                    continue
                ll += log_pdf
            except Exception as e:
                same_pred += 1  # Difficulties in computing Gaussian_KDE

        if same_pred == pred_len:
            if ignore_if_fail:
                continue  # simply not being considered for computation
            else:
                raise Exception('Failed to compute KDE-NLL for one or more trajectory. To ignore the trajectories that '
                                f'result in computation failure, supply --ignore_if_kde_nll_fails.{os.linesep}WARNING! '
                                'This will mean that some samples will be ignored, which may be unfair when comparing '
                                'with other methods whose samples do not result in error.')

        ll = ll / (pred_len - same_pred)
        nll_all_ped_list.append(ll)
    nll_all_ped = torch.tensor(nll_all_ped_list, device=gt.device)
    return nll_all_ped if mode == 'raw' else torch.sum(nll_all_ped)
