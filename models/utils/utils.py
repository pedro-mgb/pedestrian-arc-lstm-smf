"""
Created on March 12th, 2021
Can contain several utility functions to be used on the model.
Source for some of these: https://github.com/agrimgupta92/sgan/
"""
import math
import random

import numpy as np
import torch


def relative_traj_to_abs(rel_traj, start_pos):
    """
    converts a trajectory with relative displacements (which can also be seen as 'velocities') to a trajectory with
    absolute positions.
    :param rel_traj: Tensor of shape (traj_len, batch, 2). The relative trajectory
    :param start_pos: Tensor of shape (batch, 2). The position offset to apply
    :return: Tensor of shape (traj_len, batch, 2). The absolute trajectory
    """
    # convert to format (batch, traj_len, 2)
    rel_traj = rel_traj.permute(1, 0, 2)
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj.permute(1, 0, 2)


def random_angle_rotation(traj, mean=0, std=0, inplace=False):
    """
    Apply random rotation to the sequence by sampling from gaussian distribution
    :param traj: Tensor of shape (traj_len, batch, 2). The trajectory to rotate
    :param mean: mean value for the gaussian distribution (in radians)
    :param std: standard deviation value for the gaussian distribution (in radians)
    :param inplace: if True, perform rotation in situ, on traj; if False, use on another tensor
    :return: the rotated trajectory
    """
    theta = np.random.normal(loc=mean, scale=std)
    return rotate_trajectory(traj, theta, inplace), theta


def random_angle_rotation_uniform(traj, threshold=0, inplace=False):
    """
    Apply random rotation to the sequence by sampling from uniform distribution
    :param traj: Tensor of shape (traj_len, batch, 2). The trajectory to rotate
    :param threshold: threshold limit (maximum, and symmetric on minimum) on the
    :param inplace: if True, perform rotation in situ, on traj; if False, use on another tensor
    :return: the rotated trajectory
    """
    theta = np.random.uniform(low=-threshold, high=+threshold)
    return rotate_trajectory(traj, theta, inplace), theta


def rotate_trajectory(traj, angle, inplace=False):
    """
    Rotate trajectory by a certain angle. This only changes the direction of the trajectory, the starting point will
    be the same.
    :param traj: Tensor of shape (traj_len, batch, 2). The trajectory to rotate
    :param angle: rotation angle, in radians
    :param inplace: if True, perform rotation in situ, on traj; if False, use on another tensor
    :return: the rotated trajectory
    """
    # contribution in terms of x, y
    cos_t, sin_t = math.cos(angle), math.sin(angle)
    if torch.is_tensor(traj):
        traj_copy = traj.clone()
        initial_position = traj_copy[0, :, :].clone().unsqueeze(0)
    elif isinstance(traj, np.ndarray):
        traj_copy = traj.copy()
        initial_position = np.expand_dims(traj_copy[0, :, :].copy(), 0)
    else:
        raise ValueError("The argument traj in rotate_trajectory must either be a pytorch tensor or a numpy array")
    # 2D rotation matrix -
    # [  cos(theta)   sin(theta) ]
    # [ -sin(theta)   sin(theta) ]
    if inplace:
        traj_copy -= initial_position
        traj[:, :, 0] = +cos_t * traj_copy[:, :, 0] + sin_t * traj_copy[:, :, 1]
        traj[:, :, 1] = -sin_t * traj_copy[:, :, 0] + cos_t * traj_copy[:, :, 1]
        traj += initial_position
        return traj
    else:
        traj_copy[:, :, 0] = +cos_t * (traj[:, :, 0] - initial_position[:, :, 0]) + sin_t * (
                traj_copy[:, :, 1] - initial_position[:, :, 1])
        traj_copy[:, :, 1] = -sin_t * (traj[:, :, 0] - initial_position[:, :, 0]) + cos_t * (
                traj[:, :, 1] - initial_position[:, :, 1])
        return traj_copy


def normalize_0_1_min_max(data, _min, _denominator, reverse=False):
    """
    Normalize data in a [0, 1] interval, using the minmax technique.
    It involves subtracting the minimum, and then dividing by the range (maximum - minimum)
    :param data: the data to normalize; normalization is NOT performed in situ, so this data will be preserved
    :param _min: the minimum value of the data; may not actually be min(data), can be a more 'global' value
    :param _denominator: denominator, assumed to be maximum - minimum
    :param reverse: if True, will actually denormalize - multiply by (max - min), then sum min. This expects the data
    parameter to be normalized in [0, 1] interval
    :return: the normalized (or denormalized if reverse=True) data
    """
    if reverse:
        # denormalize
        return data * _denominator + _min
    else:
        # normalize
        return (data - _min) / _denominator


def remove_zeros_from_traj(traj):
    """
    Remove the start of the trajectory, if it only contains zeroes, while always keeping the last instant. This is
    especially useful for when the trajectory consists of relative displacements (or velocities), because it removes
    'redundant' instants where the agents are not moving
    :param traj: Tensor of shape (traj_len, batch, 2). The observed trajectory (may be absolute coordinates, or
    velocities)
    :return: trajectory with starting zeroes removed
    """
    t_start = 0
    for t in range(0, traj.shape[0] - 1):
        num_non_zeroes = torch.nonzero(traj[t, :, :])
        if num_non_zeroes.shape[t] == 0:
            t_start = t + 1
        else:
            break
    return traj[t_start:, :, :]


def trajs_with_inside_points(trajectories, parts=2):
    """

    :param trajectories:
    :param parts:
    :return:
    """
    trajectories_extended = torch.zeros((trajectories.shape[0] - 1) * (parts + 1), trajectories.shape[1],
                                        trajectories.shape[2])
    for t in range(trajectories.shape[0] - 1):
        # create segments, ignoring the first position (which would be the last position from the prior instant)
        # assumed that the first instant
        segment = get_inside_points(trajectories[t], trajectories[t + 1], parts)[:, :, 1:]
        # the number of intermediate points virtually goes to the time dimension
        trajectories_extended[t * (parts + 1):(t + 1) * (parts + 1), :, :] = segment.permute(2, 0, 1)
    return trajectories_extended


def get_inside_points(points1, points2, parts=2):
    """
    Build 2D line segment between start points, with a specified number of intermediate points.
    :param points1: Tensor of shape (batch, 2), starting points
    :param points2: Tensor of shape (batch, 2), ending points
    :param parts: when building the line segments that unite the two points of consecutive instants, how many
    intermediate points will be included (this excludes start and end of the segment)
    :return: Tensor of shape (batch,2,parts+2). Equally distanced points between starting and ending "control" points
    """
    points = torch.zeros(points1.shape[0], points1.shape[1], parts + 2, device=points1.device)
    for i, (p1, p2) in enumerate(zip(points1, points2)):
        points[i, :, :] = torch.cat((torch.linspace(p1[0], p2[0], parts + 2, device=points.device).unsqueeze(0),
                                     torch.linspace(p1[1], p2[1], parts + 2, device=points.device).unsqueeze(0)), dim=0)
    return points


def tensor_size_bytes(tensor, unit='MB'):
    """
    Get the size of the tensor in bytes, or a unit that's multiple of bytes
    :param tensor: the pytorch tensor
    :param unit: GigaBytes or GB (assumes GB=1e9 Bytes), MegaBytes or MB (assumes MB=1e6 Bytes),
    KiloBytes or KB (assumes KB=1e3 Bytes), Bytes or B
    :return: the size of the tensor in the desired unit
    """
    if 'G' in unit.upper():
        factor = 1e9
    elif 'M' in unit.upper():
        factor = 1e6
    elif 'K' in unit.upper():
        factor = 1e3
    else:
        factor = 1.0
    return (tensor.element_size() * tensor.nelement()) / factor


def random_rotation(xy):
    """
    Random rotation of a scene with (random) angle between 0 and 2*pi
    CREDITS: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/augmentation.py
    :param xy:
    :return:
    """
    theta = random.random() * 2.0 * math.pi
    return theta_rotation(xy, theta), theta


def theta_rotation(xy, theta):
    """
    CREDITS: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/augmentation.py
    :param xy:
    :param theta:
    :return:
    """
    ct = math.cos(theta)
    st = math.sin(theta)
    r = torch.tensor([[ct, st], [-st, ct]], device=xy.device, dtype=xy.dtype)
    return torch.einsum('ptc,ci->pti', xy, r)


def shift(xy, center):
    """
    CREDITS: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/augmentation.py
    :param xy:
    :param center:
    :return:
    """
    # theta = random.random() * 2.0 * math.pi
    xy = xy - center[np.newaxis, np.newaxis, :]
    return xy


def center_scene(xy, obs_length=9, ped_id=0):
    """
    CREDITS: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/augmentation.py
    :param xy:
    :param obs_length:
    :param ped_id:
    :return:
    """
    xy_copy = xy.clone()
    # Center
    center = xy[obs_length - 1, ped_id]  # Last Observation
    xy = shift(xy, center)
    # Rotate
    last_obs = xy[obs_length - 1, ped_id]
    second_last_obs = xy[obs_length - 2, ped_id]
    # diff = torch.tensor([last_obs[0] - second_last_obs[0], last_obs[1] - second_last_obs[1]], xy.device)
    theta = torch.atan2(last_obs[1] - second_last_obs[1], last_obs[0] - second_last_obs[0])
    # CHANGE from original implementation - east->west (left->right) instead of south->north (down->up)
    rotation = -theta  # + np.pi/2
    xy = theta_rotation(xy, rotation)
    return xy, rotation, center


def normalize_scene(_xy, rotation, center):
    """
    perform normalization of a scene or situation or set of 2D trajectories. First they are shifted, then rotation
    :param _xy: 2D trajectories to normalize
    :param rotation: angle to perform 2D rotation
    :param center: (x,y) position to shift trajectories
    :return: the trajectories, normalized (shifted then rotated)
    """
    xy_shifted = shift(_xy, center) if center is not None else _xy
    xy_rot = theta_rotation(xy_shifted, rotation) if rotation is not None else xy_shifted
    return xy_rot


def inverse_scene(_xy, rotation, center):
    """
    convert normalized trajectories back to original. First they are rotated back, then shifted back to original
    :param _xy: normalzied 2D trajectories
    :param rotation: original rotation angle
    :param center: original (x,y) position used to shift
    :return: the trajectories, de-normalized (rotated then shifted)
    """
    xy_rot = theta_rotation(_xy, -rotation) if rotation is not None else _xy
    xy_shifted = shift(xy_rot, -center) if center is not None else xy_rot
    return xy_shifted


def normalize_sequences(traj, seq_start_end, metadata_list=None, inverse=False):
    """
    SPECIFIC TO TRAJNET++ data configuration
    Perform a normalization or de-normalization of the direction of the scene as the direction of the primary
    pedestrian. Particularly useful for use with motion fields, that should receive the trajectory without its
    input normalized.
    :param traj: Tensor of shape (traj_len, num_peds, 2). The actual trajectory
    :param seq_start_end: Tensor of shape (num_seqs, 2). Indicates which trajectories belong to a certain sequence
    (that belong to the same time frame). One primary pedestrian per sequence
    :param metadata_list: list of metadata components, containing the direction of the pedestrian in question
    :param inverse: if True, will de-normalize the trajectory (original direction). If False, will normalize.
    :return: Tensor of shape (traj_len, num_peds, 2). The normalized, or de-normalized (in terms of direction of
    primary pedestrian) trajectory.
    """
    with torch.no_grad():
        if metadata_list is None or (not isinstance(metadata_list, list) and not isinstance(metadata_list, tuple)) \
                or not metadata_list:
            # no information to perform normalization or de-normalization
            return traj
        traj_out = traj.clone()
        for i, metadata in enumerate(metadata_list):
            rot = metadata.angle if hasattr(metadata, 'angle') else None
            center = metadata.center if hasattr(metadata, 'center') else None
            seq = traj[:, seq_start_end[i, 0]:seq_start_end[i, 1]]
            traj_out[:, seq_start_end[i, 0]:seq_start_end[i, 1]] = \
                normalize_scene(seq, rot, center) if not inverse else inverse_scene(seq, rot, center)
        return traj_out
