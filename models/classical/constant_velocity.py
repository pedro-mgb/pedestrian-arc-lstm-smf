"""
Created on March 18th, 2021
Predicting trajectories using a simple constant velocity model
"""
import numpy as np
import torch

from models.utils.utils import relative_traj_to_abs


def predict_const_vel(obs_traj, pred_len=1, multi_modal=False, first_multi_modal=False):
    """
    predict trajectories assuming a constant velocity
    :param obs_traj: observed trajectory (assumed to be in absolute coordinates).
    Expected to be a tensor of shape (obs_len, batch, 2)
    :param pred_len: for how many instants to perform prediction
    :param multi_modal: if True, generate based on average and standard deviation of velocities in module.
    This generates just one prediction, but it can be called more than once, and results will be different.
    :param first_multi_modal: if True, will not use standard deviation, just average value. This may improve the results
    :return: the predicted trajectories. Tensor of shape (pred_len, batch, 2)
    """
    # this actually doesn't use velocity, uses relative displacement, but this is because one assumes a constant spacing
    # in time between the positions
    last_pos = obs_traj[-1]
    obs_rel = torch.zeros(obs_traj.shape, device=obs_traj.device)
    obs_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]
    if multi_modal:
        vel = __cv_multimodal_direction__(obs_rel)
    else:
        vel = last_pos - obs_traj[-2]
    pred_traj_rel = vel.repeat(pred_len, 1, 1)
    pred_traj = relative_traj_to_abs(pred_traj_rel, last_pos)
    return pred_traj


def __cv_multimodal_direction__(obs_rel, angle_std=25):
    """
    Credits for this implementation goes to https://github.com/cschoeller/constant_velocity_pedestrian_motion
    From the paper entitled "What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction"
    To cite, use:
    @article{article,
    author = {Schöller, Christoph and Aravantinos, Vincent and Lay, Florian and Knoll, Alois},
    year = {2020},
    month = {01},
    pages = {1-1},
    title = {What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction},
    volume = {PP},
    journal = {IEEE Robotics and Automation Letters},
    doi = {10.1109/LRA.2020.2969925}
    }

    :param obs_rel: Tensor of shape [obs_traj_len, num_peds, 2]. The relative displacements (simulating velocity) for
    each of the pedestrians.
    :param angle_std: Standard deviation of the rotation angle for the velocity direction, in degrees.
    By default will use 25 degrees, which is the value used in the original implementation
    :return:
    """
    last_vel = obs_rel[-1]
    sampled_angle_deg = np.random.normal(0, angle_std, last_vel.shape[0])
    theta = (sampled_angle_deg * np.pi) / 180.  # convert from degrees to radians
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = torch.tensor([[c, s], [-s, c]], dtype=last_vel.dtype)
    # i<->2, j<->2, n<->num_peds
    new_vel = torch.einsum('ijn,jn->in', rotation_mat, last_vel.T).T
    return new_vel


def __cv_multimodal_speed__(obs_traj, obs_rel, last_pos, first_multi_modal):
    num_peds = obs_rel.shape[1]
    dims_pos = obs_rel.shape[2]
    # Use the entire observed length and the absolute velocity across instants to weight the constant velocities to
    # use. If the velocity is 0 then it will not be used.
    vel_obs_std_mean = torch.std_mean(torch.abs(obs_rel), dim=0)
    # weight_vel_obs = torch.mean(torch.abs(obs_rel), dim=0)
    vel = last_pos - obs_traj[-2]
    for idx in range(num_peds):
        for pos in range(dims_pos):
            if torch.is_nonzero(vel[idx][pos]):
                if first_multi_modal:
                    weight = vel_obs_std_mean[1][idx][pos]
                else:
                    weight = torch.normal(mean=vel_obs_std_mean[1][idx][pos], std=vel_obs_std_mean[0][idx][pos])
                vel[idx][pos] *= weight / torch.abs(vel[idx][pos])
    return vel