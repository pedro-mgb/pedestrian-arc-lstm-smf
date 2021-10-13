"""
Created on April 10th, 2021
Script that displays the trajectories from a specific dataset.
"""
import argparse
import copy
import os

import torch

import matplotlib.pyplot as plt

from models.data.loaders import get_data_loader
from models.data.environment import Environment
from models.utils.parser_options import add_parser_arguments_for_data, add_parser_arguments_plotting, \
    add_parser_arguments_for_testing
from models.utils.utils import rotate_trajectory

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_testing(parser)
parser = add_parser_arguments_for_data(parser)
parser = add_parser_arguments_plotting(parser)
parser.add_argument('--data_location', default='datasets/test/',
                    help='the relative path to the directory where the f data files are, or relative path to a file')


def main(args):
    device = torch.device('cpu')
    data_location = os.path.relpath(args.data_location)
    # get_data_loader returns a function object to retrieve the dataloader
    dataset, loader = (get_data_loader(args, data_location))(args, device, data_location)

    num_trajectories = 0
    plt.figure()
    environment, environment_plot = None, None
    num_collisions_with_env_total = num_collisions_with_env_obs = num_collisions_with_env_pred = 0
    if args.environment_location:
        # also display the static environment - in the background
        environment = Environment.load(args.environment_location)
        # make a copy, since the trajectories may change when being plotted
        environment_plot = Environment(copy.deepcopy(environment.obstacles), copy.deepcopy(environment.scene_bounds))
        environment_plot.change(args.switch_x_y, args.invert_x, args.invert_y)
        environment_plot.plot(plt)

    for batch in loader:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, seq_start_end) = batch
        # iterate over the full trajectories, one at a time (hence the permute)
        full_traj = torch.cat((obs_traj, pred_traj_gt), dim=0)
        if environment:
            collisions_full_traj = environment.compute_collisions(full_traj, combine_cse_osb=True)
            collisions_obs_traj = environment.compute_collisions(obs_traj, combine_cse_osb=True)
            # include last observed position for prediction to consider displacement caused by first position
            collisions_pred_traj = environment.compute_collisions(
                torch.cat((obs_traj[-1].unsqueeze(0), pred_traj_gt), ), combine_cse_osb=True)
            num_collisions_with_env_total += int(torch.sum(collisions_full_traj).cpu().detach().data)
            num_collisions_with_env_obs += int(torch.sum(collisions_obs_traj).cpu().detach().data)
            num_collisions_with_env_pred += int(torch.sum(collisions_pred_traj).cpu().detach().data)
        if args.rotate_by:
            full_traj = rotate_trajectory(full_traj, args.rotate_by, inplace=True)
        if args.switch_x_y:
            full_traj_c = full_traj.clone()
            full_traj[:, :, 0] = full_traj_c[:, :, 1]
            full_traj[:, :, 1] = full_traj_c[:, :, 0]
        if args.invert_x:
            full_traj[:, :, 0] = - full_traj[:, :, 0]
        if args.invert_y:
            full_traj[:, :, 1] = - full_traj[:, :, 1]
        full_traj = full_traj.permute(1, 0, 2)
        num_trajectories += full_traj.shape[0]
        for traj in full_traj:
            traj_np = traj.cpu().detach().numpy()
            plt.scatter(traj_np[:, 0], traj_np[:, 1], color='blue', s=10, alpha=0.5)
            # start and end positions
            if args.distinguish_start_end:
                plt.scatter(traj_np[0, 0], traj_np[0, 1], color='green', s=10, alpha=0.5)
                plt.scatter(traj_np[-1, 0], traj_np[-1, 1], color='red', s=10, alpha=0.5)
            plt.plot(traj_np[:, 0], traj_np[:, 1], color='blue', linewidth=1, alpha=0.5)
            plt.xlabel(f'x ({args.units})', fontdict={'fontsize': 14})
            plt.ylabel(f'y ({args.units})', fontdict={'fontsize': 14})
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
        # will always plot the last batch, even if part of it exceeds the 'max_trajectories' threshold
        if num_trajectories > args.max_trajectories:
            break
    if environment:
        print(f'Collisions with static environment: Total={num_collisions_with_env_total}; '
              f'Obs={num_collisions_with_env_obs}; Pred={num_collisions_with_env_pred}')
    plt.show()


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
