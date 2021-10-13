"""
Created March 6th, 2021
Script in which it is possible to elaborate statistics regarding the dataset(s) being used
If --trajnetpp is used (or data is in trajnetpp format - with file path having \'trajnetpp\' in it)
    then you should also supply --no_partial_trajectories

"""
import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from models.data.loaders import get_data_loader
from models.losses_and_metrics import num_collisions
from models.utils.parser_options import add_parser_arguments_for_data, add_parser_arguments_misc, \
    override_args_from_json

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_data(parser)
parser = add_parser_arguments_misc(parser)
parser.add_argument('--data_location', default='datasets_in_trajnetpp21/train/',
                    help='the relative path to the directory where the f data files are, or relative path to a file')
parser.add_argument('--data_location_abs', default='',
                    help='the absolute path to the directory where the data files are, or absolute path to a file')

parser.add_argument('--do_not_plot', action='store_true', help='Do not plot histograms')
parser.add_argument('--len_threshold_start', default=0, type=int,
                    help='start threshold of trajectory length for histograms')
parser.add_argument('--len_threshold_end', default=100, type=int,
                    help='end threshold of trajectory length for histograms')
parser.add_argument('--dont_put_bins_as_x_ticks', action='store_true',
                    help='If used, will not used the several bins created in the histogram as x ticks for the plot')
parser.add_argument('--num_bins', default=10, type=int, help='number of bins for the histogram')
parser.add_argument('--y_ticks', default=10, type=int, help='number of ticks for plot, in y axis')
parser.add_argument('--social_stats', action='store_true',
                    help='If supplied, will also perform and show some statistics regarding social aspects of '
                         'interacting pedestrians. Usually best if supplied with --fixed_len. If the dataset has lots '
                         'of trajectories, it make take some minutes to compute the statistics')
parser.add_argument('--col_thresh', type=float, nargs='+', default=0.1,
                    help='List of collision thresholds to employ in order to compute dataset statistics.')


def main(args):
    if args.use_gpu and not torch.cuda.is_available():
        args.use_gpu = False
        print("WARNING: Use GPU was activated but CUDA is not available for this pytorch version. Will use CPU instead")
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    data_location = os.path.abspath(args.data_location_abs) if args.data_location_abs else os.path.relpath(
        args.data_location)
    # get_data_loader returns a function object to retrieve the dataloader
    dataset, loader = (get_data_loader(args, data_location))(args, device, data_location)
    num_sequences = 0
    # num_distinct_primary_pedestrians only applicable for Trajnet++ datasets
    num_pedestrians = 0
    pedestrians_per_sequence_list = []
    trajectory_length_total = 0.0
    trajectory_length_list = []
    len_threshold_start, len_threshold_end = args.len_threshold_start, args.len_threshold_end
    max_traj_len, min_traj_len = 0.0, float('inf')
    obs_traj_len_total, pred_traj_len_total = 0.0, 0.0
    avg_displacement, avg_velocity = 0.0, 0.0
    obs_avg_displacement, obs_avg_velocity = 0.0, 0.0
    pred_avg_displacement, pred_avg_velocity = 0.0, 0.0
    abs_avg_distance_travelled, abs_avg_velocity = 0.0, 0.0
    abs_obs_avg_distance_travelled, abs_obs_avg_velocity = 0.0, 0.0
    abs_pred_avg_distance_travelled, abs_pred_avg_velocity = 0.0, 0.0
    velocity_list, obs_velocity_list, pred_velocity_list = torch.tensor([], device=device), \
                                                           torch.tensor([], device=device), \
                                                           torch.tensor([], device=device)
    acceleration_list, obs_acceleration_list, pred_acceleration_list = torch.tensor([], device=device), \
                                                                       torch.tensor([], device=device), \
                                                                       torch.tensor([], device=device)
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
    # list according to the number of different thresholds provided
    args.col_thresh = args.col_thresh if isinstance(args.col_thresh, list) else [args.col_thresh]
    total_num_colliding_pedestrians = [0] * len(args.col_thresh) if isinstance(args.col_thresh, list) else [0]
    obs_num_colliding_pedestrians = [0] * len(args.col_thresh) if isinstance(args.col_thresh, list) else [0]
    pred_num_colliding_pedestrians = [0] * len(args.col_thresh) if isinstance(args.col_thresh, list) else [0]
    total_num_colliding_primary_pedestrians = [0] * len(args.col_thresh) if isinstance(args.col_thresh, list) else [0]
    obs_num_colliding_primary_pedestrians = [0] * len(args.col_thresh) if isinstance(args.col_thresh, list) else [0]
    pred_num_colliding_primary_pedestrians = [0] * len(args.col_thresh) if isinstance(args.col_thresh, list) else [0]
    num_batches = len(loader)
    print(num_batches)
    for batch in loader:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _, _, seq_start_end) = batch
        _num_sequences = seq_start_end.shape[0]
        _num_pedestrians = 0
        obs_len, pred_len = obs_traj.shape[0], pred_traj_gt.shape[0]
        traj_len = obs_len + pred_len
        obs_traj_len_total += _num_pedestrians * obs_len
        pred_traj_len_total += _num_pedestrians * pred_len
        for (start, end) in seq_start_end.numpy():
            _num_pedestrians += (end - start)
            pedestrians_per_sequence_list.append(end - start)
            if args.social_stats:
                _obs_num_colliding_pedestrians = num_collisions(obs_traj[:, start:end, :],
                                                                col_thresh=args.col_thresh, mode='raw')
                _pred_num_colliding_pedestrians = num_collisions(
                    torch.cat((obs_traj[-1, start:end, :].unsqueeze(0), pred_traj_gt[:, start:end, :])),
                    col_thresh=args.col_thresh, mode='raw')
                _total_num_colliding_pedestrians = num_collisions(
                    torch.cat((obs_traj[:, start:end, :], pred_traj_gt[:, start:end, :]), dim=0),
                    col_thresh=args.col_thresh, mode='raw')
                for i in range(len(args.col_thresh)):
                    obs_num_colliding_pedestrians[i] += int(torch.sum(_obs_num_colliding_pedestrians[i]))
                    pred_num_colliding_pedestrians[i] += int(torch.sum(_pred_num_colliding_pedestrians[i]))
                    total_num_colliding_pedestrians[i] += int(torch.sum(_total_num_colliding_pedestrians[i]))
                    obs_num_colliding_primary_pedestrians[i] += int(torch.sum(_obs_num_colliding_pedestrians[i, 0]))
                    pred_num_colliding_primary_pedestrians[i] += int(torch.sum(_pred_num_colliding_pedestrians[i, 0]))
                    total_num_colliding_primary_pedestrians[i] += int(torch.sum(_total_num_colliding_pedestrians[i, 0]))
        if (not args.fixed_len and not args.variable_len) and not args.no_partial_trajectories:
            # remove partial trajectories (that contain NaNs); specific to Trajnet++
            mask = (torch.any(torch.isnan(obs_traj[:, :, 0]), dim=0) +
                    torch.any(torch.isnan(obs_traj_rel[:, :, 0]), dim=0) +
                    torch.any(torch.isnan(pred_traj_gt[:, :, 0]), dim=0) +
                    torch.any(torch.isnan(pred_traj_gt_rel[:, :, 0]), dim=0)) == 0
            obs_traj, obs_traj_rel = obs_traj[:, mask, :], obs_traj_rel[:, mask, :]
            pred_traj_gt, pred_traj_gt_rel = pred_traj_gt[:, mask, :], pred_traj_gt_rel[:, mask, :]
        batch_min_x, batch_min_y = torch.min(torch.cat((obs_traj[:, :, 0], pred_traj_gt[:, :, 0]))), \
                                   torch.min(torch.cat((obs_traj[:, :, 1], pred_traj_gt[:, :, 1])))
        batch_max_x, batch_max_y = torch.max(torch.cat((obs_traj[:, :, 0], pred_traj_gt[:, :, 0]))), \
                                   torch.max(torch.cat((obs_traj[:, :, 1], pred_traj_gt[:, :, 1])))
        min_x, min_y = batch_min_x if batch_min_x < min_x else min_x, batch_min_y if batch_min_y < min_y else min_y
        max_x, max_y = batch_max_x if batch_max_x > max_x else max_x, batch_max_y if batch_max_y > max_y else max_y
        obs_traj_len_total += _num_pedestrians * obs_len
        pred_traj_len_total += _num_pedestrians * pred_len
        num_pedestrians += _num_pedestrians
        num_sequences += _num_sequences
        acceleration_obs = torch.zeros_like(obs_traj_rel)
        acceleration_obs[1:, :, :] = obs_traj_rel[1:, :, :] - obs_traj_rel[:-1, :, :]
        acceleration_pred = torch.zeros_like(pred_traj_gt_rel)
        acceleration_pred[1:, :, :] = pred_traj_gt_rel[1:, :, :] - pred_traj_gt_rel[:-1, :, :]
        # accumulating average distance values
        abs_d_obs = torch.sum(obs_traj_rel.norm(dim=2), dim=0)
        abs_d_pred = torch.sum(pred_traj_gt_rel.norm(dim=2), dim=0)
        abs_d_total = abs_d_obs + abs_d_pred
        abs_sum_acc_obs = torch.sum(acceleration_obs.norm(dim=2), dim=0)
        abs_sum_acc_pred = torch.sum(acceleration_pred.norm(dim=2), dim=0)
        abs_sum_acc_total = abs_sum_acc_obs + abs_sum_acc_pred
        abs_avg_distance_travelled += torch.sum(abs_d_total)
        abs_obs_avg_distance_travelled += torch.sum(abs_d_obs)
        abs_pred_avg_distance_travelled += torch.sum(abs_d_pred)
        abs_avg_velocity += torch.sum(abs_d_total) / (traj_len - 1)
        abs_obs_avg_velocity += torch.sum(abs_d_obs) / (obs_len - 1 if obs_len > 1 else obs_len)
        abs_pred_avg_velocity += torch.sum(abs_d_pred) / (pred_len - 1 if pred_len > 1 else pred_len)
        velocity_list = torch.cat((velocity_list, abs_d_total / (traj_len - 1)))
        obs_velocity_list = torch.cat((obs_velocity_list, abs_d_obs / (obs_len - 1 if obs_len > 1 else obs_len)))
        pred_velocity_list = torch.cat((pred_velocity_list, abs_d_pred / (pred_len - 1 if pred_len > 1 else pred_len)))
        acceleration_list = torch.cat((acceleration_list, abs_sum_acc_total / (traj_len - 1)))
        obs_acceleration_list = torch.cat(
            (obs_acceleration_list, abs_sum_acc_obs / (obs_len - 1 if obs_len > 1 else obs_len)))
        pred_acceleration_list = torch.cat(
            (pred_acceleration_list, abs_sum_acc_pred / (pred_len - 1 if pred_len > 1 else pred_len)))
        displacement_obs = torch.norm(obs_traj[-1] - obs_traj[0], dim=1)
        displacement_pred = torch.norm(pred_traj_gt[-1] - pred_traj_gt[0], dim=1)
        displacement_total = torch.norm(pred_traj_gt[-1] - obs_traj[0], dim=1)
        avg_displacement += torch.sum(torch.abs(displacement_total))
        obs_avg_displacement += torch.sum(torch.abs(displacement_obs))
        pred_avg_displacement += torch.sum(torch.abs(displacement_pred))
        avg_velocity += torch.sum(torch.abs(displacement_total)) / (traj_len - 1)
        obs_avg_velocity += torch.sum(torch.abs(displacement_obs)) / (obs_len - 1 if obs_len > 1 else obs_len)
        pred_avg_velocity += torch.sum(torch.abs(displacement_pred)) / (pred_len - 1 if pred_len > 1 else pred_len)
        if traj_len < min_traj_len:
            min_traj_len = traj_len
        if traj_len > max_traj_len:
            max_traj_len = traj_len
        if len_threshold_start <= traj_len <= len_threshold_end:
            trajectory_length_list.extend([traj_len] * _num_pedestrians)
        trajectory_length_total += _num_pedestrians * traj_len
    trajectory_lengths = np.array(trajectory_length_list)
    avg_ped_per_sequence = float(num_pedestrians) / float(num_sequences)
    print("Total number of sequences: ", num_sequences)
    print("Total number of pedestrians: ", num_pedestrians)
    if hasattr(dataset, 'num_distinct_primary_pedestrians'):
        print("Total number of distinct primary pedestrians: ", dataset.num_distinct_primary_pedestrians)
    # print("Average number of pedestrians in each sequence: {0:.3f}".format(avg_ped_per_sequence))
    pedestrians_per_sequence_list = np.array(pedestrians_per_sequence_list)
    print(f"Pedestrians per sequence: Range {np.min(pedestrians_per_sequence_list)}-"
          f"{np.max(pedestrians_per_sequence_list)}; Avg={np.mean(pedestrians_per_sequence_list):.3f}; "
          f"Std={np.std(pedestrians_per_sequence_list):.3f}")
    if args.social_stats:
        print(f"Total number of colliding pedestrians: {total_num_colliding_pedestrians}")
        print(f"Obs number of colliding pedestrians: {obs_num_colliding_pedestrians}")
        print(f"Pred number of colliding pedestrians: {pred_num_colliding_pedestrians}")
        print("Percentage number of colliding pedestrians (%): "
              f"{[c / float(num_pedestrians) * 100.0 for c in total_num_colliding_pedestrians]}")
        print("Percentage Obs number of colliding pedestrians (%): "
              f"{[c / float(num_pedestrians) * 100.0 for c in obs_num_colliding_pedestrians]}")
        print("Percentage Pred number of colliding pedestrians: "
              f"{[c / float(num_pedestrians) * 100.0 for c in pred_num_colliding_pedestrians]}")
        print(f"Total number of colliding primary pedestrians: {total_num_colliding_primary_pedestrians}")
        print(f"Obs number of colliding primary pedestrians: {obs_num_colliding_primary_pedestrians}")
        print(f"Pred number of colliding primary pedestrians: {pred_num_colliding_primary_pedestrians}")
        print("Percentage number of colliding primary pedestrians (%): "
              f"{[c / float(num_sequences) * 100.0 for c in total_num_colliding_primary_pedestrians]}")
        print("Percentage Obs number of colliding primary pedestrians (%): "
              f"{[c / float(num_sequences) * 100.0 for c in obs_num_colliding_primary_pedestrians]}")
        print("Percentage Pred number of colliding primary pedestrians: "
              f"{[c / float(num_sequences) * 100.0 for c in pred_num_colliding_primary_pedestrians]}")
    print("")
    print("Trajectory length range: {0}-{1}. Average trajectory length: {2:.3f}, Std: {3:.3f}".format(
        min_traj_len, max_traj_len, trajectory_length_total / float(num_pedestrians), np.std(trajectory_lengths)))
    print("Average observed trajectory length: {0:.3f}.\t Average predicted trajectory length: {1:.3f}".format(
        obs_traj_len_total / float(num_pedestrians), pred_traj_len_total / float(num_pedestrians)))
    # display velocities
    print("Avg. Total distance travelled = {0:.3f}; Obs = {1:.3f}; Pred = {2:.3f}".format(
        abs_avg_distance_travelled / float(num_pedestrians), abs_obs_avg_distance_travelled / float(num_pedestrians),
        abs_pred_avg_distance_travelled / float(num_pedestrians)))
    print("Avg. velocity from total distance = {0:.3f}; Obs = {1:.3f}; Pred = {2:.3f}".format(
        abs_avg_velocity / float(num_pedestrians), abs_obs_avg_velocity / float(num_pedestrians),
        abs_pred_avg_velocity / float(num_pedestrians)))
    print("Avg. displacement = {0:.3f}; Obs = {1:.3f}; Pred = {2:.3f}".format(
        avg_displacement / float(num_pedestrians), obs_avg_displacement / float(num_pedestrians),
        pred_avg_displacement / float(num_pedestrians)))
    print("Avg. velocity from displacement = {0:.3f}; Obs = {1:.3f}; Pred = {2:.3f}".format(
        avg_velocity / float(num_pedestrians), obs_avg_velocity / float(num_pedestrians),
        pred_avg_velocity / float(num_pedestrians)))
    velocity_std_and_mean = torch.std_mean(velocity_list)
    obs_velocity_std_and_mean = torch.std_mean(obs_velocity_list)
    pred_velocity_std_and_mean = torch.std_mean(pred_velocity_list)
    acceleration_std_and_mean = torch.std_mean(acceleration_list)
    obs_acceleration_std_and_mean = torch.std_mean(obs_acceleration_list)
    pred_acceleration_std_and_mean = torch.std_mean(pred_acceleration_list)
    print("Full trajectory velocity: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
        torch.min(velocity_list).data, torch.max(velocity_list).data, velocity_std_and_mean[1].data,
        velocity_std_and_mean[0].data))
    print("Obs. trajectory velocity: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
        torch.min(obs_velocity_list).data, torch.max(obs_velocity_list).data, obs_velocity_std_and_mean[1].data,
        obs_velocity_std_and_mean[0].data))
    print("Pred trajectory velocity: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
        torch.min(pred_velocity_list).data, torch.max(pred_velocity_list).data, pred_velocity_std_and_mean[1].data,
        pred_velocity_std_and_mean[0].data))
    print("Full trajectory acceleration: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
        torch.min(acceleration_list).data, torch.max(acceleration_list).data, acceleration_std_and_mean[1].data,
        acceleration_std_and_mean[0].data))
    print("Obs. trajectory acceleration: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
        torch.min(obs_acceleration_list).data, torch.max(obs_acceleration_list).data,
        obs_acceleration_std_and_mean[1].data, obs_acceleration_std_and_mean[0].data))
    print("Pred trajectory acceleration: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
        torch.min(pred_acceleration_list).data, torch.max(pred_acceleration_list).data,
        pred_acceleration_std_and_mean[1].data, pred_acceleration_std_and_mean[0].data))
    print(f"Data between x in [{min_x:.2f},{max_x:.2f}], and y in [{min_y:.2f},{max_y:.2f}]")
    if not args.do_not_plot:
        y_ticks = args.y_ticks
        n, bins, patches = plt.hist(trajectory_lengths, bins=args.num_bins)
        if not args.dont_put_bins_as_x_ticks:
            plt.xticks(bins.astype(int))
        max_number = np.max(n)
        print("Total number of trajectories in histogram: %d" % np.sum(n))
        if max_number > 100:
            last_y = 100 * int(math.ceil(max_number / 100))
        else:
            last_y = int(max_number)
        plt.yticks(range(0, last_y, int(last_y / y_ticks)))
        plt.figure()
        plt.hist(velocity_list.detach().numpy(), bins=args.num_bins * 2)
        plt.figure()
        plt.hist(obs_velocity_list.detach().numpy(), bins=args.num_bins * 2)
        plt.figure()
        plt.hist(pred_velocity_list.detach().numpy(), bins=args.num_bins * 2)
        plt.show()


if __name__ == '__main__':
    arguments = parser.parse_args()
    if hasattr(arguments, 'load_args_from_json') and arguments.load_args_from_json:
        new_args = override_args_from_json(arguments, arguments.load_args_from_json, parser)
    else:
        new_args = arguments
    main(new_args)
