"""
Created on 20/08/2021
Script that plots predictions made by models, but unlike 'other_scripts/plot_single_predictions.py', this script
focuses on the social aspect of the predictions, considering the neighbours, and seeing in which cases collisions occur

This script is more applicable to Trajnet++ data, that assumes the existence of a primary pedestrian.
"""
import argparse
import copy
import os

import torch
import matplotlib.pyplot as plt

from models.data.loaders import load_test_data
from models.data.environment import Environment
from models.utils.parser_options import add_parser_arguments_for_data, add_parser_arguments_for_testing, \
    add_parser_arguments_plotting, add_parser_arguments_misc, override_args_from_json
from models.utils.evaluator import TrajectoryType, map_traj_type, __batch_metrics_unimodal__
from models.utils.plotting import apply_plot_args_to_trajectories, plot_situation, AnimatedSituation, \
    get_or_compute_predictions
from models.losses_and_metrics import num_collisions_between_two
from models.evaluate import load_model_s, ModelType
from models.classical.constant_velocity import predict_const_vel

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_data(parser)
parser = add_parser_arguments_for_testing(parser)
parser = add_parser_arguments_plotting(parser)
parser = add_parser_arguments_misc(parser)
parser.add_argument('--disable_gc', action='store_true')

# the animation objects must be stored so that the animation can be properly rendered
animation_list = []


def main(args):
    global animation_list
    # plt.rcParams.update({'figure.max_open_warning': 0})  # to avoid showing a warning about too much figures
    # FORCE USE OF CPU; GPU wouldn't really have any benefit here
    device = torch.device('cpu')
    test_loaders, file_names = load_test_data(args, device)
    if args.model_paths is None or args.model_labels is None:
        raise Exception('You must supply a list of models via --model_paths and a associated labels via --labels.')
    if len(args.model_paths) != len(args.model_labels):
        raise Exception(f'The number of model paths (received {len(args.model_paths)}) must be equal to the number of '
                        f'associated labels (received {len(args.model_labels)}).')
    assert args.plot_limits is None or len(args.plot_limits) == 0 or len(args.plot_limits) == 4, \
        'You must provide either 4 values for the plot limits (xmin xmax ymin ymax), or not use --plot_limits'
    models, train_args, input_types, output_types = load_all_models(args, device, file_names)
    print("Loaded all models. Beginning evaluation / display of trajectory predictions")
    observed_trajectories, observed_displacements, ground_truth_trajectories = [], [], []
    models_prediction = []
    start_end_list = []
    num_batches = sum([len(loader) for loader in test_loaders])
    curr_batch = 0
    num_trajectories_discarded, num_trajectories_plotted = 0, 0

    # prepare the static environment for plotting
    if args.environment_location:
        # also display the static environment - in the background
        environment = Environment.load(args.environment_location)
        environment_plot = Environment(copy.deepcopy(environment.obstacles), copy.deepcopy(environment.scene_bounds))
        environment_plot.change(args.switch_x_y, args.invert_x, args.invert_y)
    else:
        environment = environment_plot = None

    for loader_idx, loader in enumerate(test_loaders):
        for batch in loader:
            curr_batch += 1
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, metadata, loss_mask, seq_start_end) = batch
            print(f'\rBatch {curr_batch}/{num_batches}', end='')
            if obs_traj.shape[0] + pred_traj_gt.shape[0] <= args.length_threshold:
                num_trajectories_discarded += 1
                continue
            observed_trajectories.append(obs_traj)
            observed_displacements.append(obs_traj_rel)
            ground_truth_trajectories.append(pred_traj_gt)
            start_end_list.append(seq_start_end)
            models_prediction.append(torch.tensor([], device=device))
            for idx, model in enumerate(models):
                if args.num_samples > 1:
                    raise Exception('TODO! Displaying multiple predictions for same pedestrian not implemented yet')
                else:
                    # get predictions and compute metrics
                    obs_traj_seq = obs_traj if input_types[idx] == TrajectoryType.ABS else obs_traj_rel
                    pred_traj_len = pred_traj_gt.shape[0]
                    prediction = get_or_compute_predictions(model, loader_idx, train_args[idx], input_types[idx],
                                                            output_types[idx], obs_traj_seq, obs_traj, seq_start_end,
                                                            metadata, pred_traj_len)
                    with torch.no_grad():
                        pred_seq, _, _ = __batch_metrics_unimodal__(args, prediction, output_types[idx], obs_traj,
                                                                    pred_traj_gt, pred_traj_len, obs_traj_rel,
                                                                    seq_start_end)
                    models_prediction[-1] = torch.cat((models_prediction[-1], pred_seq[:, :, :2].unsqueeze(0)), dim=0)
            for seq_idx, (start, end) in enumerate(seq_start_end):
                predictions_for_situation = models_prediction[-1][:, :, start:end]
                gt_rel_primary = torch.norm(torch.cat((obs_traj_rel[:, start, :], pred_traj_rel_gt[:, start, :]),
                                                      dim=0), dim=1)
                if torch.all(torch.abs(torch.sum(gt_rel_primary)) <= args.displacement_threshold):
                    # small displacement below specified threshold (e.g. pedestrian is stopped)
                    num_trajectories_discarded += 1
                    continue
                if (end - start - 1) > args.max_neighbours or (end - start - 1) < args.min_neighbours:
                    num_trajectories_discarded += 1
                    continue
                plotted = plot_method_predictions_situation(args, predictions_for_situation, obs_traj[:, start:end],
                                                            pred_traj_gt[:, start:end], environment, environment_plot,
                                                            args.model_labels, models)
                if not plotted:
                    num_trajectories_discarded += 1
                    continue
                else:
                    num_trajectories_plotted += 1
                    if num_trajectories_plotted >= args.max_trajectories:
                        break
            if num_trajectories_plotted >= args.max_trajectories:
                print(f'\rReached {args.max_trajectories} plots')
                break
    plt.show()


def plot_method_predictions_situation(args, predictions, obs_traj, pred_gt, environment, environment_to_plot,
                                      model_labels, model_list):
    """
    Perform plotting of a specific situation. May be a static plot, or an animation.
    Uses plotting methods from utilities package (models.utils.plotting.py)
    :param args: command line arguments to configure plotting and/or animation
    :param predictions: tensor of shape [num_models, pred_len, num_peds, 2]. Predicted trajectories of all
    pedestrians in a situation, for num_models supplied (num_models >= 1)
    :param obs_traj: tensor of shape [obs_len, num_peds, 2]. Observed trajectories of all pedestrians in a situation
    :param pred_gt: tensor of shape [pred_len, num_peds, 2]. Ground truth trajectories of all pedestrians in a situation
    :param environment: scene-specific environment (containing obstacles and/or scene limits); this environment will
    not be plotted, but instead it will be used to evaluate if certain predictions collide with the environment.
    :param environment_to_plot: scene-specific environment to plot
    :param model_labels: list of length num_models, containing labels to identify each of the models
    :param model_list: list of length num_models, containing the instances of each model used; may contain useful
    information about particular models, like the size of FOV shape pooling being used.
    :return: True if situation was plotted/animated, False otherwise (it may not be plotted due it the situation not
    fulfilling specific conditions).
    """
    trajnetpp = not args.fixed_len and not args.variable_len
    num_models = len(model_labels)
    if args.plot_limits and len(args.plot_limits) == 4:
        x_limits, y_limits = args.plot_limits[:2], args.plot_limits[2:]
    else:
        x_limits, y_limits = None, None
    if environment and args.only_plot_collisions_static and torch.sum(
            environment.compute_collisions(predictions[:, :, 0].permute(1, 0, 2), combine_cse_osb=True)) == 0:
        return False  # all primary pedestrian predictions comply with the scene environment
    if args.only_plot_collisions_ped:
        num_cols = 0
        for i in range(num_models):
            cols_p = num_collisions_between_two(predictions[i], predictions[i], col_thresh=args.collision_threshold,
                                                inter_points=args.num_inter_pts_cols, mode='raw')
            cols_gt = num_collisions_between_two(predictions[i], pred_gt, col_thresh=args.collision_threshold,
                                                 inter_points=args.num_inter_pts_cols, mode='raw')
            num_cols += ((cols_p[0] + cols_gt[0]) if trajnetpp else (torch.sum(cols_p) + torch.sum(cols_gt)))
        if num_cols <= 0:
            return False  # there are no collisions between pedestrians for the given predictions

    last_obs = obs_traj[-1].unsqueeze(0)
    predictions = torch.cat((last_obs.unsqueeze(0).repeat(num_models, 1, 1, 1), predictions[:, :, :, :]), dim=1)
    gt = torch.cat((last_obs, pred_gt), dim=0)
    obs_traj, gt, predictions = apply_plot_args_to_trajectories(args, obs_traj, gt, predictions)
    if args.animate:
        anim = AnimatedSituation(args, obs_traj, gt, predictions, environment_to_plot, model_list, model_labels,
                                 x_limits, y_limits)
        anim.save()
        animation_list.append(anim)
    else:
        # plot static figure
        plt.figure()
        if environment_to_plot:
            environment_to_plot.plot(plt)
        plot_situation(args, obs_traj, gt, predictions, model_labels, x_limits, y_limits)
    return True


def load_all_models(args, device, data_file_names):
    """
    load all models to plot predictions, or files with pre-computed predictions
    :param args: command line arguments with information and configuration options to load models
    :param device: torch.device to map the models to (e.g. map to cpu or cuda), if required
    :param data_file_names: names of the original data files with GT data. For the case of loading pre-computed
    predictions from files, checks if the names match.
    :return: 4 lists, all of the same length (number of models), containing in each element:
    1. the model
    2. train arguments used to create / train the model (if applicable)
    3. type of input it expects (ABSOLUTE POSITIONS, VELOCITIES, ACCELERATIONS)
    4. type outputted by the model (ABSOLUTE POSITIONS, VELOCITIES, ACCELERATIONS)
    """
    models, train_args, input_types, output_types = [], [], [], []
    for idx, path in enumerate(args.model_paths):
        file_name = os.path.basename(path)
        label_name = args.model_labels[idx]
        # first, check if the file has extension .ndjson - means it consists of pre-computed Trajnet++ predictions
        if args.model_paths[idx].endswith('ndjson'):
            if args.fixed_len or args.variable_len:
                raise Exception(f'Prediction file ({args.model_paths[idx]}) is only supported for Trajnet++ data')
            test_dir_cache = args.test_dir
            args.test_dir = args.model_paths[idx]
            loaders_pred, pred_file_names = load_test_data(args, device, load_pred=True)
            if not (len(pred_file_names) == len(data_file_names) and
                    [p == d for (p, d) in zip(pred_file_names, data_file_names)]):
                raise Exception(f'The provided path to load prediction ({args.model_paths[idx]}) does not match the '
                                f'provided data path ({test_dir_cache})')
            args.test_dir = test_dir_cache
            # need to convert to iter to get one batch a time
            models.append([iter(loader) for loader in loaders_pred])
            train_args.append(None)
            input_types.append(TrajectoryType.ABS)
            output_types.append(TrajectoryType.ABS)
        # otherwise - it is an actual model
        elif 'lstm' in file_name.lower():
            args.model_path = path
            model, _, train_arg = load_model_s(args, device, ModelType.LSTM)
            assert len(model) == 1, f'You cannot supply a directory, ({args.model_path})' \
                                    'the path to the model must be a single file, that ' 'must exist!'
            model, train_arg = model[0], train_arg[0]
            use_acceleration = train_arg.use_acc if hasattr(train_args, 'use_acc') else False
            if 'interaction' in model.__class__.__name__.lower() or \
                    'social' in model.__class__.__name__.lower():  # interaction-aware model
                input_type = TrajectoryType.ABS
                output_type = TrajectoryType.VEL
            elif 'fields' in model.__class__.__name__.lower():  # model uses motion fields
                input_type = TrajectoryType.ABS
                output_type = TrajectoryType.VEL
            else:
                input_type = output_type = map_traj_type(train_arg.use_abs, use_acceleration)
            models.append(model)
            train_args.append(train_arg)
            input_types.append(input_type)
            output_types.append(output_type)
        elif 'fields' in file_name.lower():
            args.model_path = path
            model, _, _ = load_model_s(args, device, ModelType.SMF)
            assert len(model) == 1, f'You cannot supply a directory, ({args.model_path})' \
                                    'the path to the model must be a single file, that ' 'must exist!'
            models.append(model[0])
            train_args.append(None)
            input_types.append(TrajectoryType.ABS)
            output_types.append(TrajectoryType.ABS)
        else:  # == classical model, if nothing is said (constant velocity)
            models.append(predict_const_vel)
            train_args.append(None)
            input_types.append(TrajectoryType.ABS)
            output_types.append(TrajectoryType.ABS)
    return models, train_args, input_types, output_types


if __name__ == '__main__':
    arguments = parser.parse_args()
    if hasattr(arguments, 'load_args_from_json') and arguments.load_args_from_json:
        new_args = override_args_from_json(arguments, arguments.load_args_from_json, parser)
    else:
        new_args = arguments
    main(new_args)
