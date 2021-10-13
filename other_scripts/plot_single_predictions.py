"""
Script to plot predictions, possibly from several models, and compare them with ground truth.
These predictions are only regarding one pedestrian - PRIMARY in the case of Trajnet++. Other pedestrians - neighbours
for the case of Trajnet++ - are ignored.

NOT IMPLEMENTED - Option to visualize multimodality, in case models can output several samples.
"""
import argparse
import copy

import torch
import matplotlib.pyplot as plt

from models.data.loaders import load_test_data
from models.data.environment import Environment
from models.utils.parser_options import add_parser_arguments_for_data, add_parser_arguments_for_testing, \
    add_parser_arguments_plotting, add_parser_arguments_misc, override_args_from_json
from models.utils.evaluator import TrajectoryType, map_traj_type, __batch_metrics_unimodal__
from models.utils.plotting import apply_plot_args_to_trajectories, get_or_compute_predictions
from models.evaluate import load_model_s, ModelType
from models.classical.constant_velocity import predict_const_vel

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_data(parser)
parser = add_parser_arguments_for_testing(parser)
parser = add_parser_arguments_plotting(parser)
parser = add_parser_arguments_misc(parser)
parser.add_argument('--all_prediction_per_method', action='store_true',
                    help='If true, will display all predictions for each method')

lstm_label = 'LSTM'
# lstm_enc_dec_label = 'LSTM_enc_dec'
sparse_motion_fields_label = 'SMF'
const_vel_label = 'CV'
method_choices = [lstm_label, const_vel_label, sparse_motion_fields_label]


def main(args):
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
    models, train_args, input_types, output_types = load_models(args, device, file_names)
    print("Loaded all models. Beginning evaluation / display of trajectory predictions")
    trajnetpp = not args.fixed_len and not args.variable_len
    observed_trajectories, observed_displacements, ground_truth_trajectories = [], [], []
    models_prediction = []
    ades_per_method, fdes_per_method = [], []
    best_model_ade, best_model_fde, worst_model_ade, worst_model_fde = [], [], [], []
    start_end_list = []
    num_batches = sum([len(loader) for loader in test_loaders])
    curr_batch = 0
    for loader_idx, loader in enumerate(test_loaders):
        for batch in loader:
            curr_batch += 1
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, metadata, loss_mask, seq_start_end) = batch
            print(f'\rBatch {curr_batch}/{num_batches}', end='')
            if trajnetpp:
                # only for primary pedestrians
                observed_trajectories.append(obs_traj[:, seq_start_end[:, 0]])
                observed_displacements.append(obs_traj_rel[:, seq_start_end[:, 0]])
                ground_truth_trajectories.append(pred_traj_gt[:, seq_start_end[:, 0]])
            else:
                observed_trajectories.append(obs_traj)
                observed_displacements.append(obs_traj_rel)
                ground_truth_trajectories.append(pred_traj_gt)
            start_end_list.append(seq_start_end)
            models_prediction.append(torch.tensor([], device=device))
            ades_per_method.append(torch.tensor([], device=device))
            fdes_per_method.append(torch.tensor([], device=device))
            for idx, model in enumerate(models):
                if args.num_samples > 1:
                    raise Exception('TODO! Displaying multiple predictions for same pedestrian not implemented yet')
                else:
                    obs_traj_seq = obs_traj if input_types[idx] == TrajectoryType.ABS else obs_traj_rel
                    pred_traj_len = pred_traj_gt.shape[0]
                    prediction = get_or_compute_predictions(model, loader_idx, train_args[idx], input_types[idx],
                                                            output_types[idx], obs_traj_seq, obs_traj, seq_start_end,
                                                            metadata, pred_traj_len)
                    pred_seq, ade, fde = __batch_metrics_unimodal__(args, prediction, output_types[idx], obs_traj,
                                                                    pred_traj_gt, pred_traj_len, obs_traj_rel,
                                                                    seq_start_end)
                    if trajnetpp:
                        # only for primary pedestrians
                        pred_seq = pred_seq[:, seq_start_end[:, 0]]
                    models_prediction[-1] = torch.cat((models_prediction[-1], pred_seq[:, :, :2].unsqueeze(0)), dim=0)
                    ades_per_method[-1] = torch.cat((ades_per_method[-1], ade.unsqueeze(0)), dim=0)
                    fdes_per_method[-1] = torch.cat((fdes_per_method[-1], fde.unsqueeze(0)), dim=0)
            best_model_ade.append(torch.argmin(ades_per_method[-1], dim=0))
            best_model_fde.append(torch.argmin(fdes_per_method[-1], dim=0))
            worst_model_ade.append(torch.argmax(ades_per_method[-1], dim=0))
            worst_model_fde.append(torch.argmax(fdes_per_method[-1], dim=0))
    if args.all_prediction_per_method:
        display_all_predictions_per_method(args, args.model_labels, observed_trajectories, models_prediction)
    else:
        display_predictions_for_methods(args, device, args.model_labels, observed_trajectories, observed_displacements,
                                        ground_truth_trajectories, models_prediction, best_model_ade, best_model_fde,
                                        worst_model_ade, worst_model_fde, start_end_list)


def display_all_predictions_per_method(args, model_labels, observed_trajectories, models_prediction):
    """
    Create a total of "num_models" plots, each containing the entirety of all predicted trajectories. Useful to see if
    the overall prediction horizon differs to a great extend from the GT
    :param args: command line arguments to regulate plots
    :param model_labels: list of length num_models. Labels to give to each model/plot
    :param observed_trajectories: All observed trajectories
    :param models_prediction: All model predictions
    :return: nothing, plotting is done here
    """
    if args.plot_limits and len(args.plot_limits) == 4:
        x_limits, y_limits = args.plot_limits[:2], args.plot_limits[2:]
    else:
        x_limits, y_limits = None, None
    if args.environment_location:
        # also display the static environment - in the background
        environment = Environment.load(args.environment_location)
        environment_plot = Environment(copy.deepcopy(environment.obstacles), copy.deepcopy(environment.scene_bounds))
        environment_plot.change(args.switch_x_y, args.invert_x, args.invert_y)
    else:
        environment = environment_plot = None
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    obs_color, model_colors = color_cycle[0], color_cycle[1:]
    for model_idx, model_label in enumerate(model_labels):
        plt.figure()
        plt.title(f'All predictions for model {model_label}', fontdict={'fontsize': 14})
        legend_shown = False
        num_collisions_with_env = num_trajectories = 0
        if environment:
            environment_plot.plot(plt)
        for idx in range(len(observed_trajectories)):
            last_obs = observed_trajectories[idx][-1, :, :].clone().unsqueeze(0)
            obs_trajs = observed_trajectories[idx]
            num_trajectories += obs_trajs.shape[1]
            # append last observed position to make the trajectory a single dash (not have the gap between last
            # observed position and first predicted position)
            predictions = torch.cat((last_obs, models_prediction[idx][model_idx, :, :, :]), dim=0)
            if environment:
                num_collisions_with_env += torch.sum(
                    environment.compute_collisions(predictions, combine_cse_osb=True)).cpu().detach().data
            predictions = predictions
            for p in range(obs_trajs.shape[1]):
                if legend_shown:
                    plt.plot(obs_trajs[:, p, 0], obs_trajs[:, p, 1], linewidth=1, color=obs_color)
                    plt.plot(predictions[:, p, 0], predictions[:, p, 1], linewidth=1, color=model_colors[model_idx])
                else:
                    legend_shown = True
                    plt.plot(obs_trajs[:, p, 0], obs_trajs[:, p, 1], linewidth=1, color=obs_color, label='OBS')
                    plt.plot(predictions[:, p, 0], predictions[:, p, 1], linewidth=1, color=model_colors[model_idx],
                             label=model_label)
                plt.scatter(obs_trajs[:, p, 0], obs_trajs[:, p, 1], s=50, alpha=0.7, color=obs_color)
                plt.scatter(predictions[1:, p, 0], predictions[1:, p, 1], s=50, alpha=0.7,
                            color=model_colors[model_idx])
                plt.xlabel(f'x ({args.units})', fontdict={'fontsize': 14})
                plt.ylabel(f'y ({args.units})', fontdict={'fontsize': 14})
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
        if environment:
            print(f'Model {model_label} environment collisions: {num_collisions_with_env}; '
                  f'Average per trajectory: {num_collisions_with_env / float(num_trajectories):.2f}')
        if x_limits and y_limits:
            plt.xlim(x_limits)
            plt.ylim(y_limits)
        plt.legend(loc=args.legend_location)
    plt.show()


def display_predictions_for_methods(args, device, model_labels, observed_trajectories, observed_displacements,
                                    ground_truth_trajectories, models_prediction, best_model_ade, best_model_fde,
                                    worst_model_ade, worst_model_fde, start_end_list):
    """
    Plots all predictions from the supplied methods/models, one trajectory per plot.
    Will also print information to the command line, indicating for each model, how much of their predictions result
    in the best/worst ADE/FDE.
    :param args: command line arguments to configure the plots
    :param device: torch.device (cpu or cuda) to map tensors to
    :param model_labels: list with labels identifying each model
    :param observed_trajectories: all observed trajectories, in absolute positions
    :param observed_displacements: all observed trajectories, in terms of relative displacements (or velocities)
    :param ground_truth_trajectories: all future GT trajectories, in absolute positions
    :param models_prediction: all future predicted trajectories, for
    :param best_model_ade: the index for the model with best ADE, per trajectory
    :param best_model_fde: the index for the model with best FDE, per trajectory
    :param worst_model_ade: the index for the model with worst ADE, per trajectory
    :param worst_model_fde: the index for the model with worst FDE, per trajectory
    :param start_end_list: delimiters for the beginning and end of each situation or set of trajectories.
    :return: nothing, plots will be done here.
    """
    num_trajectories_discarded = 0
    num_trajectories_plotted = 0
    num_models = len(model_labels)
    num_best_ade, num_best_fde, num_worst_ade, num_worst_fde = [0] * num_models, [0] * num_models, [0] * num_models, \
                                                               [0] * num_models
    if args.environment_location:
        # also display the static environment - in the background
        environment = Environment.load(args.environment_location)
        environment_plot = Environment(copy.deepcopy(environment.obstacles), copy.deepcopy(environment.scene_bounds))
        environment_plot.change(args.switch_x_y, args.invert_x, args.invert_y)
    else:
        environment = environment_plot = None
    if args.plot_limits and len(args.plot_limits) == 4:
        x_limits, y_limits = args.plot_limits[:2], args.plot_limits[2:]
    else:
        x_limits, y_limits = None, None
    for idx, batch_obs in enumerate(observed_displacements):
        for batch_idx, obs_vel in enumerate(batch_obs.permute(1, 0, 2)):
            gt = ground_truth_trajectories[idx][:, batch_idx, :]
            if torch.all(torch.abs(torch.sum(obs_vel, dim=0)) <= args.displacement_threshold):
                # small displacement below specified threshold (e.g. pedestrian is stopped)
                num_trajectories_discarded += 1
                continue
            if obs_vel.shape[0] + gt.shape[0] <= args.length_threshold:
                continue
            last_obs = observed_trajectories[idx][-1, batch_idx, :].clone().unsqueeze(0)
            obs_traj = observed_trajectories[idx][:, batch_idx, :]
            # append last observed position to make the trajectory a single dash (not have the gap between last
            # observed position and first predicted position)
            gt = torch.cat((last_obs, gt), dim=0)
            # tensor of shape (num_models, traj_len, 2); the permute below is for num_models to simulate 'batch'
            predictions = torch.cat((last_obs.unsqueeze(0).repeat(num_models, 1, 1),
                                     models_prediction[idx][:, :, batch_idx, :]), dim=1)
            if environment and args.only_plot_collisions_static and torch.sum(
                    environment.compute_collisions(predictions[:, :, 0].permute(1, 0, 2), combine_cse_osb=True)) == 0:
                continue  # all primary pedestrian predictions comply with the scene environment
            predictions = predictions
            best_ade, best_fde, worst_ade, worst_fde = best_model_ade[idx][batch_idx], best_model_fde[idx][batch_idx], \
                                                       worst_model_ade[idx][batch_idx], worst_model_fde[idx][batch_idx]
            num_best_ade[best_ade] += 1
            num_best_fde[best_fde] += 1
            num_worst_ade[worst_ade] += 1
            num_worst_fde[worst_fde] += 1
            best_model_ade_label = model_labels[best_ade]
            # best_model_fde_label = model_labels[best_fde]
            worst_model_ade_label = model_labels[worst_ade]
            # worst_model_fde_label = model_labels[worst_fde]
            obs_traj, gt, predictions = apply_plot_args_to_trajectories(args, obs_traj, gt, predictions)
            plt.figure()
            if environment_plot:
                environment_plot.plot(plt)
            # TODO - allow background as a certain scene image
            """
            plt.title(f'Best prediction from {best_model_ade_label}, worse from {worst_model_ade_label}',
                      fontdict={'fontsize': 14})
            """
            plt.plot(obs_traj[:, 0], obs_traj[:, 1], linewidth=3, label='OBS')
            plt.scatter(obs_traj[:, 0], obs_traj[:, 1], s=50, alpha=0.7)
            plt.plot(gt[:, 0], gt[:, 1], linewidth=3, label='GT')
            plt.scatter(gt[1:, 0], gt[1:, 1], s=50, alpha=0.7)
            for label_idx, pred in enumerate(predictions):
                plt.plot(pred[:, 0], pred[:, 1], linewidth=3, label=model_labels[label_idx])
                plt.scatter(pred[1:, 0], pred[1:, 1], s=50, alpha=0.7)
            if x_limits and y_limits:
                plt.xlim(x_limits)
                plt.ylim(y_limits)
            plt.xlabel(f'x ({args.units})', fontdict={'fontsize': 14})
            plt.ylabel(f'y ({args.units})', fontdict={'fontsize': 14})
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(loc=args.legend_location)
            if args.plot_title is not None and args.plot_title:
                plt.title(args.plot_title, fontdict={'fontsize': 18})
            num_trajectories_plotted += 1
            if num_trajectories_plotted >= args.max_trajectories:
                print(f'Reached {args.max_trajectories} plots')
                for m_idx, label in enumerate(model_labels):
                    print(f"Model {label} predictions (from the plots): {num_best_ade[m_idx]} with best ADE, "
                          f"{num_best_fde[m_idx]} with best FDE, {num_worst_ade[m_idx]} with worst ADE, "
                          f"{num_worst_fde[m_idx]} with worst FDE.")
                plt.show()
                return
    print("")
    for idx, label in enumerate(model_labels):
        print(f"Model {label} predictions (from the plots): {num_best_ade[idx]} with best ADE, {num_best_fde[idx]} "
              f"with best FDE, {num_worst_ade[idx]} with worst ADE, {num_worst_fde[idx]} with worst FDE.")
    plt.show()


def load_models(args, device, data_file_names):
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
    for idx, label in enumerate(args.model_labels):
        # first, check if the file has extension .ndjson - means it consists of pre-computed Trajnet++ predictions
        if args.model_paths[idx].endswith('ndjson'):
            if args.fixed_len or args.variable_len:
                raise Exception(f'Prediction file ({args.model_paths[idx]}) is only supported for Trajnet++ data')
            test_dir_cache = args.test_dir
            args.test_dir = args.model_paths[idx]
            loaders_pred, pred_file_names = load_test_data(args, device, load_pred=True)
            assert len(pred_file_names) == len(data_file_names) and \
                   [p == d for (p, d) in zip(pred_file_names, data_file_names)], \
                    f'The provided path to load prediction ({args.model_paths[idx]}) does not match the provided ' \
                    f'data path ({test_dir_cache})'
            args.test_dir = test_dir_cache
            # need to convert to iter to get one batch a time
            models.append([iter(loader) for loader in loaders_pred])
            train_args.append(None)
            input_types.append(TrajectoryType.ABS)
            output_types.append(TrajectoryType.ABS)
        # otherwise - it is an actual model
        elif 'lstm' in label.lower():
            args.model_path = args.model_paths[idx]
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
        elif label == sparse_motion_fields_label:
            args.model_path = args.model_paths[idx]
            model, _, _ = load_model_s(args, device, ModelType.SMF)
            assert len(model) == 1, f'You cannot supply a directory, ({args.model_path})' \
                                    'the path to the model must be a single file, that ' 'must exist!'
            models.append(model[0])
            train_args.append(None)
            input_types.append(TrajectoryType.ABS)
            output_types.append(TrajectoryType.ABS)
        else:  # == const_vel_label
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
