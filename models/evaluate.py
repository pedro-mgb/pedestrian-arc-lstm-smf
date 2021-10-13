"""
Created on March 12th, 2021
Evaluation script for all models, with the possibility of several parameters being regulated as command line arguments
This is meant to be run from the parent directory, not the current one.
You may supply a model file or directory of model files via --model_path (saved with torch.load), and a testing data
directory or file, via --test_dir.
Alternatively, you may supply a files containing the predictions of the data from --test_dir
"""

import argparse
from enum import Enum

import torch

from models.lstm.loaders import load_lstm_models
from models.lstm.lstm_interactions import LSTMWithInteractionModule
from models.lstm.lstm_fields_interactions import FieldsWithInteractionModuleAndLSTM
from models.fields.sparse_motion_fields import SparseMotionFields
from models.classical.constant_velocity import predict_const_vel
from models.fields.loaders import load_fields
from models.utils.loaders import map_from_file_name_biwi_crowds
from models.utils.parser_options import add_parser_arguments_for_data, add_parser_arguments_for_testing, \
    add_parser_arguments_misc, override_args_from_json
from models.data.loaders import load_test_data, load_biwi_crowds_data_per_scene
from models.utils.evaluator import TrajectoryType, MultimodalityType, map_traj_type, compute_metrics, \
    output_overall_performance, append_and_output_per_file, pred_data_call
from models.data.environment import load_environments


class ModelType(Enum):
    LSTM = 0
    CV = 1
    SMF = 2
    FROM_FILE = 3


parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_data(parser)
parser = add_parser_arguments_for_testing(parser)
parser = add_parser_arguments_misc(parser)


def main(args):
    if args.use_gpu and not torch.cuda.is_available():
        args.use_gpu = False
        print("WARNING: Use GPU was activated but CUDA is not available for this pytorch version. "
              "Will use CPU instead")
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # note, will map dataset tensors to GPU at the beginning if the --map_gpu_beginning flag is supplied
    device_load_data = torch.device('cpu') if args.use_gpu and (not args.map_gpu_beginning) else device
    # to make sure data shuffling does not affect evaluation, especially when using --model_pred_path

    model_type = ModelType.CV if args.cv else ModelType.SMF if args.smf else ModelType.LSTM
    if args.model_pred_path:
        model_type = ModelType.FROM_FILE
        if args.fixed_len or args.variable_len:
            raise Exception('Flag --model_pred_path is currently only available for Trajnet++ data')
        test_data_location = args.test_dir
        args.test_dir = args.model_pred_path
        loader_list, file_names = load_test_data(args, device, load_pred=True)
        model_list = [iter(loader) for loader in loader_list]  # so that we can iterate over the predictions
        scene_list, train_args_list = [], [None for _ in range(len(model_list))]
        args.test_dir = test_data_location
    else:
        model_list, scene_list, train_args_list = load_model_s(args, device, model_type)
    if model_type == ModelType.LSTM:
        try:
            # if at training direction normalization of the scene is used, it will also be used at testing
            sample_train_args = train_args_list[0]
            args.normalize_scene = sample_train_args.normalize_scene
        except AttributeError as _:
            pass  # if there is no argument called normalize_scene (cross-compatibility for old models)
    if args.environment_location:
        envs, env_scenes = load_environments(args, args.environment_location)
    else:
        envs = env_scenes = None
    if len(scene_list) > 1:
        test_loaders, file_names = load_biwi_crowds_data_per_scene(args, device_load_data, scene_list)
    else:
        test_loaders, file_names = load_test_data(args, device_load_data)
        # wrap in a list for compatibility
        test_loaders, file_names = [test_loaders], [file_names]
    print("Beginning evaluation")
    with torch.no_grad():
        results_list = []
        for idx, model in enumerate(model_list):
            if model is None and model_type == ModelType.CV:
                model = predict_const_vel
            results_list_per_scene = []
            if len(scene_list) > 1:
                print(f"\tEvaluating scene {scene_list[idx]}")
            train_args = train_args_list[idx]
            if model_type == ModelType.LSTM:
                # specific to LSTM
                model_with_fields = 'fields' in model.__class__.__name__.lower()
                model_with_interactions = 'interaction' in model.__class__.__name__.lower() or 'social' in \
                                          model.__class__.__name__.lower()
            else:
                model_with_fields = model_with_interactions = False

            # to decide how to call the model forward pass for evaluation
            def model_fun_call_lstm(_model, obs_traj, pred_len, seq_start_end, metadata, _1, _2):
                if isinstance(_model, LSTMWithInteractionModule):
                    return _model(obs_traj, pred_len, seq_start_end=seq_start_end)
                elif isinstance(_model, FieldsWithInteractionModuleAndLSTM):
                    return _model(obs_traj, pred_len, seq_start_end=seq_start_end, metadata=metadata)
                    # else:
                return _model(obs_traj, pred_len)

            use_acceleration = train_args.use_acc if hasattr(train_args, 'use_acc') else False
            if model_type == ModelType.CV or model_type == ModelType.SMF or model_type == ModelType.FROM_FILE:
                input_type = TrajectoryType.ABS
                output_type = TrajectoryType.ABS
            elif model_with_fields or model_with_interactions:
                input_type = TrajectoryType.ABS
                output_type = TrajectoryType.VEL
            else:
                input_type = output_type = map_traj_type(train_args.use_abs, use_acceleration)
            if model_type == ModelType.FROM_FILE:
                # there is a 1:1 correspondence between the data files and the prediction files
                eval_results = compute_metrics(args, model, test_loaders[0][idx], input_type, output_type, device,
                                               multimodal_type=MultimodalityType.NO_DISTRIBUTION_ONE_CALL,
                                               environment=envs[map_from_file_name_biwi_crowds(
                                                   file_names[0][idx], env_scenes)] if envs is not None else None,
                                               model_fun=pred_data_call)
                append_and_output_per_file(args, file_names[0][idx], eval_results, results_list_per_scene)
            else:
                # use an actual model and compute the predictions
                multimodality = MultimodalityType.DISTRIBUTION_ONE_CALL if model_type == ModelType.LSTM \
                                    else MultimodalityType.NO_DISTRIBUTION_MULTIPLE_CALLS
                for loader, name in zip(test_loaders[idx], file_names[idx]):
                    extra_kwargs = {'multimodal_type': multimodality,
                                    'environment': envs[map_from_file_name_biwi_crowds(name, env_scenes)]
                                    if envs is not None else None,
                                    'model_fun': model_fun_call_lstm if model_type == ModelType.LSTM else
                                    __fields_model_call__ if model_type == ModelType.SMF else
                                    __const_vel_call_from_sample_info__}
                    eval_results = compute_metrics(args, model, loader, input_type, output_type, device, **extra_kwargs)
                    append_and_output_per_file(args, name, eval_results, results_list_per_scene)
            if len(scene_list) > 1:
                output_overall_performance(args, results_list_per_scene)
            results_list.extend(results_list_per_scene)
        ade_global, fde_global, statistics = output_overall_performance(args, results_list)
        # note that statistics may be None if --statistics is not supplied
    return ade_global, fde_global, statistics


def load_model_s(args, device, model_type):
    if model_type == ModelType.CV:
        return [None], [None], [None]
    print(f"Loading model(s) from {args.model_path}")
    if model_type == ModelType.SMF:
        models_and_scenes = load_fields(device, args.model_path)
        model_list, scene_list = [], []
        for [model_data, scene_label] in models_and_scenes:
            model_list.append(SparseMotionFields(model_data['Te_best'], model_data['Qe_best'], model_data['Bc_best'],
                                                 model_data['min_max'], model_data['parameters']))
            scene_list.append(scene_label)
        print(f"Done! Loaded {SparseMotionFields.__name__} model")
        return model_list, scene_list, [None for _ in range(len(model_list))]
    # model_type == ModelType.LSTM
    models_and_scenes = load_lstm_models(device, args.model_path)
    model_list, scene_list, train_arg_list = [], [], []
    for [model, scene_label] in models_and_scenes:
        model_list.append(model[0])
        train_arg_list.append(model[1])
        scene_list.append(scene_label)
    print(
        f"Done! Loaded {len(model_list)} {model_list[0].__class__.__name__} model{'s' if len(model_list) > 1 else ''}")
    assert len(model_list) == len(scene_list) and len(model_list) == len(train_arg_list), \
        f'The specified lists, resulting from loading the several models, do not have the same size. Model list has ' \
        f'size {len(model_list)}; Scene list has size {len(scene_list)}; Train args list has size {len(train_arg_list)}'
    return model_list, scene_list, train_arg_list


def __fields_model_call__(model, obs_traj, pred_len, _seq_start_end, _metadata, num_samples, _curr_sample):
    return model(obs_traj, pred_len, multi_modal=num_samples > 1)


def __const_vel_call_from_sample_info__(model, obs_traj, pred_len, _seq_start_end, _metadata, num_samples, curr_sample):
    kwargs = {'multi_modal': False, 'first_multi_modal': False}
    if num_samples > 1:
        kwargs['multi_modal'] = curr_sample > 0
        kwargs['first_multi_modal'] = curr_sample == 1
    return model(obs_traj, pred_len, **kwargs)


if __name__ == '__main__':
    arguments = parser.parse_args()
    if hasattr(arguments, 'load_args_from_json') and arguments.load_args_from_json:
        new_args = override_args_from_json(arguments, arguments.load_args_from_json, parser)
    else:
        new_args = arguments
    main(new_args)
