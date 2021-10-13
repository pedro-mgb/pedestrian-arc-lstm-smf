"""
Created on April 27th 2021.
File that contains methods to load LSTM-like models from files.
Includes the possibility of loading several models, each trained on different scenes.
"""
import os

import torch

from models.utils.loaders import get_models_per_scene_biwi_crowds
from models.lstm.lstm import VanillaLSTM, VanillaLstmEncDec
from models.lstm.lstm_fields import SimpleFieldsWithLSTM, FieldsWithLSTM
from models.lstm.lstm_interactions import LSTMWithInteractionModule
from models.lstm.lstm_fields_interactions import FieldsWithInteractionModuleAndLSTM
from models.interaction_modules.shape_based import PoolingArguments, ShapedBasedPoolingType, ShapeBasedPooling, \
    PoolingShape
from models.interaction_modules.complex_shape_config import ShapeConfigLSTM
from models.interaction_modules.simple_shape_config import ShapeConfigPedDensity, ShapeConfigNeighDist, \
    ArcShapeRadiusConfigVisiblePedDensity, ArcShapeRadiusConfigVisibleNeighDist, GrowingShapeUpToMaxPedestrians
from models.fields.sparse_motion_fields import SparseMotionFields
from models.utils.parser_options import get_output_activation, get_input_activation, get_interaction_module_label


def load_lstm_models(device, path):
    """
    Load lstm models from a file path (can be a single file, or a directory containing several files)
    :param device: The torch.device to map the pytorch data to
    :param path: the path to the lstm model(s); if it's to a directory, then it is assumed that the models are to use
    with biwi / crowds datasets, per each of the (4) scenes
    :return: list of models and scene labels; in case of a single file, the list will have one element
    """
    if os.path.isdir(path):
        model_list = get_models_per_scene_biwi_crowds(device, path, load_single_lstm_model)
    else:
        # is a single file
        # scene label refers to all the existing data (pretty much just a placeholder)
        model_list = [[load_single_lstm_model(device, path), 'all_data']]
    return model_list


def load_single_lstm_model(device, path):
    """
    Load a single LSTM model from a file path.
    :param device: The torch.device to map the pytorch data to
    :param path: the file path to the lstm model
    :return: a list with two elements, in this order: the lstm model (can be of several kinds);
    a dictionary of arguments containing the parameters use for training the model
    """
    saved_model_data = torch.load(path, map_location=device)
    train_args = saved_model_data['args']
    model = build_eval_model_from_args(train_args, saved_model_data, device)
    return [model, train_args]


def build_eval_model_from_args(args, saved_model_data, device):
    """
    Build an lstm model for evaluation, using provided arguments; the model will be built using the saved state dict
    (built at the end of model training)
    :param args: list of arguments to select and build the lstm model
    :param saved_model_data: the saved model content from file; among other things, contains the state dict
    :param device: The torch.device to map the pytorch data to
    :return: The LSTM model. Can be the simplest vanilla version, or be more complex and have other considerations.
    """
    # NOTE - this may be changed to instantiate a kind of model depending on parameters
    # FOR legacy models that did not have these attributes.
    normalize_embedding = args.normalize_embedding
    out_gaussian = args.out_gaussian
    use_history = args.feed_history
    discard_zeros = args.discard_zeros
    activation_on_output = get_output_activation(args)
    # pick the type of lstm model
    model_interaction_module_label, pooling_shape = get_interaction_module_label(args)
    if model_interaction_module_label is not None:
        # model incorporates social interactions
        interaction_module = build_interaction_module(args, model_interaction_module_label, pooling_shape)
        shape_config = build_shape_config(args, interaction_module, pooling_shape)
        if 'fields' in saved_model_data and saved_model_data['fields'] is not None:
            # uses interactions and motion fields - interaction and scene-aware
            model_data = saved_model_data['fields']
            fields = SparseMotionFields(model_data['Te'], model_data['Qe'], model_data['Bc'],
                                        [model_data['min'], model_data['max']], model_data['parameters'])
            model = FieldsWithInteractionModuleAndLSTM(fields, interaction_module, shape_config,
                                                       embedding_dim=args.embedding_dim, h_dim=args.lstm_h_dim,
                                                       activation_on_input_embedding=get_input_activation(args),
                                                       output_gaussian=out_gaussian,
                                                       activation_on_output=activation_on_output,
                                                       feed_all=args.feed_all_fields,
                                                       use_probs=args.feed_with_probabilities)
        else:
            model = LSTMWithInteractionModule(interaction_module, shape_config, embedding_dim=args.embedding_dim,
                                              h_dim=args.lstm_h_dim,
                                              activation_on_input_embedding=get_input_activation(args),
                                              output_gaussian=out_gaussian, use_enc_dec=args.use_enc_dec,
                                              activation_on_output=activation_on_output)
    elif 'fields' in saved_model_data and saved_model_data['fields'] is not None:
        model_data = saved_model_data['fields']
        fields_model = SparseMotionFields(model_data['Te'], model_data['Qe'], model_data['Bc'],
                                          [model_data['min'], model_data['max']], model_data['parameters'])
        if args.simple_fields:
            model = SimpleFieldsWithLSTM(fields=fields_model, embedding_dim=args.embedding_dim,
                                         h_dim=args.lstm_h_dim, num_layers=args.num_layers,
                                         activation_on_input_embedding=get_input_activation(args),
                                         activation_on_output=activation_on_output,
                                         normalize_embedding=normalize_embedding, output_gaussian=out_gaussian,
                                         discard_zeros=discard_zeros)
        else:
            feed_all = args.feed_all_fields
            use_probabilities = args.feed_with_probabilities
            model = FieldsWithLSTM(fields=fields_model, feed_all=feed_all or use_probabilities,
                                   use_probs=use_probabilities, embedding_dim=args.embedding_dim, h_dim=args.lstm_h_dim,
                                   num_layers=args.num_layers, activation_on_input_embedding=get_input_activation(args),
                                   activation_on_output=activation_on_output, normalize_embedding=normalize_embedding,
                                   output_gaussian=out_gaussian, discard_zeros=discard_zeros)
    # Vanilla LSTM models - no scene compliance nor social interactions
    elif hasattr(args, 'use_enc_dec') and args.use_enc_dec:
        model = VanillaLstmEncDec(args.embedding_dim, args.lstm_h_dim, num_layers=args.num_layers,
                                  activation_on_input_embedding=get_input_activation(args),
                                  activation_on_output=activation_on_output, extra_info=use_history,
                                  normalize_embedding=normalize_embedding, output_gaussian=out_gaussian,
                                  discard_zeros=discard_zeros)
    else:
        model = VanillaLSTM(args.embedding_dim, args.lstm_h_dim,
                            activation_on_input_embedding=get_input_activation(args),
                            activation_on_output=activation_on_output, history_on_pred=use_history,
                            normalize_embedding=normalize_embedding, output_gaussian=out_gaussian,
                            discard_zeros=discard_zeros)
    model.load_state_dict(saved_model_data['model_state_dict'])
    model.to(device)
    # model.eval() used to set dropout and batch normalization layers to evaluation mode before running inference.
    # Failing to do this can yield inconsistent inference results.
    model.eval()
    return model


def build_interaction_module(args, module, shape):
    """
    Build an interaction module (e.g. object of class GridBasedPooling)
    :param args: List of arguments to select and build the interaction module
    :param module: An enum identifying the interaction module to build
    :param shape: the desired pooling shape (e.g. grid, arc)
    :return: The instance of the interaction module
    """
    pooling_arguments = PoolingArguments(args, module, shape)
    if isinstance(module, ShapedBasedPoolingType):
        return ShapeBasedPooling(pooling_arguments=pooling_arguments, h_dim=args.lstm_h_dim,
                                 include_occ=args.variable_shape if hasattr(args, 'variable_shape') else False,
                                 embedding_activation=get_input_activation(args))
    raise Exception('Provided interaction module is not available')


def build_shape_config(args, module, shape):
    """

    :param args: List of arguments to select and build the interaction module
    :param module: An enum identifying the shape configuration module to build
    :param shape: the desired pooling shape (e.g. grid, arc)
    :return: instance of the module to regulate the pooling shape, or None if no module is meant to be built
    """
    var_shape_lstm = hasattr(args, 'variable_shape') and args.variable_shape
    var_shape_ped_density = hasattr(args, 'variable_shape_ped_density') and args.variable_shape_ped_density
    var_shape_neigh_dist = hasattr(args, 'variable_shape_neigh_dist') and args.variable_shape_neigh_dist
    var_shape_ped_density_visible = (hasattr(args, 'variable_shape_ped_density_visible') and
                                     args.variable_shape_ped_density_visible)
    var_shape_neigh_dist_visible = (hasattr(args, 'variable_shape_neigh_dist_visible') and
                                    args.variable_shape_neigh_dist_visible)
    variable_shape_up_to_x_ped = hasattr(args, 'variable_shape_up_to_x_ped') and args.variable_shape_up_to_x_ped
    if not ((var_shape_lstm or var_shape_ped_density or var_shape_neigh_dist or var_shape_ped_density_visible or
             var_shape_neigh_dist_visible or variable_shape_up_to_x_ped) and shape is not None):
        return None  # no shape configuration module available
    if shape == PoolingShape.GRID:
        parameters = [args.cell_side_values]
    else:
        parameters = [args.radius_values, args.angle_values]
    random_init_shape = args.random_init_shape if hasattr(args, 'random_init_shape') else False
    if var_shape_lstm:
        return ShapeConfigLSTM(module.shape_values, module.out_dim, shape, parameters, embedding_dim=args.embedding_dim,
                               h_dim=args.lstm_h_dim, activation_on_input_embedding=get_input_activation(args),
                               dropout=args.dropout, random=random_init_shape)
    elif var_shape_ped_density:
        return ShapeConfigPedDensity(module.shape_values, shape, parameters, max_num_ped=args.max_num_peds,
                                     random=random_init_shape)
    elif var_shape_neigh_dist:
        return ShapeConfigNeighDist(module.shape_values, shape, parameters, min_dist=args.min_neigh_dist,
                                    max_dist=args.max_neigh_dist, random=random_init_shape)
    elif var_shape_ped_density_visible:
        # fix the angle to a certain value
        return ArcShapeRadiusConfigVisiblePedDensity(module.shape_values, shape, [parameters[0], [args.arc_angle]],
                                                     max_num_ped=args.max_num_peds, random=random_init_shape)
    elif var_shape_neigh_dist_visible:
        # fix the angle to a certain value
        return ArcShapeRadiusConfigVisibleNeighDist(module.shape_values, shape, [parameters[0], [args.arc_angle]],
                                                    min_dist=args.min_neigh_dist, max_dist=args.max_neigh_dist,
                                                    random=random_init_shape)
    elif variable_shape_up_to_x_ped:
        # fix angle to a certain value if arc shape
        parameters = [parameters[0], [args.arc_angle]] if shape == PoolingShape.ARC else parameters
        return GrowingShapeUpToMaxPedestrians(args, module.shape_values, shape, parameters,
                                              max_num_ped=args.max_num_peds, random=random_init_shape)
    # else - should NOT reach this point, unless there is a bug somewhere
    raise Exception('Variable shape configuration module not found')
