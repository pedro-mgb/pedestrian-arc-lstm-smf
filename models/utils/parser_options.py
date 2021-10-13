"""
Created on March 12th, 2021
Adds parser options that can be common to several contexts.

Each method has as input a parser, which we assume to be of type argparse.ArgumentParser()
(see https://docs.python.org/3/library/argparse.html)
"""

from argparse import Namespace
import os
import json

import torch

from models.modules import NormalStabilizeRange
from models.interaction_modules.shape_based import ShapedBasedPoolingType, PoolingShape

OPTIMIZER_CHOICES = ['Adam', 'adam', 'SGD', 'sgd']
# '' for no activation
INPUT_ACTIVATION_CHOICES = ['', 'none', 'None', 'ReLu', 'relu', 'Relu', 'LeakyRelu', 'leaky_relu', 'leakyrelu',
                            'leakyRelu', 'LeakyRelu', 'PReLu', 'Prelu', 'PRelu', 'prelu', 'elu', 'ELU', 'GELU', 'gelu',
                            'Gelu', 'SELU', 'selu', 'Selu', 'SILU', 'SiLU', 'silu', 'Selu']
OUTPUT_ACTIVATION_CHOICES = INPUT_ACTIVATION_CHOICES.copy()
OUTPUT_ACTIVATION_CHOICES.extend(['normal_stabilize', 'NormalStabilize'])
# NLL - Negative Log-Likelihood; GL - Gaussian Likelihood
LOSS_CHOICES = ['L2', 'l2', 'nll', 'NLL', 'gl', 'GL']

# SHAPE-BASED Pooling
Occupancy_pooling_labels = ['O', 'o', 'occupancy', 'Occupancy', 'OCC', 'occ']
Occupancy_percentage_pooling_labels = ['OP', 'op', 'occupancy_percentage', 'OccupancyPercentage', 'OCC_percent',
                                       'occ_percent']
Social_pooling_labels = ['S', 's', 'social', 'social_pooling', 'social-pooling', 'social pooling', 'spooling',
                         's-pooling', 's_pooling']
Directional_pooling_labels = ['D', 'd', 'directional', 'Directional', 'dir', 'DIR']
Directional_polar_pooling_labels = ['DP', 'dp', 'directional_polar', 'DirectionalPolar', 'dir_p', 'DIR_P', 'd_p']
Distance_pooling_labels = ['dist', 'dis', 'distance']
DistanceDirectional_pooling_labels = ['dd', 'dist_dir', 'distance_directional', 'dir_dist', 'directional_distance']
POOLING_CHOICES = []
POOLING_CHOICES.extend(Occupancy_pooling_labels)
POOLING_CHOICES.extend(Occupancy_percentage_pooling_labels)
POOLING_CHOICES.extend(Social_pooling_labels)
POOLING_CHOICES.extend(Directional_pooling_labels)
POOLING_CHOICES.extend(Directional_polar_pooling_labels)
POOLING_CHOICES.extend(Distance_pooling_labels)
POOLING_CHOICES.extend(DistanceDirectional_pooling_labels)
POOLING_CHOICES.extend(['none', 'None', 'NONE'])
POOLING_SHAPE_CHOICES = ['grid', 'arc']


def add_parser_arguments_for_data(parser):
    """
    Adds the options that the command line parser will search for, regarding data and dataset processing
    :param parser: the argument parser
    :return: the same parser, but with the added options.
    """
    # Dataset options
    parser.add_argument('--loader_num_workers', default=0, type=int,
                        help='number of workers to use of the DataLoader. By default will be single-process (no extra '
                             'workers). For loading data on Windows OS, if you get an Access Denied or Operation '
                             'Not Supported for cuda, you cannot set --loader_num_workers to something > 0 '
                             '(you can\'t share CUDA tensors among Windows processes)')
    parser.add_argument('--variable_len', '--var_len', '--variable_length', action='store_true',
                        help='Consider variable length for the retrieved trajectories; this will use the original '
                             'trajectories from the dataset files. Not recommended for social models since it '
                             'currently does not include the neighbours')
    parser.add_argument('--fixed_len', '--fixed_length', action='store_true',
                        help='force fixed length for the retrieved trajectories; this means using a technique '
                             'identical to the social gan work')

    parser.add_argument('--obs_len', '--obs_length', '--o_len', default=9, type=int,
                        help='Observation length of trajectories, which corresponds to a fixed length when dealing '
                             'with Trajnet++ data is used, or --fixed_len argument is supplied (usual value is 8). '
                             'Alternatively, when --variable_len is supplied, it sets a maximum value for the '
                             'observation length (you can supply -1 to not set a limit); For --variable_len, it can '
                             'override --obs_percentage value , i.e., the actual percentage of observed trajectory is '
                             'higher, --obs_len is reached.')
    parser.add_argument('--pred_len', '--pred_length', '--p_len', default=12, type=int,
                        help='Observation length of trajectories, which corresponds to a fixed length when dealing '
                             'with Trajnet++ data is used, or --fixed_len argument is supplied (usual value is 8). '
                             'Alternatively, when --variable_len is supplied, it sets a maximum value for the '
                             'observation length (you can supply -1 to not set a limit); For --variable_len, it can '
                             'override --obs_percentage value , i.e., the actual percentage of observed trajectory is '
                             'higher, if --pred_len is reached.')
    parser.add_argument('--obs_percentage', default=0.4, type=float,
                        help='percentage of a trajectory to consider to be observed, i.e. the past. '
                             'This assumes variable trajectory length (--variable_len flag).')

    parser.add_argument('--no_partial_trajectories', action='store_true',
                        help='Do not use partial trajectories (of length smaller than '
                             'obs_len+pred_len), discard all of those. To use with Trajnet++ data '
                             '(not supported for --fixed_len or --variable_len)')
    parser.add_argument('--augment_rotation', action='store_true',
                        help='Perform rotation augmentation (randomly rotate sequences of trajectories. '
                             'To use with Trajnet++ (not supported for --fixed_len or --variable_len). '
                             'Should only be used at train time')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='Rotate scene so primary pedestrian moves from east to west at end of observation. '
                             'To use with Trajnet++. If you use this flag at train time, supply it at test time too.'
                             'Trajectories are just rotated so evaluation metrics still are the same '
                             '(except if considering scene-specific metrics)')
    parser.add_argument('--filter_tags', nargs='+', default=[1, 2, 3, 4], choices=[1, 2, 3, 4], type=int,
                        help='Types of tags to support and load from, when considering Trajnet++ data. For more '
                             'information see '
                             'https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge')
    parser.add_argument('--filter_sub_tags', nargs='+', default=[1, 2, 3, 4], choices=[1, 2, 3, 4], type=int,
                        help='Types of sub-tags to support and load from, when considering Trajnet++ data. If the '
                             'trajectory belongs to more than one sub-tag, it will be used if at least one of those '
                             'matches the ones supplied --filter_sub_tags argument. For more information see '
                             'https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge')

    parser.add_argument('--delim', default='\t',
                        help='Default delimiter between values in the same line of a data file. Used for original '
                             'ETH/UCY files, with either --fixed_len or --variable_len flags')
    parser.add_argument('--skip', default=1, type=int)
    parser.add_argument('--use_same_obs_len', action='store_true',
                        help='use the observation length equal to the --obs_len argument (will discard trajectories '
                             'where this is not possible. Avoid using with --use_same_pred_len.')
    parser.add_argument('--use_same_pred_len', action='store_true',
                        help='use the prediction length equal to the --pred_len argument (will discard trajectories '
                             'where this is not possible. Avoid using with --use_same_obs_len.')
    parser.add_argument('--split_trajectories', action='store_true',
                        help='To use with --variable_len flag, but uses constant observation length and prediction '
                             'length, and equal to parameters --obs_len and --pred_len. However, longer trajectories '
                             'are split in portions of equal length.\n'
                             'Practical example: obs_len=8 and pred_len=12; trajectory of length 90; the first '
                             '80 instants are divided into 4 portions of length 8+12=20. The last 10 instants are '
                             'discarded.')

    parser.add_argument('--add_backwards_trajectories', action='store_true',
                        help='Simple \'augmentation\' technique: Also add trajectories in reverse order (start at end,'
                             'finish and beginning). Used for --variable_len')
    parser.add_argument('--random_rotate_std', default=0, type=float,
                        help='Specifying by how much to randomly rotate by. The random rotation angle will be sampled '
                             'from a Gaussian distribution with 0 mean and standard deviation equal to this value '
                             '(expecting value to be in radians). Used for --variable_len')
    parser.add_argument('--random_rotate_thresh', default=0, type=float,
                        help='Specifying by how much to randomly rotate by. The random rotation angle will be sampled '
                             'from a Uniform distribution with values between [threshold, threshold), with \''
                             'threshold\' being this value (expecting value to be in radians). Used for --variable_len')

    parser.add_argument('--batch_size', default=8, type=int,
                        help='Number of trajectories (or set of trajectories when')
    return parser


def add_parser_arguments_for_training(parser):
    """
    Adds the options that the command line parser will search for, regarding training of models. This also includes the
    options regarding data (see add_parser_arguments_for_data) and miscellaneous (see add_parser_arguments_misc)
    :param parser: the argument parser
    :return: the same parser, but with the added options.
    """
    parser.add_argument('--train_dir', default='datasets_in_trajnetpp21/train', type=str,
                        help='directory where the training data is found')
    parser.add_argument('--val_dir', default='datasets_in_trajnetpp21/val', type=str,
                        help='directory where the validation data is found')
    parser = add_parser_arguments_for_data(parser)

    # Optimization
    parser.add_argument('--num_epochs', default=25, type=int)

    # Model Options
    parser.add_argument('--use_enc_dec', action='store_true',
                        help='If supplied, will use an Encoder-Decoder LSTM architecture. The major difference lies in'
                             'having 2 LSTM networks (with separate weights) - the Encoder, to process the past '
                             '(observed) trajectory, and the Decoder to output the predicted trajectory.')
    parser.add_argument('--lstm_h_dim', default=128, type=int, help='dimension for the LSTM hidden state tensor')
    parser.add_argument('--embedding_dim', default=64, type=int, help='dimension for the input embedding')
    parser.add_argument('--normalize_embedding', action='store_true',
                        help='normalize the embedded input, in terms of L2 norm')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of layers of LSTM network. Currently only applicable for Encoder-Decoder '
                             'architecture and without interaction pooling')
    parser.add_argument('--input_activation', choices=INPUT_ACTIVATION_CHOICES, default='prelu', type=str,
                        help='The type of activation to use on input')
    parser.add_argument('--output_activation', choices=OUTPUT_ACTIVATION_CHOICES, default='', type=str,
                        help='The type of activation to use on output')
    parser.add_argument('--dropout', default=0, type=float, help='Dropout probability (from 0 to 1); 0 for no dropout')
    parser.add_argument('--use_abs', action='store_true', help='indicates if meant to use absolute positions or not')
    parser.add_argument('--use_acc', action='store_true', help='indicates if meant to use acceleration (differences '
                                                               'between relative positions) or not')
    parser.add_argument('--feed_history', action='store_true',
                        help='DEPRECATED. (force) Feed a history of past positions to the model, besides sending just '
                             'the previous position')
    parser.add_argument('--out_gaussian', action='store_true',
                        help='Instead of outputting 2D positions (or relative displacements), the model will output a '
                             'Bi-Variate Gaussian distribution for RELATIVE displacement. This will force the --loss '
                             'to be of type \'nll\'. This ASSUMES --use_abs was not supplied, since they can\'t be '
                             'used simultaneously.')
    '''
    parser.add_argument('--discard_zeros', action='store_true',
                        help='DEPRECATED. Discard all zeros for observed trajectories. Particularly useful for when'
                             'trajectories with zero velocities are supplied.')
    parser.add_argument('--feed_history', action='store_true',
                        help='DEPRECATED. (force) Feed a history of past positions to the model, besides sending just '
                             'the previous position')
    '''

    # Model options to incorporate social interactions
    parser.add_argument('--pooling_type', type=str, default='d', choices=POOLING_CHOICES,
                        help='The type of pooling to use along with the model. These poolings are used to incorporate'
                             'social interactions between pedestrians. There are several kinds of pooling available,'
                             'such as social pooling, directional pooling, among others.')
    parser.add_argument('--pooling_shape', type=str, default='arc', choices=POOLING_SHAPE_CHOICES,
                        help='The type of shape for the pooling. Can be a grid, can be an arc split in portions.')
    parser.add_argument('--norm_pool', action='store_true',
                        help='Applicable to Directional pooling - normalize relative velocities along direction of '
                             'movement during shape-based pooling (pedestrian in question has direction along the '
                             'positive x axis).')
    parser.add_argument('--cell_size', type=float, default=0.6,
                        help='If --pooling_shape specified is of a grid-based pooling, use this argument to specify'
                             'the size of the cell (width, or height, since cell is square), in the units used for the'
                             'dataset (e.g. meters for BIWI/Crowds aka ETH/UCY, pixels for SDD usually)')
    parser.add_argument('--grid_dim', type=int, default=16,
                        help='If --pooling_shape specified is of a grid-based pooling, use this argument to specify'
                             'the dimension of the grid. The grid will have grid_dim * grid_dim cells in total')
    parser.add_argument('--pooling_out_dim', type=int, default=256,
                        help='The dimension of the pooled output from the Pooling Layer (e.g. Social-Pooling) that is'
                             'fed to the the LSTM cell as input. By default, uses the same value as --lstm_h_dim.'
                             'Common values: 64, 128, 256, 512, 1024')
    parser.add_argument('--arc_radius', type=float, default=4.0,
                        help='If --pooling_shape specified is of an arc-based pooling, specify the radius of the arc,'
                             'corresponding to the maximum distance that a neighbour can be to be included in pooling. '
                             'The value should be in the same units used for the dataset (e.g. meters for BIWI/Crowds '
                             'aka ETH/UCY, pixels for SDD usually)')
    parser.add_argument('--arc_angle', type=float, default=140.0,
                        help='If --pooling_shape specified is of an arc-based pooling, specify the angle of the arc,'
                             'half to each side of the pedestrians gaze direction (assumed to be the same as the '
                             'velocity direction). Pedestrians outside the range will not be included in the pooling. '
                             'The value is assumed to be in degrees, (0, 360] range.')
    parser.add_argument('--n_radius', type=int, default=4,
                        help='If --pooling_shape specified is of an arc-based pooling, specify the number of divisions'
                             'made across the radius component of the arc. Each portion will cover a distance given'
                             'by arc_radius / n_radius. Number of cells is given by n_radius * n_angle')
    parser.add_argument('--n_angle', type=int, default=5,
                        help='If --pooling_shape specified is of an arc-based pooling, specify the number of divisions'
                             'made across the angular component of the arc. Each portion will cover an angle given'
                             'by arc_angle / n_angle. Number of cells is given by n_radius * n_angle')
    parser.add_argument('--single_arc_row_behind', action='store_true',
                        help='Applicable for arc pooling shape. Consider a single arc row for neighbours immediately '
                             'behind the pedestrian. This row covers the angle range not covered by the arc with width '
                             '--arc_radius. Its radius is specified by argument --arc_row_behind_radius.')
    parser.add_argument('--arc_row_behind_radius', type=float, default=1,
                        help='To use with --single_arc_row_behind. The radius of the arc row behind the pedestrian.')
    parser.add_argument('--variable_shape', action='store_true',
                        help='Do a variable pooling shape, based on the past interactions between pedestrians '
                             '(if \'--pooling_type\' was supplied)')
    parser.add_argument('--radius_values', type=float, nargs='+', default=[2, 3, 4, 5, 6, 7, 8],
                        help='List of radius values (units of the scene - metres or pixels) that are possible to use. '
                             'It does not have to be supplied in a particular order.')
    parser.add_argument('--angle_values', type=float, nargs='+',
                        default=[140],
                        help='List of angle values (degrees) that are possible to use. They should be supplied in '
                             '(0, 360] interval. It does not have to be supplied in a particular order.')
    parser.add_argument('--cell_side_values', type=float, nargs='+', default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        help='List of cell side values (units of the scene - metres or pixels) that are possible to '
                             'use. It does not have to be supplied in a particular order.')
    parser.add_argument('--random_init_shape', action='store_true',
                        help='If, at first instant, meant to randomly initialize the shape with one of the possible '
                             'values (radius/angle in case of arc, cell side in case of grid)')
    parser.add_argument('--train_var_shape', action='store_true', help='Use this flag if meant to separately train the '
                                                                       'shape-configuration LSTM network')
    parser.add_argument('--shape_data_dir', type=str, default=None,
                        help='path to file or directory containing shape pooling data computed apriori for the several '
                             'possible shapes. This is used together with --train_var_shape to make a model forward '
                             'pass for each of the possible shapes, to then compare with the ones chosen by the shape'
                             'configuration module and see which ones yield the best results. Note that for a large '
                             'dataset, this may not fully fit in memory, and may need to be partitioned and accessed '
                             'through the disk (slower, but the alternative is computing it at every epoch, which is '
                             'much worse). In case of partitions, shuffling should not be used because to get a single'
                             'batch one might need to go through several partitions, which is even slower.')
    parser.add_argument('--label_smoothing', default=0, type=float,
                        help='Perform label smoothing in deciding best shape. The main goal is to make the model not'
                             'give as much confidence to what is being considered as \'Ground Truth\'. Expecting value '
                             'between 0 and 1 (0 for no label smoothing, 1 for 0% confidence in data).')
    parser.add_argument('--variable_shape_ped_density', action='store_true',
                        help='have the pooling shape be a function of the number of pedestrians currently present at '
                             'that instant. More pedestrians = smaller shape; less pedestrians = larger shape. Limits '
                             'of shape provided by the extremities of arguments --radius_values and --angle_values in '
                             'case of arc shape, or --cell_side_values in case of grid shape. This cannot be used '
                             'simultaneously with --variable_shape argument, nor with --variable_shape_neigh_dist.')
    parser.add_argument('--variable_shape_ped_density_visible', action='store_true',
                        help='have the pooling shape be a function of the number of pedestrians currently present at '
                             'that instant, but only accounts the visible pedestrians. Similar to '
                             '--variable_shape_ped_density, but only for arc shape, and using fixed angle (with '
                             'argument --arc_angle), only radius can be varied using the extremities of argument '
                             '--radius_values. This cannot be used simultaneously with --variable_shape argument, '
                             'nor with --variable_shape_ped_density, nor --variable_shape_neigh_dist.')
    parser.add_argument('--max_num_peds', default=100, type=int,
                        help='Specific to --variable_shape_ped_density_visible and  argument. '
                             'For the current instant, any number of pedestrians equal or larger than this value will '
                             'map to the minimum shape dimensions provided.')
    parser.add_argument('--variable_shape_neigh_dist', action='store_true',
                        help='have the pooling shape be a function of mean distance of neighbours to the pedestrian '
                             'that instant. Closer neighbours = smaller shape; distant neighbours = larger shape. '
                             'Limits of shape provided by the extremities of arguments --radius_values and '
                             '--angle_values in case of arc shape, or --cell_side_values in case of grid shape. This '
                             'cannot be used simultaneously with --variable_shape argument, nor with '
                             '--variable_shape_ped_density.')
    parser.add_argument('--variable_shape_neigh_dist_visible', action='store_true',
                        help='have the pooling shape be a function of mean distance of neighbours to the pedestrian '
                             'that instant, but only applying for those that the pedestrian can see. '
                             'Closer neighbours = smaller shape; distant neighbours = larger shape. '
                             'Limits of shape provided by the extremities of arguments --radius_values, with fixed '
                             'value for angle via --arc_angle (only applies for arc shape) This cannot be used '
                             'simultaneously with any other variable shape arguments.')
    parser.add_argument('--min_neigh_dist', default=1, type=float,
                        help='Specific to --variable_shape_neigh_dist argument. For the current instant, any case where'
                             ' the mean neighbour distance is smaller or equal to this value will map to the minimum '
                             'shape dimensions provided.')
    parser.add_argument('--max_neigh_dist', default=6, type=float,
                        help='Specific to --variable_shape_neigh_dist argument. For the current instant, any case where'
                             ' the mean neighbour distance is larger or equal to this value will map to the maximum '
                             'shape dimensions provided.')
    parser.add_argument('--variable_shape_up_to_x_ped', action='store_true',
                        help='The pooling shape will grow to include up to --max_num_ped pedestrians (best no used the '
                             'default value. You can use for instance values like 10 or 15). For the case of grid, its'
                             'size will increase until the maximum specified by --grid_dim times --cell_side_values. '
                             'For the case of arc, the radius will grow up to the maximum of --radius_values (arc '
                             'angle will be fixed to --arc_angle value). Cannot be used with any other of the '
                             '--variable_shape(...) arguments')

    # Model Options for incorporate sparse motion fields
    parser.add_argument('--fields_location', default='', type=str,
                        help='If supplied, will retrieve motion fields from a certain location. This will be used in'
                             'conjunction with LSTM network. Note that since motion fields require absolute '
                             'coordinates, that is what the model expects to use, even though it may output '
                             'velocities, depending on what is the desired type of loss. Also, note that these motion'
                             'fields are trained per scene, and so the train/validation data provided should be solely '
                             'of that specific scene.')
    parser.add_argument('--simple_fields', action='store_true',
                        help='Uses prediction of sparse motion fields in conjunction with LSTM prediction, for '
                             'prediction of the next position. It\'s a simple way to integrate motion fields work, that'
                             'attempt to learn scene-specific information. Note that by default, if this parameter is'
                             'not supplied, then the type of model that integrates LSTM with motion fields uses more'
                             'complex collaboration between the two sub-models.')
    parser.add_argument('--feed_all_fields', action='store_true',
                        help='Instead of feeding the most likely prediction of the sparse motion fields method, will'
                             'feed all the predictions to the LSTM, associated to the several possible fields. Cannot'
                             'be used with --simple_fields flag')
    parser.add_argument('--feed_with_probabilities', action='store_true',
                        help='To use with --feed_all_fields, but also supply the several probabilities from the motion '
                             'fields. Cannot be used with --simple_fields flag')

    # Training Options
    parser.add_argument('--optim', choices=OPTIMIZER_CHOICES, default='adam', type=str,
                        help='Optimizer choice to use in training the network')
    parser.add_argument('--lr_step_epoch', type=int, default=0,
                        help='If = 0, will not use a learning rate scheduler. Otherwise, will use a learning rate '
                             'scheduler that drops the learning rate every --lr_step_epoch epochs. '
                             'This should not be confused with the learning rate regulation that is done '
                             'by optimizers like Adam.')
    parser.add_argument('--lr_step_start', type=int, default=0,
                        help='The epoch to start the learning rate step scheduler. If = 0, will start at the beginning '
                             'of training. If > 0, will start at epoch number --lr_step_start (where --lr_step_start=1 '
                             'corresponds to the second epoch).')
    parser.add_argument('--lr_step_gamma', type=float, default=0.1, help='step to multiply the learning rate every '
                                                                         '--lr_step_epoch epochs.')
    parser.add_argument('--clipping_threshold', default=0, type=float, help='NOT IMPLEMENTED. For gradient clipping')
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0, type=float, help='Momentum value to improve plain SGD optimizer')
    parser.add_argument('--use_nesterov', action='store_true', help='Use nesterov instead of plain SGD optimizer')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help="Weight decay (L2 penalty) for Adam optimizer. See https://arxiv.org/abs/1711.05101")
    parser.add_argument('--teacher_forcing', action='store_true',
                        help='enable teacher forcing - to use the ground truth as input to the next time step, '
                             'instead of the model output of the previous time step')
    parser.add_argument('--train_files_individually', action='store_true',
                        help='Train the network with each file individually. What this means is that each batch will'
                             'only have trajectories related to a single file, and each batch training will only have '
                             'data from a single file, instead of possibly having mixed from different files.')
    parser.add_argument('--primary_ped_only', action='store_true',
                        help='when the case of Trajnet++ data, use this option to compute the loss and train the models'
                             ' only for the primary pedestrian of each \'scene\' (aka seq or mini-batch)')

    # Loss Options
    parser.add_argument('--loss', choices=LOSS_CHOICES, default='l2', type=str,
                        help='Loss function choice to use in training the network. Some of the losses imply that the '
                             'model outputs more than just positions. E.g., use of --out_gaussian flag.')
    parser.add_argument('--loss_no_len', action='store_true',
                        help='DEPRECATED. Compute loss, not considering length of predicted trajectory; with this, the '
                             'loss on predicting for 2 or 20 instants has the same contribution on the global loss')
    parser.add_argument('--loss_rel', action='store_true',
                        help='DEPRECATED. Compute L2 loss with respect to relative displacements (only if --use_abs is '
                             'not used)')

    # Output
    parser.add_argument('--save_to', default=os.path.join(os.getcwd(), 'saved_models', 'model.pt'),
                        help='path to save the model in')
    parser.add_argument('--save_best_on_train', action='store_true',
                        help='instead of saving the model at the iteration where the validation loss is smallest, save '
                             'the model at the iteration where the training loss is smallest. Be careful because if '
                             'the model overfits, performance on test data may be worse.')
    parser.add_argument('--save_best_on_train_val_avg', action='store_true',
                        help='instead of saving the model at the iteration where the validation loss is smallest, save '
                             'the model at the iteration where the average between training and validation loss is '
                             'smallest. Be careful because if the model overfits, performance on test data may be '
                             'worse.')
    parser.add_argument('--save_every', default=0, type=int,
                        help='Save the best model version (so far) every --save_every epochs. '
                             'By default it is 0 (not saving anything)')
    parser.add_argument('--save_every_after', default=0, type=int,
                        help='Only saves models every --save_every epoch AFTER the epoch supplied here. '
                             'By default is 0 (can start saving from the first epoch)')
    parser.add_argument('--load_checkpoint', default=None, type=str,
                        help='Path to load a checkpoint (.pt file), and resume training. Make sure the model '
                             'configuration is exactly the same! Should be a checkpoint obtained using '
                             '--save_last_epoch argument')
    parser.add_argument('--save_last_epoch', action='store_true',
                        help='Create a checkpoint each epoch (each overwriting the previous checkpoint). The model '
                             'from the best epoch is still saved. Uses the same path as supplied by --save_to, with a '
                             '\'checkpoint\' appended to the end (assumes file has an extension).')
    parser.add_argument('--init_with_state_dict', default=None, type=str,
                        help='Path to a .pt file containing a state dict to initialize the weights of the model. Make '
                             'sure the model configuration is exactly the same! The .pt file to load should have a '
                             'dictionary in similar structure to the checkpoints being saved')
    parser.add_argument('--save_init_state_dict', default=None, type=str,
                        help='Path to save an initial state dictionary with model weights.')
    parser.add_argument('--load_args_from', default=None, type=str,
                        help='Path to a model to load the arguments it used during training. Allows to near-replicate '
                             'the same training conditions of a past model (except for weights initialization, data '
                             'shuffling, and in the case of regularization like --weight_decay or --dropout). '
                             'Will override all non-default arguments, except for : --save_to, --load_checkpoint, '
                             '--load_args_from, --load_args_from_json, --init_with_state_dict '
                             '(these need to be supplied again). For instance, if you supply --save_best_on_train, but '
                             'args in --load_args_from path does not have --save_best_on_train, the flag '
                             '--save_best_on_train will still be used')

    # miscellaneous parameters
    parser = add_parser_arguments_misc(parser)
    parser.add_argument('--do_not_shuffle', action='store_true',
                        help='use this if you DO NOT want to shuffle training batches at every epoch')
    parser.add_argument('--plot_losses', action='store_true',
                        help='at the end of training, plot the evolution of the '
                             'losses in training and validation sets at each epoch')
    parser.add_argument('--profile', action='store_true',
                        help='Use pytorch profiler to time each training epoch.\n'
                             'It is preferable to do this on a single training epoch (maybe two), and with little data')

    return parser


def get_input_activation(args):
    """
    Get input activation function depending on supplied option.
    :param args: the command-line arguments that were parsed; contains the activation function label
    :return: the input activation function (torch.nn) or None, if no function label was supplied
    """
    if not hasattr(args, 'input_activation'):
        activation = ''
    else:
        activation = args.input_activation.lower()
    return __get_activation__(activation, INPUT_ACTIVATION_CHOICES)


def get_output_activation(args):
    """
    Get input activation function depending on supplied option.
    :param args: the command-line arguments that were parsed; contains the activation function label
    :return: the input activation function (torch.nn) or None, if no function label was supplied
    """
    if not hasattr(args, 'output_activation'):
        activation = ''
    else:
        activation = args.output_activation.lower()
    return __get_activation__(activation, OUTPUT_ACTIVATION_CHOICES)


def __get_activation__(activation, choices):
    """
    Get proper activation function depending on supplied option. Also checks if such function is available
    :param activation: label identifying the desired option for activation function
    :return: the activation function (torch.nn) or None, if no function label was supplied
    """
    # the assert shouldn't fail unless a bug is present - the assert makes sure we don't go further
    assert activation in choices, 'The input activation type ' + activation + ' is not available!'
    if not activation or activation == 'none':
        return None
    elif activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'leakyrelu' or activation == 'leaky_relu':
        return torch.nn.LeakyReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'gelu':
        return torch.nn.GELU()
    elif activation == 'selu':
        return torch.nn.SELU()
    elif activation == 'silu':
        return torch.nn.SiLU()
    elif activation == 'elu':
        return torch.nn.ELU()
    elif activation == 'normal_stabilize' or activation == 'normalstabilize':
        return NormalStabilizeRange(inplace=False)
    return None


def get_interaction_module_label(args):
    """
    Get the interaction module name depending on supplied arguments
    The interaction modules contain
    :param args: command line arguments, with which to identify the type of interaction module
    :return: the label identifying the interaction module (can be an enum)
    """
    if not hasattr(args, 'pooling_type') or args.pooling_type is None or not args.pooling_type or \
            args.pooling_type.lower() == 'none':
        args.pooling_type = ''  # standard value or this argument for no pooling
        return None, None  # no pooling
    pooling_type = args.pooling_type.lower()
    if not hasattr(args, 'pooling_shape'):
        shape = PoolingShape.GRID
    else:
        shape = PoolingShape.GRID if args.pooling_shape == 'grid' else PoolingShape.ARC
    if pooling_type in Occupancy_pooling_labels:
        type_enum = ShapedBasedPoolingType.OCCUPANCY
    elif pooling_type in Occupancy_percentage_pooling_labels:
        type_enum = ShapedBasedPoolingType.OCCUPANCY_PERCENT
    elif pooling_type in Social_pooling_labels:
        type_enum = ShapedBasedPoolingType.SOCIAL
    elif pooling_type in Directional_pooling_labels:
        type_enum = ShapedBasedPoolingType.DIRECTIONAL
    elif pooling_type in Directional_polar_pooling_labels:
        type_enum = ShapedBasedPoolingType.DIRECTIONAL_POLAR
    elif pooling_type in Distance_pooling_labels:
        type_enum = ShapedBasedPoolingType.DISTANCE
    elif pooling_type in DistanceDirectional_pooling_labels:
        type_enum = ShapedBasedPoolingType.DISTANCE_DIRECTIONAL
    else:
        raise Exception('Pooling type ' + args.pooling_type + ' is still not available')
    return type_enum, shape


def add_parser_arguments_for_testing(parser):
    """
    Adds the options that the command line parser will search for, regarding testing of models
    :param parser: the argument parser
    :return: the same parser, but with the added options.
    """
    parser.add_argument('--test_dir', default='datasets_in_trajnetpp21/test', type=str,
                        help='Path to directory (or single file) where the training data is found.')
    parser.add_argument('--test_files_individually', action='store_true',
                        help='If supplied, will test each file and display results for each file individually')
    parser.add_argument('--statistics', action='store_true',
                        help='If supplied, will perform statistics with the results')
    parser.add_argument('--cv', action='store_true',
                        help='Instead of expecting an LSTM based model, will use a classical constant velocity '
                             '(--model_path will not be used)')
    parser.add_argument('--smf', action='store_true',
                        help='Instead of expecting an LSTM based model, will expect a standalone Sparse Motion Fields '
                             '(SMF) ')
    parser.add_argument('--model_path', default=os.path.join(os.getcwd(), 'saved_models', 'model.pt'), type=str,
                        help='Path to retrieve the model from')
    parser.add_argument('--model_pred_path', default=None, type=str,
                        help='Instead of providing actual models, you can provide files containing predictions of '
                             'trajectories (accepts multimodality - can have more than one prediction for each '
                             'trajectory). Currently only available for Trajnet++ format (see '
                             'https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge for more '
                             'information on the format). The number of prediction files MUST equal the number of data '
                             'files from --test_dir.')
    parser.add_argument('--num_samples', default=1, type=int,
                        help='If the model is multimodal, number of samples to draw for each trajectory. '
                             'The metrics to be displayed will be according to the sample that has the smallest ADE')
    parser.add_argument('--eval_mode', default='min_ade', type=str,
                        choices=['min', 'min_ade', 'min_fde', 'min_both', 'max', 'max_ade', 'max_fde', 'max_both',
                                 'average', 'avg', 'mean', 'std', 'standard_deviation', 'st_dev'],
                        help='For the case of multimodal evaluation, provide a mode to pick the ADE and FDE to display'
                             '(can use minimum/maximum (min_ade<->min, max_ade<->max, and other variants), mean and '
                             'standard deviations')
    parser.add_argument('--kde_nll', action='store_true',
                        help='When --num_samples > 1, and with this option supplied, will compute a Kernel Density '
                             'Estimate Negative Log Likelihood (KDE-NLL). It is a more robust way of evaluating '
                             'multimodality than best ADE/FDE. A substantially large number of samples is required. '
                             'For instance, Trajnet++ used 50 samples. Note that more samples can be used '
                             '(Trajectron++ used 2000), but that will also increase overall computation time.')
    parser.add_argument('--ignore_if_kde_nll_fails', action='store_true',
                        help='Ignore cases where computation of KDE-NLL fails for all epochs. This may happen due to '
                             'there being no multimodality (e.g. multimodal C.V generating samples with 0 speed.')
    parser.add_argument('--environment_location', default=None, type=str,
                        help='Path to a file or directory of files containing information about the static environment '
                             '(presences of obstacles). If supplied, will also evaluate from the point of view of '
                             'collisions with the static obstacles. Note that their position is approximate, so results'
                             ' are not 100% reliable. This will turn on flag --statistics, which in turn will show'
                             'other information besides collisions with the static environment. \n'
                             'Example of path: \'datasets_utils/environment/obstacles/biwi_eth_map.txt\'\n'
                             'Or directory: \'datasets_utils/environment/obstacles\' (assumes the existence of several '
                             'scenes - several environment files)')
    parser.add_argument('--static_collisions_neighbours', action='store_true',
                        help='Specific to Trajnet++ standard - compute collisions with environment not only for '
                             'primary pedestrians, but also for neighbours.')
    parser.add_argument('--social_metrics', action='store_true',
                        help='If supplied, will also compute some social-related metrics, like percentages of'
                             'colliding pedestrians. This will turn on flag --statistics, which in turn will show'
                             'other information besides this social information. Note also that the statistics may '
                             'take some time to compute, especially for datasets with lots of pedestrians (e.g. takes '
                             'much longer for crowds_univ than biwi_eth)')
    parser.add_argument('--collision_threshold', default=0.1, type=float,
                        help='Available for --social_metrics. If two pedestrians get to a distance below this one, a '
                             'collision between those pedestrians is said to occur. For BIWI/Crowds Datasets, '
                             '0.1 results in practically no collisions in GT')
    parser.add_argument('--num_inter_pts_cols', default=2, type=int,
                        help='Available for --social_metrics. For computing collisions, this variable defines how many'
                             'intermediate points in the line segment that connects two pedestrian positions. Increase '
                             'this value to increase the accuracy of the number of collisions, at the cost of also '
                             'increasing computation time.')
    return parser


def add_parser_arguments_misc(parser):
    """
    Adds the options that the command line parser will search for, some miscellaneous parameters, like use of gpu,
    timing, etc.
    :param parser: the argument parser
    :return: the same parser, but with the added options.
    """
    parser.add_argument('--use_gpu', action='store_true',
                        help='use GPU (CUDA). For loading data on Windows OS, if you get an Access Denied or Operation '
                             'Not Supported for cuda, you must set --loader_num_workers to 0 '
                             '(you can\'t share CUDA tensors among Windows processes).')
    parser.add_argument('--gpu_num', default="0", type=str)
    parser.add_argument('--map_gpu_beginning', action='store_true',
                        help='Will map all tensors (including FULL dataset) to GPU at the start of the instance, if '
                             '--use_gpu flag is supplied and CUDA is available. This option is NOT recommended if you '
                             'have low GPU memory or if you dataset is very large, since you may quickly run out of '
                             'memory.')
    parser.add_argument('--timing', action='store_true',
                        help='if specified, will display times for several parts of training')
    parser.add_argument('--load_args_from_json', type=str, default=None,
                        help='Path to json file containing args to pass. Should be an object containing the keys of '
                             'the attributes you want to change (keys that you don\'t supply will be left unchanged) '
                             'and their values according to their type (int, str, bool, list, etc.)')
    return parser


def add_parser_arguments_plotting(parser):
    """
    Adds the options that te command line parser will search for, regarding configuration for plots
    :param parser: the argument parser
    :return: the same parser, but with the added options.
    """
    # required = parser.add_argument_group('required named arguments')
    parser.add_argument('--model_paths', nargs='+', type=str,  # required=True,
                        help='List of paths to several model(s) or file(s) containing pre-computed predictions. '
                             'If a model does not have actual path (e.g. CV), supply the word \'none\'. '
                             'This list must be supplied so that the script knows where to get the predictions.')
    parser.add_argument('--model_labels', nargs='+', type=str,  # required=True,
                        help='List of labels for each of the models. It is also required. The number of labels must'
                             'equal to the number of paths ')

    parser.add_argument('--max_trajectories', default=10, type=int,
                        help='Maximum number of trajectory plots to display. '
                             'Script will stop if this number is reached.')
    parser.add_argument('--displacement_threshold', default=0, type=float,
                        help='Any (GT -> PAST + FUTURE) trajectory with total displacement below or equal to this '
                             'value will not be plotted')
    parser.add_argument('--length_threshold', default=0, type=int,
                        help='Any primary trajectory with length below or equal to this value will not be plotted')
    parser.add_argument('--distinguish_start_end', action='store_true',
                        help='distinguish the start and end positions in '
                             'the trajectories to plot')

    parser.add_argument('--rotate_by', default=0, type=float,
                        help='Rotate the trajectories to display by a fixed angle. Note that this is not the same as '
                             'parameters like --random_rotate_std or --random_rotate_thresh, those are related to '
                             'parameters of a distribution. The angle should be supplied in radians.')
    parser.add_argument('--switch_x_y', action='store_true', help='Switch x with y coordinates for all trajectories. '
                                                                  'Applied after --rotate_by')
    parser.add_argument('--invert_x', action='store_true',
                        help='For all trajectories, do x=-x. Applied after --switch_x_y')
    parser.add_argument('--invert_y', action='store_true',
                        help='For all trajectories, do y=-y. Applied after --switch_x_y. For both biwi_eth and '
                             'biwi_hotel scenes, if one supplies --switch_x_y and --invert_y, the scene will be '
                             'oriented similar to the original video. For crowds_univ and crowds_zara, these arguments '
                             'need not be supplied (scenes are already oriented properly')
    parser.add_argument('--plot_limits', nargs='+', type=float,
                        help='List with FOUR float values, to specify the limits, in x/y values, for the plot window. '
                             f'Expected format is: min_x max_x min_y max_y{os.linesep}'
                             'Here are several values to specify so that the plot limits are APPROXIMATELY the scene '
                             f'limits, but in metric units:{os.linesep}'
                             '-> biwi_eth: [-7.3 14.7 -2.7 13.8]     OR [-2.7 13.8 -14.7 7.3] if --switch_x_y and '
                             f'--invert_y are supplied{os.linesep}'
                             '-> biwi_hotel: [-4.4 6.9 -10.5 4.5]     OR [-10.5 4.5 -6.9 4.4] if --switch_x_y and '
                             f'--invert_y are supplied{os.linesep}'
                             f'-> crowds_univ: [-1.0 16.6 -0.3 13.8]{os.linesep}'
                             f'-> crowds_zara: [-1.0 16.6 -0.4 13.7]{os.linesep}')
    parser.add_argument('--units', default='m', help='Unit to display in the coordinate system for the plots')
    parser.add_argument('--environment_path', default=None, type=str,
                        help='Path to a file containing information about the static environment (presences of '
                             'obstacles). If supplied, will also plot the obstacles along with the trajectories . '
                             'Note that their position is approximate. Example of path:'
                             '\'datasets_utils/environment/obstacles/biwi_eth_map.txt\'')
    parser.add_argument('--only_plot_collisions_static', action='store_true',
                        help='If supplied, will only plot trajectories in which collisions occur with the static '
                             'environment. This requires --environment_location argument to be provided, with the'
                             'path to the scene environment file.')
    parser.add_argument('--only_plot_collisions_ped', action='store_true',
                        help='If supplied, will only plot trajectories in which collisions between pedestrians occur. '
                             'For Trajnet++, only collisions between primary pedestrian and neighbours count.')
    parser.add_argument('--ignore_neighbours_past', action='store_true',
                        help='Do not plot partial neighbour trajectories that only exist in the past (to be observed, '
                             'and not to be ')
    parser.add_argument('--min_neighbours', default=2, type=int,
                        help='Minimum number of neighbours to consider. Discards the sequences that have less than '
                             'that number of neighbours')
    parser.add_argument('--max_neighbours', default=10, type=int,
                        help='Maximum number of neighbours to consider. Discards the sequences that have more than '
                             'that number of neighbours')
    parser.add_argument('--ignore_neighbour_pred', action='store_true',
                        help='Do not plot the predictions of the neighbours (assumes Trajnet++ format data')
    parser.add_argument('--plot_title', default=None, type=str, help='Title for each plot')

    parser.add_argument('--animate', action='store_true',
                        help='If true, will animate the plots (predictions every instant). The animations will be '
                             'saved to .mp4 files in the current working directory (where the script is being run), '
                             'unless --animation_parent_dir is changed')
    parser.add_argument('--animation_parent_dir', type=str, default=None,
                        help='The parent directory to store the .mp4 trajectory animations. By default, will save to '
                             'the current working directory (the one where the script is being run)')
    parser.add_argument('--animation_save_name_append', type=str, default='number', choices=['number', 'time'],
                        help='What to append to each animation file. If \'number\', will append the number of the '
                             'current animation drawn so far (e.g. append 3 if it is the third). If \'time\' is '
                             'sent, will append the current time. Using \'time\' means you can run multiple times '
                             'without overwriting the previous animations.')
    parser.add_argument('--legend_location', type=str, default='best',
                        choices=['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right',
                                 'center left', 'center right', 'lower center', 'upper center', 'center'],
                        help='The location for the legend in the plot/animation.')
    parser.add_argument('--anim_neighbour_motion', action='store_true',
                        help='Instead of plotting the entirety of all neighbour trajectories (which can be confusing '
                             'in crowded scenarios), will animate the current neighbour velocity, via arrows')
    parser.add_argument('--maintain_last_instant_for', default=5, type=int,
                        help='The number of instants to maintain the last prediction. This ensures that the outputted '
                             'video does not immediately end when prediction ends')
    parser.add_argument('--fps', default=1,
                        help='The number of frames per second for the generated video')
    parser.add_argument('--plot_pool_shape', action='store_true',
                        help='If supplied with --animate, will also draw the current pooling shape (arc or grid) that '
                             'is being employed, only for the primary pedestrian (assumes Trajnet++ data).'
                             ' This shape will only be drawn for the first social method supplied (so the animation '
                             'does not get confusing).')
    parser.add_argument('--timestamp_in_plot_title', action='store_true',
                        help='If supplied, will append to the current timestamp to the title of the animation plot')
    return parser


DEFAULT_EXCEPTION_LIST = ('save_to', 'load_checkpoint', 'init_with_state_dict', 'load_args_from')


def override_args(current_args, new_args, parser=None, exception_list=DEFAULT_EXCEPTION_LIST):
    """
    Override all available arguments using a new argument list. Only non-default arguments and ones from
    :param current_args: The current Namespace class with all available arguments
    :param new_args: the new Namespace with arguments to override with
    :param parser: the argument parser, that contains default values for arguments
    :param exception_list: list of exception keywords for specific arguments that will not be overridden
    :return: the new and overridden arguments, number of arguments that were changed, number of arguments with values
    different form the default (may or may not have been overridden), and the names of arguments that were changed
    """
    # convert to dictionary
    overridden_args_dict, curr_dict = vars(current_args).copy(), vars(current_args)
    new_dict = new_args if isinstance(new_args, dict) else vars(new_args)
    num_overridden_args, num_args_diff_default, names_of_overridden_args = 0, 0, []
    for arg_name, arg_value in curr_dict.items():
        if arg_name not in new_dict.keys() or arg_name in exception_list:
            continue
        default_value, new_value = parser.get_default(arg_name), new_dict[arg_name]
        if default_value != arg_value or new_value == arg_value:
            num_args_diff_default += (default_value != arg_value and new_value != arg_value)
            continue
        overridden_args_dict[arg_name] = new_value
        num_overridden_args += 1
        names_of_overridden_args.append(arg_name)
    overridden_args = Namespace(**overridden_args_dict)
    return overridden_args, num_overridden_args, num_args_diff_default, names_of_overridden_args


def override_args_from_json(current_args, json_file, parser=None, exception_list=DEFAULT_EXCEPTION_LIST):
    """
    method to override list of arguments using an existing JSON file. The JSON file should have an object/dictionary
    containing all arguments that the user wants to override.
    :param current_args: current command line arguments (should have default values).
    :param json_file: JSON file containing arguments
    :param parser: the argument parser, that contains default values for arguments
    :param exception_list: list of exception keywords for specific arguments that will not be overridden
    :return: the new and overridden arguments, number of arguments that were changed, number of arguments with values
    different form the default (may or may not have been overridden), and the names of arguments that were changed
    """
    json_args = json.load(open(json_file, ))
    new_args, _, _, _ = override_args(current_args, json_args, parser, exception_list)
    return new_args
