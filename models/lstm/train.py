"""
Created on March 7th, 2021
Script to train LSTM network, with the possibility of several parameters being regulated as command line arguments
This is meant to be run from the parent directory, not the current one
"""

import argparse
import gc
import logging
import sys
import os
import time

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.autograd.profiler as profiler
import matplotlib.pyplot as plt
import numpy as np

from models.lstm.lstm import VanillaLSTM, VanillaLstmEncDec
from models.lstm.lstm_fields import FieldsWithLSTM, SimpleFieldsWithLSTM
from models.lstm.lstm_interactions import LSTMWithInteractionModule
from models.lstm.lstm_fields_interactions import FieldsWithInteractionModuleAndLSTM
from models.lstm.loaders import build_interaction_module, build_shape_config
from models.losses_and_metrics import l2_loss, nll, gaussian_likelihood_loss
from models.utils.utils import relative_traj_to_abs
from models.utils.parser_options import add_parser_arguments_for_training, get_input_activation, \
    get_output_activation, add_parser_arguments_for_testing, OPTIMIZER_CHOICES, LOSS_CHOICES, \
    get_interaction_module_label, override_args, override_args_from_json
from models.data.loaders import load_train_val_data
from models.fields.loaders import load_fields
from models.fields.sparse_motion_fields import SparseMotionFields
from models.interaction_modules.shape_data import data_loader_shapes
from models.interaction_modules.train_aux import train_var_shape_epoch

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_training(parser)
# the testing arguments are if the user supplies to perform evaluation at the end of training
parser = add_parser_arguments_for_testing(parser)

# initialize logging options
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

# plot info
TRAIN_COLOR, VAL_COLOR = '#1f77b4', '#ffa500'
PLOT_LINE_WIDTH = 3
MAX_X_TICKS = 10


def main(args, _arg_parser=None):
    if args.use_gpu and not torch.cuda.is_available():
        args.use_gpu = False
        logger.warning("Use GPU was activated but CUDA is not available for this pytorch version. Will use CPU instead")
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    if args.load_args_from is not None:
        loaded_args = torch.load(args.load_args_from, map_location=device)['args']
        args, num_overridden, num_non_default, _ = override_args(args, loaded_args, _arg_parser)
        logger.info(f'Loaded {num_overridden} arguments from {args.load_args_from}. '
                    f'{num_non_default} args were not loaded because they were supplied in this instance')
        del loaded_args

    # args.train_var_shape = False

    if not args.pooling_type:
        # for non-Interaction based trajectories, in the case of LSTM networks, for Trajnet++
        # they will not process partial trajectories. The alternative would be to force this on the networks themselves
        # this is the 'lazy' way of achieving this
        args.no_partial_trajectories = True

    if args.variable_shape and args.train_var_shape and not args.primary_ped_only:
        args.primary_ped_only = True
        args.fixed_len = args.variable_len = False
        logger.warning("Training the shape configuration module is only currently available for the primary pedestrians"
                       " and Trajnet++ data. As such, --primary_ped_only and --trajnepp flags were forced to True")

    if args.primary_ped_only and (args.fixed_len or args.variable_len):
        logger.warning("Training for primary pedestrians is only available for Trajnet++ data. As such, --fixed_len "
                       "and --variable_len flags were disabled (forced to False)")
    # note, will map dataset tensors to GPU at the beginning if the --map_gpu_beginning flag is supplied
    if args.use_gpu:
        if args.map_gpu_beginning:
            logger.info("Will map all data to CUDA tensors. If you run out of GPU memory, then turn off "
                        "--map_gpu_beginning flag.")
            device_load_data = device
        else:
            device_load_data = torch.device('cpu')
    else:
        device_load_data = device
    train_loaders, train_file_names, val_loader = load_train_val_data(args, device_load_data, logger)
    if val_loader is None:
        args.save_best_on_train = True  # forces saving according to best loss on train set.
    total_train_num_batches = 0
    for t in train_loaders:
        total_train_num_batches += len(t)
    model, model_name = get_model(args, device)
    optimizer, optimizer_shape, scheduler = get_optimizer(args, model)
    loss_fn = get_loss(args)

    if args.variable_shape and args.train_var_shape:
        print("")  # new line
        # load the shape-based pooling data for the existing set (or compute it if it does not exist in file)
        t_load_shape_data = time.time()
        logger.info("Loading (or computing from scratch) shape-based data for training the variable shape Network")
        shape_dataset, shape_loader = data_loader_shapes(args, train_loaders, model.interaction_module, device)
        """
        if args.gt_generator_model:
            gt_model, _ = get_model(args, device)
        else:
            gt_model = None
        """
        gt_model = None
        logger.info(f"DONE! Shape data loaded. Took {time.time() - t_load_shape_data:.2f} s")
    else:
        shape_dataset, shape_loader, gt_model = None, None, None

    train_loss_history = []
    val_loss_history = []
    # train/val loss is cuda tensor with 0 dims. Need a maximum value for history, and need to append it to a list to
    #   plot graph with loss evolution.
    min_loss = torch.tensor(float('inf'), device=device)
    start_epoch = 0
    if args.init_with_state_dict:
        state_data = torch.load(args.init_with_state_dict, map_location=device)
        try:
            model.load_state_dict(state_data['model_state_dict'])
        except RuntimeError:
            # try to perform partial load of parameters
            msd = model.state_dict()
            for label, state in state_data['model_state_dict'].items():
                if label in msd:
                    msd[label] = state
            model.load_state_dict(msd)
        try:
            optimizer.load_state_dict(state_data['optimizer_state_dict'])
        except ValueError:
            # partial load for optimizer parameters, in case there was an error in the matching of some parameters
            osd = optimizer.state_dict()
            # WARNING: MIGHT SPECIFIC TO PYTORCH VERSION
            for label, state in state_data['optimizer_state_dict']['param_groups'][0].items():
                if label in osd['param_groups'][0]:
                    if label == 'params':
                        continue
                    else:
                        osd['param_groups'][0][label] = state
            optimizer.load_state_dict(osd)
        if state_data['lr_scheduler_state_dict'] is not None and scheduler is not None:
            scheduler.load_state_dict(state_data['lr_scheduler_state_dict'])
        del state_data
        logger.info(f'LOADED Initial state dictionary from {args.init_with_state_dict}')
    if args.save_init_state_dict:
        save_data = {
            'epoch': -1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
        }
        torch.save(save_data, args.save_init_state_dict)
        logger.info(f'Initial state dictionary of {model_name} saved to {args.save_init_state_dict}')
    if args.load_checkpoint:
        data = torch.load(args.load_checkpoint, map_location=device)
        """
        'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        """
        start_epoch = data['epoch']
        checkpoint = data['best']  # best model saved so far
        if start_epoch >= args.num_epochs:
            tl, vl, e = checkpoint['train_loss'], checkpoint['val_loss'], checkpoint['epoch']
            logger.error(f'Checkpoint loaded from {args.load_checkpoint} has already been trained for {start_epoch}, '
                         f'and the training was meant to be for {args.num_epochs}.{os.linesep}Saving best model from '
                         f'the supplied checkpoint (train_loss={tl:.3f}; val_loss={vl:3f}; epoch={e}) to '
                         f'{args.save_to}')
            torch.save(checkpoint, args.save_to)
            return
        # the best loss is according to the arguments supplied here, not the ones from the previous checkpoint
        if args.save_best_on_train:
            min_loss = checkpoint['train_loss']
        elif args.save_best_on_train_val_avg:
            min_loss = (checkpoint['train_loss'] + checkpoint['val_loss']) / 2
        else:
            min_loss = checkpoint['val_loss']
        model.load_state_dict(data['model_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
        if data['lr_scheduler_state_dict'] is not None and scheduler is not None:
            scheduler.load_state_dict(data['lr_scheduler_state_dict'])
        logger.info(f'Successfully loaded checkpoint from {args.load_checkpoint}, that was trained for {start_epoch} '
                    f'epochs')
        train_loss_history = checkpoint['train_loss_history'] if 'train_loss_history' in checkpoint else []
        val_loss_history = checkpoint['val_loss_history'] if 'val_loss_history' in checkpoint else []
        del checkpoint
        del data

    checkpoint = {}  # checkpoint corresponds to locally caching (in memory) model parameters at a point in training

    print(os.linesep)
    logger.info(f'Starting training. Total of {args.num_epochs - start_epoch} epochs, with a total of '
                f'{total_train_num_batches} batches for training and '
                f'{len(val_loader) if val_loader is not None else 0} batches for validation.')
    t_train_start = time.time()
    for epoch in range(start_epoch, args.num_epochs):
        if args.timing and args.use_gpu:
            torch.cuda.synchronize()  # Waits for all kernels in all streams on a CUDA device to complete.
        t_epoch_start = time.time()
        # logger.info(f'Starting train epoch {epoch + 1}')
        gc.collect()  # use garbage collector to try to keep memory usage in steady levels
        if args.variable_shape and args.train_var_shape:
            train_loss, train_main_loss, train_shape_loss = train_var_shape_epoch(args, model, device, optimizer_shape,
                                                                                  shape_loader, loss_fn,
                                                                                  __compute_loss__, logger)
        else:
            train_main_loss = train_shape_loss = 0
            if args.profile:
                # use profiler on the epoch training
                with profiler.profile(use_cuda=args.use_gpu) as prof:
                    # with profiler.record_function("model_training_1_epoch"):
                    train_loss = lstm_train_epoch(args, train_loaders, model, optimizer, loss_fn, device)
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))
            else:
                train_loss = lstm_train_epoch(args, train_loaders, model, optimizer, loss_fn, device)
            if args.use_gpu:
                torch.cuda.synchronize()  # Waits for all kernels in all streams on a CUDA device to complete.
        if args.use_gpu:
            torch.cuda.synchronize()  # Waits for all kernels in all streams on a CUDA device to complete.
        t_epoch_train_end = time.time()
        # TODO possibly implement proper early stopping - https://github.com/Bjarten/early-stopping-pytorch
        if val_loader is not None:
            val_loss = lstm_loss_on_set(args, model, val_loader, loss_fn, device)
        else:
            val_loss = torch.tensor(float('nan'))
        if scheduler and epoch >= args.lr_step_start:  # apply scheduler to reduce learning rate
            scheduler.step()
        if args.timing:
            if args.use_gpu:
                torch.cuda.synchronize()  # Waits for all kernels in all streams on a CUDA device to complete.
            t_epoch_end = time.time()
            if args.variable_shape and args.train_var_shape:
                epoch_summary = 'Epoch {0} finished in {1:.2f} s - {2:.2f} s for forward/backward prop; {3:.2f} s ' \
                                'for validation loss calculation.' + \
                                '{4}Training loss = {5:.3f} WITH MainLoss={6:.3f}; ShapeLoss={7:.3f}; ' \
                                'Validation loss = {8:.3f}'
                logger.info(
                    epoch_summary.format(epoch + 1, t_epoch_end - t_epoch_start, t_epoch_train_end - t_epoch_start,
                                         t_epoch_end - t_epoch_train_end, os.linesep, train_loss, train_main_loss,
                                         train_shape_loss, val_loss))
            else:
                epoch_summary = 'Epoch {0} finished in {1:.2f} s - {2:.2f} s for forward/backward prop, {3:.2f} s for' \
                                'validation +loss calculation).{4}Training loss = {5:.3f}; Validation loss = {6:.3f}'
                logger.info(epoch_summary.format(epoch + 1, t_epoch_end - t_epoch_start,
                                                 t_epoch_train_end - t_epoch_start, t_epoch_end - t_epoch_train_end,
                                                 os.linesep, train_loss, val_loss))
        else:
            if args.variable_shape and args.train_var_shape:
                logger.info('Epoch {0} finished. Training loss = {1:.3f} WITH MainLoss={2:.3f}; ShapeLoss{3:.3f}; '
                            'Validation loss = {4:.3f}; '.format(epoch + 1, train_loss, train_main_loss,
                                                                 train_shape_loss, val_loss))
            else:
                logger.info('Epoch {0} finished. Training loss = {1:.3f}; Validation loss = {2:.3f}; '.format(
                    epoch + 1, train_loss, val_loss))
        # Append losses to history list.
        if args.use_gpu:
            # map to cpu before appending; note that .detach() is not needed for validation, because there were no
            # gradients computed for these losses (use of torch.no_grad())
            train_loss_history.append(train_loss.cpu().detach().numpy())
            val_loss_history.append(val_loss.cpu().numpy())
        else:
            train_loss_history.append(train_loss.detach().numpy())
            val_loss_history.append(val_loss.numpy())
        # decide if meant to save checkpoint based on validation loss
        if args.save_best_on_train:
            loss_cmp = train_loss
        elif args.save_best_on_train_val_avg:
            loss_cmp = (train_loss + val_loss) / 2
        else:
            loss_cmp = val_loss
        if loss_cmp < min_loss:
            min_loss = loss_cmp
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history,
                'model_name': model_name,
                'args': args,
                'fields': model.motion_fields.__dict__ if args.fields_location else None
            }
        if args.save_last_epoch:
            save_epoch = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                'best': checkpoint
            }
            extension_index = args.save_to.index('.')
            torch.save(save_epoch, args.save_to[:extension_index] + '_checkpoint' + args.save_to[extension_index:])
        if args.save_every > 0 and max(args.save_every_after - 1, 0) <= epoch < args.num_epochs - 1 and \
                (epoch + 1) % args.save_every == 0:
            # save the best checkpoint so far
            extension_index = args.save_to.index('.')
            path_x_epoch = args.save_to[:extension_index] + f'_epoch{epoch + 1}' + args.save_to[extension_index:]
            torch.save(checkpoint, path_x_epoch)
            logger.info(f'Saved best checkpoint so far to {path_x_epoch}.{os.linesep}\t'
                        f'Used checkpoint from epoch {checkpoint["epoch"]}, with train_loss='
                        f'{checkpoint["train_loss"]:.3f} and val_loss={checkpoint["val_loss"]:.3f}')
    print(os.linesep)
    if args.timing:
        end_train_time = time.time()
        logger.info('End of training - took a total of {0:.3f} seconds; average epoch time was {1:.3f} seconds.'.format(
            end_train_time - t_train_start, (end_train_time - t_train_start) / float(args.num_epochs)))
    else:
        logger.info('End of training')
    if args.plot_losses:
        # plot training and val losses throughout the epochs
        horizontal_axis = np.linspace(1, args.num_epochs, num=args.num_epochs).astype(int)
        plt.plot(horizontal_axis, np.array(train_loss_history), color=TRAIN_COLOR, linewidth=PLOT_LINE_WIDTH,
                 label='train loss')
        plt.plot(horizontal_axis, np.array(val_loss_history), color=VAL_COLOR, linewidth=PLOT_LINE_WIDTH,
                 label='val loss')
        plt.title(f'Training and Validation Loss (Type={args.loss.upper()})', fontdict={'fontsize': 18})
        plt.ylabel('Loss', fontdict={'fontsize': 14})
        plt.xlabel('Epoch', fontdict={'fontsize': 14})
        plt.legend()
        plt.show()
    # save model
    if checkpoint is not None:
        print(os.linesep)
        logger.info('Saving model to {0}.{1}Used checkpoint from epoch {2}, with train_loss={3:.3f} '
                    'and val_loss={4:.3f}'.format(args.save_to, os.linesep, checkpoint['epoch'],
                                                  checkpoint['train_loss'], checkpoint['val_loss']))
        # add history of losses;
        checkpoint['train_loss_history'] = train_loss_history
        checkpoint['val_loss_history'] = val_loss_history
        torch.save(checkpoint, args.save_to)


def lstm_train_epoch(args, train_loaders, model, optimizer, loss_fn, device):
    """
    Computes one epoch of training
    :param args: several arguments that may be used to configure the manner in which the training epoch will be done
    :param train_loaders: list of loaders for the training dataset (may contain just one loader with all the data
    :param model: model to be trained
    :param optimizer: optimizer to use for training (e.g. SGD, Adam)
    :param loss_fn: type of loss function to apply
    :param device: device on which the tensors should be stored
    :return: the overall loss for this epoch
    """
    model.train()
    full_loss_history = torch.tensor([], device=device)
    full_num_peds_per_batch_history = torch.tensor([], device=device)
    full_pred_traj_len_per_batch_history = torch.tensor([], device=device)
    curr_batch = 0
    num_batch = np.sum(np.array([len(t) for t in train_loaders]))
    i = 0
    for train_loader in train_loaders:
        current_batch_size, loss, num_peds_per_batch, pred_traj_len_per_batch = __init_batch_configuration__(device)
        for batch in train_loader:
            curr_batch += 1
            print(f'\r Train Batch: {curr_batch}/{num_batch}', end='')
            if args.use_gpu and not args.map_gpu_beginning:
                # map tensors in batch to the desired device
                batch = [tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, metadata, loss_mask, seq_start_end) = batch
            # all trajectory same length and vanilla lstm doesn't care about interactions - no need to use seq_start_end
            if args.use_abs or args.fields_location or args.pooling_type:
                # use absolute positions; also applies to when motion fields are incorporated
                # also applies for use of social interactions pooling (note that the output of this is displacements)
                obs_traj_seq = obs_traj
                pred_traj_seq = pred_traj_gt
            elif args.use_acc:
                # use acceleration - differences between relative displacements
                obs_traj_seq, pred_traj_seq = torch.zeros_like(obs_traj_rel), torch.zeros_like(pred_traj_rel_gt)
                obs_traj_seq[1:, :, :] = obs_traj_rel[1:, :, :] - obs_traj_rel[:-1, :, :]
                pred_traj_seq[1:, :, :] = pred_traj_rel_gt[1:, :, :] - pred_traj_rel_gt[:-1, :, :]
            else:
                # use relative displacements
                obs_traj_seq = obs_traj_rel
                pred_traj_seq = pred_traj_rel_gt
            obs_traj_len = obs_traj_seq.shape[0]
            pred_traj_len = pred_traj_seq.shape[0]
            # model forward pass
            if args.teacher_forcing:
                pred_len_in, pred_seq_in = -1, pred_traj_seq.clone()
            else:
                pred_len_in, pred_seq_in = pred_traj_len, None
            if args.pooling_type:
                # use social interactions
                if args.fields_location:
                    # also use fields - requires passing metadata as well since it might be required
                    pred_seq = model(obs_traj_seq, pred_len_in, pred_traj_gt=pred_seq_in, seq_start_end=seq_start_end,
                                     metadata=metadata)
                else:
                    pred_seq = model(obs_traj_seq, pred_len_in, pred_traj_gt=pred_seq_in, seq_start_end=seq_start_end)
            else:
                # don't use social interactions - discard
                pred_seq = model(obs_traj_seq, pred_len_in, pred_traj_gt=pred_seq_in)
            if args.primary_ped_only:
                seq_mask = torch.zeros(obs_traj_seq.shape[1], dtype=torch.bool, device=obs_traj_seq.device)
                seq_mask[seq_start_end[:, 0]] = True
            else:
                # just don't consider partial trajectories for computing loss
                seq_mask = (torch.any(torch.isnan(obs_traj_seq[:, :, 0]), dim=0) +
                            torch.any(torch.isnan(pred_traj_gt[:, :, 0]), dim=0) +
                            torch.any(torch.isnan(pred_seq[:, :, 0]), dim=0)) == 0  # tensor of shape (batch)
            # apply mask to discard incomplete trajectories - cannot compute loss with them
            pred_seq, pred_traj_gt, pred_traj_rel_gt = pred_seq[:, seq_mask, :], pred_traj_gt[:, seq_mask, :], \
                                                       pred_traj_rel_gt[:, seq_mask, :]
            loss_aux = __compute_loss__(args, loss_fn, pred_seq, pred_traj_gt, pred_traj_rel_gt,
                                        loss_mask[seq_mask, obs_traj_len:], obs_traj[-1, seq_mask],
                                        obs_traj_rel[-1, seq_mask])
            # Alternatively, seq_start_end.shape[0] can be used - batch not being a set of pedestrian trajectories, but
            # instead a set of sequences (trajectories in the same time frame)
            current_batch_size += obs_traj.shape[1]
            # aggregate data in a batch
            loss, num_peds_per_batch, pred_traj_len_per_batch = \
                __cat_losses__(loss, num_peds_per_batch, pred_traj_len_per_batch, loss_aux, pred_seq.size(1),
                               pred_traj_len, device)
            # since batches may not actually have specified batch size, we make sure we only do backward propagation
            # once we have a big enough batch of sequence of trajectories (that may have different length, depending on
            # other arguments that were supplied)
            if current_batch_size >= args.batch_size:
                full_loss_history, full_num_peds_per_batch_history, full_pred_traj_len_per_batch_history = \
                    __train_with_batch__(args, loss, num_peds_per_batch, pred_traj_len_per_batch, full_loss_history,
                                         full_num_peds_per_batch_history, full_pred_traj_len_per_batch_history,
                                         optimizer)
                # re-initialize fields to train another batch
                current_batch_size, loss, num_peds_per_batch, pred_traj_len_per_batch = \
                    __init_batch_configuration__(device)
        # in case last batch doesn't get to the batch size, we still want to use those values for training
        if 0 < current_batch_size < args.batch_size:
            full_loss_history, full_num_peds_per_batch_history, full_pred_traj_len_per_batch_history = \
                __train_with_batch__(args, loss, num_peds_per_batch, pred_traj_len_per_batch, full_loss_history,
                                     full_num_peds_per_batch_history, full_pred_traj_len_per_batch_history, optimizer)
    # get the overall average loss of the epoch
    loss = full_loss_history
    num_peds_per_batch = full_num_peds_per_batch_history
    pred_traj_len_per_batch = full_pred_traj_len_per_batch_history
    loss_epoch = torch.dot(loss, (num_peds_per_batch * pred_traj_len_per_batch) /
                           torch.dot(num_peds_per_batch, pred_traj_len_per_batch))
    print('\r', end='')
    return loss_epoch


def __train_with_batch__(args, loss, num_peds_per_batch, pred_traj_len_per_batch, full_loss_list,
                         full_num_peds_per_batch_list, full_pred_traj_len_per_batch_list, optimizer):
    """
    train a single batch, returning the complete loss values and their "weights" using existing list from previous
    batches.
    :param args: several arguments that may be used to configure the manner in which the training will be done
    :param loss: Tensor of shape (num_batches). Contains loss for potentially several batches (to reach the number of
    args.batch_size in case some batches are smaller).
    :param num_peds_per_batch: Tensor of shape (num_batches). Number of pedestrians per batch (for average of losses
    for existing batches)
    :param pred_traj_len_per_batch: Tensor of shape (num_batches). Average prediction length per batch (for weighted
    average of losses for existing batches)
    :param full_loss_list: The current loss list for previously trained batches. The loss list for the training of
    these batches will be appended here.
    :param full_num_peds_per_batch_list: The current list of number of pedestrians for previously trained batches.
    The number of pedestrians for these batches will be appended here.
    :param full_pred_traj_len_per_batch_list: The current list of prediction lengths for previously trained batches.
    The average prediction lengths for these batches will be appended here.
    :param optimizer: Provided optimizer to perform training
    :return: the lists for losses, number of pedestrians, and prediction lengths, with the newly appended data
    """
    # average of the errors with respect to number of pedestrians (or number of trajectories), and also the
    # length of the the trajectories
    # can be done for several "batches". Note that grad needs scalar outputs
    if args.loss_no_len:
        loss_train = torch.dot(loss, num_peds_per_batch / torch.sum(num_peds_per_batch))
    else:
        loss_train = torch.dot(loss, (num_peds_per_batch * pred_traj_len_per_batch) /
                               torch.dot(num_peds_per_batch, pred_traj_len_per_batch))
    # backpropagation - computing gradients
    loss_train.backward()
    # update weights - backward pass was done prior to this
    optimizer.step()
    # zero the gradients for training
    optimizer.zero_grad()
    full_loss_list = torch.cat((full_loss_list, loss.detach()), dim=0)
    full_num_peds_per_batch_list = torch.cat((full_num_peds_per_batch_list, num_peds_per_batch), dim=0)
    full_pred_traj_len_per_batch_list = torch.cat((full_pred_traj_len_per_batch_list, pred_traj_len_per_batch), dim=0)
    return full_loss_list, full_num_peds_per_batch_list, full_pred_traj_len_per_batch_list


def lstm_loss_on_set(args, model, loader, loss_fn, device):
    """
    Compute the loss on a certain dataset
    :param args: several arguments that may be used to configure the manner in which the losses will be computed
    :param model: prediction model to compute the loss on
    :param loader: loader for the dataset to compute the loss on
    :param loss_fn: type of loss function to apply
    :param device: device on which the tensors should be stored
    :return: the computed loss (will be an average of losses computed on batches)
    """
    model.eval()
    _, loss, num_peds_per_batch, pred_traj_len_per_batch = __init_batch_configuration__(device)
    # no gradient and backward pass here - just forward pass to compute loss
    curr_batch, num_batch = 0, len(loader)
    with torch.no_grad():
        for batch in loader:
            curr_batch += 1
            print(f'\r Val Batch: {curr_batch}/{num_batch}', end='')
            if args.use_gpu and not args.map_gpu_beginning:
                # map tensors in batch to the desired device
                batch = [tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, metadata, loss_mask, seq_start_end) = batch
            # for the case of args.fields_location, if sparse motion fields were supplied, then trajectory to supply
            # should be in absolute coordinates
            obs_traj_seq = obs_traj if args.use_abs or args.fields_location or args.pooling_type else obs_traj_rel
            pred_traj_len = pred_traj_gt.shape[0]
            obs_traj_len = obs_traj_seq.shape[0]
            # model forward pass - and compute loss
            if args.pooling_type:
                if args.fields_location:
                    # also use fields - requires passing metadata as well since it might be required
                    pred_seq = model(obs_traj_seq, pred_traj_len, seq_start_end=seq_start_end, metadata=metadata)
                else:
                    pred_seq = model(obs_traj_seq, pred_len=pred_traj_len, seq_start_end=seq_start_end)
            else:
                pred_seq = model(obs_traj_seq, pred_len=pred_traj_len)
            if args.primary_ped_only:
                seq_mask = torch.zeros(obs_traj_seq.shape[1], dtype=torch.bool, device=obs_traj_seq.device)
                seq_mask[seq_start_end[:, 0]] = True
            else:
                # just don't consider partial trajectories for computing loss
                seq_mask = (torch.any(torch.isnan(obs_traj_seq[:, :, 0]), dim=0) +
                            torch.any(torch.isnan(pred_traj_gt[:, :, 0]), dim=0) +
                            torch.any(torch.isnan(pred_seq[:, :, 0]), dim=0)) == 0  # tensor of shape (batch)
            # apply mask to discard incomplete trajectories - cannot compute loss with them
            pred_seq, pred_traj_gt, pred_traj_rel_gt = pred_seq[:, seq_mask, :], pred_traj_gt[:, seq_mask, :], \
                                                       pred_traj_rel_gt[:, seq_mask, :]
            loss_aux = __compute_loss__(args, loss_fn, pred_seq, pred_traj_gt, pred_traj_rel_gt,
                                        loss_mask[seq_mask, obs_traj_len:], obs_traj[-1, seq_mask],
                                        obs_traj_rel[-1, seq_mask])
            loss, num_peds_per_batch, pred_traj_len_per_batch = \
                __cat_losses__(loss, num_peds_per_batch, pred_traj_len_per_batch, loss_aux, loss_mask.size(0),
                               pred_traj_len, device)
    if args.loss_no_len:
        avg_loss = torch.dot(loss, num_peds_per_batch / torch.sum(num_peds_per_batch))
    else:
        avg_loss = torch.dot(loss, (num_peds_per_batch * pred_traj_len_per_batch) /
                             torch.dot(num_peds_per_batch, pred_traj_len_per_batch))
    print('\r', end='')
    return avg_loss


def __init_batch_configuration__(device):
    """
    initialize some parameters that are used to configure and store the training in one batch
    :param device: the torch.device to store created tensors on
    :return: empty parameters for:
        - the current batch size
        - training loss
        - how many pedestrians are there in each batch
        - what is the prediction length, in each batch
    """
    current_batch_size = 0
    # for loss computation (will accumulate if batch size is below desired)
    loss = torch.tensor([], device=device)
    # number of pedestrians per batch (will accumulate if batch size is below desired)
    num_peds_per_batch = torch.tensor([], device=device)
    # expected pedestrian length per batch
    pred_traj_len_per_batch = torch.tensor([], device=device)
    return current_batch_size, loss, num_peds_per_batch, pred_traj_len_per_batch


def __cat_losses__(loss, num_peds_per_batch, pred_traj_len_per_batch, loss_new, num_peds_new, pred_len_new, device):
    """
    concatenate prior losses with new ones, as well as other parameters
    :param loss: the list with prior losses
    :param num_peds_per_batch: list with number of pedestrians for prior batches
    :param pred_traj_len_per_batch:
    :param loss_new:
    :param num_peds_new:
    :param pred_len_new:
    :param device:
    :return:
    """
    loss = torch.cat((loss, loss_new.view(1).to(device)), dim=0)
    num_peds_per_batch = torch.cat((num_peds_per_batch, torch.tensor([num_peds_new], dtype=torch.float,
                                                                     device=device).view(1)), dim=0)
    pred_traj_len_per_batch = torch.cat((pred_traj_len_per_batch, torch.tensor([pred_len_new], dtype=torch.float,
                                                                               device=device).view(1)), dim=0)
    return loss, num_peds_per_batch, pred_traj_len_per_batch


def get_model(args, device):
    """
    build and initialize the model to be trained
    :param args: several arguments required to build the model
    :param device: device for the model to be mapped to (e.g. cuda, cpu)
    :return: the LSTM model
    """
    t1 = time.time()
    logger.info(f"Initializing LSTM model")
    activation_on_output = get_output_activation(args)
    model_interaction_module_label, pooling_shape = get_interaction_module_label(args)
    if model_interaction_module_label is not None:
        # model incorporates social interactions
        interaction_module = build_interaction_module(args, model_interaction_module_label, pooling_shape)
        shape_config = build_shape_config(args, interaction_module, pooling_shape)
        if args.fields_location:
            # uses interactions and motion fields - interaction and scene-aware
            fields_model = __get_fields_model__(args, device)
            model = FieldsWithInteractionModuleAndLSTM(fields_model, interaction_module, shape_config,
                                                       embedding_dim=args.embedding_dim, h_dim=args.lstm_h_dim,
                                                       activation_on_input_embedding=get_input_activation(args),
                                                       output_gaussian=args.out_gaussian,
                                                       activation_on_output=activation_on_output,
                                                       feed_all=args.feed_all_fields,
                                                       use_probs=args.feed_with_probabilities)
        else:
            model = LSTMWithInteractionModule(interaction_module, shape_config=shape_config,
                                              embedding_dim=args.embedding_dim, h_dim=args.lstm_h_dim,
                                              activation_on_input_embedding=get_input_activation(args),
                                              output_gaussian=args.out_gaussian, use_enc_dec=args.use_enc_dec,
                                              activation_on_output=activation_on_output, dropout=args.dropout)
    elif args.fields_location:
        fields_model = __get_fields_model__(args, device)
        if args.simple_fields:
            model = SimpleFieldsWithLSTM(fields=fields_model, embedding_dim=args.embedding_dim, h_dim=args.lstm_h_dim,
                                         activation_on_input_embedding=get_input_activation(args),
                                         activation_on_output=activation_on_output, dropout=args.dropout,
                                         num_layers=args.num_layers, normalize_embedding=args.normalize_embedding,
                                         output_gaussian=args.out_gaussian, discard_zeros=args.discard_zeros)
        else:
            model = FieldsWithLSTM(fields=fields_model, feed_all=args.feed_all_fields or args.feed_with_probabilities,
                                   use_probs=args.feed_with_probabilities, embedding_dim=args.embedding_dim,
                                   h_dim=args.lstm_h_dim, activation_on_input_embedding=get_input_activation(args),
                                   activation_on_output=activation_on_output, dropout=args.dropout,
                                   num_layers=args.num_layers, normalize_embedding=args.normalize_embedding,
                                   output_gaussian=args.out_gaussian, discard_zeros=args.discard_zeros)
    elif args.use_enc_dec:
        model = VanillaLstmEncDec(args.embedding_dim, args.lstm_h_dim,
                                  activation_on_input_embedding=get_input_activation(args),
                                  activation_on_output=activation_on_output, dropout=args.dropout,
                                  num_layers=args.num_layers, extra_info=args.feed_history,
                                  normalize_embedding=args.normalize_embedding, output_gaussian=args.out_gaussian,
                                  discard_zeros=args.discard_zeros)
    else:
        model = VanillaLSTM(args.embedding_dim, args.lstm_h_dim,
                            activation_on_input_embedding=get_input_activation(args),
                            activation_on_output=activation_on_output, dropout=args.dropout,
                            history_on_pred=args.feed_history, normalize_embedding=args.normalize_embedding,
                            output_gaussian=args.out_gaussian, discard_zeros=args.discard_zeros)
    # model should always be mapped to cuda before assigning optimizer
    model_name = model.__class__.__name__
    model.to(device)
    model.train()
    if args.timing:
        logger.info("Done! Took {0:.2f} s to initialize model of type {1}".format(time.time() - t1, model_name))
        # to check that model parameters are on cuda
        # print(next(model.parameters()).is_cuda)

    return model, model_name


def __get_fields_model__(args, device):
    fields_list = load_fields(device, args.fields_location)
    model_data = fields_list[0][0]
    if len(fields_list) > 1:
        logger.warning(f'Received more than one model for motion fields, but will only use the first one: '
                       f'specific to {fields_list[0][1]} scene')
    fields_model = SparseMotionFields(model_data['Te_best'], model_data['Qe_best'], model_data['Bc_best'],
                                      model_data['min_max'], model_data['parameters'])
    return fields_model


def get_optimizer(args, model):
    """
    get the optimizer to be used in training the model
    see https://pytorch.org/docs/stable/optim.html for the kind of optimizers available (not all have been used here)
    :param args: several arguments required to choose and build the optimizer
    :param model: the model on which the optimizer will act
    :return: the torch.optim class for the desired optimizer
    """
    # the assert shouldn't fail unless a bug is present - the assert makes sure we don't go further
    assert args.optim in OPTIMIZER_CHOICES, 'The optimizer type ' + args.optim + ' is not available!'
    optimizer_type = args.optim.lower()
    model_params = model.parameters()
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model_params, lr=args.learning_rate, momentum=args.momentum,
                              nesterov=args.use_nesterov)
    else:
        # optimizer_type == 'adam'
        optimizer = optim.Adam(model_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    # zero the gradients before any training begins
    optimizer.zero_grad()
    if args.lr_step_epoch and args.lr_step_epoch > 0:
        return optimizer, optimizer, \
               lr_scheduler.StepLR(optimizer, step_size=args.lr_step_epoch, gamma=args.lr_step_gamma)
    # else, no learning rate scheduler
    return optimizer, optimizer, None


def get_loss(args):
    """
    get the desired loss function to be used in training the model
    :param args: several arguments required to choose the loss function
    :return: the loss function (it is an actual function)
    """
    # the assert shouldn't fail unless a bug is present - the assert makes sure we don't go further
    loss = args.loss.lower()
    assert loss in LOSS_CHOICES, 'The loss type ' + loss + ' is not available!'
    if args.out_gaussian:
        if loss == 'l2':
            # the one used by default
            args.loss = 'NLL'
            return nll
        elif loss == 'nll':
            return nll
        elif loss == 'gl':
            return gaussian_likelihood_loss
    if loss == 'l2':
        return l2_loss
    return l2_loss


def __compute_loss__(args, loss_fn, pred_seq, pred_traj_gt, pred_traj_rel_gt, loss_mask, last_obs_pos, last_obs_vel,
                     mode=None):
    """
    compute the loss, which can be done in a different format depending on specified arguments
    :param args: several arguments that may be used to specify how the loss is meant to be computed
    :param loss_fn: the actual loss function to use (expects predictions and ground truth, with same dimensions)
    :param pred_seq: Tensor of shape (pred_traj_len, batch, output_shape). Contains the model prediction, which can be
    sequence of positions/velocities (output_shape=2), or for instance a probability distribution (output_shape=5 in the
    case of a bi-variate gaussian).
    :param pred_traj_gt: Tensor of shape (pred_traj_len, batch, 2). The ground truth trajectory, in absolute
    positions, to compare with the prediction.
    :param pred_traj_rel_gt: Tensor of shape (pred_traj_len, batch, 2). The ground truth trajectory, in relative
    displacements (velocities), to compare with the prediction.
    :param loss_mask: Tensor of shape (batch, traj_len). Applies a mask to the loss values (in case one doesn't want to
    consider some trajectories for the sake of loss computation).
    :param last_obs_pos: Tensor of shape (batch, 2). The last observed (past) position for each trajectory.
    :param last_obs_vel: Tensor of shape (batch, 2). The last observed (past) velocity, or relative displacement, for
    each trajectory.
    :return: the computed loss, a tensor whose shape can be different depending on the kind of loss.
    Usually, it's a tensor of shape (batch), having one loss value per trajectory.
    """
    if not args.use_abs:
        if args.use_acc:
            # convert from acceleration to relative displacements
            pred_seq = relative_traj_to_abs(pred_seq, last_obs_vel)
        if args.loss_rel or args.out_gaussian:
            # applies for both cases of l2 loss between relative displacements and nll loss
            # use of ground truth relative displacements instead of the ground truth absolute positions
            loss = loss_fn(pred_seq, pred_traj_rel_gt, loss_mask) if mode is None else \
                loss_fn(pred_seq, pred_traj_rel_gt, loss_mask, mode=mode)
        else:
            pred_seq = relative_traj_to_abs(pred_seq, last_obs_pos)
            loss = loss_fn(pred_seq, pred_traj_gt, loss_mask) if mode is None else \
                loss_fn(pred_seq, pred_traj_gt, loss_mask, mode=mode)
    else:
        loss = loss_fn(pred_seq, pred_traj_gt, loss_mask) if mode is None else \
            loss_fn(pred_seq, pred_traj_gt, loss_mask, mode=mode)
    return loss


if __name__ == '__main__':
    arguments = parser.parse_args()
    if hasattr(arguments, 'load_args_from_json') and arguments.load_args_from_json:
        new_args = override_args_from_json(arguments, arguments.load_args_from_json, parser)
    else:
        new_args = arguments
    main(new_args, parser)
