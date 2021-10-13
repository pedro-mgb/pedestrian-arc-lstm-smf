"""
Created on June 28th 2021
"""
import copy

import numpy as np
import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    """
    Create a NLL loss but by employing label smoothing. Particularly useful when the data is not 100% reliable, and
    there is a probability that other labels can actually be the real one.
    Credits to github users @ https://github.com/pytorch/pytorch/issues/7455
    """

    def __init__(self, smoothing=0.0, dim=-1):
        """
        Create the label smoothing loss
        :param smoothing: smooting probability (0 for no smoothing, 1 for full smoothing)
        :param dim: dimension to apply the loss sum
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        """
        Compute the forward pass, to compute the loss
        :param pred: predictions from a model
        :param target: target data (ground truth)
        :return: the overall mean of the loss
        """
        # assuming last index of pred has number of classes
        # pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.shape[-1] - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class SimulatedInteractionModule(nn.Module):
    """
    Simulates the ShapeBasedPooling class (../interacion_modules/shape_based.py) to return pooling of of neighbour
    motion. The pooling data has been pre-computed, which is why this module simulates it.
    The module knows how many sequences there are, and sequentially returns the correct portion of the pooling data
    assuming that they come in the same order as they were sent (increasing the sequence ID, then increasing the time
    instant t once all sequences have been returned)
    """

    def __init__(self, obs, pred, seq_start_end, pool_data, occ_data, interaction_module):
        """
        Initialize the SimulatedInteractionModule with a batch of data
        :param obs: Tensor of shape (obs_len, num_peds, 2). Observed (past) trajectories
        :param pred: Tensor of shape (pred_len, num_peds, 2). Predicted (or future) trajectories
        :param seq_start_end: Tensor of shape (batch_size). Delimitting for each sequence apart of the batch
        :param pool_data: Tensor of shape (obs_len + pred_len, num_peds, *(shape_dims), 2). Pooling data (assumed it is
        directional pooling) to return
        :param occ_data: Tensor of shape (obs_len + pred_len, num_peds, *(shape_dims), 1). Occupancy pooling data
        (assumed it is percentage-wise) to return
        :param interaction_module: The actual interaction module, that contains the embedding layers to employ. If
        desired, the pooling data will be passed through such embedding layer
        """
        super(SimulatedInteractionModule, self).__init__()
        self.pool_data, self.occ_data, self.start_end = pool_data, occ_data, seq_start_end
        # first position is not used, because it is only used to compute the displacement
        self.traj_len, self.num_seqs = obs.shape[0] - 1 + pred.shape[0], seq_start_end.shape[0]
        self.t, self.s = 0, 0  # marking current instant for time and
        # this may result in an error if embedding occ does not exist
        self.embedding, self.embedding_occ = interaction_module.embedding, interaction_module.embedding_occ

    def forward(self, h, positions, past_positions=None, embed=True):
        """
        Compute forward pass, and retrieve the relevant amount of pooling data using the current sequence index, and
        the current time index
        The arguments are documented in forward method of ShapeBasedPooling class (../interacion_modules/shape_based.py)
        :return: the pooling data (directional and occupancy), possibly embedded
        """
        pool = self.pool_data[self.t, self.start_end[self.s, 0]:self.start_end[self.s, 1]]
        occ = self.occ_data[self.t, self.start_end[self.s, 0]:self.start_end[self.s, 1]]
        self.s += 1
        if self.s >= self.num_seqs:
            self.t += 1
            self.s = 0
        if embed:
            return self.embedding(pool.reshape([pool.shape[0], np.prod(pool.shape[1:])])), \
                   self.embedding_occ(occ.reshape([occ.shape[0], np.prod(occ.shape[1:])]))
        else:
            return pool, occ


def train_var_shape_epoch(args, model, device, optimizer, shape_loader, main_loss_fn, loss_call, logger):
    """
    train the variable shape configuration model (which may partially include the rest of the model) for one epoch, with
    pre-loaded pooling shape data (to decrease the overall training time)
    :param args: command line arguments to configure this auxiliary training
    :param model: the trajectory forecasting model, including the shape configuration module (shape_config attribute)
    :param device: the torch.device (e.g. cpu, cuda) to map all tensors to
    :param optimizer: the torch.optimizer used to update the weights
    :param shape_loader: the DataLoader of pre-computed pooling shape data (directional and occupancy)
    :param main_loss_fn: the type of loss to use (e.g. NLL)
    :param loss_call: The wrapper of the main_loss_fn call to use. Should be a method from the main train script
    :param logger: the logger to log relevant info regarding
    :return: the overall loss, and the part of loss regarding the trajectory prediction, and the one regarding the
    picking of a pooling shape
    """
    # also disable training for parts of model
    gt_generator, interaction_module = create_gt_generator(model)
    # only perform parameter update for the shape configuration LSTM network
    model.train()
    curr_batch, num_batches = 0, len(shape_loader)
    if args.label_smoothing > 0:
        nll_classification_loss = LabelSmoothingLoss(smoothing=0.3)
    else:
        nll_classification_loss = torch.nn.NLLLoss()
    full_loss, full_main_loss, full_shape_loss = torch.tensor([], device=device), torch.tensor([], device=device), \
                                                 torch.tensor([], device=device)
    # torch.autograd.set_detect_anomaly(True)  # FOR DEBUGGING AND DETECTING ERROR IN AUTOGRAD AND BACKPROPAGATION
    for batch in shape_loader:
        curr_batch += 1
        print(f'\r Train Main+Shape - Batch: {curr_batch}/{num_batches}', end='')
        (obs_traj, pred_traj_gt, obs_pp, pred_pp, pred_pp_rel, pool_data, occ_data, seq_start_end, shape_seq_start_end,
         shape_list) = batch
        pred_len = pred_traj_gt.shape[0]
        # get the "ground truth" data, by passing the trajectories via LSTM with ALL shape pooling values derived from
        # the several possible values for shapes
        # THIS part can result in an ERROR if the shape pooling data is loaded from file and its characteristics don't
        # match the data being used in some aspect (different trajectories, different options for shapes, etc.)
        # the interaction module will only embed this data
        gt_generator.interaction_module = SimulatedInteractionModule(obs_pp.clone(), pred_pp.clone(),
                                                                     shape_seq_start_end, pool_data,
                                                                     occ_data, interaction_module)
        with torch.no_grad():
            gt_shape_trajs = gt_generator(obs_pp, pred_traj_gt=pred_pp.clone(), seq_start_end=shape_seq_start_end,
                                          full_teacher_forcing=True, return_obs=True)
            # The predictions for the observed trajectory are also returned. This is to also pick what is the best
            # shape for those pedestrians and feed it as input data to the model. The goal is to assist it in training
            obs_portion_used = gt_shape_trajs.shape[0] - pred_pp.shape[0]
            new_obs = obs_pp[:-obs_portion_used]
            # portion of observed trajectory that simulates predictions
            obs_becoming_pred = obs_pp[-obs_portion_used:]
            obs_becoming_pred_rel = torch.zeros_like(obs_becoming_pred)
            obs_becoming_pred_rel[0] = obs_becoming_pred[0] - new_obs[-1]
            obs_becoming_pred_rel[1:] = obs_becoming_pred[1:] - obs_becoming_pred[:-1]
            new_pred, new_pred_rel = torch.cat((obs_becoming_pred, pred_pp)), \
                                     torch.cat((obs_becoming_pred_rel, pred_pp_rel))
            # apply the training loss using a specified call (coming from a parent training script, for instance)
            loss_gt = loss_call(args, main_loss_fn, gt_shape_trajs, new_pred, new_pred_rel,
                                torch.ones_like(new_pred[:, :, 0]), new_obs[-1], new_obs[-1] - new_obs[-2],
                                mode='log_raw_or_plain_raw')
            idx_probs_all = torch.tensor([], device=loss_gt.device)
            best_idx = torch.tensor([], device=loss_gt.device)
            for i, (start, end) in enumerate(shape_seq_start_end):
                loss_portion = loss_gt[:, start:end]
                min_idx = torch.argmin(loss_portion, dim=1)
                # best shape is THE SAME for all pedestrians in the same sequence (primary and neighbours)
                best_idx_curr = min_idx.unsqueeze(1).repeat(1, seq_start_end[i, 1] - seq_start_end[i, 0])
                best_idx = torch.cat((best_idx, best_idx_curr), dim=1)
                # single value equal to one, which corresponds to the index of the shape in which the loss is smaller
                # since the values used in the model go through a log softmax, values should be in [-inf, 0) range
                idx_probs_portion = torch.full_like(loss_portion, float('-inf'))
                idx_probs_portion[torch.arange(start=0, end=min_idx.shape[0], step=1), min_idx] = 0
                idx_probs_all = torch.cat((idx_probs_all, idx_probs_portion), dim=1)
                # get the best shape idx data to feed to the model as input, and the gt data is separated
            obs_idx = best_idx[:-pred_len]
            diff_idx = best_idx.shape[0] - pred_len - obs_traj.shape[0]
            if diff_idx > 0:
                # there are too many times steps returned as ground truth data, which should not happen
                obs_idx = obs_idx[diff_idx:]
            elif diff_idx < 0:
                # there are too few time steps. This should happen because the model does not return "ground truth"
                # for best shape data in the initial time steps of the observed data. So we will say that those
                # instants will have the best shape equal to the best shape of the first available instant
                num_repeats = - diff_idx
                obs_idx = torch.cat((obs_idx[0:1].repeat(num_repeats, 1), obs_idx), dim=0)
            idx_probs = idx_probs_all[-pred_len:]  # this data becomes only relative to the predicted trajectory
            idx_in = torch.cat((obs_idx, best_idx[-pred_len:])) if args.teacher_forcing else obs_idx
        optimizer.zero_grad()
        # perform prediction with the original model and standard trajectory batch
        if args.teacher_forcing:
            pred_seq = model(obs_traj, pred_traj_gt=pred_traj_gt.clone(),
                             seq_start_end=seq_start_end, idx_shape_data=idx_in)
        else:
            pred_seq = model(obs_traj, pred_len=pred_traj_gt.shape[0],
                             seq_start_end=seq_start_end, idx_shape_data=idx_in)
        probs_pp_s, probs_pp_r, probs_pp_a = model.shape_config.probs_pp_s, model.shape_config.probs_pp_r, \
                                             model.shape_config.probs_pp_a
        if args.pooling_shape == 'grid':
            raise Exception('NOT IMPLEMENTED YET')
        else:
            # shape == arc
            r_probs, a_probs = probs_pp_r[-pred_len:], probs_pp_a[-pred_len:]
            n_r, n_a = r_probs.shape[-1], a_probs.shape[-1]
            n_probs = r_probs.shape[0] * r_probs.shape[1]  # = a_probs.shape[0] * a_probs.shape[1]
            gt_r_probs, gt_a_probs = torch.tensor([], device=loss_gt.device), torch.tensor([], device=loss_gt.device)
            # GT TARGETS FOR NLL LOSS
            all_r_idx, all_a_idx = torch.tensor([], device=loss_gt.device, dtype=torch.long), \
                                   torch.tensor([], device=loss_gt.device, dtype=torch.long)
            for (start, end) in shape_seq_start_end:
                idx_portion = idx_probs[:, start:end]
                max_idx = torch.argmax(idx_portion, dim=1)
                # assuming values start first in the same radius and go through all possible angles - see shape_data.py
                r_idx, a_idx = (max_idx / n_a).to(torch.long), (max_idx % n_a).to(torch.long)
                all_r_idx, all_a_idx = torch.cat((all_r_idx, r_idx.unsqueeze(1)), dim=1), \
                                       torch.cat((all_a_idx, a_idx.unsqueeze(1)), dim=1)
                gt_r_probs_portion = torch.full((r_idx.shape[0], n_r), float('-inf'), device=max_idx.device)
                gt_a_probs_portion = torch.full((a_idx.shape[0], n_a), float('-inf'), device=max_idx.device)
                gt_r_probs_portion[torch.arange(start=0, end=r_idx.shape[0], step=1), r_idx] = 0
                gt_a_probs_portion[torch.arange(start=0, end=a_idx.shape[0], step=1), a_idx] = 0
                gt_r_probs = torch.cat((gt_r_probs, gt_r_probs_portion.unsqueeze(1)), dim=1)
                gt_a_probs = torch.cat((gt_a_probs, gt_a_probs_portion.unsqueeze(1)), dim=1)
            batch_r_loss = nll_classification_loss(r_probs.view(n_probs, -1), all_r_idx.view(n_probs))
            batch_a_loss = nll_classification_loss(a_probs.view(n_probs, -1), all_a_idx.view(n_probs))
            batch_shape_loss = batch_r_loss + batch_a_loss
        # loss with predicted trajectory only computed for primary pedestrians
        seq_mask = torch.zeros(obs_traj.shape[1], dtype=torch.bool, device=obs_traj.device)
        seq_mask[seq_start_end[:, 0]] = True
        # apply mask to discard trajectories that are not belonging to primary pedestrian
        pred_seq, pred_traj_gt = pred_seq[:, seq_mask, :], pred_traj_gt[:, seq_mask, :]
        pred_traj_rel_gt = torch.zeros_like(pred_traj_gt)
        pred_traj_rel_gt[0] = pred_traj_gt[0] - obs_traj[-1, seq_mask]
        pred_traj_rel_gt[1:] = pred_traj_gt[1:] - pred_traj_gt[:-1]
        batch_main_loss = loss_call(args, main_loss_fn, pred_seq, pred_traj_gt, pred_traj_rel_gt,
                                    torch.ones_like(pred_traj_rel_gt[:, :, 0]), obs_traj[-1, seq_mask],
                                    obs_traj[-1, seq_mask] - obs_traj[-2, seq_mask])
        # mean of the loss (divide by number of pedestrians and number of prediction instants)
        full_main_loss = torch.cat((full_main_loss, batch_main_loss.detach().unsqueeze(0)))
        full_shape_loss = torch.cat((full_shape_loss, batch_shape_loss.detach().unsqueeze(0)))
        # full_shape_loss = torch.cat((full_shape_loss, batch_shape_loss.detach().unsqueeze(0)))
        # loss = torch.mean(batch_main_loss)  # should yield mean, but it's to force this
        loss = torch.mean(batch_shape_loss) + torch.mean(batch_main_loss)  # should yield mean, but it's to force this
        loss.backward()
        optimizer.step()
    print(f'\r', end='')
    return torch.mean(full_main_loss) + torch.mean(full_shape_loss), torch.mean(full_main_loss), \
           torch.mean(full_shape_loss)


def create_gt_generator(model):
    """
    Create a "gt generator". This will be a copy of the model, minus the shape configuration. It will be fixed
    throughout one epoch, and will generate the ground truth to train the shape configuration model. This ground truth
    will be the pooling shape dimensions that yielded the lowest loss - simulating what could be the real shape
    dimension of the pedestrian.
    May also disable gradient computation for parts of the original model (that are not relevant from a point of view
    of training).
    :param model: the original model
    :return: the gt generator, and the interaction model
    """
    # gt_model a model that will be used to generate "predictions" that will be simulated as ground truth data
    shape_config = model.shape_config
    model.shape_config = None  # deepcopy might not work without this line, due to there being some extra tensors
    gt_generator = copy.deepcopy(model)
    gt_generator.eval()
    gt_generator.shape_config = None
    # disable all gradients - except for the ones associated to the shape configuration network
    model.shape_config = shape_config
    for param in gt_generator.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = True
    # because the actual pooling data will come from a dataset, and not computed by the module, the goal of this
    # interaction module is just to embed the data, and send it as direct input to the LSTM cell
    interaction_module = copy.deepcopy(gt_generator.interaction_module)
    return gt_generator, interaction_module
