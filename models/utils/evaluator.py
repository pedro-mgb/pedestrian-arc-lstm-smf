"""
Created on March 18th 2021
"""
import enum
import os

import torch
import torch.distributions.multivariate_normal as multivariate_normal
import numpy as np

from models.losses_and_metrics import final_displacement_error, displacement_error, num_collisions, \
    num_collisions_between_two, compute_kde_nll
from models.utils.utils import relative_traj_to_abs, normalize_sequences


class EvaluationResults:
    """
    Class that stores evaluation results (computed here). May also optionally store some statistics
    Can be used to retrieve the summary of the results
    """

    def __init__(self, device, args, environment=None):
        """
        initialize parameters for evaluation results.
        - ade_per_batch and fde_per_batch contain the main displacement errors (average and final) used for evaluation
        per batch
        - num_peds_per_batch contains the number of pedestrians in each bach
        - pred_traj_len_per_batch contains the prediction length per batch (assumed unique for each batch)
        - obs_traj_len_per_batch contains the observed trajectory length per batch (assumed unique for each batch)
        num_peds_per_batch and pred_traj_len_per_batch are used to perform average and weighted average of the metrics
        :param device: torch.device to map the tensors to
        :param args: command line arguments to configure this evaluator
        :param environment: Environment object containing static obstacles, with which one can evaluate the existence
        of collisions with environment; by default is none, in the case no environment is used.
        """
        self.device = device
        # if meant to use extra data besides standard evaluation with ADE and FDE
        use_statistics = args.statistics or args.social_metrics or environment is not None or args.kde_nll
        self.statistics = StatsResults(device, args, environment) \
            if use_statistics else None

        # for errors computation (will accumulate if batch size is below desired)
        self.ade_per_batch = torch.tensor([], device=self.device)
        self.fde_per_batch = torch.tensor([], device=self.device)
        # how much the errors contribute per batch
        self.num_peds_per_batch = torch.tensor([], device=self.device)
        self.pred_traj_len_per_batch = torch.tensor([], device=self.device)
        self.obs_traj_len_per_batch = torch.tensor([], device=self.device)

        # cached values
        self.ade, self.ade_no_len, self.fde, self.contribution_ade, self.contribution_fde = None, None, None, None, None

    def update(self, ade_raw, fde_raw, num_peds, pred_traj_len, obs_traj_len, pred_traj_rel_gt, pred_seq_rel, pred_seq,
               pred_traj_gt, kde_nll, seq_start_end, metadata):
        """
        update the evaluation results with some tensors (e.g. from evaluation of a batch)
        :param ade_raw: the ADE, in mean values
        :param fde_raw: the FDE, in mean values
        :param num_peds: the number of pedestrians (or of trajectories) relating to this update
        :param pred_traj_len: length of the predicted trajectories
        :param obs_traj_len: length of the observed trajectories
        :param pred_traj_rel_gt: the ground truth relative displacements (or velocities)
        :param pred_seq_rel: the predicted relative displacements (or velocities)
        :param pred_seq: the predicted trajectories, in absolute coordinates. Should include the last observed position,
        for the sake of computing collisions.
        :param pred_traj_gt: the real ground truth trajectories in absolute coordinates. Used to compute metrics along
        with the model predictions, like percentage of collisions
        :param kde_nll: Multimodal metric for Kernel Density Estimate Negative Log Likelihood
        :param seq_start_end: delimits beginning and end of pedestrians that are in the same time frame
        :param metadata: list of length num_seqs. Contains extra metadata regarding each sequence, that may be useful
        for the model's prediction
        :return: nothing
        """
        ade = torch.sum(ade_raw) / num_peds
        fde = torch.sum(fde_raw) / num_peds
        self.ade_per_batch = torch.cat((self.ade_per_batch, ade.view(1).to(self.device)), dim=0)
        self.fde_per_batch = torch.cat((self.fde_per_batch, fde.view(1).to(self.device)), dim=0)
        self.num_peds_per_batch = torch.cat((self.num_peds_per_batch, torch.tensor([num_peds], dtype=torch.float,
                                                                                   device=self.device).view(1)), dim=0)
        self.pred_traj_len_per_batch = torch.cat((self.pred_traj_len_per_batch,
                                                  torch.tensor([pred_traj_len], dtype=torch.float,
                                                               device=self.device).view(1)), dim=0)
        self.obs_traj_len_per_batch = torch.cat((self.obs_traj_len_per_batch,
                                                 torch.tensor([obs_traj_len], dtype=torch.float,
                                                              device=self.device).view(1)), dim=0)
        if self.statistics is not None:
            self.statistics.update(ade_raw, fde_raw, num_peds, pred_traj_len, obs_traj_len, pred_seq_rel,
                                   pred_traj_rel_gt, pred_seq, pred_traj_gt, kde_nll, seq_start_end, metadata)

    def get(self, refresh=False):
        """
        retrieve the summary of the evaluation results.
        :param refresh: if True, compute results to send from scratch; if False, retrieve cached values (if available)
        :return: several values:
        - ade: the Average Displacement Error, from the whole data present (weighted mean based on trajectory length,
        single value)
        - ade_no_len: the Average Displacement Error, from the whole data present (mean, dividing by the total number
        of trajectories (not trajectory length, single value).
        - fde: the Final Displacement Error, from the whole data present (mean, single value)
        - contribution_ade, contribution_fde: the contribution for each of the metrics (the fde one is used for
        ade_no_len); they are used for when performing weighted averages of several EvaluationResults objects
        """
        if not refresh and self.ade is not None:
            # get cached values
            return self.ade, self.ade_no_len, self.fde, self.contribution_ade, self.contribution_fde, self.statistics
        # average of the errors with respect to number of pedestrians (or number of trajectories)
        # for ADE, the length of the trajectories is also taken into account
        self.ade = torch.sum(torch.dot(self.ade_per_batch,
                                       (self.num_peds_per_batch * self.pred_traj_len_per_batch) / torch.dot(
                                           self.num_peds_per_batch, self.pred_traj_len_per_batch))).data.cpu().numpy()
        self.contribution_ade = torch.dot(self.num_peds_per_batch, self.pred_traj_len_per_batch).data.cpu().numpy()
        # considering all ADE's with the same contribution, independent of trajectory length
        self.ade_no_len = torch.sum(
            torch.dot(self.ade_per_batch,
                      self.num_peds_per_batch / torch.sum(self.num_peds_per_batch))).data.cpu().numpy()
        self.fde = torch.sum(
            torch.dot(self.fde_per_batch,
                      self.num_peds_per_batch / torch.sum(self.num_peds_per_batch))).data.cpu().numpy()
        # this contribution is the same that is used for 'ade_file_no_len'
        self.contribution_fde = torch.sum(self.num_peds_per_batch).data.cpu().numpy()
        return self.ade, self.ade_no_len, self.fde, self.contribution_ade, self.contribution_fde, self.statistics


class StatsResults:
    """
    Stores statistics, associated to the evaluation results
    """

    def __init__(self, device, args=None, environment=None):
        """
        initialize parameters for evaluation results.
        - ade_all and fde_all contain a full list of errors
        - velocities_all and velocities_gt_all contains list of predicted and ground truth velocities, respectively
        - obs_len_all and pred_len_all contains list of observed and predicted trajectory lengths, respectively
        - obs_traj_len_per_batch contains the observed trajectory length per batch (assumed unique for each batch)
        :param device: torch.device to map the tensors to
        :param args: command line arguments to configure this evaluator
        :param environment: Environment object containing static obstacles, with which one can evaluate the existence
        of collisions with environment; by default is none, in the case no environment is used.
        """
        self.device = device
        self.trajnetpp = not args.fixed_len and not args.variable_len
        self.social = args.social_metrics
        self.environment = environment
        self.environment_for_neighbours = args.static_collisions_neighbours  # only makes difference for Trajnet++
        self.col_t, self.inter_pts = args.collision_threshold, args.num_inter_pts_cols
        # used only when statistics is set to true
        self.ade_all = torch.tensor([], device=self.device)
        self.fde_all = torch.tensor([], device=self.device)
        self.kde_nll_all = torch.tensor([], device=self.device)
        self.velocities_gt_all = torch.tensor([], device=self.device)
        self.velocities_all = torch.tensor([], device=self.device)
        self.obs_len_all = torch.tensor([], device=self.device)
        self.pred_len_all = torch.tensor([], device=self.device)
        # social collisions
        self.colliding_peds_pred = torch.tensor([], device=self.device, dtype=torch.long)
        self.colliding_peds_gt = torch.tensor([], device=self.device, dtype=torch.long)
        # collisions with scene environment (obstacles, unwalkable regions)
        self.cse = torch.tensor([], device=self.device, dtype=torch.long)
        # trajectories out of scene bounds
        self.osb = torch.tensor([], device=self.device, dtype=torch.long)

    def __bool__(self):
        """
        :return: True if this object has content (i.e. any of the tensors has content). False otherwise
        """
        return self.ade_all.nelement() > 0 or self.fde_all.nelement() > 0 or self.velocities_gt_all.nelement() > 0 or \
               self.velocities_all.nelement() > 0 or self.obs_len_all.nelement() > 0 or self.pred_len_all.nelement() > 0

    def update(self, ade_raw, fde_raw, num_peds, pred_traj_len, obs_traj_len, pred_traj_rel_gt, pred_seq_rel, pred_seq,
               pred_traj_gt, kde_nll, seq_start_end, metadata):
        """
        update the statistics
        :param ade_raw: the ADE, in mean values
        :param fde_raw: the FDE, in mean values
        :param num_peds: the number of pedestrians (or of trajectories) relating to this update
        :param pred_traj_len: length of the predicted trajectories
        :param obs_traj_len: length of the observed trajectories
        :param pred_traj_rel_gt: the ground truth relative displacements (or velocities)
        :param pred_seq_rel: the predicted relative displacements (or velocities)
        :param pred_seq: the predicted trajectories, in absolute coordinates. Should include the last observed position,
        for the sake of computing collisions.
        :param pred_traj_gt: the real ground truth trajectories in absolute coordinates. Used to compute metrics along
        with the model predictions, like percentage of collisions
        :param kde_nll: Multimodal metric for Kernel Density Estimate Negative Log Likelihood
        :param seq_start_end: delimits beginning and end of pedestrians that are in the same time frame
        :param metadata: list of length num_seqs. Contains extra metadata regarding each sequence, that may be useful
        for the model's prediction
        :return: nothing
        """
        self.ade_all = torch.cat((self.ade_all, ade_raw))
        self.fde_all = torch.cat((self.fde_all, fde_raw))
        avg_velocities_gt = torch.mean(torch.norm(pred_traj_rel_gt[:, seq_start_end[:, 0]] if self.trajnetpp else
                                                  pred_traj_rel_gt, dim=2), dim=0)
        self.velocities_gt_all = torch.cat((self.velocities_gt_all, avg_velocities_gt))
        avg_velocities = torch.mean(torch.norm(pred_seq_rel[:, seq_start_end[:, 0]] if self.trajnetpp else
                                               pred_seq_rel, dim=2), dim=0)
        self.velocities_all = torch.cat((self.velocities_all, avg_velocities))
        self.obs_len_all = torch.cat((self.obs_len_all, torch.tensor([obs_traj_len], dtype=torch.float,
                                                                     device=self.device).repeat(num_peds)))
        self.pred_len_all = torch.cat((self.pred_len_all, torch.tensor([pred_traj_len], dtype=torch.float,
                                                                       device=self.device).repeat(num_peds)))
        pred_seq_real = normalize_sequences(pred_seq, seq_start_end, metadata_list=metadata, inverse=True)
        if kde_nll is not None:
            self.kde_nll_all = torch.cat((self.kde_nll_all, kde_nll))
        if self.social:
            for (start, end) in seq_start_end.cpu().numpy():
                collisions_pred = num_collisions(pred_seq[:, start:end, :], self.col_t, self.inter_pts, mode='raw')
                collisions_gt = num_collisions_between_two(pred_seq[:, start:end, :], pred_traj_gt[:, start:end, :],
                                                           self.col_t, self.inter_pts, mode='raw')
                if self.trajnetpp:
                    # only use the collision statistic for primary pedestrian - first in seq_start_end
                    collisions_pred = collisions_pred[0:1]
                    collisions_gt = collisions_gt[0:1]
                self.colliding_peds_pred = torch.cat((self.colliding_peds_pred, collisions_pred))
                self.colliding_peds_gt = torch.cat((self.colliding_peds_gt, collisions_gt))
        if self.environment:
            # the predicted sequence may be direction-normalized - de-normalize
            if self.trajnetpp:
                if not self.environment_for_neighbours:
                    # only for primary pedestrians
                    cse, osb = self.environment.compute_collisions(pred_seq_real[:, seq_start_end[:, 0], :])
                else:
                    cse, osb = self.environment.compute_collisions(pred_seq_real)
            else:
                # all trajectories
                cse, osb = self.environment.compute_collisions(pred_seq_real)
            if cse is not None:  # can be None if scene has no obstacles
                self.cse = torch.cat((self.cse, cse))
            if osb is not None:  # can be None if scene has no bounds
                self.osb = torch.cat((self.osb, osb))

    def update_from_existing(self, ades, fdes, velocities_gt, velocities, obs_lens, pred_lens, collisions_pred,
                             collisions_gt, collisions_scene_environment, out_of_scene_bounds, kde_nll):
        """
        update the statistics from existing statistics values
        :param ades: a list of average displacement errors
        :param fdes: a list of final displacement errors
        :param velocities_gt: a list of ground truth velocities
        :param velocities: a list of predicted velocities
        :param obs_lens: a list of observed trajectory lengths
        :param pred_lens: a list of predicted trajectory lengths
        :param collisions_scene_environment: a list with number of collisions with scene environment, which includes
        obstacles and unwalkable regions (1 value per pedestrian)
        :param out_of_scene_bounds: a list with number of predictions going out of scene bounds (1 value per pedestrian)
        :param collisions_pred: (social) collisions between predicted trajectories of pedestrians
        :param collisions_gt: (social) collisions between predicted trajectories of pedestrians and ground truth
        :param kde_nll: a list of Kernel Density Estimate Negative Log Likelihood values
        :return: nothing
        """
        self.ade_all = torch.cat((self.ade_all, ades))
        self.fde_all = torch.cat((self.fde_all, fdes))
        self.kde_nll_all = torch.cat((self.kde_nll_all, kde_nll))
        self.velocities_gt_all = torch.cat((self.velocities_gt_all, velocities_gt))
        self.velocities_all = torch.cat((self.velocities_all, velocities))
        self.obs_len_all = torch.cat((self.obs_len_all, obs_lens))
        self.pred_len_all = torch.cat((self.pred_len_all, pred_lens))
        self.colliding_peds_pred = torch.cat((self.colliding_peds_pred, collisions_pred))
        self.colliding_peds_gt = torch.cat((self.colliding_peds_gt, collisions_gt))
        self.cse = torch.cat((self.cse, collisions_scene_environment))
        self.osb = torch.cat((self.osb, out_of_scene_bounds))

    def update_from_existing_stats(self, _statistics):
        """
        update statistics from existing StatsResults object
        :param _statistics: the StatsResults object
        :return: nothing
        """
        if not isinstance(_statistics, StatsResults) or not _statistics:
            return
        ades, fdes, velocities_gt, velocities, obs_lens, pred_lens, colliding_peds_pred, colliding_peds_gt, \
        cse, osb, kde_nll = _statistics.get()
        self.update_from_existing(ades, fdes, velocities_gt, velocities, obs_lens, pred_lens, colliding_peds_pred,
                                  colliding_peds_gt, cse, osb, kde_nll)

    def get(self):
        """
        get the statistics
        :return: the lists; see __init__ method of this class for what each one is
        """
        return self.ade_all, self.fde_all, self.velocities_gt_all, self.velocities_all, self.obs_len_all, \
               self.pred_len_all, self.colliding_peds_pred, self.colliding_peds_gt, self.cse, self.osb, \
               self.kde_nll_all


def map_traj_type(use_abs, use_acc):
    """
    map the trajectory type (see TrajectoryType class) given some options
    :param use_abs: if there is use of absolute positions
    :param use_acc: if there is use of accelerations (or difference between relative displacements)
    :return: the object of type TrajectoryType
    """
    if use_abs:
        return TrajectoryType.ABS
    elif use_acc:
        return TrajectoryType.ACC
    return TrajectoryType.VEL


class TrajectoryType(enum.Enum):
    """
    Type of trajectory. Can be one of the following:
    - ABS: Absolute positions
    - VEL: Velocity, or relative displacements
    - ACC: Acceleration, or differences between the relative displacements
    """
    ABS = 1
    VEL = 2
    ACC = 3


class MultimodalityType(enum.Enum):
    """
    Type of the Multimodality being employed by the model. Regulates the computation of multimodal metrics. Can be:
    - DISTRIBUTION_ONE_CALL: Model returns a probabilistic distribution for the predicted trajectories of pedestrians
    (e.g. s bivariate gaussian), and a single model call is done to get this distribution, from which samples are taken
    - NO_DISTRIBUTION_ONE_CALL: Model does not return a distribution. Instead, it returns samples. Only one call is
    necessary to return the --num_samples trajectory prediction samples.
    - NO_DISTRIBUTION_MULTIPLE_CALLS: Model does not return a distribution. Instead, it returns samples. Each sample
    requires a new call. So, --num_samples model calls are required for --num_samples trajectory prediction samples.
    """
    DISTRIBUTION_ONE_CALL = 0
    NO_DISTRIBUTION_ONE_CALL = 1
    NO_DISTRIBUTION_MULTIPLE_CALLS = 2


def compute_metrics(args, model, loader, input_type, output_type, device, multimodal_type, environment=None,
                    model_fun=None):
    """
    Compute the evaluation metrics, doing a forward pass on the model with the test data
    :param args: command line arguments containing several parameters regarding evaluation
    :param model: the model, that receives an observed trajectory, and prediction length, and outputs a predicted
    trajectory, or distribution (if num_samples > 1)
    :param loader: the loader for the test data
    :param input_type: The type of trajectory that the model should receive
    :param output_type: The type of trajectory that the model outputs (does not account for the fact of being
    distribution or not)
    :param device: torch.device to map the tensors to
    :param multimodal_type: type of multimodality to follow. See class MultimodalType for more information
    :param environment: Class of type Environment, containing static obstacles and unwalkable area limits
    :param model_fun: function that decides how the model forward pass is done, to perform prediction. This function
    will receive several things (although the caller does not have to use everything) like: the actual model, the
    observed sequence (may be absolute, relative coordinates, depends on input_type), the prediction length,
    seq_start_end which is specially useful for methods that consider multiple pedestrians in the same time frame,
    possibly interacting; also the number of samples and current sample number for the case of multimodality.
    If no model fun is supplied, then a standard call is made, with only observed trajectory and prediction length.
    :return: the evaluation results, object of class EvalResults
    """
    eval_results = EvaluationResults(device, args, environment)
    for batch in loader:
        if args.use_gpu and not args.map_gpu_beginning:
            # map tensors in batch to the desired device
            batch = [tensor.to(device) if torch.is_tensor(tensor) else tensor for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt, metadata, loss_mask, seq_start_end) = batch
        num_peds = loss_mask.size(0)
        obs_traj_len = obs_traj.shape[0]
        pred_traj_len = pred_traj_gt.shape[0]
        kde_nll = None
        if args.num_samples > 1 and multimodal_type != MultimodalityType.DISTRIBUTION_ONE_CALL:
            # this use case is regarding computation for when a model is not outputting parameters of a
            # probabilistic distribution, but instead returns samples (all at once or one at a time)
            multiple_model_calls = multimodal_type != MultimodalityType.NO_DISTRIBUTION_ONE_CALL
            pred_seq, ade_raw, fde_raw, kde_nll = \
                __batch_metrics_multimodal_no_distribution__(args, model, input_type, output_type, obs_traj,
                                                             pred_traj_gt, pred_traj_len, obs_traj_rel, seq_start_end,
                                                             metadata, model_fun, multiple_model_calls)
        else:
            obs_traj_seq = obs_traj if input_type == TrajectoryType.ABS else obs_traj_rel
            # model forward pass - and compute metrics
            pred_seq_full = __model_forward_pass__(args, model, obs_traj_seq, pred_traj_len, seq_start_end, metadata,
                                                   0, model_fun)
            if args.num_samples > 1:
                pred_seq, ade_raw, fde_raw, kde_nll = \
                    __batch_metrics_multimodal__(args, pred_seq_full, output_type, obs_traj, pred_traj_gt,
                                                 pred_traj_len, obs_traj_rel, seq_start_end)
            else:
                pred_seq, ade_raw, fde_raw = __batch_metrics_unimodal__(args, pred_seq_full, output_type, obs_traj,
                                                                        pred_traj_gt, pred_traj_len, obs_traj_rel,
                                                                        seq_start_end)
        if not args.fixed_len and not args.variable_len:
            # for trajnetpp, only primary pedestrian counts
            num_peds = seq_start_end.shape[0]
        eval_results. \
            update(ade_raw, fde_raw, num_peds, pred_traj_len, obs_traj_len, pred_traj_rel_gt,
                   pred_seq[1:, :, :] - pred_seq[:-1, :, :] if pred_traj_len > 1 else torch.zeros_like(pred_seq),
                   torch.cat((obs_traj[-1].unsqueeze(0), pred_seq), dim=0),
                   torch.cat((obs_traj[-1].unsqueeze(0), pred_traj_gt), dim=0), kde_nll, seq_start_end, metadata)
    return eval_results


def __batch_metrics_unimodal__(args, pred_seq_orig, output_type, obs_traj, pred_traj_gt, pred_traj_len, obs_traj_rel,
                               seq_start_end):
    """
    compute metrics on a batch, for unimodal format (i.e. one sample per trajectory)
    :param pred_seq_orig: Tensor of shape (obs_len, batch, 2); last dimension can be more than 2, but only the first two
    are used, since this expects the prediction to be of a 2D trajectory.
        The original predicted sequence from the model (may suffer changes before it is used to compute the metrics);
    :param output_type: The type of trajectory that the model outputs (does not account for the fact of being
    distribution or not)
    :param obs_traj: Tensor of shape (obs_traj_len, batch, 2). The observed trajectory
    :param pred_traj_gt: Tensor of shape (pred_traj_len, batch, 2). The ground truth future trajectory
    :param pred_traj_len: length of each trajectory in pred_traj_gt
    :param obs_traj_rel: Tensor of shape (obs_traj_len, batch, 2). The relative displacements (or velocities)
    :return: 3 tensors:
    - pred_seq: Tensor of shape (pred_traj_len, batch, 2). The predicted sequence - that may be different than the one
    sent as input to this method
    - ade_raw, fde_raw: Tensors, each of shape (batch). Raw average and final displacement errors, per trajectory.
    """
    trajnetpp = not args.fixed_len and not args.variable_len  # Trajnet++ data being used
    pred_seq = pred_seq_orig[:, :, :2]
    if not output_type == TrajectoryType.ABS:
        if output_type == TrajectoryType.ACC:
            pred_seq = relative_traj_to_abs(pred_seq, obs_traj_rel[-1])
        pred_seq = relative_traj_to_abs(pred_seq, obs_traj[-1])
    # without aggregating the batches
    # note if it's in Trajnet++ configuration, will only compute metrics for primary pedestrian
    pred_seq_for_metrics = pred_seq[:, seq_start_end[:, 0], :] if trajnetpp else pred_seq
    seq_gt_for_metrics = pred_traj_gt[:, seq_start_end[:, 0], :] if trajnetpp else pred_traj_gt
    ade_raw = displacement_error(pred_seq_for_metrics, seq_gt_for_metrics, mode='raw') / pred_traj_len
    fde_raw = final_displacement_error(pred_seq_for_metrics[-1], seq_gt_for_metrics[-1], mode='raw')
    return pred_seq, ade_raw, fde_raw


def __batch_metrics_multimodal__(args, pred_seq_orig, output_type, obs_traj, pred_traj_gt, pred_traj_len,
                                 obs_traj_rel, seq_start_end):
    """
    compute metrics on a batch, for multimodal format (i.e. several samples per trajectory)
    :param args: command line arguments that contain several parameters to configure the evaluation
    :param pred_seq_orig: Tensor of shape (obs_len, batch, 5); the last dimension is 5 because this expects the
    prediction to be the parameters of a bi-variate gaussian distribution (2D mean and standard deviation, and also a
    correlation factor), so that one can sample several times from the distribution.
        The original predicted sequence from the model (may suffer changes before it is used to compute the metrics);
    :param output_type: The type of trajectory that the model outputs (does not account for the fact of being
    distribution or not)
    :param obs_traj: Tensor of shape (obs_traj_len, batch, 2). The observed trajectory
    :param pred_traj_gt: Tensor of shape (pred_traj_len, batch, 2). The ground truth future trajectory
    :param pred_traj_len: length of each trajectory in pred_traj_gt
    :param obs_traj_rel: Tensor of shape (obs_traj_len, batch, 2). The relative displacements (or velocities)
    :return: 3 tensors:
    - pred_seq: Tensor of shape (pred_traj_len, batch, 2). The predicted sequence - that may be different than the one
    sent as input to this method
    - ade_raw, fde_raw: Tensors, each of shape (batch). Raw average and final displacement errors, per trajectory.
    """
    pred_seq_list = torch.tensor([], device=pred_seq_orig.device)
    ade_list = torch.tensor([], device=pred_seq_orig.device)
    fde_list = torch.tensor([], device=pred_seq_orig.device)
    # create the distribution to sample values from
    #   CREDITS to https://github.com/pedro-mgb/Social-STGCNN/blob/master/test.py
    mean = pred_seq_orig[:, :, :2]
    sx = torch.exp(pred_seq_orig[:, :, 2])
    sy = torch.exp(pred_seq_orig[:, :, 3])
    corr_sx_sy = torch.tanh(pred_seq_orig[:, :, 4]) * sx * sy
    covariance = torch.zeros(pred_seq_orig.shape[0], pred_seq_orig.shape[1], 2, 2, device=pred_seq_orig.device)
    covariance[:, :, 0, 0] = sx * sx
    covariance[:, :, 0, 1] = corr_sx_sy
    covariance[:, :, 1, 0] = corr_sx_sy
    covariance[:, :, 1, 1] = sy * sy
    seq_mask = ~torch.isnan(mean[0, :, 0])
    mv_normal = multivariate_normal.MultivariateNormal(mean[:, seq_mask], covariance[:, seq_mask])
    for _ in range(args.num_samples):
        pred_seq_sample = torch.full_like(mean, float('nan'))
        pred_seq_sample[:, seq_mask] = mv_normal.sample()
        pred_seq, ade_raw, fde_raw = __batch_metrics_unimodal__(args, pred_seq_sample, output_type, obs_traj,
                                                                pred_traj_gt,
                                                                pred_traj_len, obs_traj_rel, seq_start_end)
        ade_list = torch.cat((ade_list, ade_raw))
        fde_list = torch.cat((fde_list, fde_raw))
        pred_seq_list = torch.cat((pred_seq_list, pred_seq))
    return compute_multimodal_metrics(ade_list, fde_list, pred_seq_list, pred_traj_gt, seq_start_end, args)


def __batch_metrics_multimodal_no_distribution__(args, model, input_type, output_type, obs_traj, pred_traj_gt,
                                                 pred_traj_len, obs_traj_rel, seq_start_end, metadata, model_fun,
                                                 multiple_model_calls=True):
    """
    compute metrics on a batch, for multimodal format (i.e. several samples per trajectory); this is different from
    __batch_metrics_multimodal__ method because this assumes the model outputs a single sample, instead of a
    distribution (to have multimodality, several model forward passes need to be done)
    :param args: command line arguments that contain several parameters to configure the evaluation
    :param model: the model, that receives an observed trajectory, and prediction length, and outputs a predicted
    trajectory, or distribution (if num_samples > 1)
    :param input_type: The type of trajectory that the model should receive
    :param output_type: The type of trajectory that the model outputs (does not account for the fact of being
    distribution or not)
    :param obs_traj: Tensor of shape (obs_traj_len, batch, 2). The observed trajectory
    :param pred_traj_gt: Tensor of shape (pred_traj_len, batch, 2). The ground truth future trajectory
    :param pred_traj_len: length of each trajectory in pred_traj_gt
    :param obs_traj_rel: Tensor of shape (obs_traj_len, batch, 2). The relative displacements (or velocities)
    :param seq_start_end: Tensor of shape (2, num_seqs). Indicates which trajectories belong to a certain sequence
    (that belong to the same time frame)
    :param metadata: list of length num_seqs. Contains extra metadata regarding each sequence, that may be useful
    for the model's prediction
    :param model_fun: function that decides how the model forward pass is done, to perform prediction. This function
    will receive several things (although the caller does not have to use everything) like: the actual model, the
    observed sequence (may be absolute, relative coordinates, depends on input_type), the prediction length,
    seq_start_end which is specially useful for methods that consider multiple pedestrians in the same time frame,
    possibly interacting; also the number of samples and current sample number for the case of multimodality.
    If no model fun is supplied, then a standard call is made, with only observed trajectory and prediction length.
    :param multiple_model_calls: If True, will make args.num_samples calls to get each sample. Otherwise, it assumes a
    single call will give the args.num_samples trajectory samples.
    :return: 3 tensors:
    - best_trajectories: Tensor of shape (pred_traj_len, batch, 2). The best predicted trajectory (lowest ADE)
    - best_ade, best_fde: Tensors, each of shape (batch). Raw average and final displacement errors, per trajectory. The
    ADE is the lowest possible, the FDE is the one of the trajectory correspondent with that lowest ADE
    """
    pred_seq_list = torch.tensor([], device=obs_traj.device)
    ade_list = torch.tensor([], device=obs_traj.device)
    fde_list = torch.tensor([], device=obs_traj.device)
    obs_traj_seq = obs_traj if input_type == TrajectoryType.ABS else obs_traj_rel
    pred_seq_all_samples = __model_forward_pass__(args, model, obs_traj_seq, pred_traj_len, seq_start_end, metadata, 0,
                                                  model_fun) if not multiple_model_calls else None
    for s in range(args.num_samples):
        pred_seq_full = __model_forward_pass__(args, model, obs_traj_seq, pred_traj_len, seq_start_end, metadata, s,
                                               model_fun) if pred_seq_all_samples is None else pred_seq_all_samples[s]
        # the [:,:,:2] should not be necessary, more of a safety measure
        pred_seq, ade_raw, fde_raw = __batch_metrics_unimodal__(args, pred_seq_full[:, :, :2], output_type, obs_traj,
                                                                pred_traj_gt, pred_traj_len, obs_traj_rel,
                                                                seq_start_end)
        ade_list = torch.cat((ade_list, ade_raw))
        fde_list = torch.cat((fde_list, fde_raw))
        pred_seq_list = torch.cat((pred_seq_list, pred_seq))
    return compute_multimodal_metrics(ade_list, fde_list, pred_seq_list, pred_traj_gt, seq_start_end, args)


def __model_forward_pass__(args, model, obs_traj_seq, pred_traj_len, seq_start_end, metadata, curr_sample, model_fun):
    """
    Wrapper call to the model forward pass, which depending on the exact model, may require different kinds of input
    :param args: command line arguments that contain several parameters to configure the evaluation
    :param model: the model, that receives an observed trajectory, and prediction length, and outputs a predicted
    trajectory, or distribution (if num_samples > 1)
    :param obs_traj_seq: Tensor of shape (obs_traj_len, batch, 2). The observed trajectory (can be absolute positions,
    or relative displacements)
    :param pred_traj_len: length of each trajectory in pred_traj_gt
    :param seq_start_end: Tensor of shape (2, num_seqs). Indicates which trajectories belong to a certain sequence
    (that belong to the same time frame)
    :param metadata: list of length num_seqs. Contains extra metadata regarding each sequence, that may be useful
    for the model's prediction
    :param curr_sample: in the case of multimodal evaluation, the number of the current sample. Useful when models
    don't use the same prediction for each trajectory.
    :param model_fun: function that decides how the model forward pass is done, to perform prediction. This function
    will receive several things (although the caller does not have to use everything) like: the actual model, the
    observed sequence (may be absolute, relative coordinates, depends on input_type), the prediction length,
    seq_start_end which is specially useful for methods that consider multiple pedestrians in the same time frame,
    possibly interacting; also the number of samples and current sample number for the case of multimodality.
    If no model fun is supplied, then a standard call is made, with only observed trajectory and prediction length.
    :return: Tensor of shape (obs_traj_len, batch, output_shape). The predicted trajectories, via the model forward pass
    """
    if not model_fun:
        return model(obs_traj_seq, pred_traj_len)
    # else, there is a function to be called with additional arguments
    return model_fun(model, obs_traj_seq, pred_traj_len, seq_start_end, metadata, args.num_samples, curr_sample)


def compute_multimodal_metrics(_ade_list, _fde_list, _pred_seq_list, _pred_traj_gt, seq_start_end, args):
    """
    Evaluation helper for multi-modality. From the several samples , the trajectory with lowest ADE is chosen, and then
    the FDE is the one correspondent to that trajectory.
    :param _ade_list: Tensor of shape (num_samples * num_peds). List of average displacement errors, for all the samples
    :param _fde_list: Tensor of shape (num_samples * num_peds). List of final displacement errors, for all the samples
    :param _pred_seq_list: Tensor of shape (pred_traj_len * num_samples, num_peds, 2). The samples of predictions
    :param _pred_traj_gt: Tensor of shape (pred_traj_len, num_peds, 2). GT, to compute some multimodal metrics
    :param seq_start_end: Tensor of shape (batch). Separation of sequences by delimiting start and end
    :param args: provided arguments to decide how the evaluation process goes
    :return: 3 tensors:
    - best_trajectories: Tensor of shape (pred_traj_len, batch, 2). The best predicted trajectory (lowest ADE)
    - best_ade, best_fde: Tensors, each of shape (batch). Raw average and final displacement errors, per trajectory. The
    ADE is the lowest possible, the FDE is the one of the trajectory correspondent with that lowest ADE
    """
    # convert to shape (batch, num_samples)
    ade_list = _ade_list.view(args.num_samples, -1).permute(1, 0)
    fde_list = _fde_list.view(args.num_samples, -1).permute(1, 0)
    ade_chosen, fde_chosen, idx = pick_metrics_according_to_multimodal_mode(ade_list, fde_list, args.eval_mode)
    pred_seq_list = _pred_seq_list.view(-1, args.num_samples, _pred_seq_list.shape[1], _pred_seq_list.shape[2])
    if args.num_samples > 1 and args.kde_nll:
        if not args.fixed_len and not args.variable_len:  # Trajnet++, only primary pedestrian
            pred, gt = pred_seq_list[:, :, seq_start_end[:, 0]], _pred_traj_gt[:, seq_start_end[:, 0]]
            kde_nll = compute_kde_nll(pred, gt, ignore_if_fail=args.ignore_if_kde_nll_fails)
        else:
            kde_nll = compute_kde_nll(pred_seq_list, _pred_traj_gt, ignore_if_fail=args.ignore_if_kde_nll_fails)
    else:
        kde_nll = None
    # best_trajectories = pred_seq_list[:, idx, range(_pred_seq_list.shape[1]), :]
    if not args.fixed_len and not args.variable_len:  # Trajnet++, only primary pedestrian
        # repeat index for all neighbours apart of the same sequence
        full_idx = torch.cat([i.unsqueeze(0).repeat(
            seq_start_end[repeat_times, 1] - seq_start_end[repeat_times, 0]) for repeat_times, i in enumerate(idx)])
    else:
        full_idx = idx
    best_trajectories = pred_seq_list[:, full_idx, range(_pred_seq_list.shape[1]), :]
    return best_trajectories, ade_chosen, fde_chosen, kde_nll


def pick_metrics_according_to_multimodal_mode(ade_list, fde_list, mode='min'):
    """
    Pick the ADE and FDE values for multimodal models, according to a certain provided choice mode.
    :param ade_list: Tensor of shape (batch, num_samples). List of ADE values
    :param fde_list: Tensor of shape (batch, num_samples). List of FDE values
    :param mode: How to pick the ADE and FDE values. One of the following values: 'min', 'min_ade', 'min_fde',
    'min_both', 'max', 'max_ade', 'max_fde', 'max_both', 'avg', 'std'
    :return: The picked ADE and FDE values - each is a tensor of shape (batch). Depending on the mode, it may be
    relevant to supply the index (from 0 to num_samples-1) of the trajectory that was picked - also shape (batch)
    """
    mode = mode.lower()
    torch_fun, torch_fun_both = None, None
    if 'min' in mode:
        if 'both' in mode:
            torch_fun_both = torch.min
        else:
            torch_fun = torch.min
    elif 'max' in mode:
        if 'both' in mode:
            torch_fun_both = torch.max
        else:
            torch_fun = torch.max
    elif 'average' in mode or 'avg' in mode or 'mean' in mode:
        torch_fun_both = torch.mean
    elif 'std' in mode or 'standard' in mode or 'deviation' in mode or 'dev' in mode:
        torch_fun_both = torch.std
    else:
        raise Exception(f'The evaluation mode {mode} is not available')

    if torch_fun_both:  # std or mean or doing min/max for both
        ades = torch_fun_both(ade_list, dim=1)
        fdes = torch_fun_both(fde_list, dim=1)
        # when applying the function to both ade and fde
        return ades, fdes, 0
    # else -  torch_fun is not None - min or max
    if 'fde' in mode:
        fde_choice, idx = torch_fun(fde_list, dim=1)
        ade_choice = ade_list[range(fde_choice.shape[0]), idx]
    else:
        ade_choice, idx = torch_fun(ade_list, dim=1)
        fde_choice = fde_list[range(ade_choice.shape[0]), idx]
    return ade_choice, fde_choice, idx


def output_overall_performance(args, results_list):
    """
    output the overall performance of the model (possibily including several files/scenes; note that there may be
    several files beloning to the same 'scene', where scene here refers to the physical context in which the
    trajectories are inserted (the location of it, the presence of obstacles or un-walkable areas, etc.)
    :param args: command line arguments, that contain several options to influence how to output the results
    :param results_list: list of results for the several files/scenes
    :return: the average and final displacement errors, in single values (weighted mean considering number of
    trajectories and prediction length, for each file/scene)
    """
    if len(results_list) > 1:
        col_pred = torch.tensor([], device=results_list[0].device)
        col_gt = torch.tensor([], device=results_list[0].device)
        col_env_total = torch.tensor([], device=results_list[0].device)
        col_env = torch.tensor([], device=results_list[0].device)
        o_s_b_total = torch.tensor([], device=results_list[0].device)
        o_s_b = torch.tensor([], device=results_list[0].device)
        # create lists for errors from the several results
        ade_list, ade_no_len_list, fde_list, contribution_ade_list, contribution_fde_list = [], [], [], [], []
        total_statistics = StatsResults(device=results_list[0].device, args=args)
        for result in results_list:
            ade, ade_no_len, fde, c_ade, c_fde, statistics = result.get()
            ade_list.append(ade)
            ade_no_len_list.append(ade_no_len)
            fde_list.append(fde)
            contribution_ade_list.append(c_ade)
            contribution_fde_list.append(c_fde)
            if statistics is not None:
                _, _, _, _, _, _, colliding_peds_pred, colliding_peds_gt, cse, osb, _ = statistics.get()
                if args.social_metrics:
                    col_pred = torch.cat((col_pred, torch.mean(colliding_peds_pred.float()).unsqueeze(0)), dim=0)
                    col_gt = torch.cat((col_gt, torch.mean(colliding_peds_gt.float()).unsqueeze(0)), dim=0)
                if args.environment_location:
                    if torch.is_tensor(cse) and cse.nelement() > 0:
                        col_env_total = torch.cat((col_env_total, torch.sum(cse.float()).unsqueeze(0)), dim=0)
                        col_env = torch.cat((col_env, torch.mean(cse.float()).unsqueeze(0)), dim=0)
                    if torch.is_tensor(osb) and osb.nelement() > 0:
                        o_s_b_total = torch.cat((o_s_b_total, torch.sum(osb.float()).unsqueeze(0)), dim=0)
                        o_s_b = torch.cat((o_s_b, torch.mean(osb.float()).unsqueeze(0)), dim=0)
            total_statistics.update_from_existing_stats(statistics)
        mean_ade = np.mean(ade_list)
        mean_ade_no_len = np.mean(ade_no_len_list)
        mean_fde = np.mean(fde_list)
        weighted_mean_ade = np.dot(ade_list, contribution_ade_list) / np.sum(contribution_ade_list)
        weighted_mean_ade_no_len = np.dot(ade_no_len_list, contribution_fde_list) / np.sum(contribution_fde_list)
        weighted_mean_fde = np.dot(fde_list, contribution_fde_list) / np.sum(contribution_fde_list)
        ade_to_return, fde_to_return = weighted_mean_ade, weighted_mean_fde
        print(f"Global results:{os.linesep}" +
              f"ADE_mean={mean_ade:.3f};\tADE_weighted_mean={weighted_mean_ade:.3f};      " +
              f"Without length influencing: mean={mean_ade_no_len:.3f};\t weighted_mean={weighted_mean_ade_no_len:.3f}"
              + f"{os.linesep}FDE_mean={mean_fde:.3f};\tFDE_weighted_mean={weighted_mean_fde:.3f}")
        if args.social_metrics:
            print("Average percentage of colliding pedestrians (with other predictions): "
                  f"{torch.mean(col_pred.float()).data * 100.0:.2f}.")
            print("Average percentage of colliding pedestrians (with ground truth): "
                  f"{torch.mean(col_gt.float()).data * 100.0:.2f}.")
        if args.environment_location:
            print("Per-scene average Collisions with scene environment (CSE): "
                  f"Total={int(torch.mean(col_env_total).data) if col_env_total.nelement() > 0 else 'N/A'}.    "
                  "Average=" + (f"{float(torch.mean(col_env)):.3f}" if col_env.nelement() > 0 else "N/A"))
            print("Per-scene average Trajectories going out of scene bounds (OSB): Total="
                  f"{int(torch.mean(o_s_b_total).data) if o_s_b_total.nelement() > 0 else 'N/A'}.    "
                  "Average=" + (f"{float(torch.mean(o_s_b_total)):.3f}" if o_s_b_total.nelement() > 0 else 'N/A'))
        __compute_statistics__(args, total_statistics, 'all_data')
        statistics_return = total_statistics
    else:
        ade_to_return, ade_no_len, fde_to_return, _, _, statistics_return = results_list[0].get()
        if not args.test_files_individually:
            print(f"ADE={ade_to_return:.3f};      Without length influencing:{ade_no_len:.3f}{os.linesep}"
                  f"FDE={fde_to_return:.3f}{os.linesep}")
            __compute_statistics__(args, statistics_return, 'all_data')
    return ade_to_return, fde_to_return, statistics_return


def append_and_output_per_file(args, dataset_name, eval_results, results_list):
    """
    append results to a list, and output them; these results are associated to a single file
    :param args: the command line arguments, containing several options
    :param dataset_name: name of the dataset (or file) to which these results refer to
    :param eval_results: the results on this file
    :param results_list: the full lists of results in several files; eval_results appended here
    :return: nothing
    """
    ade, ade_no_len, fde, contribution_ade, contribution_fde, statistics = eval_results.get()
    results_list.append(eval_results)
    if args.test_files_individually:
        print(f"File {dataset_name}:{os.linesep}ADE={ade:.3f}; Without length influencing: {ade_no_len:.3f}"
              f"{os.linesep}FDE={fde:.3f}")
        __compute_statistics__(args, statistics, dataset_name)
        print("")  # empty line


def __compute_statistics__(args, statistics, label):
    """
    computes and displays statistics
    :param statistics: Object of type StatsResults. Must not be empty
    :param label: Label identifying to what these statistics refer to (used for plotting titles)
    :return: nothing
    """
    if not statistics:
        return
    all_ades, all_fdes, all_velocities_gt, all_velocities, all_obs_lens, all_pred_lens, all_colliding_peds_pred, \
    all_colliding_peds_gt, all_cse, all_osb, all_kde_nll = statistics.get()
    if args.statistics:
        # only compute and display these statistics if the statistics flag was actually sent
        # the case
        ade_std_and_mean = torch.std_mean(all_ades)
        fde_std_and_mean = torch.std_mean(all_fdes)
        velocity_std_and_mean = torch.std_mean(all_velocities)
        velocity_diff_between_gt = torch.abs(all_velocities_gt - all_velocities)
        velocity_diff_std_and_mean = torch.std_mean(velocity_diff_between_gt)
        print("ADE statistics: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
            torch.min(all_ades).data, torch.max(all_ades).data, ade_std_and_mean[1].data, ade_std_and_mean[0].data))
        print("FDE statistics: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
            torch.min(all_fdes).data, torch.max(all_fdes).data, fde_std_and_mean[1].data, fde_std_and_mean[0].data))
        print("Pred trajectory speed: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
            torch.min(all_velocities).data, torch.max(all_velocities).data, velocity_std_and_mean[1].data,
            velocity_std_and_mean[0].data))
        print("Difference in speed compared to GT: Min={0:.3f} ; Max={1:.3f}; Mean={2:.3f}; Std={3:.3f}".format(
            torch.min(velocity_diff_between_gt).data, torch.max(velocity_diff_between_gt).data,
            velocity_diff_std_and_mean[1].data, velocity_diff_std_and_mean[0].data))
    if all_kde_nll.nelement() > 0:
        print(f"KDE-NLL (from {args.num_samples} samples): {torch.mean(all_kde_nll).data:.2f}")
    if all_colliding_peds_pred.nelement() > 0:
        print("WEIGHTED Average percentage of colliding pedestrians (with other predictions): "
              f"{torch.mean(all_colliding_peds_pred.float()).data * 100.0:.2f}.")
    if all_colliding_peds_gt.nelement() > 0:
        print("WEIGHTED Average percentage of colliding pedestrians (with ground truth): "
              f"{torch.mean(all_colliding_peds_gt.float()).data * 100.0:.2f}.")
    if args.environment_location:
        if all_cse.nelement() > 0:
            print(f"Total collisions with scene environment (CSE): {int(torch.sum(all_cse).data)}.    "
                  f"Average={float(torch.mean(all_cse)):.3f},    "
                  f"Std={float(torch.std(all_cse)):.3f}")
        else:
            print(f"Total collisions with scene environment (CSE): N/A")
        if all_osb.nelement() > 0:
            print(f"Total trajectories going out of scene bounds (OSB): {int(torch.sum(all_osb).data)}.    "
                  f"Average={float(torch.mean(all_osb)):.3f},    "
                  f"Std={float(torch.std(all_osb)):.3f}")
        else:
            print(f"Total trajectories going out of scene bounds (OSB): N/A")


def pred_data_call(_loader, obs_traj, _pred_len, _seq_start_end, _metadata, _num_samples, _curr_sample):
    prediction, pred_seq_start_end, pred_metadata = next(_loader)
    p_dim = 1 if prediction.ndim == 3 else 2  # dimension for number of pedestrians (may change for multiple samples)
    if not all([type(pm) is type(m) and pm.id == m.id for pm, m in zip(pred_metadata, _metadata)]):
        raise Exception('Past data and loaded prediction data do not belong to the same minibatch')
    if prediction.shape[p_dim] < obs_traj.shape[1]:
        # some pedestrians in obs are present, while not being present in prediction
        # possible reason for this - there was no data at prediction time for pedestrians to access
        pred_shape = list(prediction.shape)
        pred_shape[p_dim] = obs_traj.shape[1]
        new_prediction = torch.zeros(pred_shape, device=obs_traj.device)
        seq_start_end_diff = _seq_start_end - pred_seq_start_end
        # append sequences of NaN's to the neighbours that are missing
        for i, (start, end) in enumerate(_seq_start_end):
            (diff_start, diff_end) = seq_start_end_diff[i]
            if p_dim == 2:
                new_prediction[:, :, start:(end - diff_end + diff_start)] = \
                    prediction[:, :, (start - diff_start):(end - diff_end)]
                new_prediction[:, :, (end - diff_end + diff_start):end] = float('nan')
            else:
                new_prediction[:, start:(end - diff_end + diff_start)] = \
                    prediction[:, (start - diff_start):(end - diff_end)]
                new_prediction[:, (end - diff_end + diff_start):end] = float('nan')
        return new_prediction
    # else all pedestrians are present in prediction
    return prediction
