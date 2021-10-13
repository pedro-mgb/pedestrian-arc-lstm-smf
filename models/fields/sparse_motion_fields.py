"""
Created on 17/April/2021

Implementation of sparse motion fields model with the goal of trajectory forecast
NOTE: This model is only to be used for evaluation purposes. Training this model on data is not provided.
The original source code was in MATLAB, this is a conversion to python, using pytorch and numpy frameworks.
"""
import numpy as np
import torch

from models.fields.interpolation import phi_2d
from models.utils.utils import normalize_0_1_min_max


class SparseMotionFields:

    def __init__(self, te_best, qe_best, bc_best, min_max, extra_parameters):
        """
        Initializes the SparseMotionFields model
        :param te_best: Tensor of shape (2, n^2, nK). The actual motion fields for x/y coordinates, defined in a square
        grid
        :param qe_best: Tensor of shape (4, n^2, nK). A 2x2 covariance matrix, to include uncertainty in model
        predictions. Also defined in a square grid.
        :param bc_best: Tensor of shape (nK^2, n^2). Commutation matrix with probabilities of switching between the
        several motion fields (there's only one active motion field per instant, per trajectory). Also defined in a
        square grid.
        :param min_max: a 2D array with the minimum and maximum position values (both x/y). The SparseMotionFields model
        needs trajectories to be normalized in [0, 1] interval. The normalization is done by subtracting min, then
        dividing by max
        :param extra_parameters: a dictionary with several parameters chosen for training the SparseMotionFields model:
        n is the dimension of the grid in which the model is defined; nK is the number of different active motion fields
        """
        self.parameters = extra_parameters
        self.n = np.asscalar(self.parameters[1, np.where(self.parameters[0, :] == 'n')[0]].astype('int32'))
        self.grid_size = np.square(self.n)
        self.nK = np.asscalar(self.parameters[1, np.where(self.parameters[0, :] == 'nK')[0]].astype('int32'))
        self.cov_type = np.asscalar(self.parameters[1, np.where(self.parameters[0, :] == 'cov_type')[0]])

        # actual model obtained through training (not available in this repository)
        self.Te = te_best.to(torch.double)
        self.Qe_orig = qe_best.to(torch.double)
        if self.Qe_orig.size(1) < self.grid_size:
            self.Qe = torch.zeros(self.Qe_orig.shape[0], self.grid_size, self.Qe_orig.shape[2],
                                  device=self.Qe_orig.device)
            # for each field, convert the covariances to the dimension equal to number of fields.
            for k in range(self.Qe_orig.shape[2]):
                q_aux = self.Qe_orig[:, :, k]
                # From MATLAB - Q_aux(:,:, k) = repmat(Qa(:), 1, n ^ 2);
                self.Qe[:, :, k] = q_aux.repeat(self.grid_size / q_aux.shape[1], dim=1)
        else:
            self.Qe = self.Qe_orig
        self.Bc = bc_best.to(torch.double)

        # [0,1] normalization with minmax: (value - min) / (max - min)
        self.min, self.max = min_max[0], min_max[1]
        self.denominator = self.max - self.min

        self.parameters = extra_parameters

    def __call__(self, obs_traj, pred_len, multi_modal=False, out_displacement=False, return_extra=False,
                 all_fields=False):
        """
        callable class (for cross-compatibility with other models)
        model forward pass, to predict a trajectory (or batch of trajectories)
        Doubles are used here to tune precision issues, that can influence the results
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory. May be relative
        trajectory (displacement) or absolute. The processing method depends on flag self.use_abs
        :param pred_len: Indicates for how long should prediction be made
        :param multi_modal: Indicates if meant to use a multimodal-like prediction, via the covariance matrices
        :param out_displacement: If true, will output the displacement applied by sparse motion fields method, instead
        of the whole absolute position
        :param return_extra: whether or not to return extra information
        :param all_fields: instead of returning trajectories for just one position, return
        :return: Tensor of shape (pred_len, batch, output_shape). The predicted trajectories (or distributions)
        """
        assert pred_len > 0, "The provided prediction length must be greater than 0, in order for the model to " \
                             "actually make a prediction"
        # normalize
        obs_traj = obs_traj.double()
        obs_traj_norm = normalize_0_1_min_max(obs_traj, self.min, self.denominator)
        # considering partial trajectories - the cases where there are NaN's in some values of positions
        seq_mask = ~torch.isnan(obs_traj[:, :, 0])  # tensor of shape [obs_traj_len, batch]
        seq_mask_per_ped = torch.all(seq_mask, dim=0)  # tensor of shape [batch]
        if ~torch.all(seq_mask_per_ped):
            has_partial_trajectories = True  # there are partial trajectories
            weights, alpha, active_fields, bc_times_phi_past = \
                self.process_past_with_partial(obs_traj_norm, seq_mask, seq_mask_per_ped, pred_len)
            # the ones that have NaN values for the last instants will not have predictions
            seq_mask = (torch.isnan(weights[-1, :, 0]) + torch.isnan(obs_traj_norm[-1, :, 0])) == 0
            mask_index = [i for i, m in enumerate(seq_mask) if m]
            position = obs_traj_norm[-1, seq_mask, :]
            alpha = alpha[:, seq_mask, :]
            active_fields = active_fields[:, seq_mask]
            bc_times_phi_past = bc_times_phi_past[:, seq_mask, :]
            pass
            # raise Exception('Not yet implemented!')
        else:
            has_partial_trajectories = False  # no partial trajectories
            weights, alpha, active_fields, bc_times_phi_past = self.process_past(obs_traj_norm, pred_len)
            position = obs_traj_norm[-1, :, :]
        # now, we perform prediction - instant by instant (for the whole batch)
        if all_fields:
            # tensor of shape (pred_len, batch, nK, 2)    where nK is the number of different types of motion fields
            pred_traj = torch.full((pred_len, obs_traj.shape[1], self.Te.shape[2], obs_traj.shape[2]),
                                   float('nan'), device=obs_traj.device)
        else:
            # tensor of shape (pred_len, batch, 2)
            pred_traj = torch.full((pred_len, obs_traj.shape[1], obs_traj.shape[2]), float('nan'),
                                   device=obs_traj.device)
        for t in range(weights.shape[0], weights.shape[0] + pred_len):
            position, position_append, alpha[t, :, :], active_fields[t, :] = \
                self.single_prediction(position.clone(), alpha[t - 1, :, :], multi_modal, out_displacement, False,
                                       all_fields)
            # the actual output is the position denormalized, in actual coordinate values
            if all_fields:
                # the minimum is the offset to convert to absolute coordinates; for displacement it should not be used
                pos_out = normalize_0_1_min_max(position_append, self.min if not out_displacement else 0,
                                                self.denominator, reverse=True)
            else:
                pos_out = normalize_0_1_min_max(position_append, self.min if not out_displacement else 0,
                                                self.denominator, reverse=True)
            if has_partial_trajectories:
                pred_traj[t - weights.shape[0], seq_mask] = pos_out.float()
            else:
                pred_traj[t - weights.shape[0]] = pos_out.float()
        if return_extra:
            if has_partial_trajectories:
                alpha_all = torch.full((alpha.shape[0], obs_traj.shape[1], alpha.shape[2]), float('nan'),
                                       dtype=torch.double, device=alpha.device)
                alpha_all[:, seq_mask, :] = alpha
                active_fields_all = torch.full((active_fields.shape[0], obs_traj.shape[1]), float('nan'),
                                               dtype=torch.double, device=alpha.device)
                active_fields_all[:, seq_mask] = active_fields
                return pred_traj, alpha_all, active_fields_all
            return pred_traj, alpha, active_fields
        else:
            return pred_traj

    def process_past_with_partial(self, obs_traj_norm, seq_mask, seq_mask_per_ped, pred_len):
        # process the full length trajectories separately - 'f' stands for 'full'
        weights_f, alpha_f, active_fields_f, bc_times_phi_past_f = \
            self.process_past(obs_traj_norm[:, seq_mask_per_ped, :], pred_len)
        # build the parameters for the trajectories
        weights = torch.full((weights_f.shape[0], seq_mask.shape[1], weights_f.shape[2]), float('nan'),
                             dtype=torch.double, device=seq_mask.device)
        alpha = torch.full((alpha_f.shape[0], seq_mask.shape[1], alpha_f.shape[2]), float('nan'),
                           dtype=torch.double, device=seq_mask.device)
        active_fields = torch.full((active_fields_f.shape[0], seq_mask.shape[1]), float('nan'),
                                   dtype=torch.double, device=seq_mask.device)
        bc_times_phi_past = torch.full((bc_times_phi_past_f.shape[0], seq_mask.shape[1],
                                        bc_times_phi_past_f.shape[2], bc_times_phi_past_f.shape[3]), float('nan'),
                                       dtype=torch.double, device=seq_mask.device)
        mask_f_index = [i for i, m in enumerate(seq_mask_per_ped) if m]
        mask_index = [i for i, m in enumerate(seq_mask_per_ped) if not m]
        for i, p in enumerate(mask_f_index):
            weights[:, p] = weights_f[:, i]
            alpha[:, p] = alpha_f[:, i]
            active_fields[:, p] = active_fields_f[:, i]
            bc_times_phi_past[:, p] = bc_times_phi_past_f[:, i]
        # now process each of the partial trajectories individually
        for p in mask_index:
            # the last instant is only process after, hence the ':-1'
            mask_time_index = [i for i, m in enumerate(seq_mask[:-1, p]) if m]
            if torch.any(~seq_mask[-2:, p]):
                # one of the last 2 positions is empty - should not perform prediction for these trajectories
                continue
                # weights_p, alpha_p, active_fields_p, bc_times_phi_past_p
            obs_traj_p = obs_traj_norm[seq_mask[:, p], p]
            weights_p, alpha_p, active_fields_p, bc_times_phi_past_p = \
                self.process_past(obs_traj_p.unsqueeze(1), pred_len)
            for i, t in enumerate(mask_time_index):
                weights[t, p] = weights_p[i, 0]
                alpha[t, p] = alpha_p[i, 0]
                active_fields[t, p] = active_fields[i, 0]
                bc_times_phi_past[t, p] = bc_times_phi_past_p[i, 0]
        return weights, alpha, active_fields, bc_times_phi_past

    def process_past(self, obs_traj_norm, pred_len):
        # interpolation (except for last position)
        phi_past = phi_2d(obs_traj_norm[:-1, :, :], self.grid_size)
        # compute probabilities (weights) of each motion field
        # from MATLAB (one specific field) - uncert = reshape(sum(Q_aux(:,:, k)*Phi, 2), 2, 2);
        # i<->4; j<->n^2; k<->nK; l<->obs_traj_len-1; m<->batch      tensor of shape (4, nK, obs_traj_len-1, batch)
        qe_times_phi_past = torch.einsum('ijk,jlm->ilmk', self.Qe, phi_past)
        # tensor of shape (obs_traj_len-1,batch,nK,2,2)
        uncertainty = qe_times_phi_past.view(2, 2, qe_times_phi_past.shape[1], qe_times_phi_past.shape[2],
                                             qe_times_phi_past.shape[3]).permute(2, 3, 4, 0, 1)
        # tensor of shape (obs_traj_len-1,batch,nK)
        c = 1 / (2 * np.pi * torch.sqrt(torch.linalg.det(uncertainty)))
        s = torch.inverse(uncertainty)
        # tensor of shape (obs_traj_len-1, batch, 2, nK);
        # i<->n^2; j<->obs_traj_len-1; k<->batch; l<->2; m<->nK
        fields_contribution = torch.einsum('lim,ijk -> jklm', self.Te, phi_past)
        # from MATLAB (one specific field/instant) - e = trajectory(:, t + 1)-trajectory(:, t)-Te_best(:,:, k)*Phi;
        obs_traj_norm_repeat = obs_traj_norm.unsqueeze(3).repeat(1, 1, 1, self.Te.shape[2])
        # difference between actual displacement and the displacement provided by the motion fields
        e = obs_traj_norm_repeat[1:, :, :, :] - obs_traj_norm_repeat[:-1, :, :, :] - fields_contribution
        et = torch.transpose(e, 2, 3)
        # from MATLAB (one specific field/instant) - P(k, 1) = C. * exp(-0.5 * e'*S*e);
        # c_repeat = c.unsqueeze(0).unsqueeze(0).repeat(phi_past.shape[1], phi_past.shape[2], 1)
        # j<->obs_traj_len-1; k<->batch; l<->p<->2; m<->nK
        s_e = torch.einsum('jkmpl,jklm->jkmp', s, e)
        et_s_e = torch.einsum('jkml,jkml->jkm', et, s_e)
        weights = c * torch.exp(-0.5 * et_s_e)

        # to decide what are the active fields to apply at each time step, for each trajectory
        # tensor of shape (obs_traj_len+pred_traj_len, batch)
        active_fields = torch.zeros(obs_traj_norm.shape[0] + pred_len, obs_traj_norm.shape[1], dtype=torch.long,
                                    device=weights.device)
        # tensor of shape (obs_traj_len+pred_traj_len, batch, nK)
        alpha = torch.zeros(obs_traj_norm.shape[0] + pred_len, weights.shape[1], weights.shape[2],
                            dtype=torch.double, device=weights.device)
        # initialize alpha values for first position: alpha(t=0) = P(t=0) / nK; then normalize (to ge probabilities)
        alpha[0, :, :] = weights[0, :, :] / weights.size(2)
        alpha[0, :, :] = torch.div(alpha[0, :, :].permute(1, 0), torch.sum(alpha[0, :, :], dim=1)).permute(1, 0)
        active_fields[0, :] = torch.argmax(alpha[0, :, :], dim=1)
        # h<->nK^2 ; i<->n^2; j<->obs_traj_len-1; k<->batch;
        # After operations result will be: tensor of shape [obs_traj_len-1,batch,nK,nK]
        bc_times_phi_past = torch.einsum('hi,ijk->hjk', self.Bc, phi_past) \
            .view(self.Te.shape[2], self.Te.shape[2], phi_past.shape[1], phi_past.shape[2]).permute(2, 3, 0, 1)
        for t in range(1, weights.shape[0]):
            # from MATLAB alpha = P.*(reshape(BC_best*Phi,nK,nK)'*alpha);           h,i<->nK ; k<->batch;
            bc_phi_alpha_prev = torch.einsum('khi,ki->kh', bc_times_phi_past[t, :, :, :], alpha[t - 1, :, :])
            alpha[t, :, :] = weights[t, :, :] * bc_phi_alpha_prev
            # normalize to obtain probabilities
            alpha[t, :, :] = torch.div(alpha[t, :, :].permute(1, 0), torch.sum(alpha[t, :, :], dim=1)).permute(1, 0)
            active_fields[t, :] = torch.argmax(alpha[t, :, :], dim=1)
        return weights, alpha, active_fields, bc_times_phi_past

    def single_prediction(self, position, alpha, multi_modal=False, out_displacement=False, normalize_pos=False,
                          all_fields=False):
        """
        perform a prediction for a single time step
        :param position: the position in the previous time step
        :param alpha: the probabilities (or weights) for each motion field
        :param multi_modal: Indicates if meant to use a multimodal-like prediction, via the covariance matrices
        :param out_displacement: If true, will output the displacement applied by sparse motion fields method, instead
        of the whole absolute position
        :param normalize_pos: if supplied, it assumes that position does not come normalized in [0,1] interval, and as
        such, the output will not also be normalized (will be in whatever units/scale it was supplied)
        :param all_fields: instead of returning trajectories for just one position, return
        :return: by this order:
        - absolute_position (normalized)
        - the prediction (can be displacement or absolute, normalized or not), or predictions using all fields (if flag
        all_fields=True)
        - new probabilities for each field (alpha)
        - active field chosen for the trajectory (or batch of trajectories)
        """
        if normalize_pos:
            position = normalize_0_1_min_max(position, self.min, self.denominator)
        phi = phi_2d(position.unsqueeze(0), self.grid_size)
        bc_times_phi = torch.einsum('hi,ijk->hjk', self.Bc, phi).view(
            self.Te.shape[2], self.Te.shape[2], phi.shape[1], phi.shape[2]).permute(2, 3, 0, 1)
        alpha = torch.einsum('khi,ki->kh', bc_times_phi[0, :, :, :], alpha)
        # normalize to obtain probabilities
        alpha = torch.div(alpha.permute(1, 0), torch.sum(alpha, dim=1)).permute(1, 0)
        active_fields = torch.argmax(alpha, dim=1)
        # actual prediction - may add noise depending on if multimodal is intended or not
        # remove time dimension, since this phi is specific to one instant
        phi = phi.squeeze(dim=1)
        active_te = self.Te[:, :, active_fields]  # tensor of shape (2, n^2, batch)
        # i<->2; j<->n^2; k<->batch;                tensor of shape (batch, 2)
        fields_displacement = torch.einsum('ijk,jk->ki', active_te, phi)
        fields_displacement_all, position_append_all = None, None
        if all_fields:
            # i<->2; j<->n^2; k<->batch; l<->nK     tensor of shape (batch, nK, 2)
            fields_displacement_all = torch.einsum('ijkl,jkl->kli',
                                                   self.Te.unsqueeze(2).repeat(1, 1, phi.shape[1], 1),
                                                   phi.unsqueeze(2).repeat(1, 1, self.Te.shape[2]))
        if multi_modal:
            # TODO implement use of all_fields flag with multimodal; useful for further integration of noise
            #  when interacting with other models (such as providing this noise with a LSTM trained with NLL)
            # i<->4; j<->n^2; k<->nK; m<->batch      tensor of shape (4, batch)
            active_qe_times_phi = torch.einsum('ijm,jm->im', self.Qe[:, :, active_fields], phi)
            # tensor of shape (batch,2,2)
            uncertainty = active_qe_times_phi.view(2, 2, active_qe_times_phi.shape[1]).permute(2, 0, 1)
            # see https://www.mathworks.com/help/matlab/ref/chol.html for analogous function in MatLab
            # m<->batch; i<->j<->2
            uncertainty_contribution = torch.einsum(
                'mi,mij->mj', torch.randn(uncertainty.shape[0], uncertainty.shape[1], dtype=torch.double),
                torch.cholesky(uncertainty))
            # append noise to position; make sure values are in [0, 1] interval
            position_abs = torch.clamp(position + fields_displacement + uncertainty_contribution, min=0, max=1)
            position_append = fields_displacement + uncertainty_contribution if out_displacement else position_abs
        else:
            # make sure values are in [0, 1] interval
            position_abs = torch.clamp(position + fields_displacement, min=0, max=1)
            position_append = fields_displacement if out_displacement else position_abs
            if all_fields:
                position_abs_all = torch.clamp(
                    position.unsqueeze(1).repeat(1, self.Te.shape[2], 1) + fields_displacement_all, min=0, max=1)
                position_append_all = fields_displacement_all if out_displacement else position_abs_all
        if normalize_pos:
            position_abs = normalize_0_1_min_max(position_abs, self.min, self.denominator, reverse=True)
            position_append = normalize_0_1_min_max(position_append, self.min, self.denominator, reverse=True)
        # the actual output is the position denormalized, in actual coordinate values
        if all_fields:
            return position_abs, position_append_all, alpha, active_fields
        else:
            return position_abs, position_append, alpha, active_fields
