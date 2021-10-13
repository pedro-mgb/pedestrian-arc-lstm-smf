"""
Created on May 19th 2021
Contains LSTM networks with extended variants to incorporate interactions between several pedestrians.
Some interaction modules are present in other files, like models/interaction_modules/shape_based.py
"""
import copy

import torch

from models.lstm.lstm import VanillaLSTM, POS_2D
from models.interaction_modules.simple_shape_config import ShapeConfigNeighDist, ArcShapeRadiusConfigVisibleNeighDist, \
    ArcShapeRadiusConfigVisiblePedDensity, GrowingShapeUpToMaxPedestrians


class LSTMWithInteractionModule(VanillaLSTM):
    """
    LSTM model that contains an interaction model, to consider the influence of neighbour pedestrians (which is not
    done by the vanilla lstm model). Usually such interaction modules output a pooled tensor, a sort of "compact"
    representation of the interactions between pedestrians.
    """

    def __init__(self, module, shape_config=None, embedding_dim=64, h_dim=64, activation_on_input_embedding=None,
                 activation_on_output=None, dropout=0, output_gaussian=False, use_enc_dec=True):
        """
        Initializes the LSTM network, with a module that considers social interactions (usually outputting a pooled
        context for the several pedestrians in a sequence).
        :param module: A module that considers interactions between pedestrians (e.g. of type GridBasedPooling)
        :param use_enc_dec: If meant or not to use encoder-decoder architecture for LSTM
        For the description of the remaining parameters, see documentation of VanillaLSTM
        """
        # LSTM cell meant to receive both embedding of input, and embedding of the interaction module
        super(LSTMWithInteractionModule, self).__init__(embedding_dim + module.out_dim, h_dim,
                                                        activation_on_input_embedding, dropout=dropout,
                                                        output_gaussian=output_gaussian,
                                                        activation_on_output=activation_on_output)
        self.interaction_module = module
        self.shape_config = shape_config
        self.input_embedding = self.__change_input_embedding_dims__(POS_2D, embedding_dim)
        self.use_enc_dec = use_enc_dec
        if use_enc_dec:
            self.encoder = self.lstm
            self.decoder = copy.deepcopy(self.lstm)  # duplicate, but does not share the same parameters as encoder

    @staticmethod
    def build_seq_mask_for_step(rel_positions, positions, past_positions, state, extra=None):
        # mask for pedestrians absent from this instant (partial trajectories) consider only the hidden states of
        # pedestrians present in in that instant (no nan values) - Tensor of shape (num_peds)
        seq_mask = (torch.isnan(past_positions[:, 0]) + torch.isnan(positions[:, 0]) +
                    torch.isnan(rel_positions[:, 0])) == 0
        if extra is not None:
            seq_mask = seq_mask * (~torch.isnan(extra))
        # stack the state, removing the pedestrians not present in scene, via the mask
        state_masked = [
            torch.stack([h for m, h in zip(seq_mask, state[0]) if m], dim=0),
            torch.stack([c for m, c in zip(seq_mask, state[1]) if m], dim=0),
        ]
        return seq_mask, state_masked

    def step_interaction_layer(self, positions, past_positions, state, seq_start_end, seq_mask):
        pool_data = torch.tensor([], device=seq_start_end.device)
        hh = state[0].clone()  # CLONE is necessary to retain the original state for backpropagation (grad requires it)
        for (start, end) in seq_start_end:
            start_end_seq_mask = seq_mask[start:end]
            indexes_ped = torch.stack([start + i for i, m in enumerate(start_end_seq_mask) if m])
            past_pos = past_positions[start:end][start_end_seq_mask]
            if self.shape_config is not None:
                self.shape_config.update_shape(indexes_ped)
            seq_pool_data, occ_data = self.interaction_module(hh[start:end][start_end_seq_mask],
                                                              positions[start:end][start_end_seq_mask],
                                                              past_positions=past_pos)
            pool_data = torch.cat((pool_data, seq_pool_data), dim=0)
            if self.shape_config is not None:
                # use the shape_config module to compute the new shape pooling parameters for the pooling layer
                if isinstance(self.shape_config, ArcShapeRadiusConfigVisiblePedDensity) or \
                        isinstance(self.shape_config, ShapeConfigNeighDist) or \
                        isinstance(self.shape_config, ArcShapeRadiusConfigVisibleNeighDist) or \
                        isinstance(self.shape_config, GrowingShapeUpToMaxPedestrians):
                    self.shape_config(past_pos, positions[start:end, :2][start_end_seq_mask], indexes_ped)
                else:
                    self.shape_config(occ_data, indexes_ped)
        return pool_data

    def step_output(self, lstm_cell, embedded, pool_data, state, state_masked, seq_mask):
        # input_dim=[num_peds, embedding_dim+self.interaction_module.out_dim];  state_dim=[num_peds, h_dim]
        state_masked = lstm_cell(torch.cat((embedded, pool_data), dim=1), state_masked)
        out_embedding = self.output_embedding(state_masked[0])
        output = torch.full((seq_mask.size(0), out_embedding.size(1)), float("Nan"), device=out_embedding.device)
        mask_index = [i for i, m in enumerate(seq_mask) if m]
        state_out = (state[0].clone(), state[1].clone())
        # unmask [Update hidden-states and output]
        for i, h, c, o in zip(mask_index, state_masked[0], state_masked[1], out_embedding):
            state_out[0][i] = h
            state_out[1][i] = c
            output[i] = o
        return state_out, output

    def step(self, lstm, state, rel_positions, positions, seq_start_end, past_positions):
        """
        Performs a single step in the LSTM processing the trajectory, including use of the LSTM module
        CREDITS - Adapted trajnetpp repository:
        https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/lstm.py#L71
        :param lstm: the lstm cell to use. If self.use_enc_dec, can use either encoder or decoder
        :param state: list with 2 tensors of shape (batch, self.h_dim). The hidden LSTM tensor h, and the inner LSTM
        cell state c.
        :param rel_positions: Tensor of shape (batch, 2). The relative displacements of pedestrians
        :param positions: Tensor of shape (batch, 2). Absolute positions for pedestrians
        :param seq_start_end: Tensor of shape (2, num_seqs). Indicates which trajectories belong to a certain sequence
        (that belong to the same time frame)
        :param past_positions: Tensor of shape (batch, 2). Absolute past positions for pedestrians
        :return: the updated LSTM state (same shape as state variable going in); and the predicted output of lstm,
        which is a tensor of shape (batch, output_shape) (output_shape=2 or 5 in terms of 2D positions or gaussian)
        """
        seq_mask, state_masked = self.build_seq_mask_for_step(rel_positions, positions, past_positions, state)
        # the ':2' is used in case the tensor has more info (e.g. when output_gaussian in __init__ is True)
        embedded = self.input_embedding(rel_positions[seq_mask, :2])
        embedded = self.dropout(embedded)

        pool_data = self.step_interaction_layer(positions, past_positions, state, seq_start_end, seq_mask)

        return self.step_output(lstm, embedded, pool_data, state, state_masked, seq_mask)

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None, seq_start_end=(), full_teacher_forcing=False,
                return_obs=False, idx_shape_data=None, return_state=False):
        """
        Model forward pass, processing the past trajectories and outputting the predictions.
        :param obs_traj: Tensor of shape (obs_traj_len, num_peds, 2). Sequence of trajectories in the same time frame.
        Should be in  absolute coordinates, due to the need of the interaction module to compute relative distances
        between pedestrians.
        :param pred_len: (when gt not available) Indicates for how long should prediction be made
        :param pred_traj_gt: Tensor of shape (pred_traj_len, num_peds, 2); Should only be used in training. Ground truth
        prediction data, used to enable teacher forcing. May be relative displacement or absolute positions.
        :param seq_start_end: Tensor of shape (2, num_seqs). Indicates which trajectories belong to a certain sequence
        (that belong to the same time frame)
        :param full_teacher_forcing: use teacher forcing on the trajectories of primary pedestrians too
        :param return_obs:
        :param idx_shape_data:
        :param return_state: also return the most recent lstm state
        :return: The predicted trajectory, which is a tensor of shape (pred_traj_len, batch, output_shape)
        (output_shape=2 or 5 in terms of 2D positions or gaussian)
        """
        obs_traj_rel = torch.zeros_like(obs_traj)
        obs_traj_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]
        num_peds, device = obs_traj.size(1), obs_traj.device
        if self.shape_config is not None:
            # reset the information in the shape configuration module
            if idx_shape_data is not None:
                self.shape_config.reset(num_peds, seq_start_end, obs_traj.device, idx_shape_data)
            else:
                self.shape_config.reset(num_peds, seq_start_end, obs_traj.device)
        state = self.init_hidden(num_peds, obs_traj.device)
        prev_positions = obs_traj[0, :, :]
        rel_positions_pred = None
        obs_pred_out = torch.zeros(obs_traj.shape[0] - 1, num_peds, self.output_dim, device=device) if return_obs else 0
        t = 0
        # process past trajectory, including last instant; the last state will be used to obtain the first prediction
        for rel_positions, positions in zip(obs_traj_rel[1:, :, :], obs_traj[1:, :, :]):
            state, rel_positions_pred = self.step(self.encoder if self.use_enc_dec else self.lstm, state, rel_positions,
                                                  positions, seq_start_end, past_positions=prev_positions)
            if return_obs:
                obs_pred_out[t] = rel_positions_pred
            t += 1
            prev_positions = positions
        if pred_len > 0:
            pred_traj_out = torch.zeros(pred_len, num_peds, self.output_dim, device=obs_traj.device)
            rel_positions = pred_traj_out[0] = rel_positions_pred
            positions = obs_traj[-1] + rel_positions_pred[:, :2]
            for t in range(1, pred_len):
                state, rel_positions = self.step(self.decoder if self.use_enc_dec else self.lstm, state,
                                                 rel_positions.detach(), positions.detach(), seq_start_end,
                                                 past_positions=prev_positions.detach())
                pred_traj_out[t] = rel_positions
                prev_positions = positions
                positions = positions + rel_positions[:, :2]
        else:
            pred_traj_out = torch.zeros(pred_traj_gt.shape[0], num_peds, self.output_dim, device=obs_traj.device)
            pred_traj_out[0] = rel_positions_pred
            relative_displacements_gt = pred_traj_gt - \
                                        torch.cat((obs_traj[-1].unsqueeze(0), pred_traj_gt[:-1]), dim=0)
            positions_pred = obs_traj[-1] + rel_positions_pred[:, :2]
            # teacher forcing
            for t in range(1, pred_traj_gt.shape[0]):
                # the primary pedestrian will make use of the prediction, instead of the real data. SOURCE:
                # https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/lstm.py#L235
                rel_displacement = relative_displacements_gt[t]
                positions_in = pred_traj_gt[t - 1]
                if not full_teacher_forcing:
                    # primary pedestrians will not have ground truth trajectory, since it gets the model too adjusted
                    #   to the training set; only the social context is retrieved from the ground truth
                    rel_displacement[seq_start_end[:, 0], :] = pred_traj_out[t - 1, seq_start_end[:, 0], :2].detach()
                    positions_in[seq_start_end[:, 0], :] = positions_pred[seq_start_end[:, 0], :].detach()
                state, pred_traj_out[t] = self.step(self.decoder if self.use_enc_dec else self.lstm, state,
                                                    rel_displacement, positions_in, seq_start_end,
                                                    past_positions=prev_positions)
                prev_positions = positions_in  # gt for all trajectories except those of primary pedestrian
                positions_pred = prev_positions + pred_traj_out[t, :, :2]
        # the last output in 'obs_pred_out' corresponds to the first prediction - so it is on 'pred_traj_out'
        out_traj = torch.cat((obs_pred_out[:-1], pred_traj_out), dim=0) if return_obs else pred_traj_out
        if return_state:
            return out_traj, state[0]
        # else - just return prediction
        return out_traj
