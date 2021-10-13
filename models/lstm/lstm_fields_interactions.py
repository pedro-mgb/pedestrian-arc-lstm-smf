"""
Created on July 25th, 2021

LSTM model (several variants of it) to incorporate motion fields with use of interactions.
It can be seen as a combinations of the model in lstm_fields.py and lstm_interactions.py
It uses an Encoder-Decoder architecture and has the possibility of having different integrations of SparseMotionFields

The only scene and interaction-aware model in this repo.
"""
import torch
from torch import nn

from models.lstm.lstm import POS_2D
from models.lstm.lstm_interactions import LSTMWithInteractionModule
from models.fields.sparse_motion_fields import SparseMotionFields
from models.utils.utils import normalize_sequences


class FieldsWithInteractionModuleAndLSTM(nn.Module):
    def __init__(self, fields, module, shape_config=None, embedding_dim=64, h_dim=64,
                 activation_on_input_embedding=None, activation_on_output=None, dropout=0, output_gaussian=False,
                 feed_all=False, use_probs=False):
        """
        Initializes the LSTM network, by creating an encoder and decoder with an interaction module
        For the definition
        """
        super(FieldsWithInteractionModuleAndLSTM, self).__init__()
        assert isinstance(fields, SparseMotionFields), 'The fields supplied must be of type SparseMotionFields'
        self.motion_fields = fields
        self.interaction_module = module
        self.shape_config = shape_config

        self.feed_all, self.use_probs = feed_all, use_probs

        # past trajectories does not use motion fields - just interactions between pedestrians
        self.encoder = LSTMWithInteractionModule(module, shape_config, embedding_dim, h_dim,
                                                 activation_on_input_embedding, activation_on_output, dropout,
                                                 output_gaussian, use_enc_dec=False)
        self.decoder = DecoderWithFieldsAndInteractions(fields, module, shape_config, embedding_dim, h_dim, dropout,
                                                        activation_on_input_embedding, activation_on_output,
                                                        output_gaussian, feed_all, use_probs)

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None, seq_start_end=(), full_teacher_forcing=False,
                idx_shape_data=None, metadata=None):
        _, final_h = self.encoder(obs_traj, pred_len=1, seq_start_end=seq_start_end, return_obs=True,
                                  idx_shape_data=idx_shape_data, return_state=True)
        k = 1
        return self.decoder(obs_traj, pred_len, pred_traj_gt, seq_start_end, full_teacher_forcing,
                            idx_shape_data=idx_shape_data, state_final=final_h, metadata=metadata)


class DecoderWithFieldsAndInteractions(LSTMWithInteractionModule):
    def __init__(self, fields, module, shape_config=None, embedding_dim=64, h_dim=64, dropout=0,
                 activation_on_input_embedding=None, activation_on_output=None, output_gaussian=False,
                 feed_all=False, use_probs=False):
        super(DecoderWithFieldsAndInteractions, self).__init__(module, shape_config, embedding_dim, h_dim,
                                                               activation_on_input_embedding, activation_on_output,
                                                               dropout, output_gaussian, use_enc_dec=False)
        self.motion_fields = fields
        self.feed_all, self.use_probs = feed_all, use_probs
        if self.feed_all:
            # include the embedding of lstm prediction + embedding of K field prediction (two separate layers)
            self.input_embedding = self.__change_input_embedding_dims__(POS_2D, int(embedding_dim / 2))
            self.fields_embedding = self.__change_input_embedding_dims__(
                self.motion_fields.Te.shape[2] * self.motion_fields.Te.shape[0], int(embedding_dim / 2))
        else:
            # include the embedding of lstm prediction + most likely fields prediction
            self.input_embedding = self.__change_input_embedding_dims__(POS_2D * 2, embedding_dim)
            self.fields_embedding = None

    def step(self, lstm, state, rel_positions, positions, seq_start_end, past_positions, fields_displacement=None,
             alpha=None, metadata=None):
        """
        Performs a single step of processing the predicted trajectory with the decoder.
        :param lstm: the lstm cell to use. Should be the encoder
        :param state: list with 2 tensors of shape (batch, self.h_dim). The hidden LSTM tensor h, and the inner LSTM
        cell state c.
        :param rel_positions: Tensor of shape (batch, 2). The relative displacements of pedestrians
        :param positions: Tensor of shape (batch, 2). Absolute positions for pedestrians
        :param seq_start_end: Tensor of shape (2, num_seqs). Indicates which trajectories belong to a certain sequence
        (that belong to the same time frame)
        :param past_positions: Tensor of shape (batch, 2). Absolute past positions for pedestrians
        :param fields_displacement: Tensor of shape (batch, 2) or (batch, nK, 2) if self.feed_all. The displacements
        outputted by the motion fields model
        :param alpha: Tensor of shape (batch, nK). The probabilities associated to each motion field regime
        :param metadata: list of metadata objects for each sequence, with relevant information to convert between
        the motion fields
        :return: the updated LSTM state (same shape as state variable going in); and the predicted output of lstm,
        which is a tensor of shape (batch, output_shape) (output_shape=2 or 5 in terms of 2D positions or gaussian)
        """
        seq_mask, state_masked = self.build_seq_mask_for_step(rel_positions, positions, past_positions, state,
                                                              extra=alpha[:, 0])

        if self.feed_all:
            # pass the LSTM prediction and the motion fields predictions by separate embeddings, and then concat
            pos_embedded = self.input_embedding(rel_positions[seq_mask, :2])
            # normalize the direction of the motion fields displacement
            nk = fields_displacement.shape[1]  # number different regimes for motion fields
            fields_pos = fields_displacement + \
                         normalize_sequences(past_positions[:, :2].unsqueeze(0), seq_start_end, metadata,
                                             inverse=True).permute(1, 0, 2).repeat(1, nk, 1)
            fields_displacement_use = fields_displacement.clone()
            for k in range(nk):
                fields_displacement_use[:, k] = normalize_sequences(fields_pos[:, k].unsqueeze(0), seq_start_end,
                                                                    metadata).squeeze(0) - past_positions
            # .clone() is necessary here because torch.autograd detects change. Results in RuntimeError: one of the
            # variables needed for gradient computation has been modified by an inplace operation
            if self.use_probs:
                displacement_times_probs = fields_displacement_use[seq_mask] * alpha.unsqueeze(2).repeat(1, 1, 2)
                fields_embedded = self.fields_embedding(displacement_times_probs.float().view(
                    fields_displacement_use[seq_mask].shape[0], -1).clone())
            else:
                fields_embedded = self.fields_embedding(fields_displacement_use[seq_mask].view(
                    fields_displacement_use[seq_mask].shape[0], -1).float().clone())
            embedded = torch.cat((pos_embedded, fields_embedded), dim=1)
        else:
            """
            embedded = self.input_embedding(
                torch.cat((rel_positions[seq_mask, :2], fields_displacement[seq_mask, :2]), dim=1).to(torch.float))
            """
            # this should be made to work make this work with metadata
            raise Exception('TODO not implemented use of Arc-LSTM-SMF model without --feed_all_fields_flag')
        embedded = self.dropout(embedded)

        pool_data = self.step_interaction_layer(positions, past_positions, state, seq_start_end, seq_mask)

        fields_displacement_new, alpha_new = fields_displacement.clone(), alpha.clone()
        # compute the next motion fields displacement
        with torch.no_grad():
            positions_real = normalize_sequences(positions.unsqueeze(0), seq_start_end, metadata,
                                                 inverse=True).squeeze(0)
            _, fields_displacement_new[seq_mask], alpha_new[seq_mask], _ = \
                self.motion_fields.single_prediction(positions_real[seq_mask], alpha[seq_mask], out_displacement=True,
                                                     normalize_pos=True, all_fields=self.feed_all)

        state_out, output = self.step_output(lstm, embedded, pool_data, state, state_masked, seq_mask)

        return state_out, output, fields_displacement_new, alpha

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None, seq_start_end=(), full_teacher_forcing=False,
                return_obs=False, idx_shape_data=None, return_state=False, state_final=None, metadata=None):
        """

        :param: the final LSTM state originating from the decoder, which will be sent
        For the remaining parameters, see LSTMWithInteractionModule. Some of the parameters, like return_obs and
        return_state are here for compatibility with LSTMWithInteractionModule, but they are not actually used here
        :return:
        """
        assert state_final is not None, 'An LSTM state from the encoder must be supplied'
        num_peds, device = obs_traj.size(1), obs_traj.device
        if self.shape_config is not None:
            # reset the information in the shape configuration module for a new batch
            if idx_shape_data is not None:
                self.shape_config.reset(num_peds, seq_start_end, device, idx_shape_data)
            else:
                self.shape_config.reset(num_peds, seq_start_end, device)
        with torch.no_grad():
            # denormalize past trajectory (if applicable) to feed motion fields
            traj_in = normalize_sequences(obs_traj, seq_start_end, metadata, inverse=True)
            displacement, alpha_total, active_field_total = \
                self.motion_fields(traj_in, 1, out_displacement=True, return_extra=True, all_fields=self.feed_all)
        # remove the considerations related to the past - not useful here
        alpha = alpha_total[obs_traj.shape[0] - 1, :, :]
        # active_field = active_field_total[obs_traj.shape[0] - 1, :]  # not used
        displacement = displacement.squeeze(dim=0)
        pred_duration = pred_len if pred_len > 0 else pred_traj_gt.shape[0]
        if self.feed_all:
            # tensor of shape (pred_len, batch, nK, 2)      where nK is the number of different types of motion fields
            fields_displacement = torch.full_like(displacement, float('nan'), dtype=alpha.dtype,
                                                  device=device).unsqueeze(0).repeat(pred_duration, 1, 1, 1)
        else:
            fields_displacement = torch.full((pred_duration, num_peds, obs_traj.shape[2]), float('nan'),
                                             dtype=alpha.dtype, device=device)
        seq_mask = (torch.isnan(alpha[:, 0]) + torch.isnan(obs_traj[-1, :, 0])) == 0
        fields_displacement[0] = displacement
        state = (state_final, torch.zeros_like(state_final))  # initialize the INNER LSTM state c^t (no memory)
        # get first predicted position from encoder state (regarding last observed position)
        out_rel = self.output_embedding(state_final[seq_mask, :].view(-1, self.h_dim))
        rel_positions_pred = torch.full((obs_traj.shape[1], self.output_dim), float("Nan"), device=device)
        rel_positions_pred[seq_mask] = out_rel
        prev_positions = obs_traj[-1]
        if pred_len > 0:
            pred_traj_out = torch.zeros(pred_len, num_peds, self.output_dim, device=device)
            rel_positions = pred_traj_out[0] = rel_positions_pred
            positions = obs_traj[-1] + rel_positions_pred[:, :2]
            for t in range(1, pred_len):
                state, rel_positions, fields_displacement[t], alpha = \
                    self.step(self.lstm, state, rel_positions.detach(), positions.detach(),
                              seq_start_end, past_positions=prev_positions.detach(),
                              fields_displacement=fields_displacement[t - 1], alpha=alpha,
                              metadata=metadata)
                pred_traj_out[t] = rel_positions
                prev_positions = positions
                positions = positions + rel_positions[:, :2]
        else:
            pred_traj_out = torch.zeros(pred_traj_gt.shape[0], num_peds, self.output_dim, device=device)
            pred_traj_out[0] = rel_positions_pred
            relative_displacements_gt = pred_traj_gt - \
                                        torch.cat((obs_traj[-1].unsqueeze(0), pred_traj_gt[:-1]), dim=0)
            positions_pred = obs_traj[-1] + rel_positions_pred[:, :2]
            # teacher forcing
            for t in range(1, pred_traj_gt.shape[0]):
                # print(f'\r{t}', end='')
                rel_displacement, positions_in = relative_displacements_gt[t], pred_traj_gt[t - 1]
                if not full_teacher_forcing:
                    # primary pedestrians will not have ground truth trajectory, since it gets the model too adjusted
                    #   to the training set; only the social context is retrieved from the ground truth
                    rel_displacement[seq_start_end[:, 0], :] = pred_traj_out[t - 1, seq_start_end[:, 0], :2].detach()
                    positions_in[seq_start_end[:, 0], :] = positions_pred[seq_start_end[:, 0], :].detach()
                state, pred_traj_out[t], fields_displacement[t], alpha = \
                    self.step(self.lstm, state, rel_displacement, positions_in, seq_start_end,
                              past_positions=prev_positions, fields_displacement=fields_displacement[t - 1],
                              alpha=alpha.clone(), metadata=metadata)
                prev_positions = positions_in  # gt for all trajectories except those of primary pedestrian
                positions_pred = prev_positions + pred_traj_out[t, :, :2]
        return pred_traj_out
