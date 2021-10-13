"""
Created on April 25th, 2021

LSTM model (several variants of it) to incorporate motion fields.
The Sparse Motion Fields model is trained with data from a single scene.
The goal of the model is to learn information regarding the scene (presence of obstacles/unwalkable areas).
These LSTM models incorporate the predicted displacements of Sparse Motion Fields into its own prediction.
"""
import torch
import torch.nn.functional as fun

from models.fields.sparse_motion_fields import SparseMotionFields
from models.lstm.lstm import VanillaLstmEncDec, Decoder, POS_2D
from models.utils.utils import remove_zeros_from_traj


class SimpleFieldsWithLSTM(VanillaLstmEncDec):
    """
    Class that performs prediction via LSTM network (Encoder-Decoder architecture, meaning 2 LSTMs), plus a simple
    usage of the sparse motion fields method.
    """

    def __init__(self, fields, embedding_dim=64, h_dim=64, num_layers=1, dropout=0,
                 activation_on_input_embedding=None, activation_on_output=None, normalize_embedding=False,
                 output_gaussian=False, discard_zeros=False):
        """
        see models.lstm.lstm.VanillaLstmEncDec for description of the rest of the parameters
        Initialization of LSTM model that also includes the Sparse Motion Fields model in it
        :param fields: Motion fields, assumed to be of class models.fields.sparse_motion_fields.SparseMotionFields
        """
        super(SimpleFieldsWithLSTM, self).__init__(embedding_dim, h_dim, num_layers, dropout,
                                                   activation_on_input_embedding, activation_on_output, True,
                                                   normalize_embedding, output_gaussian, discard_zeros)
        assert isinstance(fields, SparseMotionFields), 'The fields supplied must be of type SparseMotionFields'
        self.motion_fields = fields
        self.__obs_cache__ = []
        self.__pred_cache__ = []

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None):
        """
        Forward pass through the network, to predict a trajectory (or batch of trajectories)
        To perform the prediction (done in the decoder) at time t+1, the input will be the LSTM prediction at t, plus
        the SparseMotionFields prediction at t. The objective is for the LSTM network to learn which one weighs the
        most in particular cases.
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory. Should be absolute
        coordinates, because it's what Motion fields must receive.
        :param pred_len: Indicates for how long should prediction be made
        :param pred_traj_gt: (NOT SUPPORTED HERE) Tensor of shape (pred_len, batch, 2).
        Ground truth prediction data, used to enable teacher forcing. May be relative displacement or absolute.
        :return: Tensor of shape (pred_len, batch, output_shape). The predicted trajectories (or distributions)
        """
        assert pred_len > 0 and not pred_traj_gt, \
            'Teacher forcing is not available for this model. Please disable it'
        final_encoder_h, obs_traj, obs_traj_rel = self.__encoder_forward__(obs_traj)
        # displacement prediction via motion fields method
        tensor_in_cache = False
        pred_seq_fields = None
        for idx, t in enumerate(self.__obs_cache__):
            if t.shape == obs_traj.shape and torch.all(t == obs_traj):
                pred_seq_fields = self.__pred_cache__[idx]
                tensor_in_cache = True
                break
        if not tensor_in_cache:
            pred_seq_fields = self.motion_fields(obs_traj, pred_len, out_displacement=True)
            self.__obs_cache__.append(obs_traj)
            self.__pred_cache__.append(pred_seq_fields)
        # Perform prediction via LSTM decoder (may or may not use teacher forcing)
        # the additional info supplied will be the the prediction from SparseMotionFields model (except for last
        # instant, that does not serve a purpose here).
        pred_traj, _ = self.decoder(obs_traj_rel, final_encoder_h, pred_len, pred_traj_gt, pred_seq_fields[:-1, :, :])
        return pred_traj

    def __encoder_forward__(self, obs_traj):
        obs_traj_rel = torch.zeros_like(obs_traj)
        obs_traj_rel[1:, :, :] = obs_traj[1:, :, :] - obs_traj[:-1, :, :]
        if self.discard_zeros:
            obs_traj_rel = remove_zeros_from_traj(obs_traj_rel)
            # leave at least the last 2 positions for the observed trajectory
            obs_traj_start = min(obs_traj.shape[0] - obs_traj_rel.shape[0], obs_traj.shape[0] - 2)
            obs_traj = obs_traj[obs_traj_start:, :, :]
        # Encode sequence with LSTM encoder
        final_encoder_h = self.encoder(obs_traj_rel.clone())
        return final_encoder_h, obs_traj, obs_traj_rel


class FieldsWithLSTM(SimpleFieldsWithLSTM):
    def __init__(self, fields, feed_all=False, use_probs=False, embedding_dim=64, h_dim=64, num_layers=1, dropout=0,
                 activation_on_input_embedding=None, activation_on_output=None, normalize_embedding=False,
                 output_gaussian=False, discard_zeros=False):
        """
        see models.lstm.lstm.VanillaLstmEncDec for description of the rest of the parameters
        Initialization of LSTM model that also includes the Sparse Motion Fields model in it
        :param fields: Motion fields, assumed to be of class models.fields.sparse_motion_fields.SparseMotionFields
        :param feed_all: whether or not to feed to the LSTM all Motion Fields predictions, instead of just the most
        probable one
        :param use_probs: (only if feed_all=True) use probabilities also, to weigh each prediction.
        """
        super(FieldsWithLSTM, self).__init__(fields, embedding_dim, h_dim, num_layers, dropout,
                                             activation_on_input_embedding, activation_on_output, normalize_embedding,
                                             output_gaussian, discard_zeros)
        # override the decoder for one that uses the sparse motion fields model
        self.decoder = DecoderWithFields(fields, feed_all, use_probs, embedding_dim, h_dim, num_layers, dropout,
                                         activation_on_input_embedding, activation_on_output, normalize_embedding,
                                         output_gaussian)

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None):
        """
        forward pass, to compute the predicted trajectory
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory
        (in absolute coordinates)
        :param pred_len: how many steps to perform prediction
        :param pred_traj_gt: here for cross-compatibility with super class - NOT USED
        :return: pred_traj - tensor of shape (pred_len, batch, output_shape); where output_shape is usually 2 in case of
        a 2D trajectory, or 5 in case of parameters of a bi-variate gaussian distribution
        """
        assert pred_len > 0 and not pred_traj_gt, \
            'Teacher forcing is not available for this model. Please disable it'
        final_encoder_h, obs_traj, obs_traj_rel = self.__encoder_forward__(obs_traj)
        pred_traj, _ = self.decoder(obs_traj_rel, final_encoder_h, pred_len)
        return pred_traj


class DecoderWithFields(Decoder):
    """
    LSTM decoder class to perform trajectory forecast. This is an extension to the standard models.lstm.lstm.Decoder
    class, integrating motion fields prediction directly into it + using this prediction as input to the motion fields
    """

    def __init__(self, fields, feed_all=False, use_probs=False, embedding_dim=64, h_dim=64, num_layers=1, dropout=0,
                 activation_on_input_embedding=None, activation_on_output=None, normalize_embedding=False,
                 output_gaussian=False):
        """
        See models.lstm.lstm.Decoder for the description of the other parameters
        :param fields: the motion fields to use with the decoder
        :param feed_all: whether or not to feed to the LSTM all Motion Fields predictions, instead of just the most
        probable one
        """
        super(DecoderWithFields, self).__init__(embedding_dim, h_dim, num_layers, dropout,
                                                activation_on_input_embedding, activation_on_output,
                                                normalize_embedding, extra_info=True, output_gaussian=output_gaussian)
        self.motion_fields = fields
        self.feed_all = feed_all
        self.use_probs = use_probs
        # fields_embedding - only used if self.feed_all is active
        if self.feed_all:
            self.fields_embedding = self.__change_input_embedding_dims__(
                self.motion_fields.Te.shape[2] * self.motion_fields.Te.shape[0], int(embedding_dim / 2))
            self.input_embedding = self.__change_input_embedding_dims__(POS_2D, int(embedding_dim / 2))
            return
        else:
            self.fields_embedding = None

    def forward(self, obs_traj, state_final, pred_len=-1, pred_traj_gt=None, extra_info=None):
        """
        Forward propagation to predict a trajectory, using the prior LSTM prediction and state, as well the
        prediction(s) from SMF model, and then feeding the SMF with the LSTM prediction, in a sort of correcting the
        motion fields prediction, that may be erroneous.
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory
        (in absolute coordinates)
        :param state_final: the state obtained from the encoder - each hh of shape (num_layers, batch, h_dim)
        :param pred_len: how many steps to perform prediction
        :param pred_traj_gt: here for cross-compatibility with super class - NOT USED
        :param extra_info: here for cross-compatibility with super class - NOT USED
        :return: pred_traj - tensor of shape (pred_len, batch, output_shape); where output_shape is usually 2 in case of
        a 2D trajectory, or 5 in case of parameters of a bi-variate gaussian distribution
            WARNING: use of bi-variate gaussian not tested here
        """
        assert pred_len > 0, "The pred_len argument must be > 0 (teacher forcing not supported)"
        batch = obs_traj.size(1)
        with torch.no_grad():
            displacement, alpha_total, active_field_total = \
                self.motion_fields(obs_traj, 1, out_displacement=True, return_extra=True, all_fields=self.feed_all)
        # remove the considerations related to the past - not useful here
        alpha = alpha_total[obs_traj.shape[0] - 1, :, :]
        # active_field = active_field_total[obs_traj.shape[0] - 1, :]  # not used
        displacement = displacement.squeeze(dim=0)
        if self.feed_all:
            # tensor of shape (pred_len, batch, nK, 2)      where nK is the number of different types of motion fields
            fields_displacement = torch.full_like(displacement, float('nan'), dtype=alpha.dtype,
                                                  device=obs_traj.device).unsqueeze(0).repeat(pred_len, 1, 1, 1)
        else:
            fields_displacement = torch.full((pred_len, batch, obs_traj.shape[2]), float('nan'),
                                             dtype=alpha.dtype, device=obs_traj.device)
        seq_mask = (torch.isnan(alpha[:, 0]) + torch.isnan(obs_traj[-1, :, 0])) == 0
        alpha = alpha[seq_mask]
        mask_index = [i for i, m in enumerate(seq_mask) if m]
        fields_displacement[0] = displacement
        state = (state_final, torch.zeros_like(state_final))
        pred_traj = []
        # get first predicted position from state (regarding last observed position)
        # the 'state_final[-1, ...' is because if the LSTM network has multiple layers, we only want the output
        #   from the final LSTM layer; the 'seq_mask' is to not have cases that don't have the last observed position
        pos_partial = self.hidden2pos(state_final[-1, seq_mask, :].view(-1, self.h_dim))
        pos = torch.full((obs_traj.shape[1], pos_partial.shape[1]), float('nan'), device=obs_traj.device)
        pos[seq_mask] = pos_partial
        pred_traj.append(pos)
        pos_abs = pos_partial[:, :2] + obs_traj[-1, seq_mask]
        for t in range(1, pred_len):
            if self.feed_all:
                # pass the LSTM prediction and the motion fields predictions by separate embeddings, and then concat
                pos_embedded = self.input_embedding(pos_partial[:, :2])
                # .clone() is necessary here because torch.autograd detects change. Results in RuntimeError: one of the
                # variables needed for gradient computation has been modified by an inplace operation
                if self.use_probs:
                    displacement_times_probabilities = fields_displacement[t - 1, seq_mask] \
                                                       * alpha.unsqueeze(2).repeat(1, 1, 2)
                    fields_embedded = self.fields_embedding(
                        displacement_times_probabilities.float().view(
                            fields_displacement[seq_mask].shape[1], -1).clone())
                else:
                    fields_embedded = self.fields_embedding(
                        fields_displacement[t - 1, seq_mask].view(fields_displacement[:, seq_mask].shape[1],
                                                                  -1).float().clone())
                decoder_input = torch.cat((pos_embedded, fields_embedded), dim=1)
            else:
                # the ':2' is used in case the tensor has more info (e.g. when output_gaussian in __init__ is True)
                embedding_input = torch.cat((pos_partial[:, :2], fields_displacement[t - 1, seq_mask]), dim=1)
                decoder_input = self.input_embedding(embedding_input)
            decoder_input = decoder_input.view(1, pos_partial.shape[0], self.embedding_dim)
            if self.normalize_embedding:
                decoder_input = fun.normalize(decoder_input, p=2, dim=0)
            state_masked = [
                torch.stack([h for m, h in zip(seq_mask,
                                               state[0].permute(1, 0, 2)) if m], dim=0).permute(1, 0, 2),
                torch.stack([c for m, c in zip(seq_mask,
                                               state[1].permute(1, 0, 2)) if m], dim=0).permute(1, 0, 2)
            ]
            output, state_masked = self.decoder(decoder_input, state_masked)
            # update the motion fields prediction using LSTM prediction
            with torch.no_grad():
                _, fields_displacement[t, seq_mask], alpha, _ = \
                    self.motion_fields.single_prediction(pos_abs, alpha, out_displacement=True,
                                                         normalize_pos=True, all_fields=self.feed_all)
            pos_partial = self.hidden2pos(output.view(-1, self.h_dim))
            state_out = (state[0].clone(), state[1].clone())
            pos = torch.full((obs_traj.shape[1], pos_partial.shape[1]), float('nan'), device=obs_traj.device)
            # unmask [Update hidden-states and output]
            for j, h, c, o in zip(mask_index, state_masked[0].permute(1, 0, 2), state_masked[1].permute(1, 0, 2),
                                  pos_partial):
                state_out[0][:, j, :] = h
                state_out[1][:, j, :] = c
                pos[j] = o
            state = state_out
            pos_abs += pos_partial[:, :2]
            pred_traj.append(pos)
        pred_traj = torch.stack(pred_traj, dim=0)

        return pred_traj, state[0]
