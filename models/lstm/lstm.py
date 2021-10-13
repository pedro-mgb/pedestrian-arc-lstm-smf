"""
Created on 06/March/2021

Contains an implementation of a vanilla LSTM network for trajectory forecast.

"""
import torch
from torch import nn
import torch.nn.functional as fun

# just x,y values
from models.utils.utils import remove_zeros_from_traj

POS_2D = 2
# just x,y values
POS_AND_EXTRA_2D = 4
# Bi-variate Gaussian for velocity, with 5 values: mean in x,y; std in x,y; correlation factor
VELOCITY_GAUSSIAN = 5


class LSTMWithVariableInputEmbedding(nn.Module):
    """
    DO NOT USE THIS CLASS. ONLY USE ITS IMPLEMENTATIONS
    """

    def __change_input_embedding_dims__(self, in_features, out_features):
        """
        Create a new layer for input embedding, with different dimensions for input and output
        :param in_features: the dimension of the input (remember that the layer can output a batch of input)
        :param out_features: the dimension of the output
        :return: a new input embedding layer (without changing the original one)
        """
        if isinstance(self.input_embedding, nn.Linear):
            return nn.Linear(in_features, out_features, bias=self.input_embedding.bias)
        # else - it is torch.nn.Sequential, with first pass being the Linear layer; the rest of the layers are the same
        return nn.Sequential(nn.Linear(in_features, out_features, bias=self.input_embedding[0].bias is not None),
                             *self.input_embedding[1:])


class VanillaLSTM(LSTMWithVariableInputEmbedding):
    def __init__(self, embedding_dim=64, h_dim=64, activation_on_input_embedding=None, activation_on_output=None,
                 dropout=0, history_on_pred=False, normalize_embedding=False, output_gaussian=False,
                 discard_zeros=False):
        """
        Initializes the VanillaLSTM model. This model does not use nor learn any scene-specific information (at least
        no such information is provided explicitly). Important modules:
            - self.input_embedding: layer that embeds the input (usually a position/velocity)
            - self.dropout: Dropout layer with a certain probability (supplied as input)
            - self.lstm: An LSTM cell, receiving and embedded input, plus state of previous time step, and outputs the
            state at the next time step. Search about nn.LSTMCell for more information.
            - self.output_embedding: Performs a sort of 'decoding' of the lstm state at certain time step to obtain a
            prediction in the next time step. Depending on the dimensions, this prediction can be a position/velocity,
            or parameters of a probability distribution (e.g. bi-variate gaussian)
        :param embedding_dim: dimension of the embedded input to use (goes from 2 to embedding_dim)
        :param h_dim: dimension of the hidden state tensor, h
        :param activation_on_input_embedding: activation on the input embedding layer, should be from torch.nn (e.g.
        torch.nn.ReLu)
        :param activation_on_output: activation on the output layer (to obtain a position/distribution), should be from
        torch.nn (e.g. torch.nn.ReLu)
        :param dropout: regularization - dropout probability (0 if not meant to use dropout)
        :param history_on_pred: whether to feed a history of the past (observed trajectory) as input, instead of just
        the previous instant
        :param normalize_embedding: if True, will normalize what comes out of self.input_embedding, in terms of having
        L2 norm equal to 1
        :param output_gaussian: if True, instead of outputting 2 values, will output 5, as parameters of a bi-variate
        gaussian distribution.
        :param discard_zeros: Whether to discard parts of the observed trajectory when zeros are received, or not
        """
        super(VanillaLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim

        self.normalize_embedding = normalize_embedding

        self.use_history = history_on_pred
        self.input_dim = POS_AND_EXTRA_2D if history_on_pred else POS_2D
        self.output_dim = VELOCITY_GAUSSIAN if output_gaussian else POS_2D
        self.discard_zeros = discard_zeros

        if activation_on_input_embedding is None:
            self.input_embedding = nn.Linear(self.input_dim, embedding_dim)
        else:
            self.input_embedding = nn.Sequential(nn.Linear(self.input_dim, embedding_dim),
                                                 activation_on_input_embedding)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTMCell(embedding_dim, h_dim)
        if activation_on_output is None:
            self.output_embedding = nn.Linear(h_dim, self.output_dim)
        else:
            self.output_embedding = nn.Sequential(nn.Linear(h_dim, self.output_dim), activation_on_output)

    def init_hidden(self, batch, device):
        """
        Initializes the hidden state tensor used as input to the LSTM encoder
        :param batch: size of the batch being used
        :param device: device for the tensor to be mapped (e.g.: cpu, cuda)
        :return: the hidden state tensor of shape 2*(batch, self.h_dim); containing both the hidden
        state and the (inner) cell state of the LSTM, on this order
        """
        return (torch.zeros(batch, self.h_dim, device=device),
                torch.zeros(batch, self.h_dim, device=device))

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None):
        """
        forward pass through the network, to predict a trajectory (or batch of trajectories)
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory. May be relative
        trajectory (displacement) or absolute.
        :param pred_len: (when gt not available) Indicates for how long should prediction be made
        :param pred_traj_gt: (should only be used in training) Ground truth prediction data, used to enable teacher
        forcing. May be relative displacement or absolute. The processing method depends on flag self.use_abs
        :return: Tensor of shape (pred_len, batch, output_shape). The predicted trajectories (or distributions)
        """
        assert pred_len > 0 or pred_traj_gt is not None, \
            "Either the pred_len argument must be > 0, or you must supply ground truth prediction trajectories " \
            "(enabling teacher forcing) in order for the LSTM to actually do a forward pass"
        if self.discard_zeros:
            obs_traj = remove_zeros_from_traj(obs_traj)
        batch = obs_traj.size(1)
        state = self.init_hidden(batch, obs_traj.device)
        # process observed trajectory, except for last position
        # for observed trajectory, the history in terms of positions is not used
        empty_history_obs = torch.zeros_like(obs_traj[0, :, :])
        for positions in obs_traj[:-1]:
            seq_mask = torch.isnan(positions[:, 0]) == 0
            state_masked = [
                torch.stack([h for m, h in zip(seq_mask, state[0]) if m], dim=0),
                torch.stack([c for m, c in zip(seq_mask, state[1]) if m], dim=0),
            ]
            if self.use_history:
                input_with_history = torch.cat((positions[seq_mask, :], empty_history_obs[seq_mask, :]), dim=1)
                embedded = self.input_embedding(input_with_history)
            else:
                embedded = self.input_embedding(positions[seq_mask, :])
            if self.normalize_embedding:
                embedded = fun.normalize(embedded, p=2, dim=1)
            embedded = self.dropout(embedded)
            state_masked = self.lstm(embedded, state_masked)
            state_out = (state[0].clone(), state[1].clone())
            mask_index = [i for i, m in enumerate(seq_mask) if m]
            # unmask [Update hidden-states and output]
            for i, h, c in zip(mask_index, state_masked[0], state_masked[1]):
                state_out[0][i] = h
                state_out[1][i] = c
            state = state_out
        positions = obs_traj[-1]
        history_obs = torch.mean(torch.abs(obs_traj), dim=0)
        if pred_len > 0:
            pred_traj = torch.zeros(pred_len, batch, self.output_dim, device=obs_traj.device)
            for i in range(pred_len):
                seq_mask = torch.isnan(positions[:, 0]) == 0
                state_masked = [
                    torch.stack([h for m, h in zip(seq_mask, state[0]) if m], dim=0),
                    torch.stack([c for m, c in zip(seq_mask, state[1]) if m], dim=0),
                ]
                # the ':2' is used in case the tensor has more info (e.g. when output_gaussian in __init__ is True)
                if self.use_history:
                    input_with_history = torch.cat((positions[seq_mask, :2],
                                                    empty_history_obs[seq_mask, :] if i == 0 else history_obs[seq_mask,
                                                                                                  :]), dim=1)
                    embedded = self.input_embedding(input_with_history)
                else:
                    embedded = self.input_embedding(positions[seq_mask, :2])
                if self.normalize_embedding:
                    embedded = fun.normalize(embedded, p=2, dim=1)
                embedded = self.dropout(embedded)
                state_masked = self.lstm(embedded, state_masked)
                positions_temp = self.output_embedding(state_masked[0])
                mask_index = [j for j, m in enumerate(seq_mask) if m]
                state_out = (state[0].clone(), state[1].clone())
                # unmask [Update hidden-states and output]
                for j, h, c, o in zip(mask_index, state_masked[0], state_masked[1], positions_temp):
                    state_out[0][j] = h
                    state_out[1][j] = c
                    positions[j] = o
                pred_traj[i] = positions
                state = state_out
        else:
            # use ground truth - teacher forcing - no normalization nor use of history
            pred_traj = torch.zeros(pred_traj_gt.shape[0], batch, self.output_dim, device=obs_traj.device)
            positions_pred = torch.full_like(positions, float('nan'), device=positions.device)
            for i, positions in enumerate(pred_traj_gt):
                seq_mask = torch.isnan(positions[:, 0]) == 0
                state_masked = [
                    torch.stack([h for m, h in zip(seq_mask, state[0]) if m], dim=0),
                    torch.stack([c for m, c in zip(seq_mask, state[1]) if m], dim=0),
                ]
                embedded = self.input_embedding(positions[seq_mask, :2])
                state_masked = self.lstm(embedded, state_masked)
                positions_temp = self.output_embedding(state_masked[0])
                mask_index = [j for j, m in enumerate(seq_mask) if m]
                state_out = (state[0].clone(), state[1].clone())
                # unmask [Update hidden-states and output]
                for j, h, c, o in zip(mask_index, state_masked[0], state_masked[1], positions_temp):
                    state_out[0][j] = h
                    state_out[1][j] = c
                    positions_pred[j] = o
                pred_traj[i] = positions_pred
                state = state_out
        return pred_traj


class VanillaLstmEncDec(nn.Module):
    """
    Model similar to VanillaLSTM.
    The big difference here is the use of a encoder-decoder architecture. This means that there is are two separate
    LSTM networks (each one has their own weights). The Encoder processes the past trajectory, and the Decoder outputs
    the predicted trajectory.
    """

    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0, activation_on_input_embedding=None,
                 activation_on_output=None, extra_info=False, normalize_embedding=False, output_gaussian=False,
                 discard_zeros=False):
        """
        For definition on the fields and parameters, refer to class VanillaLSTM
        :param extra_info: equivalent to history_on_pred parameter, but this one can have a different meaning in classes
        extending this one.
        """
        super(VanillaLstmEncDec, self).__init__()

        self.embedding_dim = embedding_dim
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.discard_zeros = discard_zeros
        self.extra_info = extra_info

        self.encoder = Encoder(embedding_dim, h_dim, num_layers, dropout, activation_on_input_embedding,
                               normalize_embedding)
        self.decoder = Decoder(embedding_dim, h_dim, num_layers, dropout, activation_on_input_embedding,
                               activation_on_output, normalize_embedding, extra_info, output_gaussian)

    def forward(self, obs_traj, pred_len=-1, pred_traj_gt=None):
        """
        forward pass through the network, to predict a trajectory (or batch of trajectories)
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory. May be relative
        trajectory (displacement) or absolute. The processing method depends on flag self.use_abs
        :param pred_len: (when gt not available) Indicates for how long should prediction be made
        :param pred_traj_gt: (should only be used in training) Tensor of shape (pred_len, batch, 2).
        Ground truth prediction data, used to enable teacher forcing. May be relative displacement or absolute.
        :return: Tensor of shape (pred_len, batch, output_shape). The predicted trajectories (or distributions)
        """
        if self.discard_zeros:
            obs_traj = remove_zeros_from_traj(obs_traj)
        # Encode sequence with LSTM encoder
        final_encoder_h = self.encoder(obs_traj)
        # Perform prediction via LSTM decoder (may or may not use teacher forcing)
        pred_traj, _ = self.decoder(obs_traj, final_encoder_h, pred_len, pred_traj_gt)
        return pred_traj


class Encoder(LSTMWithVariableInputEmbedding):
    """
    Embeds an observed trajectory into a higher dimension vector and processes it to retrieve a state vector, that on
    the last instant possesses information regarding the history of the observed trajectory. This module consists of:
    - An embedding layer (linear), to embed the 2D positions into a higher dimension
    - The actual encoder, which is an LSTM network that together with a state tensor, outputs the state in the next
    iteration. The nn.LSTM module is used to process a full trajectory (rather than one by one like nn.LSTMCell).
    """

    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0, activation=None, normalize_embedding=False):
        """
        creates the LSTM encoder module
        :param embedding_dim: dimension of the embedded input to use (goes from 2 to embedding_dim)
        :param h_dim: dimension of the hidden state tensor, h
        :param num_layers: Number of recurrent layers. Use value > 1 if meant to use a stacked LSTM
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
        with dropout probability equal to this value
        """
        super(Encoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.normalize_embedding = normalize_embedding

        if activation is None:
            self.input_embedding = nn.Linear(2, embedding_dim)
        else:
            self.input_embedding = nn.Sequential(nn.Linear(2, embedding_dim), activation)

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

    def init_hidden(self, batch, device):
        """
        Initializes the hidden state tensor used as input to the LSTM encoder
        :param batch: size of the batch being used
        :param device: device for the tensor to be mapped (e.g.: cpu, cuda)
        :return: the hidden state tensor of shape 2*(self.num_layers, batch, self.h_dim); containing both the hidden
        state and the (inner) cell state of the LSTM, on this order
        """
        return (
            torch.zeros(self.num_layers, batch, self.h_dim, device=device),
            torch.zeros(self.num_layers, batch, self.h_dim, device=device)
        )

    def forward(self, obs_traj):
        """
        Forward pass through the module - encode the observed trajectory in a hidden state h
        :param obs_traj: Tensor of shape (obs_len, batch, 2). The observed trajectory (may be of displacements)
        :return: final state - Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        obs_len = obs_traj.size(0)
        seq_mask = ~torch.isnan(obs_traj[:, :, 0])
        batch = obs_traj.size(1)
        initial_state = self.init_hidden(batch, obs_traj.device)
        if torch.all(seq_mask):
            # No partial trajectories - can perform the computations directly
            batch = obs_traj.size(1)
            obs_traj_embedding = self.input_embedding(torch.reshape(obs_traj, (obs_len, batch, 2)))
            obs_traj_embedding = obs_traj_embedding.view(obs_len, batch, self.embedding_dim)
            if self.normalize_embedding:
                obs_traj_embedding = fun.normalize(obs_traj_embedding, p=2, dim=0)
            output, state = self.encoder(obs_traj_embedding, initial_state)
            return state[0]
        else:
            # need to discard where there are no positions (NaN) for the partial trajectories - via sequence mask
            state = initial_state
            for t in range(obs_len):
                position = obs_traj[t, :, :]
                position_in = position[seq_mask[t], :]
                obs_traj_embedding = self.input_embedding(position_in)
                if self.normalize_embedding:
                    obs_traj_embedding = fun.normalize(obs_traj_embedding, p=2, dim=0)
                state_masked = [
                    torch.stack([h for m, h in zip(seq_mask[t],
                                                   state[0].permute(1, 0, 2)) if m], dim=0).permute(1, 0, 2),
                    torch.stack([c for m, c in zip(seq_mask[t],
                                                   state[1].permute(1, 0, 2)) if m], dim=0).permute(1, 0, 2)
                ]
                # the unsqueeze is to re-obtain the time dimension (single instant)
                _, state_masked = self.encoder(obs_traj_embedding.unsqueeze(0), state_masked)
                mask_index = [i for i, m in enumerate(seq_mask[t]) if m]
                state_out = (state[0].clone(), state[1].clone())
                # unmask [Update hidden-states and output]
                for i, h, c in zip(mask_index, state_masked[0].permute(1, 0, 2), state_masked[1].permute(1, 0, 2), ):
                    state_out[0][:, i] = h
                    state_out[1][:, i] = c
                state = state_out
            return state[0]


class Decoder(LSTMWithVariableInputEmbedding):
    """
    LSTM-Based Decoder for an LSTM network of Encoder-Decoder architecture.
    The Decoder will be responsible for building the actual predicted trajectory.
    The module is composed of:
    - An embedding layer for the positions, similar to Encoder
    - A LSTM network, similar to Encoder
    - An output (linear/affine) layer to extract a prediction from the state.
    """

    def __init__(self, embedding_dim=64, h_dim=64, num_layers=1, dropout=0, activation_on_input_embedding=None,
                 activation_on_output=None, normalize_embedding=False, extra_info=False, output_gaussian=False):
        """
        To see what each parameter is, see __init__ on class VanillaLstmEncDec
        """
        super(Decoder, self).__init__()

        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.extra = extra_info
        self.input_dim = POS_AND_EXTRA_2D if extra_info else POS_2D
        self.output_dim = VELOCITY_GAUSSIAN if output_gaussian else POS_2D

        self.normalize_embedding = normalize_embedding

        if activation_on_input_embedding is None:
            self.input_embedding = nn.Linear(self.input_dim, embedding_dim)
        else:
            self.input_embedding = nn.Sequential(nn.Linear(self.input_dim, embedding_dim),
                                                 activation_on_input_embedding)

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if activation_on_output is None:
            self.hidden2pos = nn.Linear(h_dim, self.output_dim)
        else:
            self.hidden2pos = nn.Sequential(nn.Linear(h_dim, self.output_dim), activation_on_output)

    def forward(self, obs_traj, state_final, pred_len=-1, pred_traj_gt=None, extra_info=None):
        """
        Forward pass through the module to decode the hidden state - generates the prediction, one instant at a time
        :param obs_traj: Tensor of shape (obs_len, batch, 2) containing the observed trajectory. May be relative
        trajectory (displacement) or absolute.
        :param state_final: hh each tensor of shape (num_layers, batch, h_dim)
        :param pred_len: length of the sequence - indicates how long should the decoder forward pass goes for
        :param pred_traj_gt: Tensor of shape (pred_len, batch, 2). Ground truth data - to use for teacher forcing
        :param extra_info: Tensor of shape (pred_len, batch, 2). Extra information, to be used if parameter extra info
        was supplied (if nothing is sent here, a default kind of extra info is used - history of past positions)
        :return: pred_traj - tensor of shape (pred_len, batch, output_shape); where output_shape is usually 2 in case of
        a 2D trajectory, or 5 in case of parameters of a bi-variate gaussian distribution
        """
        assert pred_len > 0 or pred_traj_gt is not None, \
            "Either the pred_len argument must be > 0, or you must supply ground truth prediction trajectories " \
            "(enabling teacher forcing) in order for the LSTM to actually do a forward pass"
        state = (state_final, torch.zeros_like(state_final))
        seq_mask_o = ~torch.isnan(obs_traj[:, :, 0])
        seq_mask = seq_mask_o[-1, :]
        mask_index = [i for i, m in enumerate(seq_mask) if m]
        if pred_len > 0:
            pred_traj = []
            # get first predicted position from state (regarding last observed position)
            # -> the 'state_final[-1, :, :]' is used because if the LSTM network has multiple layers, we only want the
            # output from the final LSTM layer
            pos_partial = self.hidden2pos(state_final[-1, seq_mask, :].view(-1, self.h_dim))
            pos = torch.full_like(obs_traj[-1], float('nan'), device=obs_traj.device)
            # unmask [Update hidden-states and output]
            for i, p in zip(mask_index, pos_partial):
                pos[i] = p
            pred_traj.append(pos)
            if extra_info is None:
                # by default the extra information used is the mean of the past velocities, in module
                extra_info = torch.mean(torch.abs(obs_traj[seq_mask_o]), dim=0).unsqueeze(0).repeat(pred_len - 1, 1, 1)
            for t in range(1, pred_len):
                seq_mask = ~torch.isnan(pos[:, 0])
                # the ':2' is used in case the tensor has more info (e.g. when output_gaussian in __init__ is True)
                if self.extra:
                    embedding_input = torch.cat((pos[seq_mask, :2], extra_info[t - 1, seq_mask, :]), dim=1)
                else:
                    embedding_input = pos[seq_mask, :2]
                decoder_input = self.input_embedding(embedding_input)
                batch_at_t = torch.sum(seq_mask)
                decoder_input = decoder_input.view(1, decoder_input.shape[0], self.embedding_dim)
                if self.normalize_embedding:
                    decoder_input = fun.normalize(decoder_input, p=2, dim=0)
                state_masked = [
                    torch.stack([h for m, h in zip(seq_mask,
                                                   state[0].permute(1, 0, 2)) if m], dim=0).permute(1, 0, 2),
                    torch.stack([c for m, c in zip(seq_mask,
                                                   state[1].permute(1, 0, 2)) if m], dim=0).permute(1, 0, 2)
                ]
                output, state_masked = self.decoder(decoder_input, state_masked)
                pos_partial = self.hidden2pos(output.view(-1, self.h_dim))
                pos = torch.full_like(obs_traj[-1], float('nan'), device=obs_traj.device)
                state_out = (state[0].clone(), state[1].clone())
                # unmask [Update hidden-states and output]
                for j, h, c, o in zip(mask_index, state_masked[0].permute(1, 0, 2), state_masked[1].permute(1, 0, 2),
                                      pos_partial):
                    state_out[0][:, j, :] = h
                    state_out[1][:, j, :] = c
                    pos[j] = o
                state = state_out
                pred_traj.append(pos)
            pred_traj = torch.stack(pred_traj, dim=0)
        else:
            # TEACHER FORCING NOT AVAILABLE WITH PARTIAL TRAJECTORIES
            traj = torch.cat((obs_traj[-1].unsqueeze(0), pred_traj_gt), dim=0)
            decoder_input = self.input_embedding(traj)
            if self.normalize_embedding:
                decoder_input = fun.normalize(decoder_input, p=2, dim=0)
            output, state = self.decoder(decoder_input, state)
            pred_traj = self.hidden2pos(output[1:, :, :])

        return pred_traj, state[0]
