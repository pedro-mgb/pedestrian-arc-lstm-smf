import numpy as np
import torch
from torch import nn

from models.interaction_modules.shape_based import PoolingShape


class ShapeConfigLSTM(nn.Module):
    """
    LSTM-based module that allows to learn what dimensions of the pooling shape to use. This module keeps a history
    of past pooling shape dimensions per pedestrian (this assumes it can vary once every instant), and uses it along
    with the provided social context to estimate the most probable shape dimensions at the next instant. This notion
    of selecting the most probable dimensions can be thought of as a classification problem, but since this network
    is not explicitly trained, it becomes a case of unsupervised learning.
    """

    def __init__(self, module_shape_values, out_dim, shape, values, embedding_dim=64, h_dim=64,
                 activation_on_input_embedding=None, dropout=0, random=False):
        """
        Initializes the LSTM-based shape configuration module
        :param module_shape_values: the shape-based interaction module values object (NOT the module itself)
        :param shape: the kind of shape (grid, arc) being employed
        :param values: list with several values for the possible different shape sizes. In case a shape is defined by
        more than one parameters (e.g. arc is defined by radius and spread<->angle), it expected a list of list is
        provided (with each sub-list containing all the values for a specific parameter). In the case of arc shape, two
        sub-lists should be provided, in this order: a list with arc radius values, and a list with spread values.
        :param embedding_dim: dimension of the embedded input to use (goes from 2 to embedding_dim)
        :param h_dim: dimension of the hidden state tensor, h
        :param activation_on_input_embedding: not actually used for this LSTM version. No activation function is used
        :param dropout: regularization - dropout probability (0 if not meant to use dropout)
        :param random: if meant to do random initialization of shape parameters or not. At the first instant of
        trajectory, a random value for each of the shape parameters is chosen
        """
        super(ShapeConfigLSTM, self).__init__()
        self.lstm = nn.LSTMCell(embedding_dim + out_dim, h_dim)
        self.module_shape_values = module_shape_values
        self.shape = shape
        self.embedding_dim, self.h_dim = embedding_dim, h_dim
        self.random_init = random
        if shape == PoolingShape.GRID:
            self.input_dim = 1  # side of the grid
            self.radii, self.angles = None, None
            self.sides = sorted(values[0])
            self.out_dim = len(self.sides)
            self.default_side = self.sides[int(len(self.sides) / 2)]
            self.default_side_idx = int(len(self.sides) / 2)
            if activation_on_input_embedding is None:
                self.input_embedding1 = nn.Embedding(len(self.sides), int(self.embedding_dim / 2))
            else:
                self.input_embedding1 = nn.Sequential(nn.Embedding(len(self.sides), int(self.embedding_dim / 2)),
                                                      activation_on_input_embedding)
        elif shape == PoolingShape.ARC:
            self.input_dim = 2  # radius and angle
            self.radii, self.angles = sorted(values[0]), sorted([a for a in values[1]])
            self.out_dim = len(self.radii) + len(self.angles)
            self.default_radius = self.radii[int(len(self.radii) / 2)]
            # angles from degrees to radians
            self.default_angle = self.angles[int(len(self.angles) / 2)] * np.pi / 180
            self.angles = [a * np.pi / 180 for a in values[1]]
            self.default_radius_idx, self.default_angle_idx = int(len(self.radii) / 2), int(len(self.angles) / 2)
            # normalizing angle values in [-pi, pi) interval; will not be done here - LSTM will process values from
            # 0 to 2pi (assuming angles in degrees are in 0 to 360 format)
            """
            self.angles = [(a + np.pi) % (2 * np.pi) - np.pi for a in values[1]]
            self.default_angle = (self.radii[-1] - self.radii[0]) / 2
            """
            if activation_on_input_embedding is None:
                self.input_embedding1 = nn.Embedding(len(self.radii), int(self.embedding_dim / 2))
                self.input_embedding2 = nn.Embedding(len(self.angles), int(self.embedding_dim / 2))
            else:
                self.input_embedding1 = nn.Sequential(nn.Embedding(len(self.radii), int(self.embedding_dim / 2)),
                                                      activation_on_input_embedding)
                self.input_embedding2 = nn.Sequential(nn.Embedding(len(self.angles), int(self.embedding_dim / 2)),
                                                      activation_on_input_embedding)
            self.sides = None
            # convert to pytorch tensor
            # self.radii, self.angles = torch.tensor(self.radii), sorted([a for a in values[1]])
        else:
            raise Exception('Pooling shapes available: GRID or ARC.')
        self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(self.h_dim, self.out_dim)
        self.softmax = nn.LogSoftmax(dim=1)  # outputs will be converted to probabilities using softmax

        # extra information defined in __init__, but only properly initialized through self.reset()
        self.state, self.input, self.best_shapes = None, None, None
        self.all_sides, self.all_radii, self.all_angles = None, None, None
        self.curr_step_pp_probs_s, self.probs_pp_s = None, None
        self.curr_step_pp_probs_r, self.curr_step_pp_probs_a, self.probs_pp_r, self.probs_pp_a = None, None, None, None
        self.curr_step_pp_idx, self.idx_pp, self.curr_num_pp, self.curr_pp = None, None, None, None

        self.input_idx, self.t, self.s, self.total_seqs = None, 0, 0, 0

    def reset(self, num_pedestrians, seq_start_end, device, input_idx=None):
        """
        Since this module stores the state and input to the cell in the object, it must be reset when meant to process
        a new batch of trajectories. This method should be called every time a new batch is processed, in order to
        reset the state, and the input values
        :param num_pedestrians: the number of pedestrians in the batch
        :param seq_start_end:
        :param device: the torch.device to map the pytorch tensors to
        :return: nothing
        """
        self.input_idx = input_idx
        # reset the hidden lstm state
        self.state = (torch.zeros(num_pedestrians, self.h_dim, device=device),
                      torch.zeros(num_pedestrians, self.h_dim, device=device))
        if self.shape == PoolingShape.GRID:
            if self.random_init or self.input_idx is not None:
                if self.random_init:
                    # random values for shape
                    s_idx = (torch.rand(num_pedestrians, device=device) * len(self.sides)).to(torch.long)
                else:
                    s_idx = (self.input_idx[0] / len(self.sides)).to(torch.long)
                self.input = s_idx.unsqueeze(1)
                self.all_sides = torch.tensor(self.sides, device=device).unsqueeze(0).repeat(
                    s_idx.shape[0], 1).gather(1, s_idx.unsqueeze(1)).squeeze(1)
            else:
                self.all_sides = torch.ones(num_pedestrians, device=device) * self.default_side
                self.input = torch.full((num_pedestrians, 1), self.default_side_idx, dtype=torch.long, device=device)
            self.best_shapes = self.all_sides.unsqueeze(1)
            self.curr_step_pp_probs_s = torch.torch.tensor([], device=device)
            self.probs_pp_s = torch.torch.tensor([], device=device)
        else:
            # shape == PoolingShape.ARC
            if self.random_init or self.input_idx is not None:
                if self.random_init:
                    # random values for shape
                    r_idx = (torch.rand(num_pedestrians, device=device) * len(self.radii)).to(torch.long)
                    a_idx = (torch.rand(num_pedestrians, device=device) * len(self.angles)).to(torch.long)
                else:
                    r_idx = (self.input_idx[0] / len(self.angles)).to(torch.long)
                    a_idx = (self.input_idx[0] % len(self.angles)).to(torch.long)
                self.input = torch.cat((r_idx.unsqueeze(1), a_idx.unsqueeze(1)), dim=1)
                self.all_radii = torch.tensor(self.radii, device=device).unsqueeze(0).repeat(
                    r_idx.shape[0], 1).gather(1, r_idx.unsqueeze(1)).squeeze(1)
                self.all_angles = torch.tensor(self.angles, device=device).unsqueeze(0).repeat(
                    a_idx.shape[0], 1).gather(1, a_idx.unsqueeze(1)).squeeze(1)
            else:
                self.all_radii = torch.ones(num_pedestrians, device=device) * self.default_radius
                self.all_angles = torch.ones(num_pedestrians, device=device) * self.default_angle
                self.input = torch.cat((
                    torch.full((num_pedestrians, 1), self.default_radius_idx, dtype=torch.long, device=device),
                    torch.full((num_pedestrians, 1), self.default_angle_idx, dtype=torch.long, device=device)), dim=1)
            self.best_shapes = torch.cat((self.all_radii.unsqueeze(1), self.all_angles.unsqueeze(1)), dim=1)
            self.curr_step_pp_probs_r, self.curr_step_pp_probs_a = torch.tensor([], device=device), \
                                                                   torch.tensor([], device=device)
            self.probs_pp_r, self.probs_pp_a = torch.torch.tensor([], device=device), \
                                               torch.torch.tensor([], device=device)

        # FOR TRAINING WITH A SECONDARY LOSS
        self.curr_step_pp_idx = torch.tensor([], device=device)
        self.idx_pp = torch.tensor([], device=device)
        self.curr_num_pp, self.curr_pp = seq_start_end.shape[0], 0

        self.t, self.s, self.total_seqs = 1, 0, seq_start_end.shape[0]

    def update_shape(self, indexes):
        """
        Update the shape size (by changing the parameters) of the pooling layer (self.module_shape_values), using the
        current values stored in this shape configuration module
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). To select the shape parameters only for those pedestrians
        :return: nothing
        """
        if self.shape == PoolingShape.GRID:
            self.module_shape_values.all_sides = self.all_sides[indexes]
        else:
            # shape == PoolingShape.ARC
            # update the parameters for the arc-based pooling - max radius and the angle (spread of the arc)
            self.module_shape_values.all_radius = self.all_radii[indexes]
            self.module_shape_values.all_angles = self.all_angles[indexes]

    def forward(self, pooling_in, indexes):
        """
        Forward pass through the LSTM shape configuration network. Use the current state and the prior shape to compute
        the most likely (via output linear layer + softmax to get probabilities) shape parameters or the next instant
        :param pooling_in: Tensor of shape [num_pedestrians, out_dim]. The (embedded) of the pooling layer,
        containing information regarding the presence and influence of the pedestrians neighbourhood.
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). Forward pass done only for those pedestrians
        :return: Nothing. The updated shape parameters are stored internally, and can be applied using self.update_shape
        """
        if self.state is None:
            raise Exception("The 'shape_config.reset()' method must be called prior to doing a forward pass")
        if self.shape == PoolingShape.GRID:
            emb = self.input_embedding1(self.input[indexes, :])
        else:
            # shape is arc - do radius and angle separately
            emb = torch.cat((self.input_embedding1(self.input[indexes, 0]),
                             self.input_embedding2(self.input[indexes, 1])), dim=1)
        state_indexed = [
            torch.stack([self.state[0][i].clone() for i in indexes], dim=0),
            torch.stack([self.state[1][i].clone() for i in indexes], dim=0),
        ]
        state_out = self.lstm(torch.cat((emb, pooling_in), dim=1), state_indexed)
        out = self.out_layer(state_out[0])
        if self.shape == PoolingShape.GRID:
            # TODO confirm if this is correct
            prob = self.softmax(out)
            if self.input_idx is not None and self.t < self.input_idx.shape[0]:
                # parameters coming directly from data
                s_idx = (self.input_idx[self.t] / len(self.sides)).to(torch.long)[indexes]
                self.s += 1
                if self.s >= self.total_seqs:
                    self.s = 0
                    self.t += 1
            else:
                s_idx = torch.argmax(prob, dim=1)
            best_side = torch.tensor(self.sides, device=out.device).unsqueeze(0).repeat(s_idx.shape[0], 1).gather(
                1, s_idx.unsqueeze(1)).squeeze(1)
            self.all_sides[indexes] = best_side
            self.input[indexes, 0] = s_idx
            self.curr_step_pp_probs_s = torch.cat(self.curr_step_pp_probs_s, prob[0:1])
        else:
            # shape == PoolingShape.ARC
            r, a = out[:, :len(self.radii)], out[:, len(self.radii):]
            r_prob, a_prob = self.softmax(r), self.softmax(a)  # probabilities for each of the parameters - [0, 1]
            if self.input_idx is not None and self.t < self.input_idx.shape[0]:
                # parameters coming directly from data
                # assuming a matrix where row index gives radius and column index gives angle
                r_idx = (self.input_idx[self.t] / len(self.angles)).to(torch.long)[indexes]
                a_idx = (self.input_idx[self.t] % len(self.angles)).to(torch.long)[indexes]
                self.s += 1
                if self.s >= self.total_seqs:
                    self.s = 0
                    self.t += 1
            else:
                r_idx, a_idx = torch.argmax(r_prob, dim=1), torch.argmax(a_prob, dim=1)  # most probable parameters
            best_r = torch.tensor(self.radii, device=r.device).unsqueeze(0).repeat(r_idx.shape[0], 1).gather(
                1, r_idx.unsqueeze(1)).squeeze(1)
            best_a = torch.tensor(self.angles, device=a.device).unsqueeze(0).repeat(a_idx.shape[0], 1).gather(
                1, a_idx.unsqueeze(1)).squeeze(1)
            # update the radii and angles to use for the pooling shape, and as input at the next instant
            self.all_radii[indexes] = best_r
            self.all_angles[indexes] = best_a
            # update input to this LSTM
            self.input[indexes, :] = torch.cat((r_idx.unsqueeze(1), a_idx.unsqueeze(1)), dim=1)
            # update debug info for LSTM network
            self.best_shapes[indexes, :] = torch.cat((best_r.unsqueeze(1), best_a.unsqueeze(1)), dim=1)
            self.curr_step_pp_probs_r = torch.cat((self.curr_step_pp_probs_r, r_prob[0:1]))
            self.curr_step_pp_probs_a = torch.cat((self.curr_step_pp_probs_a, a_prob[0:1]))
        # update state
        self.state[0][indexes] = state_out[0].clone()
        self.state[1][indexes] = state_out[1].clone()

        # RELEVANT FOR TRAINING THIS MODULE - UPDATE information for primary pedestrian
        # Assumed that it is one per forward pass
        self.curr_step_pp_idx = torch.cat((self.curr_step_pp_idx, self.input[indexes[0], :].unsqueeze(0)), dim=0)
        self.curr_pp += 1
        if self.curr_pp >= self.curr_num_pp:
            self.curr_pp = 0
            self.__update_shapes_next_step__()

    def __update_shapes_next_step__(self):
        # FOR DEBUG IN EVALUATION - update the retrieved shape parameters
        """

        :return:
        """
        '''
        self.idx_this_batch = torch.cat((self.idx_this_batch, self.best_shapes.unsqueeze(0).clone()), dim=0)
        self.params_pp = torch.cat((self.params_pp, self.curr_step_pp_params.unsqueeze(0).clone()), dim=0)
        # reset this value for the next time step
        self.curr_step_pp_params = torch.tensor([], device=self.params_pp.device)
        '''
        if self.shape == PoolingShape.GRID:
            self.probs_pp_s = torch.cat((self.probs_pp_s, self.curr_step_pp_probs_s.unsqueeze(0)))
            self.curr_step_pp_probs_s = torch.tensor([], device=self.probs_pp_s.device)
        else:
            # shape == PoolingShape.ARC
            self.probs_pp_r = torch.cat((self.probs_pp_r, self.curr_step_pp_probs_r.unsqueeze(0)))
            self.probs_pp_a = torch.cat((self.probs_pp_a, self.curr_step_pp_probs_a.unsqueeze(0)))
            self.curr_step_pp_probs_r = torch.tensor([], device=self.probs_pp_r.device)
            self.curr_step_pp_probs_a = torch.tensor([], device=self.probs_pp_a.device)
        self.idx_pp = torch.cat((self.idx_pp, self.curr_step_pp_idx.unsqueeze(0)))
        self.curr_step_pp_idx = torch.tensor([], device=self.idx_pp.device)