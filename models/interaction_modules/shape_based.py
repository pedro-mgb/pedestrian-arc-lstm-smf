"""
Created on May 19th 2021
"""
from enum import Enum

import numpy as np
import torch
from torch import nn


class PoolingArguments:
    """
    Contains several arguments (several are hyper-parameters) used to define the Pooling-Based architecture to
    incorporate social interactions in a data-driven model.
    To see what each of these parameters mean, see their help value (can be seen in parser_options.py)
    """

    def __init__(self, args, pooling_type, pooling_shape):
        """
        Builds the PoolingArguments object, aggregating all necessary arguments here
        :param args: Command line arguments with several options and hyperparameters regarding the Pooling Architecture
        :param pooling_type: The type of pooling to perform (or what information of the neighbours to use).
        Obtained from args.pooling_type
        :param pooling_shape: Shape of the pooling to use (e.g. grid). Obtained from args.pooling_shape
        """
        self.cell_size = args.cell_size
        self.n = args.grid_dim
        self.arc_radius = args.arc_radius if hasattr(args, 'arc_radius') else 4.0
        self.arc_angle = args.arc_angle if hasattr(args, 'arc_angle') and args.arc_angle > 0 else 140.0
        # value assumed to be in degrees - convert to radians
        self.arc_angle = np.clip(self.arc_angle * np.pi / 180, 0, np.pi * 2)
        self.n_r = args.n_radius if hasattr(args, 'n_radius') else 4
        self.n_a = args.n_angle if hasattr(args, 'n_radius') else 5
        self.behind_ped = args.single_arc_row_behind if hasattr(args, 'single_arc_row_behind') else False
        self.arc_radius_behind = args.arc_row_behind_radius if hasattr(args, 'arc_row_behind_radius') else 0
        self.out_dim = args.pooling_out_dim if hasattr(args, 'pooling_out_dim') and args.pooling_out_dim > 0 else None
        self.out_dim = args.lstm_h_dim if self.out_dim is None else self.out_dim
        self.pooling_type = pooling_type
        self.shape = pooling_shape
        # specific to directional pooling
        self.norm_pool = args.norm_pool if hasattr(args, 'norm_pool') else False


class ShapedBasedPoolingType(Enum):
    """
    Available types for pooling layer (in terms of what inputs are used; for shape - see PoolingShape class
    - OCCUPANCY: Simple occupancy pooling, where each cell of the grid indicates if there are pedestrians there (or not)
    - SOCIAL: Social Pooling, used on the popular Social-LSTM architecture - each cell of the grid
    - DIRECTIONAL: Directional Pooling - Instead of incorporating hidden LSTM state like the Social-LSTM model,
    will consider the relative velocities between pedestrians
    - DIRECTIONAL_POLAR: Same as DIRECTIONAL, but velocities are in polar coordinates (radius<->speed and
    angle<->direction) instead of cartesian (x and y components)
    """
    OCCUPANCY = 0
    OCCUPANCY_PERCENT = 6
    SOCIAL = 1
    DIRECTIONAL = 2
    DIRECTIONAL_POLAR = 3
    DISTANCE = 4
    DISTANCE_DIRECTIONAL = 5
    # SOCIAL_DIRECTIONAL = 99


class PoolingShape(Enum):
    """
    Available types of shapes for the Shaped based pooling class
    - Grid based pooling - Square grid surrounding the pedestrian. Used in models like Social-LSTM.
    - Arc based pooling - Consider an arc of a certain angle and maximum radius going from the direction where the
    pedestrian is facing (assumption that it's the gaze direction). The arc will be divided in several portions along
    the angle and distance.
    """
    GRID = 0
    ARC = 1


class ShapeValues:
    def __init__(self):
        # For variable grid-based shape
        self.all_sides = None
        # For variable arc-based shape
        self.all_radius, self.all_angles = None, None


class ShapeBasedPooling(nn.Module):
    """
    Implementation of Shape-Based Pooling types, with more than one option to choose from.
    CREDITS: These pooling type model were not created here, it was adapted from existing repositories:
    Main: https://github.com/vita-epfl/trajnetplusplusbaselines/
    Others:
        https://github.com/quancore/social-lstm/
        https://github.com/agrimgupta92/sgan/
    """

    def __init__(self, pooling_arguments, h_dim, embedding_activation=nn.ReLU(), include_occ=False):
        """
        Constructs the Grid Pooling, with total size (cell_size * n)
        :param pooling_arguments: object of type PoolingArguments
        :param h_dim: hidden state dimension, which by default will be the dimension of the output of this pooling
        layer. This applies for the case of social pooling
        :param embedding_activation: the activation function to use in input. By default ReLu will be used, which is
        also the same activation function used in the paper
        :param include_occ: include computation and embedding of standard occupancy (in case this layer uses
        non-occupancy pooling)
        """
        super(ShapeBasedPooling, self).__init__()

        self.args = pooling_arguments
        self.h_dim = h_dim
        self.out_dim = self.args.out_dim
        self.normalize = False  # to divide the values by the total number of neighbours in the WHOLE shape
        self.norm_cell = False  # to divide the values by the total number of neighbours PER cell in the shape
        self.include_occ = include_occ

        if self.args.pooling_type == ShapedBasedPoolingType.OCCUPANCY or \
                self.args.pooling_type == ShapedBasedPoolingType.OCCUPANCY_PERCENT:
            if self.args.pooling_type == ShapedBasedPoolingType.OCCUPANCY_PERCENT:
                self.normalize = True  # percent values - between 0 and 1
            self.pooling_dim = 1  # just has pedestrian or not
        elif self.args.pooling_type == ShapedBasedPoolingType.SOCIAL:
            """
            # Source: 
            https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/gridbased_pooling.py
            ## Encode hidden-dim into latent-dim vector (faster computation)
            self.hidden_dim_encoding = torch.nn.Linear(hidden_dim, latent_dim)
            self.pooling_dim = latent_dim
            """
            self.pooling_dim = self.h_dim
        elif self.args.pooling_type == ShapedBasedPoolingType.DIRECTIONAL or \
                self.args.pooling_type == ShapedBasedPoolingType.DISTANCE:
            # DIRECTIONAL
            self.pooling_dim = 2  # relative pedestrians
            self.norm_cell = True
        elif self.args.pooling_type == ShapedBasedPoolingType.DIRECTIONAL_POLAR:
            # DIRECTIONAL_POLAR
            self.pooling_dim = 3
        else:
            # DISTANCE_DIRECTIONAL
            self.pooling_dim = 4
            self.norm_cell = True
        if self.args.shape == PoolingShape.ARC:
            self.input_dim = self.args.n_r * self.args.n_a * self.pooling_dim
            self.input_dim_occ = self.args.n_r * self.args.n_a
        else:
            # == PoolingShape.GRID
            self.input_dim = self.args.n * self.args.n * self.pooling_dim
            self.input_dim_occ = self.args.n * self.args.n

        self.shape_values = ShapeValues()

        # there are alternatives to this embedding type, like multiple
        self.embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.out_dim),
            embedding_activation, )
        if self.include_occ:
            self.embedding_occ = nn.Sequential(
                nn.Linear(self.input_dim_occ, self.out_dim),
                embedding_activation, )
        else:
            self.embedding_occ = nn.Identity()

    def forward(self, h, positions, past_positions=None, embed=True):
        """
        Forward pass through the model, to compute the pooling context (through an embedding layer).
        Will pick the appropriate type of pooling
        :param h: Tensor of shape [num_peds, h_dim]. LSTM hidden state
        :param positions: Tensor of shape [num_peds, 2]. The current positions (absolute coordinates)
        :param past_positions: Tensor of shape [num_peds, 2]. The past positions (absolute coordinates)
        :param embed: whether to returned the pooled tensor through an embedding layer, or not
        :return: Tensor of shape [num_peds, self.out_dim]. Embedded pooled context
        """
        if self.args.pooling_type == ShapedBasedPoolingType.OCCUPANCY or \
                self.args.pooling_type == ShapedBasedPoolingType.OCCUPANCY_PERCENT:
            grid, o = self.occup(past_positions, positions)
        elif self.args.pooling_type == ShapedBasedPoolingType.SOCIAL:
            grid, o = self.social(h, past_positions, positions)
        else:
            # DIRECTIONAL or DIRECTIONAL POLAR or DISTANCE or DIRECTIONAL_DISTANCE
            grid, o = self.directional(past_positions, positions)
        if embed:
            return self.embedding(grid.reshape([grid.shape[0], np.prod(grid.shape[1:])])), \
                   self.embedding_occ(o.reshape([o.shape[0], np.prod(o.shape[1:])])) if self.include_occ else None
        else:
            return grid, o if self.include_occ else None

    def occup(self, past_pos, curr_pos):
        """
        Generate an occupancy pooling map for the provided positions of several pedestrians. This is a simpler version
        of the social pooling grid (below)
        :param past_pos: Tensor of shape [num_peds, 2]. The past positions (absolute coordinates)
        :param curr_pos: Tensor of shape [num_peds, 2]. The current positions (absolute coordinates), to compute the
        grid
        :return: Tensor of shape [num_peds, 1, n1, self.n1] - grid resulting of the occupancy pooling map. Still
        needs to go through an embedding layer
        """
        if self.args.shape == PoolingShape.ARC:
            vel = curr_pos - past_pos
            angle_vel = torch.atan2(vel[:, 1], vel[:, 0])
            return self.arc_pooling(curr_pos, angle_offsets=angle_vel)
        # else - GRID
        return self.grid_pooling(curr_pos, past_obs=past_pos)

    def social(self, hidden_state, past_pos, curr_pos):
        """
        Generate a social pooling map for the provided positions of several pedestrians. For each pedestrian, creates
        a grid where each cell has the contribution of states of neighbours present in that cell.
        :param hidden_state: Tensor of shape [num_peds, h_dim]. LSTM hidden state
        :param past_pos: Tensor of shape [num_peds, 2]. The past positions (absolute coordinates)
        :param curr_pos: Tensor of shape [num_peds, 2]. The current positions (absolute coordinates), to compute the
        grid
        :return: Tensor of shape [num_peds, h_dim, n1, n2] - grid resulting of the social pooling map. Still
        needs to go through an embedding layer
        """
        num_peds = curr_pos.size(0)

        # if only primary pedestrian present
        if num_peds == 1:
            if self.args.shape == PoolingShape.GRID:
                return self.grid_pooling(curr_pos, None, past_obs=past_pos)
            else:
                # PoolingShape.ARC
                return self.arc_pooling(curr_pos, None)

        # Source: Trajnet++ version of social LSTM
        # https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/gridbased_pooling.py
        # Generate values to input in hidden state grid tensor
        # [num_peds, hidden_dim] --> [num_peds, num_peds-1, hidden_dim]         (pooling_dim <=> h_dim)
        hidden_state_grid = hidden_state.repeat(num_peds, 1).view(num_peds, num_peds, -1)
        # eliminate entries of a pedestrian with respect to itself - remove diagonal
        hidden_state_grid = hidden_state_grid[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, -1)
        # (compressed hidden-states in this case) - NOT used here
        # hidden_state_grid = self.hidden_dim_encoding(hidden_state_grid)

        if self.args.shape == PoolingShape.ARC:
            vel = curr_pos - past_pos
            angle_vel = torch.atan2(vel[:, 1], vel[:, 0])
            return self.arc_pooling(curr_pos, hidden_state_grid, angle_offsets=angle_vel)
        # else - GRID Occupancy Map
        return self.grid_pooling(curr_pos, hidden_state_grid, past_obs=past_pos)

    def directional(self, past_pos, curr_pos):
        """
        Makes the Directional Pooling, considering relative displacements between pedestrians.
        :param past_pos: Tensor of shape [num_peds, 2]. The past positions (absolute coordinates)
        :param curr_pos: Tensor of shape [num_peds, 2]. The current positions (absolute coordinates)
        :return: Tensor of shape [num_peds, h_dim, n1, n2] - grid resulting of the social pooling map. Still
        needs to go through an embedding layer
        """
        num_peds = curr_pos.size(0)
        if num_peds == 1:
            if self.args.shape == PoolingShape.GRID:
                return self.grid_pooling(curr_pos, None, past_obs=past_pos)
            else:
                # PoolingShape.ARC
                return self.arc_pooling(curr_pos, None)
        # Generate values to input in directional grid tensor (relative velocities in this case)
        vel = curr_pos - past_pos
        if self.args.pooling_type == ShapedBasedPoolingType.DISTANCE:
            unfolded = curr_pos.unsqueeze(0).repeat(curr_pos.size(0), 1, 1)
            # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
            relative = unfolded - curr_pos.unsqueeze(1)
            # Deleting Diagonal (Ped wrt itself): [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks-1, 2]
            relative_in = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
        else:
            relative = self.compute_relative_vel(vel)
            if self.args.pooling_type == ShapedBasedPoolingType.DIRECTIONAL_POLAR:
                # DIRECTIONAL_POLAR - use polar coordinates (radius<->speed and angle<->direction)
                relative_in = torch.zeros(relative.shape[0], relative.shape[1], 3)
                relative_in[:, :, 0] = torch.norm(relative, dim=2)
                angles = torch.atan2(relative[:, :, 1], relative[:, :, 0])
                relative_in[:, :, 1] = torch.cos(angles)
                relative_in[:, :, 2] = torch.sin(angles)
            elif self.args.pooling_type == ShapedBasedPoolingType.DISTANCE_DIRECTIONAL:
                unfolded_pos = curr_pos.unsqueeze(0).repeat(curr_pos.size(0), 1, 1)
                # [num_tracks, 2] --> [num_tracks, num_tracks, 2]
                relative_pos = unfolded_pos - curr_pos.unsqueeze(1)
                # Deleting Diagonal (Ped wrt itself): [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks-1, 2]
                relative_pos = relative_pos[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
                relative_in = torch.cat((relative_pos, relative), dim=2)
            else:
                # DIRECTIONAL - use cartesian coordinates (x/y)
                relative_in = relative
        if self.args.shape == PoolingShape.ARC:
            angle_vel = torch.atan2(vel[:, 1], vel[:, 0])
            return self.arc_pooling(curr_pos, relative_in, angle_offsets=angle_vel)
        # else - GRID Occupancy Map
        return self.grid_pooling(curr_pos, relative_in, past_obs=past_pos)

    def compute_relative_vel(self, vel):
        """
        Normalize pooling grid along direction of pedestrian motion
        Adapted from (with vertical direction as reference, here we do along positive horizontal axis):
        https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/gridbased_pooling.py#L232
        :param vel: the velocities in (cartesian) world coordinates
        :return: the relative velocities, normalized across the direction of motion of each reference pedestrian
        """
        num_peds = vel.size(0)
        unfolded = vel.unsqueeze(0).repeat(num_peds, 1, 1)
        if self.args.norm_pool:
            orientation = torch.atan2(vel[:, 1], vel[:, 0])
            # clockwise rotation matrix - tensor of shape [num_peds, 2 , 2]
            rot_matrix = torch.stack([torch.stack([torch.cos(orientation), torch.sin(orientation)]),
                                      torch.stack([-torch.sin(orientation), torch.cos(orientation)])]).permute(2, 0, 1)
            rot_matrix_unfolded = rot_matrix.unsqueeze(1).repeat(1, num_peds, 1, 1)
            # perform 2D rotation - [num_peds, num_peds, 2, 2] * [num_peds, num_peds, 2] -> [num_peds, num_peds, 2]
            rotated_vel_unfolded = torch.einsum('uvxy,uvy->uvx', rot_matrix_unfolded, unfolded)
            ref_vel_idx = torch.arange(end=num_peds)  # index for each pedestrian velocity reference
            # [num_peds, 2] --> [num_peds, num_peds, 2]
            relative = rotated_vel_unfolded - rotated_vel_unfolded[ref_vel_idx, ref_vel_idx].unsqueeze(1)
        else:
            # [num_peds, 2] --> [num_peds, num_peds, 2]
            relative = unfolded - vel.unsqueeze(1)
        # Deleting Diagonal (Ped wrt itself): [num_tracks, num_tracks, 2] --> [num_tracks, num_tracks-1, 2]
        relative = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
        """
        if self.args.norm_pool:
            ## Normalize pooling grid along direction of pedestrian motion
            diff = torch.cat([curr_poss[:, 1:] - past_pos[:, 1:], curr_poss[:, 0:1] - past_pos[:, 0:1]], dim=1)
            theta = np.arctan2(diff[:, 1].clone(), diff[:, 0].clone())  # orientation of relative velocity
            ct = torch.cos(theta)
            st = torch.sin(theta)
            ## Cleaner?
            relative = torch.stack(
                [torch.einsum('tc,ci->ti', pos_instance, torch.Tensor([[ct[i], st[i]], [-st[i], ct[i]]])) for
                 i, pos_instance in enumerate(relative)], dim=0)
        """
        return relative

    def grid_pooling(self, obs, other_values=None, past_obs=None):
        """
        Returns the occupancy map filled with respective attributes.
        A different occupancy map with respect to each pedestrian
        :param obs: Tensor of shape [num_peds, 2]. Current x-y positions of all pedestrians, used to construct occupancy
        map. The positions must be in absolute coordinates
        :param other_values: Tensor of shape [num_peds, num_peds-1,  2]. Attributes (pooling_dim) of the neighbours
        relative to pedestrians, to be filled in the occupancy map, e.g., relative velocities of pedestrians
        :param past_obs: Tensor of shape [num_peds, 2]. Previous x-y positions of all pedestrians, used to construct
        occupancy map. Useful for normalizing the grid tensor. Currently NOT used here
        :return: grid: Tensor of shape [num_peds, pooling_dim, n, n] - grid resulting of the occupancy
        map. Still needs to go through an embedding layer
        """
        num_peds = obs.size(0)
        # mask unseen (the calling model should take care of this, but this is just a safe-guard)
        mask = torch.any(torch.isnan(obs), dim=1)
        obs[mask] = 0
        # if only one pedestrian is present
        if num_peds == 1:
            # return self.constant * torch.ones(1, self.pooling_dim, self.args.n, self.args.n, device=obs.device)
            return torch.zeros(1, self.pooling_dim, self.args.n, self.args.n, device=obs.device), \
                   torch.zeros(1, 1, self.args.n, self.args.n, device=obs.device) if self.include_occ else None
        # Get relative position of the pedestrians
        # [num_peds, 2] --> [num_peds, num_peds, 2]
        unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
        relative = unfolded - obs.unsqueeze(1)
        # Deleting Diagonal (to not consider a pedestrian with respect to itself)
        # [num_peds, num_peds, 2] --> [num_peds, num_peds-1, 2]
        relative = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
        # In case of 'occupancy' pooling
        if other_values is None:
            other_values = torch.ones(num_peds, num_peds - 1, self.pooling_dim, device=obs.device)
        # force use of standard occupancy to facilitate some operations
        other_values_o = torch.ones(num_peds, num_peds - 1, 1, device=obs.device) if self.include_occ \
                                                                                     or self.norm_cell else None
        if self.shape_values.all_sides is None:
            cell_sizes = torch.ones_like(relative) * self.args.cell_size
        else:
            cell_sizes = self.shape_values.all_sides.unsqueeze(1).unsqueeze(2).repeat(1, num_peds - 1, 2)
        """
        # Source: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/gridbased_pooling.py
        # Normalize pooling grid along direction of pedestrian motion
        if self.norm_pool:
            relative = self.normalize(relative, obs, past_obs)

        if self.front:
            oij = (relative / (cell_sizes) + torch.Tensor([self.n / 2, 0]))
        else:
            oij = (relative / (cell_sizes) + self.n / 2)
        """
        oij = (relative / cell_sizes + self.args.n / 2)
        # if range_no_violations has 0 value - index is good (neighbour in a specific cell)
        range_no_violations = torch.sum((oij < 0) + (oij >= self.args.n), dim=2)
        range_mask = range_no_violations == 0
        # range_mask == False -> outside of shape - will not count for pooling
        oij[~range_mask] = 0
        other_values[~range_mask] = 0
        if self.include_occ or self.norm_cell:
            other_values_o[~range_mask] = 0
        # other_values[~range_mask] = self.constant # self.constant = 0 by default; not implemented here
        oij = oij.long()
        # Flatten
        oi = oij[:, :, 0] * self.args.n + oij[:, :, 1]
        # faster occupancy
        occ = torch.zeros(num_peds, self.args.n ** 2, self.pooling_dim, device=obs.device)
        occ_o = torch.zeros(num_peds, self.args.n ** 2, 1, device=obs.device) if self.include_occ \
                                                                                 or self.norm_cell else None
        # occ = self.constant*torch.ones(num_tracks, self.n**2 * self.pool_size**2, self.pooling_dim, device=obs.device)
        # Fill occupancy map with attributes <-> sum values for consecutive positions
        # PREVIOUSLY: occ[torch.arange(num_peds).unsqueeze(1), oi] = other_values
        # CHANGED FROM ORIGINAL IMPLEMENTATION: Because this only considers one value (one pedestrian) per cell
        for i in range(num_peds):
            # use index_add_ so that values of multiple agents can be accumulated on the same cell
            # this requires the indexes to be one-dimensional, hence the above for loop
            occ[i].index_add_(0, oi[i], other_values[i])
            if self.include_occ or self.norm_cell:
                occ_o[i].index_add_(0, oi[i], other_values_o[i])
        if self.normalize:
            occ /= torch.clamp(torch.sum(occ, dim=1), min=1).unsqueeze(1).repeat(1, occ.shape[1], 1)
        elif self.norm_cell:
            occ /= torch.clamp(occ_o, min=1).repeat(1, 1, occ.shape[2])
        if self.include_occ:
            occ_o /= torch.clamp(torch.sum(occ_o, dim=1), min=1).unsqueeze(1).repeat(1, occ_o.shape[1], 1)
        occ, occ_o = torch.transpose(occ, 1, 2), torch.transpose(occ_o, 1, 2) if self.include_occ else None
        occ_2d, occ_o_2d = occ.view(num_peds, -1, self.args.n, self.args.n), \
                           occ_o.view(num_peds, -1, self.args.n, self.args.n) if self.include_occ else None
        """
        # Source: 
        # https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/gridbased_pooling.py
        if self.blur_size == 1:
            occ_blurred = occ_2d
        else:
            occ_blurred = fun.avg_pool2d(
                occ_2d, self.blur_size, 1, int(self.blur_size / 2), count_include_pad=True)
        occ_summed = fun.lp_pool2d(occ_2d, 1, 1)
        # occ_summed = fun.avg_pool2d(occ_blurred, 1)  # faster?
        """
        return occ_2d, occ_o_2d

    def arc_pooling(self, obs, other_values=None, angle_offsets=None):
        """

        :param obs:
        :param other_values:
        :param angle_offsets:
        :return:
        """
        num_peds = obs.size(0)
        # mask unseen values (the calling model should take care of this, but this is just a safe-guard)
        obs[torch.any(torch.isnan(obs), dim=1)] = 0
        # if only one pedestrian is present
        if num_peds == 1:
            # return self.constant * torch.ones(1, self.pooling_dim, self.args.n, self.args.n, device=obs.device)
            return torch.zeros(1, self.pooling_dim, self.args.n_r, self.args.n_a, device=obs.device), \
                   torch.zeros(1, 1, self.args.n_r, self.args.n_a, device=obs.device) if self.include_occ else None
        # Get relative position of the pedestrians
        # [num_peds, 2] --> [num_peds, num_peds, 2]
        unfolded = obs.unsqueeze(0).repeat(obs.size(0), 1, 1)
        relative_cart = unfolded - obs.unsqueeze(1)
        # convert to polar coordinates 0<->radius, 1<->angle
        relative = torch.cat((torch.norm(relative_cart, dim=2, keepdim=True),
                              torch.atan2(relative_cart[:, :, 1], relative_cart[:, :, 0]).unsqueeze(2)), dim=2)
        # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
        relative[:, :, 1] -= angle_offsets.unsqueeze(1).repeat(1, obs.size(0))
        # Deleting Diagonal (to not consider a pedestrian with respect to itself)
        # [num_peds, num_peds, 2] --> [num_peds, num_peds-1, 2]
        relative = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
        # normalize angle values in [-pi, pi) interval
        relative[:, :, 1] = (relative[:, :, 1] + np.pi) % (2 * np.pi) - np.pi
        if other_values is None:  # In case of 'occupancy' pooling
            other_values = torch.ones(num_peds, num_peds - 1, self.pooling_dim, device=obs.device)
        # force use of standard occupancy to facilitate some operations
        other_values_o = torch.ones(num_peds, num_peds - 1, 1, device=obs.device) if self.include_occ \
                                                                                     or self.norm_cell else None
        oij = torch.zeros_like(relative)
        if self.shape_values.all_radius is None or self.shape_values.all_angles is None:
            all_radius = torch.ones_like(oij[:, :, 0]) * self.args.arc_radius
            all_angles = torch.ones_like(oij[:, :, 0]) * self.args.arc_angle
        else:
            # use existing supplied values for an exterior configuration (most likely from ShapeConfigLSTM)
            all_radius = self.shape_values.all_radius.unsqueeze(1).repeat(1, num_peds - 1)
            all_angles = self.shape_values.all_angles.unsqueeze(1).repeat(1, num_peds - 1)
        if self.args.behind_ped:
            oij[:, :, 0] = relative[:, :, 0] / all_radius * (self.args.n_r - 1) + 1  # one less radius row forward
            oij[:, :, 1] = self.args.n_a / 2.0 * (relative[:, :, 1] / (all_angles / 2) + 1)
        else:
            oij[:, :, 0] = relative[:, :, 0] / all_radius * self.args.n_r
            oij[:, :, 1] = self.args.n_a / 2.0 * (relative[:, :, 1] / (all_angles / 2) + 1)
        # if range_no_violations has 0 value - index is good (neighbour in a specific cell)
        range_no_violations = torch.zeros_like(oij)
        range_no_violations[:, :, 0] = (oij[:, :, 0] < 0) + (oij[:, :, 0] >= self.args.n_r)
        range_no_violations[:, :, 1] = (oij[:, :, 1] < 0) + (oij[:, :, 1] >= self.args.n_a)
        range_mask = torch.sum(range_no_violations, dim=2) == 0
        if self.args.behind_ped:
            index_r_behind_ped = relative[:, :, 0] / self.args.arc_radius_behind
            # flip the "gaze" direction and see what pedestrians are included on that portion
            arc_angles_behind = (np.pi * 2) - all_angles  # value in [0, 2pi) range
            relative_angle_flipped = (relative[:, :, 1]) % (2 * np.pi) - np.pi  # value in [-pi, pi) range, but flipped
            index_a_behind_ped = self.args.n_a / 2.0 * (relative_angle_flipped / (arc_angles_behind / 2) + 1)
            oij[:, :, 0][~range_mask] = index_r_behind_ped[~range_mask]
            oij[:, :, 1][~range_mask] = index_a_behind_ped[~range_mask]
            # reset the range mask with the consideration of pedestrians immediately behind
            range_no_violations_full = range_no_violations.clone()
            range_no_violations_full[:, :, 0][~range_mask] = \
                ((oij[:, :, 0][~range_mask] < 0) + (oij[:, :, 0][~range_mask] >= 1)).float()
            range_no_violations_full[:, :, 1][~range_mask] = \
                ((oij[:, :, 1][~range_mask] < 0) + (oij[:, :, 1][~range_mask] >= self.args.n_a)).float()
            range_mask_full = torch.sum(range_no_violations_full, dim=2) == 0
        else:
            range_mask_full = range_mask
        # range_mask == False -> outside of shape - will not count for pooling
        oij[~range_mask_full] = 0
        other_values[~range_mask_full] = 0
        if self.include_occ or self.norm_cell:
            other_values_o[~range_mask_full] = 0
        # other_values[~range_mask] = self.constant # self.constant = 0 by default; not implemented here
        oij = oij.long()
        # Flatten - numbering goes across the same radius, and going counter-clockwise (neg to pos) in angle
        oi = oij[:, :, 0] * self.args.n_a + oij[:, :, 1]
        # faster occupancy
        occ = torch.zeros(num_peds, self.args.n_a * self.args.n_r, self.pooling_dim, device=obs.device)
        occ_o = torch.zeros(num_peds, self.args.n_a * self.args.n_r, 1, device=obs.device) if self.include_occ \
                                                                                              or self.norm_cell else None
        # occ = self.constant*torch.ones(num_tracks, self.n**2 * self.pool_size**2, self.pooling_dim, device=obs.device)
        # Fill occupancy map with attributes - sum for elements in the same cell
        for i in range(num_peds):
            # use index_add_ so that values of multiple agents can be accumulated on the same cell
            # this requires the indexes to be one-dimensional, hence the above for loop
            occ[i].index_add_(0, oi[i], other_values[i])
            if self.include_occ or self.norm_cell:
                occ_o[i].index_add_(0, oi[i], other_values_o[i])
        if self.normalize:
            occ /= torch.clamp(torch.sum(occ, dim=1), min=1).unsqueeze(1).repeat(1, occ.shape[1], 1)
        elif self.norm_cell:
            occ /= torch.clamp(occ_o, min=1).repeat(1, 1, occ.shape[2])
        if self.include_occ:
            occ_o /= torch.clamp(torch.sum(occ_o, dim=1), min=1).unsqueeze(1).repeat(1, occ_o.shape[1], 1)
        occ, occ_o = torch.transpose(occ, 1, 2), torch.transpose(occ_o, 1, 2) if self.include_occ else None
        occ_2d, occ_o_2d = occ.view(num_peds, -1, self.args.n_r, self.args.n_a), \
                           occ_o.view(num_peds, -1, self.args.n_r, self.args.n_a) if self.include_occ else None
        return occ_2d, occ_o_2d
