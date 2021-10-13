"""
Created on July 4th, 2021
"""
import numpy as np
import torch
from torch import nn

from models.interaction_modules.shape_based import PoolingShape


class ShapeConfigPedDensity(nn.Module):

    def __init__(self, module_shape_values, shape, values, max_num_ped, random=False):
        """

        :param module_shape_values:
        :param shape:
        :param values:
        :param max_num_ped:
        :param random:
        """
        super(ShapeConfigPedDensity, self).__init__()
        self.module_shape_values = module_shape_values
        self.shape = shape
        self.random_init = random
        self.max_ped = max_num_ped

        if shape == PoolingShape.GRID:
            self.radii = self.angles = self.min_radius = self.max_radius = self.min_angle = self.max_angle = \
                self.default_radius = self.default_angle = None
            self.sides = sorted(values[0])
            self.min_side, self.max_side = self.sides[0], self.sides[-1]
            self.default_side = self.min_side + (self.max_side - self.min_side) / 2
        else:
            self.sides = self.min_side = self.max_side = self.default_side = None
            self.radii, self.angles = sorted(values[0]), sorted([a for a in values[1]])
            self.angles = [a * np.pi / 180 for a in values[1]]  # convert to radians
            self.min_radius, self.max_radius = self.radii[0], self.radii[-1]
            self.min_angle, self.max_angle = self.angles[0], self.angles[-1]
            self.default_radius = self.min_radius + (self.max_radius - self.min_radius) / 2
            self.default_angle = self.min_angle + (self.max_angle - self.min_angle) / 2

        # extra values that will be properly initialized in other method calls
        self.all_sides = self.all_radii = self.all_angles = None

    def reset(self, num_pedestrians, seq_start_end, device):
        if self.shape == PoolingShape.GRID:
            if self.random_init:
                self.all_sides = self.min_side + torch.rand(num_pedestrians, device=device) * \
                                 (self.max_side - self.min_side)
            else:
                for (start, end) in seq_start_end:
                    curr_num_peds = end - start
                    ped_density = np.clip(end - start, a_min=0, a_max=self.max_ped) / self.max_ped  # in percentage
                    seq_sides = self.min_side + torch.full((curr_num_peds,), ped_density, device=device) * \
                                (self.max_side - self.min_side)
                    self.all_sides = torch.cat((self.all_sides, seq_sides))
        else:
            # shape == PoolingShape.ARC
            if self.random_init:
                self.all_radii = self.min_radius + torch.rand(num_pedestrians, device=device) * \
                                 (self.max_radius - self.min_radius)
                self.all_angles = self.min_angle + torch.rand(num_pedestrians, device=device) * \
                                  (self.max_angle - self.min_angle)
            else:
                # self.all_radii = torch.ones(num_pedestrians, device=device) * self.default_radius
                # self.all_angles = torch.ones(num_pedestrians, device=device) * self.default_angle
                # assuming that the current pedestrian density is the one given by seq_start_end
                self.all_radii, self.all_angles = torch.tensor([], device=device), torch.tensor([], device=device)
                for (start, end) in seq_start_end:
                    curr_num_peds = (end - start).cpu().data.numpy()  # convert to single value
                    ped_density = np.clip(curr_num_peds, a_min=0, a_max=self.max_ped) / self.max_ped  # in percentage
                    seq_radii = self.min_radius + torch.full((curr_num_peds,), ped_density, device=device) * \
                                (self.max_radius - self.min_radius)
                    seq_angles = self.min_angle + torch.full((curr_num_peds,), ped_density, device=device) * \
                                 (self.max_angle - self.min_angle)
                    self.all_radii, self.all_angles = torch.cat((self.all_radii, seq_radii)), \
                                                      torch.cat((self.all_angles, seq_angles))

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

    def forward(self, _pooling_out, indexes):
        """
        Forward pass through this shape configuration network. Use the current state and the prior shape to compute
        the most likely (via output linear layer + softmax to get probabilities) shape parameters or the next instant
        :param _pooling_out: Tensor of shape [num_pedestrians, out_dim]. The (embedded) of the pooling layer,
        containing information regarding the presence and influence of the pedestrians neighbourhood. Not used here.
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). Forward pass done only for those pedestrians
        :return: Nothing. The updated shape parameters are stored internally, and can be applied using self.update_shape
        """
        curr_num_peds = indexes.nelement()
        ped_density = np.clip(curr_num_peds, a_min=0, a_max=self.max_ped) / self.max_ped  # in percentage - range (0, 1]

        if self.shape == PoolingShape.GRID:
            self.all_sides[indexes] = self.min_side + torch.full((curr_num_peds,), ped_density,
                                                                 device=indexes.device) * \
                                      (self.max_side - self.min_side)
        else:
            # shape == PoolingShape.ARC
            # update the parameters for the arc-based pooling - max radius and the angle (spread of the arc)
            self.all_radii[indexes] = self.min_radius + torch.full((curr_num_peds,), ped_density,
                                                                   device=indexes.device) * \
                                      (self.max_radius - self.min_radius)
            self.all_angles[indexes] = self.min_angle + torch.full((curr_num_peds,), ped_density,
                                                                   device=indexes.device) * \
                                       (self.max_angle - self.min_angle)


class ShapeConfigNeighDist(nn.Module):

    def __init__(self, module_shape_values, shape, values, min_dist, max_dist, random=False):
        """

        :param module_shape_values:
        :param shape:
        :param values:
        :param min_dist:
        :param max_dist:
        :param random:
        """
        super(ShapeConfigNeighDist, self).__init__()
        self.module_shape_values = module_shape_values
        self.shape = shape
        self.random_init = random
        self.min_dist = min_dist
        self.max_dist = max_dist

        if shape == PoolingShape.GRID:
            self.radii = self.angles = self.min_radius = self.max_radius = self.min_angle = self.max_angle = \
                self.default_radius = self.default_angle = None
            self.sides = sorted(values[0])
            self.min_side, self.max_side = self.sides[0], self.sides[-1]
            self.default_side = self.min_side + (self.max_side - self.min_side) / 2
        else:
            self.sides = self.min_side = self.max_side = self.default_side = None
            self.radii, self.angles = sorted(values[0]), sorted([a for a in values[1]])
            self.angles = [a * np.pi / 180 for a in values[1]]  # convert to radians
            self.min_radius, self.max_radius = self.radii[0], self.radii[-1]
            self.min_angle, self.max_angle = self.angles[0], self.angles[-1]
            self.default_radius = self.min_radius + (self.max_radius - self.min_radius) / 2
            self.default_angle = self.min_angle + (self.max_angle - self.min_angle) / 2

        # extra values that will be properly initialized in other method calls
        self.all_sides = self.all_radii = self.all_angles = None

    def reset(self, num_pedestrians, _seq_start_end, device):
        if self.shape == PoolingShape.GRID:
            if self.random_init:
                self.all_sides = self.min_side + torch.rand(num_pedestrians, device=device) * \
                                 (self.max_side - self.min_side)
            else:
                self.all_sides = torch.ones(num_pedestrians, device=device) * self.default_side
        else:
            # shape == PoolingShape.ARC
            if self.random_init:
                self.all_radii = self.min_radius + torch.rand(num_pedestrians, device=device) * \
                                 (self.max_radius - self.min_radius)
                self.all_angles = self.min_angle + torch.rand(num_pedestrians, device=device) * \
                                  (self.max_angle - self.min_angle)
            else:
                self.all_radii = torch.ones(num_pedestrians, device=device) * self.default_radius
                self.all_angles = torch.ones(num_pedestrians, device=device) * self.default_angle
                # assuming that the current pedestrian density is the one given by seq_start_end

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

    def forward(self, _past_ped_positions, ped_positions, indexes):
        """
        Forward pass through this shape configuration network. Use the current state and the prior shape to compute
        the most likely (via output linear layer + softmax to get probabilities) shape parameters or the next instant
        :param ped_positions: Tensor of shape [num_pedestrians, 2]. The 2D positions of pedestrians
        :param _past_ped_positions: past pedestrian positions. Not used here.
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). Forward pass done only for those pedestrians
        :return: Nothing. The updated shape parameters are stored internally, and can be applied using self.update_shape
        """

        num_peds = ped_positions.shape[0]
        if num_peds == 1:
            avg_distance = torch.full((1,), self.max_dist, device=ped_positions.device)
        else:
            unfolded = ped_positions.unsqueeze(0).repeat(num_peds, 1, 1)
            relative = unfolded - ped_positions.unsqueeze(1)
            # [num_peds, num_peds, 2] --> [num_peds, num_peds-1, 2]
            relative_dist = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
            avg_distance = torch.mean(torch.norm(relative_dist, dim=2), dim=1)

        # regulate the shape values based on the distance
        avg_distance = torch.clamp(avg_distance, min=self.min_dist, max=self.max_dist)
        shape_regulator = (avg_distance - self.min_dist) / (self.max_dist - self.min_dist)

        if self.shape == PoolingShape.GRID:
            self.all_sides[indexes] = self.min_side + shape_regulator * (self.max_side - self.min_side)
        else:
            # shape == PoolingShape.ARC
            # update the parameters for the arc-based pooling - max radius and the angle (spread of the arc)
            self.all_radii[indexes] = self.min_radius + shape_regulator * (self.max_radius - self.min_radius)
            self.all_angles[indexes] = self.min_angle + shape_regulator * (self.max_angle - self.min_angle)


class ArcShapeRadiusConfigVisiblePedDensity(nn.Module):
    """
    Similar to ShapeConfigPedDensity, but applies for arc shape, and only varying radius.
    It will be varying with what the pedestrian can see (with the supplied arc angle
    """

    def __init__(self, module_shape_values, shape, values, max_num_ped, random=False):
        """

        :param module_shape_values:
        :param shape:
        :param values:
        :param max_num_ped:
        :param random:
        """
        super(ArcShapeRadiusConfigVisiblePedDensity, self).__init__()
        self.module_shape_values = module_shape_values
        self.shape = shape
        self.random_init = random
        self.max_ped = max_num_ped

        if self.shape == PoolingShape.GRID:
            raise Exception('This variable shape module is only available for ARC shape.')
        else:
            self.radii = sorted(values[0])
            self.min_radius, self.max_radius = self.radii[0], self.radii[-1]
            self.default_radius = self.min_radius + (self.max_radius - self.min_radius) / 2
            # assumed angle is sent as the first value in the second list, in degrees - convert to radians
            self.default_angle = values[1][0] * np.pi / 180
        # extra values that will be properly initialized in other method calls
        self.all_radii = self.all_angles = None

    def reset(self, num_pedestrians, _seq_start_end, device):
        # only for shape == PoolingShape.ARC
        if self.random_init:
            self.all_radii = self.min_radius + torch.rand(num_pedestrians, device=device) * \
                             (self.max_radius - self.min_radius)
        else:
            self.all_radii = torch.ones(num_pedestrians, device=device) * self.default_radius
        self.all_angles = torch.ones(num_pedestrians, device=device) * self.default_angle

    def update_shape(self, indexes):
        """
        Update the shape size (by changing the parameters) of the pooling layer (self.module_shape_values), using the
        current values stored in this shape configuration module
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). To select the shape parameters only for those pedestrians
        :return: nothing
        """
        # only for shape == PoolingShape.ARC
        # update the parameters for the arc-based pooling - max radius and the angle (spread of the arc)
        self.module_shape_values.all_radius = self.all_radii[indexes]
        self.module_shape_values.all_angles = self.all_angles[indexes]

    def forward(self, past_ped_positions, ped_positions, indexes):
        """
        Forward pass through this shape configuration network. Use the current state and the prior shape to compute
        the most likely (via output linear layer + softmax to get probabilities) shape parameters or the next instant
        :param ped_positions: Tensor of shape [num_pedestrians, 2]. The 2D positions of pedestrians
        :param past_ped_positions: Tensor of shape [num_pedestrians, 2]. The past 2D positions of pedestrians
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). Forward pass done only for those pedestrians
        :return: Nothing. The updated shape parameters are stored internally, and can be applied using self.update_shape
        """
        # only for shape == PoolingShape.ARC
        num_peds = ped_positions.shape[0]
        if num_peds == 1:
            self.all_radii[indexes] = self.max_radius  # radius value won't matter; pooling will be empty
        angle_offsets = torch.atan2(ped_positions[:, 1] - past_ped_positions[:, 1],
                                    ped_positions[:, 0] - past_ped_positions[:, 0])
        # Get relative position of the pedestrians
        # [num_peds, 2] --> [num_peds, num_peds, 2]
        unfolded = ped_positions.unsqueeze(0).repeat(num_peds, 1, 1)
        relative_cart = unfolded - ped_positions.unsqueeze(1)
        # convert to angle in polar coordinates AND
        # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
        relative_angles = torch.atan2(relative_cart[:, :, 1], relative_cart[:, :, 0]) - angle_offsets.unsqueeze(1)
        # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
        # relative_angles -= angle_offsets.unsqueeze(1).repeat(1, num_peds)
        # normalize angle values in [-pi, pi) interval
        relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi

        in_sight = (relative_angles >= (-self.default_angle / 2)) * (relative_angles < (self.default_angle / 2))
        # Do not consider a pedestrian with respect to itself   [num_peds, num_peds] --> [num_peds, num_peds-1]
        in_sight = in_sight[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1)
        num_ped_in_sight = torch.sum(in_sight, dim=1)
        ped_density_in_sight = np.clip(num_ped_in_sight, a_min=0, a_max=self.max_ped) / self.max_ped  # in percentage

        # update the parameters for the arc-based pooling - max radius, with equal value of angle for all
        # maximum ped density will yield smallest radius; minimum density will yield large radius
        self.all_radii[indexes] = self.min_radius + (1 - ped_density_in_sight) * (self.max_radius - self.min_radius)
        # not necessary to update the angle, since it will not change
        # self.all_angles[indexes] = torch.ones(num_peds, device=in_sight.device) * self.default_angle


class ArcShapeRadiusConfigVisibleNeighDist(nn.Module):
    """
    Similar to ShapeConfigNeighDist, but applies for arc shape, and only varying radius.
    It will be varying with the mean distance between the visible neighbours (up to certain limits)
    """

    def __init__(self, module_shape_values, shape, values, min_dist, max_dist, random=False):
        """

        :param module_shape_values:
        :param shape:
        :param values:
        :param min_dist:
        :param max_dist:
        :param random:
        """
        super(ArcShapeRadiusConfigVisibleNeighDist, self).__init__()
        self.module_shape_values = module_shape_values
        self.shape = shape
        self.random_init = random
        self.min_dist = min_dist
        self.max_dist = max_dist
        if self.shape == PoolingShape.GRID:
            raise Exception('This variable shape module is only available for ARC shape.')
        else:
            self.radii = sorted(values[0])
            self.min_radius, self.max_radius = self.radii[0], self.radii[-1]
            self.default_radius = self.min_radius + (self.max_radius - self.min_radius) / 2
            # assumed angle is sent as the first value in the second list, in degrees - convert to radians
            self.default_angle = values[1][0] * np.pi / 180
            # extra values that will be properly initialized in other method calls
        self.all_radii = self.all_angles = None

    def reset(self, num_pedestrians, _seq_start_end, device):
        # only for shape == PoolingShape.ARC
        if self.random_init:
            self.all_radii = self.min_radius + torch.rand(num_pedestrians, device=device) * \
                             (self.max_radius - self.min_radius)
        else:
            self.all_radii = torch.ones(num_pedestrians, device=device) * self.default_radius
        self.all_angles = torch.ones(num_pedestrians, device=device) * self.default_angle

    def update_shape(self, indexes):
        """
        Update the shape size (by changing the parameters) of the pooling layer (self.module_shape_values), using the
        current values stored in this shape configuration module
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). To select the shape parameters only for those pedestrians
        :return: nothing
        """
        # only for shape == PoolingShape.ARC
        # update the parameters for the arc-based pooling - max radius and the angle (spread of the arc)
        self.module_shape_values.all_radius = self.all_radii[indexes]
        self.module_shape_values.all_angles = self.all_angles[indexes]

    def forward(self, past_ped_positions, ped_positions, indexes):
        """
        Forward pass through this shape configuration network. Use the current state and the prior shape to compute
        the most likely (via output linear layer + softmax to get probabilities) shape parameters or the next instant
        :param ped_positions: Tensor of shape [num_pedestrians, 2]. The 2D positions of pedestrians
        :param past_ped_positions: Tensor of shape [num_pedestrians, 2]. The past 2D positions of pedestrians
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). Forward pass done only for those pedestrians
        :return: Nothing. The updated shape parameters are stored internally, and can be applied using self.update_shape
        """
        # only for shape == PoolingShape.ARC
        num_peds = ped_positions.shape[0]
        if num_peds == 1:
            self.all_radii[indexes] = self.max_radius  # radius value won't matter; pooling will be empty
        angle_offsets = torch.atan2(ped_positions[:, 1] - past_ped_positions[:, 1],
                                    ped_positions[:, 0] - past_ped_positions[:, 0])
        # Get relative position of the pedestrians
        # [num_peds, 2] --> [num_peds, num_peds, 2]
        unfolded = ped_positions.unsqueeze(0).repeat(num_peds, 1, 1)
        relative_cart = unfolded - ped_positions.unsqueeze(1)
        relative_dist = torch.norm(relative_cart, dim=2)
        # convert to angle in polar coordinates AND
        # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
        relative_angles = torch.atan2(relative_cart[:, :, 1], relative_cart[:, :, 0]) - angle_offsets.unsqueeze(1)
        # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
        # relative_angles -= angle_offsets.unsqueeze(1).repeat(1, num_peds)
        # normalize angle values in [-pi, pi) interval
        relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi

        in_sight = (relative_angles >= (-self.default_angle / 2)) * (relative_angles < (self.default_angle / 2))
        # Do not consider a pedestrian with respect to itself   [num_peds, num_peds] --> [num_peds, num_peds-1]
        in_sight = in_sight[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1)
        relative_dist = relative_dist[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1)
        mean_distance = torch.tensor([], device=in_sight.device)
        for p in range(in_sight.shape[0]):
            mean_distance = torch.cat((mean_distance, torch.mean(relative_dist[p, in_sight[p]]).unsqueeze(0)))
        mean_distance[torch.isnan(mean_distance)] = 0  # for the cases with no neighbours in sight
        # the shape_regulator indicates how much the radius will grow as a linear function of the median distance
        mean_distance = torch.clamp(mean_distance, min=self.min_dist, max=self.max_dist)
        shape_regulator = (mean_distance - self.min_dist) / (self.max_dist - self.min_dist)
        # update the parameters for the arc-based pooling - max radius, with equal value of angle for all
        # maximum distance will yield smallest radius; minimum distance will yield large radius
        self.all_radii[indexes] = self.min_radius + shape_regulator * (self.max_radius - self.min_radius)
        # not necessary to update the angle, since it will not change
        # self.all_angles[indexes] = torch.ones(num_peds, device=in_sight.device) * self.default_angle


class GrowingShapeUpToMaxPedestrians(nn.Module):
    """
    radius that grows to include up to x pedestrians, with also limits on the max (maybe min?) radius
    """

    def __init__(self, args, module_shape_values, shape, values, max_num_ped, random=False):
        """
        :param args: command line arguments containing some options to configure this module
        :param module_shape_values:
        :param shape:
        :param values:
        :param max_num_ped:
        :param random:
        """
        super(GrowingShapeUpToMaxPedestrians, self).__init__()
        self.module_shape_values = module_shape_values
        self.shape = shape
        self.random_init = random
        self.max_ped = max_num_ped

        self.grid_dim = args.grid_dim  # necessary for variable grid shape

        if self.shape == PoolingShape.GRID:
            self.radii = self.angles = self.min_radius = self.max_radius = self.min_angle = self.max_angle = \
                self.default_radius = self.default_angle = None
            self.sides = sorted(values[0])
            self.min_side, self.max_side = self.sides[0], self.sides[-1]
            self.default_side = self.min_side + (self.max_side - self.min_side) / 2
        else:
            self.sides = self.min_side = self.max_side = self.default_side = None
            self.radii = sorted(values[0])
            self.min_radius, self.max_radius = self.radii[0], self.radii[-1]
            self.default_radius = self.min_radius + (self.max_radius - self.min_radius) / 2
            # assumed angle is sent as the first value in the second list, in degrees - convert to radians
            self.default_angle = values[1][0] * np.pi / 180
            self.min_angle = self.max_angle = self.default_angle
        # extra values that will be properly initialized in other method calls
        self.all_sides = self.all_radii = self.all_angles = None

    def reset(self, num_pedestrians, _seq_start_end, device):
        if self.shape == PoolingShape.GRID:
            if self.random_init:
                self.all_sides = self.min_side + torch.rand(num_pedestrians, device=device) * \
                                 (self.max_side - self.min_side)
            else:
                self.all_sides = torch.ones(num_pedestrians, device=device) * self.default_side
        else:
            # shape == PoolingShape.ARC:
            if self.random_init:
                self.all_radii = self.min_radius + torch.rand(num_pedestrians, device=device) * \
                                 (self.max_radius - self.min_radius)
            else:
                self.all_radii = torch.ones(num_pedestrians, device=device) * self.default_radius
            self.all_angles = torch.ones(num_pedestrians, device=device) * self.default_angle

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

    def forward(self, past_ped_positions, ped_positions, indexes):
        """
        Forward pass through this shape configuration network. Use the current state and the prior shape to compute
        the most likely (via output linear layer + softmax to get probabilities) shape parameters or the next instant
        :param ped_positions: Tensor of shape [num_pedestrians, 2]. The 2D positions of pedestrians
        :param past_ped_positions: Tensor of shape [num_pedestrians, 2]. The past 2D positions of pedestrians
        :param indexes: Tensor of shape [num_pedestrians]. Boolean indexes for the current pedestrians present in the
        scene (True if present, false otherwise). Forward pass done only for those pedestrians
        :return: Nothing. The updated shape parameters are stored internally, and can be applied using self.update_shape
        """
        # only for shape == PoolingShape.ARC
        num_peds = ped_positions.shape[0]
        if self.shape == PoolingShape.GRID:
            if num_peds <= self.max_ped:
                # not enough pedestrians to limit the maximum side length
                self.all_sides[indexes] = self.max_side
                return
            # Get relative position of the pedestrians      [num_peds, 2] --> [num_peds, num_peds, 2]
            unfolded = ped_positions.unsqueeze(0).repeat(num_peds, 1, 1)
            relative = unfolded - ped_positions.unsqueeze(1)
            # remove own pedestrian [num_peds, num_peds, 2] -> [num_peds, num_peds - 1, 2]
            relative = relative[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1, 2)
            # NOTE: the grid is not rotated along the pedestrian's view direction. It's aligned with x axis
            # the major distance to the axis will contribute to the grid size (to contain the neighbour the grid length
            # must be equal to have the major axis distance).
            # TODO verify if this is correct
            major_axis_dist = torch.max(torch.abs(relative), dim=2)
            sides_chosen = torch.tensor([], device=unfolded.device)
            for p in range(unfolded.shape[0]):
                # sort the distances from smallest to largest; the radius will be the one that includes max_ped
                # divide by the number of cells to get an associated cell dimension
                dist_sorted = torch.sort(major_axis_dist[p]).values / self.grid_dim
                # mean of those two values so that side is slightly larger than distance to max_ped, but smaller
                #   than distance to the next pedestrian; '* 2' because shape is square, and distance would be half side
                side = (dist_sorted[self.max_ped - 1] + dist_sorted[self.max_ped]) / 2 * 2
                sides_chosen = torch.cat((sides_chosen, torch.clamp(side, min=self.min_side,
                                                                    max=self.max_side).unsqueeze(0)))
        else:
            # self.shape == PoolingShape.ARC:
            if num_peds == 1:
                self.all_radii[indexes] = self.max_radius  # radius value won't matter; pooling will be empty
                return
            angle_offsets = torch.atan2(ped_positions[:, 1] - past_ped_positions[:, 1],
                                        ped_positions[:, 0] - past_ped_positions[:, 0])
            # Get relative position of the pedestrians      [num_peds, 2] --> [num_peds, num_peds, 2]
            unfolded = ped_positions.unsqueeze(0).repeat(num_peds, 1, 1)
            relative_cart = unfolded - ped_positions.unsqueeze(1)
            relative_dist = torch.norm(relative_cart, dim=2)
            # convert to angle in polar coordinates AND
            # subtract the angle offsets (so that the angles are in accordance with pedestrian's gaze direction)
            relative_angles = torch.atan2(relative_cart[:, :, 1], relative_cart[:, :, 0]) - angle_offsets.unsqueeze(1)
            # normalize angle values in [-pi, pi) interval
            relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi

            in_sight = (relative_angles >= (-self.default_angle / 2)) * (relative_angles < (self.default_angle / 2))
            # Do not consider a pedestrian with respect to itself   [num_peds, num_peds] --> [num_peds, num_peds-1]
            in_sight = in_sight[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1)
            relative_dist = relative_dist[~torch.eye(num_peds).bool()].reshape(num_peds, num_peds - 1)

            radii_chosen = torch.tensor([], device=in_sight.device)
            for p in range(in_sight.shape[0]):
                dist_peds_in_sight = relative_dist[p, in_sight[p]]
                if dist_peds_in_sight.shape[0] <= self.max_ped:
                    # radius has less than the specified number of pedestrians, so its only threshold will be the
                    # maximum possible radius
                    radii_chosen = torch.cat((radii_chosen, torch.tensor([self.max_radius], device=in_sight.device)))
                else:
                    # sort the distances from smallest to largest; the radius will be the one that includes max_ped
                    dist_sorted = torch.sort(dist_peds_in_sight).values
                    # mean of those two values so that radius is slightly larger than distance to max_ped, but smaller
                    #   than distance to the next pedestrian, at larger distance
                    r = (dist_sorted[self.max_ped - 1] + dist_sorted[self.max_ped]) / 2
                    radii_chosen = torch.cat((radii_chosen, torch.clamp(r, min=self.min_radius,
                                                                        max=self.max_radius).unsqueeze(0)))
            self.all_radii[indexes] = radii_chosen
            # not necessary to update the angle, since it will not change
            # self.all_angles[indexes] = torch.ones(num_peds, device=in_sight.device) * self.default_angle
