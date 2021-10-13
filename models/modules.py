"""
Created on April 4th, 2021
Contains modules that are common to several models
"""
from collections import OrderedDict

import torch
from torch import nn


def __forward_normal_stabilize_range__(normal):
    # CREDITS: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/modules.py#L51
    normal[:, 2] = 0.01 + 0.2 * torch.sigmoid(normal[:, 2])  # sigma x
    normal[:, 3] = 0.01 + 0.2 * torch.sigmoid(normal[:, 3])  # sigma y
    normal[:, 4] = 0.7 * torch.sigmoid(normal[:, 4])  # rho
    return normal


class NormalStabilizeRange(nn.Module):
    """
    Process the normal distribution using numerically stable output ranges
    CREDITS: https://github.com/vita-epfl/trajnetplusplusbaselines/blob/master/trajnetbaselines/lstm/modules.py#L51
    """

    def __init__(self, inplace=True):
        """
        initialize the module
        :param inplace: if meant to change the tensor inplace or not
        """
        super(NormalStabilizeRange, self).__init__()
        self.inplace = inplace

    def forward(self, normal):
        """
        Stabilizes the range of values for normal distribution
        :param normal: Tensor of shape (batch, 5). The normal values, mean, std and correlation
        :return: the tensor with the stabilized value range
        """
        if self.inplace:
            return __forward_normal_stabilize_range__(normal)
        else:
            normal_copy = torch.clone(input=normal)
            normal_copy = __forward_normal_stabilize_range__(normal_copy)
            return normal_copy


class LastUpdatedOrderedDict(OrderedDict):
    """Store items in the order the keys were last added"""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
