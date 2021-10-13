"""
Created on March 14th 2021
Common method and utilities for dataset processing
"""

import numpy as np
from scipy.io import loadmat
import trajnetplusplustools


class Metadata(object):
    """
    Class to store metadata-related information regarding a trajectory or sequence of trajectories. This data is not
    100% essential for training / inference, but can provide additional information for auxiliary tasks
    """

    def __init__(self, origin_label, size=0):
        """
        builds the Metadata object
        :param origin_label: label identifying the source of the trajectory (e.g. the file name)
        :param size: size (assumed MegaBytes/MB) in memory of the trajectories this metadata refers to
        """
        super(Metadata, self).__init__()
        self.origin = origin_label
        self.size = size


def read_file(_path, delim='\t'):
    """
    read contents of a file, line by line, assuming each value is separated in the same format
    :param _path: file path
    :param delim: the delimiter between values
    :return: data content, as a numpy array
    """
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            # if you obtain a "ValueError: could not convert string to float: 'f p x y'"
            # then it might be because you have the wrong delim; by default it's tab;
            #   use command line argument: --delim space
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def read_ndjson_file(_path):
    """
    Read file in ndjson format, as used by Trajnet++ data
    :param _path: path to file
    :return: a trajnetplusplustools Reader object, containing scenes and tracks
    """
    return trajnetplusplustools.Reader(_path, scene_type='paths')


def poly_fit(traj, traj_len, threshold):
    """
    fit the trajectory (a sequence of positions), to a polynomial, using least squares; indicate if it is linear or not
    by comparing it with a threshold
    :param traj: the trajectory to fit to - Numpy array of shape (2, traj_len)
    :param traj_len: the length of the trajectory
    :param threshold: threshold to decide if trajectory is linear or not
    :return: 0 if trajectory is linear, 1 otherwise
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


def read_mat_file(path):
    """
    read a file in .mat format (popular in MATLAB)
    :param path: path to the file
    :return: the loaded data, in dictionary format, where arrays/matrices are instantiated with numpy.
    """
    return loadmat(path)
