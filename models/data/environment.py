"""
Created on May 8th 2021
File containing useful classes about static environment (e.g. presence of obstacles or un-walkable areas)
"""
import math
import os

import numpy as np
import torch

from models.utils.loaders import __assert_scenes_and_models_files__


def line_segment_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    """
    See if two line segments (AB and CD) intersect. This method expects x/y components of each point to come separately
    Think of two line segments AB, and CD. These intersect if and only if points A and B are separated by segment CD and
    points C and D are separated by segment AB. If points A and B are separated by segment CD then ACD and BCD should
    have opposite orientation meaning either ACD or BCD is counterclockwise but not both
    Source: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    :return: True if AB and CD intersect, false otherwise
    """
    return ccw(ax, ay, cx, cy, dx, dy) != ccw(bx, by, cx, cy, dx, dy) and \
           ccw(ax, ay, bx, by, cx, cy) != ccw(ax, ay, bx, by, dx, dy)


def ccw(ax, ay, bx, by, cx, cy):
    """
    If the slope of the line AB is less than the slope of the line AC then the three points are listed in a
    counterclockwise order. This method expects x/y components of each point to come separately
    Source: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    :return: True
    """
    return (cy - ay) * (bx - ax) < (by - ay) * (cx - ax)


def segment_circle_intersect(ax, ay, bx, by, cx, cy, r):
    """
    See if a line segment AB intersects a circle of center C and radius r.
    :return: True if line intersects circle, false otherwise
    """
    length_ab = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)  # euclidean distance
    dx, dy = (bx - ax) / length_ab, (by - ay) / length_ab  # direction vector from A to B
    # the equation of the line AB is x = Dx * t + Ax, y = Dy * t + Ay with 0 <= t <= length_ab.
    # compute the distance between the points A and E, where
    # E is the point of AB closest the circle center (Cx, Cy)t = Dx * (Cx-Ax) + Dy * (Cy-Ay)
    t = dx * (cx - ax) + dy * (cy - ay)
    ex, ey = t * dx + ax, t * dy + ay
    length_ec = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)  # euclidean distance between closest point and center
    return length_ec < r


class LineSegment:
    def __init__(self, x_start, y_start, x_end, y_end, thickness):
        self.x_start, self.y_start, self.x_end, self.y_end = x_start, y_start, x_end, y_end
        self.__slope__ = (y_end - y_start) / (x_end - x_start) if x_end != x_start else float('inf')
        self.length = math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)  # euclidean distance
        self.thickness = thickness

    def compute_collisions(self, seq):
        """
        Compute the total number of collisions of sequence with the line
        A collision occurs with a line if a segment defined by two consecutive positions intersects this line segment
        :param seq: Pytorch Tensor of shape (traj_len, batch, 2). The sequence to compare the number of collisions with
        :return: tensor of shape (batch) containing the number of collisions per trajectory
        """
        num_collisions = torch.zeros(seq.shape[1], device=seq.device)
        for p in range(0, seq.shape[1]):
            for t in range(1, seq.shape[0]):
                # a collision occurs if two line segments intersect
                if line_segment_intersect(seq[t - 1, p, 0], seq[t - 1, p, 1], seq[t, p, 0], seq[t, p, 1],
                                          self.x_start, self.y_start, self.x_end, self.y_end):
                    num_collisions[p] += 1
                    break  # only one collisions per line per trajectory
        return num_collisions

    def change(self, switch_x_y, invert_x, invert_y):
        if switch_x_y:
            self.x_start, self.y_start = self.y_start, self.x_start
            self.x_end, self.y_end = self.y_end, self.x_end
        if invert_x:
            self.x_start = - self.x_start
            self.x_end = - self.x_end
        if invert_y:
            self.y_start = - self.y_start
            self.y_end = - self.y_end

    def plot(self, _plt, color='k'):
        """
        plot the line segment
        :param _plt: the plt from matplotlib.pyplot
        :param color: color to plot the line; default is black
        :return: Nothing
        """
        _plt.plot(np.array([self.x_start, self.x_end]), np.array([self.y_start, self.y_end]), color=color,
                  linewidth=self.thickness)

    def __str__(self):
        return f'Line Segment with x going from {self.x_start} to {self.x_end} and y from {self.y_start} to ' \
               f'{self.y_end}; line thickness of {self.thickness}'


class Circle:
    def __init__(self, x_center, y_center, radius):
        self.x_center, self.y_center, self.radius = x_center, y_center, radius

    def compute_collisions(self, seq):
        """
        Compute the total number of collisions of sequence with the environment
        :param seq: Pytorch Tensor of shape (traj_len, batch, 2). The sequence to compare the number of collisions with
        :return: tensor of shape (batch) containing the number of collisions per trajectory
        """
        num_collisions = torch.zeros(seq.shape[1], device=seq.device)
        for p in range(0, seq.shape[1]):
            for t in range(0, seq.shape[0]):
                # a collision occurs if the person is INSIDE the circle
                if math.sqrt((seq[t, p, 0] - self.x_center) ** 2 + (seq[t, p, 1] - self.y_center) ** 2) < self.radius:
                    num_collisions[p] += 1
                    break  # only one collision per circle per trajectory
        return num_collisions

    def change(self, switch_x_y, invert_x, invert_y):
        if switch_x_y:
            self.x_center, self.y_center = self.y_center, self.x_center
        if invert_x:
            self.x_center = - self.x_center
        if invert_y:
            self.y_center = - self.y_center

    def plot(self, _plt, color='k'):
        """
        plot the circle
        :param _plt: the plt from matplotlib.pyplot
        :param color: color to plot the circle; default is black
        :return: Nothing
        """
        ax = _plt.gcf().gca()  # get axis of current figure; circle must be plotted by adding a patch
        plt_circle = _plt.Circle((self.x_center, self.y_center), self.radius, color=color)
        ax.add_patch(plt_circle)

    def __str__(self):
        return f'Circle centered in x={self.x_center}, y={self.y_center}, with radius r={self.radius}'


class Environment:
    """
    Class to store environment info, originated from a file from a file.
    For a single scene, this environment class will have information about existence of obstacles in the scene.
    Also, one can check if a trajectory collides with those obstacles here.
    """

    def __init__(self, obstacles, scene_bounds):
        """
        Create environment object, using list of files
        :param obstacles: list of obstacles
        :param scene_bounds: list of delimiters for scene bounds
        """
        assert len(obstacles) > 0 or len(scene_bounds) > 0, \
            "To create a scene Environment object, there must be at least on physical obstacle or scene bounds " \
            "delimiter. However, empty lists were received for both obstacles and scene bounds."
        self.obstacles = obstacles
        self.scene_bounds = scene_bounds

    @staticmethod
    def load(file_path, delim='\t'):
        """
        Load environment information from a file
        :param file_path: path to the file containing information about environment (assuming it's for a specific scene)
        Currently allowed forms of obstacles (first value in the line is the identifier for the type of obstacle):
        - LineSegment: A single line in the file, describing the segment:
            l x_start y_start x_end y_end thickness
        - Circle: A single line in the file, describing the filled circle (not a circumference):
            c x_center y_center radius
        :param delim: delimiter between values in the same line of the file
        """
        obstacles, scene_bounds = [], []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        list_append = obstacles
        with open(file_path, 'r') as f:
            for line in f:
                if line[0] == '-':
                    # list of obstacles read - move to list of scene bounds
                    list_append = scene_bounds
                    continue
                line = line.strip().split(delim)
                obstacle_type = line[0].lower()
                if obstacle_type == 'l':
                    # is a line segment - should have an extra 5 values
                    list_append.append(LineSegment(*(float(i) for i in line[1:])))
                elif obstacle_type == 'c':
                    # is a circle - should have an extra 3 values
                    list_append.append(Circle(*(float(i) for i in line[1:])))
                # NOTE: if you obtain a "ValueError: could not convert string to float: 'f p x y'"
                # then it might be because you have the wrong delim; by default it's tab;
                #   use command line argument: --delim space
                else:
                    raise Exception(f'Unsupported obstacle type: {line[0]}. Currently only \'l\' and \'c\' types are'
                                    f'supported')
        return Environment(obstacles, scene_bounds)

    def compute_collisions(self, seq, combine_cse_osb=False):
        """
        Compute the total number of collisions of sequence with the environment
        :param seq: Pytorch Tensor of shape (traj_len, batch, 2). The sequence to compare the number of collisions with
        :param combine_cse_osb: if True, will combine collisions with obstacles and out of scene bounds (CSE and OSB
        metrics). If False (DEFAULT), will return them separately .
        :return: 2 tensors of shape (batch) containing:
        - the number of collisions with obstacles per trajectory;
        - the number of times each trajectory goes out of scene bounds
        None can be returned if the scene either have no obstacles (1st tensor) or scene bounds  (2nd tensor)
        Alternatively, ff combine_cse_osb is True, these tensors are summed and a single tensor is returned
        """
        if combine_cse_osb:
            return sum([o.compute_collisions(seq) for o in (self.obstacles + self.scene_bounds)])

        # cse<-> collisions with scene environment (obstacles) ; osb <-> out of scene bounds
        num_cse = sum([o.compute_collisions(seq) for o in self.obstacles]) if len(self.obstacles) > 0 else None
        num_osb = sum([s.compute_collisions(seq) for s in self.scene_bounds]) if len(self.scene_bounds) > 0 else None

        return num_cse, num_osb

    def plot(self, _plt, color='k'):
        """
        plot all the obstacles, in the same color
        :param _plt: the plt from matplotlib.pyplot
        :param color: color to plot the obstacles; default is black
        :return: Nothing
        """
        for o in (self.obstacles + self.scene_bounds):
            o.plot(_plt, color)

    def change(self, switch_x_y, invert_x, invert_y):
        for o in (self.obstacles + self.scene_bounds):
            o.change(switch_x_y, invert_x, invert_y)

    def __str__(self):
        string = f'Environment with a total of {len(self.obstacles)} obstacles and ' \
                 f'{len(self.scene_bounds)} delimiters of scene bounds.' + os.linesep + 'Obstacles:'
        for i, obstacle in enumerate(self.obstacles):
            string += os.linesep + f'\t{i + 1}. ' + str(obstacle)
        string += os.linesep + 'Scene bounds:'
        for i, delimiter in enumerate(self.scene_bounds):
            string += os.linesep + f'\t{i + 1}. ' + str(delimiter)
        return string


def load_environments_biwi_crowds(args, parent_folder_path):
    """
    Load the environment (containing scene-specific elements like obstacles) files from the BIWI and Crowds datasets
    :param args: command line arguments containing options on how to load the paths
    :param parent_folder_path: path to the parent folder containing the files
    :return: list of environment information (object of class Environment), and list of corresponding scenes (labels)
    """
    # first two belong to biwi (ETH dataset), the other two to crowds (UCY dataset)
    scene_labels = ['eth', 'hotel', 'univ', 'zara']
    return load_environments_per_scene(args, parent_folder_path, scene_labels)


def load_environments_per_scene(args, parent_folder_path, scene_labels):
    """
    Load the environment (containing scene-specific elements like obstacles) files for several scenes
    :param args: command line arguments containing options on how to load the paths
    :param parent_folder_path: path to the parent folder containing the files
    :param scene_labels: labels identifying each of the scenes
    :return: list of environment information (object of class Environment), and list of corresponding scenes (labels)
    """
    file_list = sorted(os.listdir(parent_folder_path))  # sorted alphabetically
    files_and_scenes = __assert_scenes_and_models_files__(parent_folder_path, file_list, scene_labels)
    # checks performed successfully, can read the fields file
    env_list, scene_list = [], []
    for [file_name, scene_label] in files_and_scenes:
        full_path = os.path.join(parent_folder_path, file_name)
        env = Environment.load(full_path, args.delim)
        env_list.append(env)
        scene_list.append(scene_label)
    return env_list, scene_list


def load_environments(args, path):
    """
    Load one or more environment (containing scene-specific elements like obstacles) files
    :param args: command line arguments containing options on how to load the paths
    :param path: path to the folder containing the files
    :return: list of environment information (object of class Environment), and list of corresponding scenes (labels)
    """
    if os.path.isdir(path):
        environments, scene_labels = load_environments_biwi_crowds(args, path)
    else:
        # is a single file
        environments = [Environment.load(path, args.delim)]
        scene_labels = ['all_data']
    return environments, scene_labels


if __name__ == '__main__':
    obstacles_biwi_eth_file_path = 'datasets_utils/environment/biwi_eth.txt'
    e = Environment.load(obstacles_biwi_eth_file_path)
    print(e, os.linesep, os.linesep)
    # sample collision detection
    seq_collides = torch.tensor([[5.0, -2.0], [5.0, 2.0]]).cpu().unsqueeze(1)
    cse, osb = e.compute_collisions(seq_collides)
    print(f'Number of collisions with scene environment (CSE): {int(torch.sum(cse).data) if cse is not None else None}')
    print(f'Number of trajectories out of scene bounds (OSB): {int(torch.sum(osb).data) if osb is not None else None}')
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        e.change(True, False, True)  # ETH scene needs x switched with y, then y inverted
        e.plot(plt)
        plt.show()
    except ModuleNotFoundError as e:
        print(f"Attempted to retrieve matplotlib.pyplot to plot scene environment, but it was not found: {e}")
