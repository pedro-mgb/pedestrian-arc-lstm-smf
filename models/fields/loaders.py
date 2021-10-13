import os
from os.path import sep

from models.data.common_dataset import read_mat_file
from models.utils.loaders import get_models_per_scene_biwi_crowds

import torch


def get_fields_from_file(device, file_path):
    """
    retrieve the motion fields from a file. These motion fields are expected to be in a matlab file (.mat extension),
    and will be read in dictionary format. The conversion to pytorch objects will be done here
    :param device: The torch.device to map the pytorch tensors to
    :param file_path: path to the file (assumed to be .mat file)
    :return: dictionary with several parameters:
    - Te_best: a Tensor of shape (2, n^2, nK). The actual motion fields for x/y coordinates, defined in a square grid
    - Qe_best: a Tensor of shape (4, n^2, nK). A 2x2 covariance matrix, to include uncertainty in model predictions.
    Also defined in a square grid.
    - Bc_best: a Tensor of shape (nK^2, n^2). Commutation matrix with probabilities of switching between the several
    motion fields (there's only one active motion field per instant, per trajectory). Also defined in a square grid.
    - min_max: a 2D array with the minimum and maximum position values (both x/y). The SparseMotionFields model needs
    trajectories to be normalized in [0, 1] interval. The normalization is done by subtracting min, then dividing by max
    - parameters: a dictionary with several parameters chosen for training the Sparse Motion Fields model: n is the
    dimension of the grid in which the model is defined; nK is the number of different active motion fields
    """
    fields_content = read_mat_file(file_path)
    if device:
        te_best = torch.from_numpy(fields_content['Te_best']).to(device)
        qe_best = torch.from_numpy(fields_content['Qe_best']).to(device)
        bc_best = torch.from_numpy(fields_content['Bc_best']).to(device)
        min_max = torch.from_numpy(fields_content['min_max']).to(device).squeeze()
        parameters = fields_content['parameters']
        fields_content = {'Te_best': te_best, 'Qe_best': qe_best, 'Bc_best': bc_best, 'min_max': min_max,
                          'parameters': parameters}
        return fields_content
    # else - don't map to torch
    return fields_content


def load_fields(device, path):
    """
    Loads motion fields from a file path; can be a single file, or a directory containing multiple models (possibly
    trained on different scenes)
    :param device: a torch.device to map the data to
    :param path: path to a file of fields or a directory containing several fields
    :return: list of motion fields (may contain a single element, if path is to a file)
    """
    if os.path.isdir(path):
        fields_list = get_models_per_scene_biwi_crowds(device, path, get_fields_from_file)
    else:
        # is a single file
        # scene label refers to all the existing data (pretty much just a placeholder)
        fields_list = [[get_fields_from_file(device, path), 'all_data']]
    return fields_list


if __name__ == '__main__':
    fields = load_fields(torch.device('cpu'), 'saved_models' + sep + 'fields')
    # since this involves getting tensors, the print may be very large
    print(fields)
