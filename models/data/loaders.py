"""
Created on March 18th 2021
"""
import os
import time

from models.data.dataset import data_loader_variable_len
from models.data.dataset_fixed import data_loader_fixed_len
from models.data.dataset_trajnetpp import data_loader_trajnetpp


def load_train_val_data(args, device, logger):
    """
    retrieve the data used for model training and validation
    :param args: several arguments required to read the data and configure how it should be loaded
    :param device: torch.device to map the tensors to (e.g. 'cpu', 'cuda')
    :param logger: log info
    :return: loaders for training and validation data. The loader for training data, even if it's just one, will be
    wrapped in a list for cross compatibility for training options where like --train_files_individually
    """
    # decide on which kind of data_loader to use - depending on supplied arguments
    data_loader = get_data_loader(args, args.train_dir)
    #  assert data_loader == get_data_loader(args, args.val_dir)
    t1 = time.time()
    if logger:
        logger.info("Retrieving train dataset in {}".format(args.train_dir))
    if hasattr(args, 'train_files_individually') and args.train_files_individually:
        train_loaders, train_file_names = load_data_from_multiple_files(args, device, args.train_dir, data_loader)
    else:
        _, train_loader = data_loader(args, device, args.train_dir)
        train_loaders = [train_loader]
        train_file_names = ['all_files']
    if args.val_dir == 'none':
        args.val_dir = None
    if args.val_dir is not None and args.val_dir:
        if logger:
            logger.info("Retrieving val dataset in {}".format(args.val_dir))
        _, val_loader = data_loader(args, device, args.val_dir)
    else:
        val_loader = None
    if args.timing and logger:
        logger.info("Done! Took {0:.2f} s to load the data{1}".format(time.time() - t1, os.linesep))
    return train_loaders, train_file_names, val_loader


def load_test_data(args, device, load_pred=False):
    """
    load data for testing, with configuration depending on arguments
    :param args: several arguments required to read the data and configure how it should be loaded
    :param device: torch.device to map the tensors to (e.g. 'cpu', 'cuda')
    :param load_pred: if meant to load predictions from file instead of actual data
    :return: a list of loaders with the trajectories (can be just one)
    """
    # decide on which kind of data_loader to use - depending on supplied arguments
    data_loader = get_data_loader(args, args.test_dir)
    t1 = time.time()
    print(f"Retrieving test " + ("MODEL PREDICTIONS" if load_pred else "dataset") + f" in {args.test_dir} " +
          (", getting each file individually" if args.test_files_individually else ""))
    if args.test_files_individually:
        # get each file into a particular loader
        test_loaders, file_names = load_data_from_multiple_files(args, device, args.test_dir, data_loader, load_pred)
    else:
        file_names = ['all files']
        _, loader = data_loader(args, device, args.test_dir, load_pred)
        test_loaders = [loader]
    if args.timing:
        print("Done! Took {0:.2f} s to load the data".format(time.time() - t1))
    else:
        print("Done!")
    return test_loaders, file_names


def load_data_from_multiple_files(args, device, data_dir, data_loader, load_pred=False):
    """
    loading of trajectory data from one or multiple files
    :param args: several arguments required to read the data and configure how it should be loaded
    :param device: torch.device to map the tensors to (e.g. 'cpu', 'cuda')
    :param data_dir: directory containing the data files, or path to file (in case of single file)
    :param data_loader: specific data_loader method, using a specific configuration
    :param load_pred: if meant to load predictions from file instead of actual data
    :return: the data loaders objects, and the file names.
    """
    if os.path.isdir(data_dir):
        file_names = sorted(os.listdir(data_dir))
        all_files = [os.path.join(data_dir, _path) for _path in file_names]
    else:
        # is a single file
        all_files = [data_dir]
        file_names = [os.path.basename(data_dir)]
    loaders = []
    for f in all_files:
        _, loader = data_loader(args, device, f, load_pred)
        loaders.append(loader)
    return loaders, file_names


def load_biwi_crowds_data_per_scene(args, device, _scene_labels=None, load_pred=False):
    """
    load data for biwi (aka ETH) and crowds (aka UCY) datasets, with one dataset loader for each of the scenes of both
    datasets: 2 for biwi (eth, hotel), 2 for crowds (univ, zara)
    :param args: command line arguments containing parameters that regulate how the data is loaded
    :param device: torch.device to map the tensors to (e.g. 'cpu', 'cuda')
    :param _scene_labels: desired labels identifying each scene from biwi and crowds datasets, that may be mapped to
    the corresponding one that should be present in the path to the data
    :param load_pred: if meant to load predictions from file instead of actual data
    :return: two lists, per scene: the first with the data loaders, the second with the file names
    """
    return __load_data_per_scene__(args, device, args.test_dir, __map_scene_labels_biwi_crowds__(_scene_labels),
                                   load_pred)


def __load_data_per_scene__(args, device, data_location, scene_labels, load_pred=False):
    """
    load data, separate for each existing scene
    :param args: command line arguments containing parameters that regulate how the data is loaded
    :param device: torch.device to map the tensors to (e.g. 'cpu', 'cuda')
    :param scene_labels: labels identifying each scene in terms of the file path
    :param load_pred: if meant to load predictions from file instead of actual data
    :return: two lists, per scene: the first with the data loaders, the second with the file names
    """
    scene_loaders, scene_file_names = [[] for _ in range(len(scene_labels))], [[] for _ in range(len(scene_labels))]
    # decide on which kind of data_loader to use - depending on supplied arguments
    data_loader = get_data_loader(args, data_location)
    for idx, label in enumerate(scene_labels):
        # append the label to the right location of the data, using the supplied path separator
        slash, idx_last_slash = '/', data_location.rfind('/') + 1
        if idx_last_slash <= 0:
            slash, idx_last_slash = '\\', data_location.rfind('\\') + 1
            if idx_last_slash <= 0:
                slash, idx_last_slash = os.pathsep, 0
        location = data_location[:idx_last_slash] + label + slash + data_location[idx_last_slash:]
        if args.test_files_individually:
            # get each file of each scene into a particular loader
            scene_file_names[idx] = sorted(os.listdir(location))
            all_files = [os.path.join(location, _path) for _path in scene_file_names[idx]]
            for f in all_files:
                _, loader = data_loader(args, device, f, load_pred)
                scene_loaders[idx].append(loader)
        else:
            scene_file_names[idx] = [scene_labels[idx]]
            _, loader = data_loader(args, device, location, load_pred)
            scene_loaders[idx] = [loader]

    return scene_loaders, scene_file_names


def __map_scene_labels_biwi_crowds__(_labels=None):
    """
    map labels from scenes in biwi and crowds dataset to a list of labels that are expected to coincide with the labels
    present in the paths to the data
    :param _labels: actual provided labels; if nothing is provided, a default list order is used; if an actual list is
    provided, then the returned list will be in the same order as the provided one
    :return: a list of scene labels that are expected to coincide with the labels present in the paths to the data
    """
    eth_label, hotel_label, univ_label, zara_label = 'biwi_eth', 'biwi_hotel', 'crowds_univ', 'crowds_zara'
    if not _labels:
        return [eth_label, hotel_label, univ_label, zara_label]
    scene_labels = []
    for label in _labels:
        label_low = label.lower()
        if 'eth' in label_low:
            scene_labels.append(eth_label)
        elif 'hotel' in label_low:
            scene_labels.append(hotel_label)
        elif 'univ' in label_low:
            scene_labels.append(univ_label)
        elif 'zara' in label_low:
            scene_labels.append(zara_label)
        else:
            raise Exception(f'Received an invalid scene label for a biwi/crowds scene - {label}')
    return scene_labels


def get_data_loader(args, data_location):
    """
    decide on which kind of data_loader to use - depending on supplied arguments
    :param args: Object containing command line arguments
    :param data_location: path to the data
    :return: the appropriate data_loader function. The function receives 3 arguments: args, device, and data_location
    """
    if args.fixed_len:
        return data_loader_fixed_len
    elif args.variable_len:
        return data_loader_variable_len
    else:
        # Trajnet++ data
        if '11' in data_location:  # length=11; so that these arguments do not have to be supplied everytime
            args.trajnetpp_obs_len = 5
            args.trajnetpp_pred_len = 6
        return data_loader_trajnetpp
