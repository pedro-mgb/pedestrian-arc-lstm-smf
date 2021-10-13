"""
Created on 04/March/2021

File that contains processing of trajectory datasets, available in '../datasets(...)' folders.
Unlike ./dataset_fixed, there is no fixation of trajectory length on the data.
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.data.common_dataset import read_file, Metadata
from models.utils.utils import random_angle_rotation, random_angle_rotation_uniform


class MetadataVariableLen(Metadata):
    """
    Class to store metadata-related information regarding each trajectory that can have variable length.
    This data is not 100% essential for training / inference, but can provide additional information for auxiliary tasks
    """
    def __init__(self, origin_label, rot_angle, size=0):
        """
        builds the Metadata object
        :param origin_label: label identifying the source of the trajectory (e.g. the file name)
        :param rot_angle: The rotation angle of trajectories, in radius. Applies for cases where --normalize_scene or
        --augment_rotation command line arguments are supplied
        :param size: size (assumed MegaBytes/MB) in memory of the trajectories this metadata refers to
        """
        super(MetadataVariableLen, self).__init__(origin_label, size)
        self.angle = rot_angle


class TrajectoryDatasetVariableLen(Dataset):
    """Dataset for the Trajectory datasets - Can handle variable length of trajectories.
    Each sequence is said to be a sort of sub-batch (not to be confused with a batch returned at each step, that can
    contain multiple sequences in it), and comprises set of trajectories of the same length, synchronized in time.
    This idea of synchronized in time means that all those trajectories are present in the exact same time frame.
    Regarding the format of the trajectory files - each line should contain 4 values separated by a delimiter (<delim>)
    frame_id<delim>ped_id<delim>x<delim>y
    Where:
        - frame_id identifies the corresponding frame or instant
        - ped_id identifies the pedestrian to which this position refers to
        - x,y correspond to the 2D position of the pedestrian in the scene. Units should not be put explicitly (meaning
        that if one uses meters, the values to put are in format (example) - 4.0<delim>5.0; NOT 4.0m<delim>5.0
    """

    def __init__(self, data_location, device=torch.device('cpu'), obs_percentage=0.4, max_obs_len=None,
                 max_pred_len=None, same_obs_len=False, same_pred_len=False, min_obs_len=2, min_traj_len=3, skip=1,
                 min_ped=0, delim='\t', use_partial_trajectories=False, split_trajectories=False, also_backwards=False,
                 batch_size=64, random_rotation_angle_std=0, random_rotation_angle_threshold=0, threshold=0.002):
        """
        Initializes the dataset contents from the file(s)
        :param data_location: Location of the dataset files in the format (can be a directory or a single file).
        Format: <frame_id> <ped_id> <x> <y>
        :param device: device to map the tensors to. By default will go to cpu
        :param obs_percentage: The percentage of the trajectory the is to be observed (corresponding to the past) -
        will round to the nearest integer
        :param max_obs_len: maximum observation length for trajectories (None or <= 0 if no maximum intended). If a
        trajectory is found with observation length > max_obs_len, then only the most recent 'max_obs_len' instants of
        the observed trajectory will be used (the older positions will be discarded). This value can be experimented
        with to analyze the influence of the past (and length of said past) in the model predictions.
        :param max_pred_len: maximum prediction length for trajectories (None or <= 0 if no maximum intended)
        :param same_obs_len: use always the same observation length (equal to max_obs_len). If the trajectory length
        is such that trajectory_length - max_pred_len < max_obs_len (if max_pred_len was supplied)
        then -> discard the trajectory (not enough instants to retrieve the predicted trajectory).
            NOTE: this field should only be active if max_obs_len has an actual value
        :param same_pred_len: use always the same prediction length (equal to max_pred_len). If the trajectory length
        is such that
            (1-obs_percentage)*trajectory_length < max_pred_len OR
            trajectory_length - min_obs_len < max_pred_len
        then -> discard the trajectory (not enough instants to retrieve the observed trajectory).
            NOTE #1: this field should only be active if max_pred_len has an actual value.
            NOTE #2: having same_pred_len and same_obs_len active at the same time may reduce drastically the number
            of trajectories that the dataset can support
        :param min_obs_len: minimum observed trajectory length (should be smaller than min_traj_len)
        :param min_traj_len: Minimum trajectory length allowed (whole trajectory -> obs_len + pred_len)
        :param skip: Number of frames to skip while making the dataset
        :param min_ped: minimum number of pedestrians that should be in a sequence - set of trajectories of equal length
        :param delim: Delimiter in the dataset files (the separation between values)
        :param split_trajectories: Assumed observation length and prediction length constant, and equal to parameters
        max_obs_len and max_pred_len. However, longer trajectories are split in portions of equal length. By default,
        max_obs_len will have value 8 assigned (instead of None).
        Practical example: max_obs_len=8 and max_pred_len=12; trajectory of length 90; The first 80 instants are divided
        into 4 portions of length 8+12=20. The last 10 instants are discarded.
        :param also_backwards: simple a 'data augmentation' technique, to also consider the same trajectories, but going
        in opposite order (starting at last instant, ending in beggining).
        :param batch_size: the desired size for batches (a group of sequences); can be slightly different than this
        :param random_rotation_angle_std: if larger than 0, will rotate each batch using a random angle sampled from a
        gaussian distribution with 0 mean and standard deviation equal to this value (in radians)
        :param random_rotation_angle_threshold: if larger than 0, will rotate each batch using a random angle sampled
        from an uniform distribution with extremes equal to -threshold and +threshold (in radians)
        :param threshold: Minimum error to be considered for non linear trajectories when using a linear predictor
        """
        super(TrajectoryDatasetVariableLen, self).__init__()

        self.data_location = data_location
        self.obs_percentage = obs_percentage
        self.pred_percentage = 1 - self.obs_percentage
        self.skip = skip
        self.delimiter = delim
        self.split_trajectories = split_trajectories
        self.batch_size = batch_size
        self.max_obs_len = max_obs_len
        self.max_pred_len = max_pred_len
        self.min_obs_len = min_obs_len
        if self.split_trajectories and not self.max_obs_len:
            self.max_obs_len = 8  # default value for observation length; prediction length already has default value
        self.also_backwards = also_backwards
        self.random_rotation_angle_std = random_rotation_angle_std
        self.random_rotation_angle_threshold = random_rotation_angle_threshold

        self.total_num_seqs = 0  # total number of sequences of trajectories spanned across time.

        if os.path.isdir(self.data_location):
            self.all_files = sorted(os.listdir(self.data_location))
            self.all_files = [os.path.join(self.data_location, _path) for _path in self.all_files]
        else:
            # is a single file
            self.all_files = [self.data_location]
        self.num_peds_in_seq = []
        self.seq_list = []
        self.seq_list_rel = []
        self.obs_seq_list = []
        self.obs_seq_list_rel = []
        self.pred_seq_list = []
        self.pred_seq_list_rel = []
        self.seq_start_end_list = []
        self.metadata_list = []

        for path in self.all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0])
            pedestrians = np.unique(data[:, 1])
            frame_data = [data[f == data[:, 0], :] for f in iter(frames)]
            pedestrian_data = [data[p == data[:, 1], :] for p in iter(pedestrians)]
            pedestrians_apart_of_seqs = []

            for frame in frame_data:
                # the actual pedestrian id is abstracted, what we retrieve is the index of pedestrian_data
                pedestrians_in_frame = [np.where((np.isclose(pedestrians, p)))[0][0] for p in iter(frame[:, 1])]
                if len(pedestrians_in_frame) <= min_ped:
                    continue
                # create batches for trajectories, with the requirement that the trajectories must have the same length
                #   and be synchronized through time (present in the exact same frames) to be apart of the same batch
                for p in pedestrians_in_frame:
                    if p in pedestrians_apart_of_seqs:
                        continue
                    # as of right now, a pedestrian will not be apart of two different batches!
                    pedestrians_in_same_seq = __get_pedestrians_for_same_seq__(pedestrians_in_frame, p, pedestrian_data)
                    if len(pedestrians_in_same_seq) <= min_ped:
                        continue
                    # seq - the set of trajectories (only x/y positions)
                    seq = np.array([pedestrian_data[p][:, 2:] for p in pedestrians_in_same_seq])
                    if self.also_backwards:
                        seq = np.concatenate((seq, np.flip(seq, axis=1)), axis=0)
                    seq_len = seq.shape[1]
                    # determine how much of the trajectory is to observe and to predict, given the provided parameters
                    if seq_len < min_traj_len or seq_len + 1 < min_obs_len:
                        # the sequence should have at least the minimum trajectory length, and one more length than the
                        # minimum observed trajectory length.
                        continue
                    if self.split_trajectories:
                        const_traj_len = self.max_obs_len + self.max_pred_len
                        if seq_len < const_traj_len:
                            # cannot split these trajectories, not long enough
                            continue
                        new_seq = None
                        for curr_seq_len in range(const_traj_len, seq_len + 1, const_traj_len):
                            seq_portion = seq[:, (curr_seq_len - const_traj_len):curr_seq_len, :]
                            new_seq = seq_portion if new_seq is None else np.concatenate((new_seq, seq_portion), axis=0)
                        obs_len = self.max_obs_len
                        pred_len = self.max_pred_len
                        seq = new_seq
                    else:
                        obs_len = int(round(self.obs_percentage * seq_len, 0))
                        if obs_len < min_obs_len:
                            obs_len = min_obs_len
                        pred_len = seq_len - obs_len
                        if self.max_pred_len is not None and self.max_pred_len > 0:
                            if pred_len > self.max_pred_len:
                                pred_len = self.max_pred_len
                                obs_len = seq_len - pred_len
                            elif same_pred_len:
                                if obs_len + self.max_pred_len <= seq_len:
                                    pred_len = self.max_pred_len
                                    obs_len = seq_len - pred_len
                                else:
                                    # trajectory length not enough to have specified number of predicted instants
                                    # without having to take those instants from observed trajectory
                                    continue
                        if self.max_obs_len is not None and self.max_obs_len > 0:
                            if same_obs_len and pred_len + self.max_obs_len > seq_len:
                                continue
                            if obs_len > self.max_obs_len:
                                diff = obs_len - self.max_obs_len
                                # discard older observed positions
                                seq = seq[:, diff:, :]
                                seq_len = seq_len - diff
                                obs_len = self.max_obs_len
                    pedestrians_apart_of_seqs.extend(pedestrians_in_same_seq)
                    self.num_peds_in_seq.append(len(pedestrians_in_same_seq))
                    seq = __convert_tensor_to_lstm_format__(seq, device)
                    if self.random_rotation_angle_std > 0:
                        # apply random rotation to the sequence by sampling from gaussian distribution
                        seq, angle = random_angle_rotation(seq, std=self.random_rotation_angle_std, inplace=True)
                    elif self.random_rotation_angle_threshold > 0:
                        # apply random rotation to the sequence by sampling from uniform distribution
                        seq, angle = random_angle_rotation_uniform(seq, threshold=self.random_rotation_angle_threshold,
                                                                   inplace=True)
                    else:
                        angle = 0
                    seq_rel = torch.zeros_like(seq)
                    # relative positions (relative displacement) - subtract previous position to current (and first
                    # position becomes 0.0 for x and y)
                    seq_rel[1:, :, :] = seq[1:, :, :] - seq[:-1, :, :]
                    obs_seq = seq[:obs_len, :, :]
                    obs_seq_rel = seq_rel[:obs_len, :, :]
                    pred_seq = seq[obs_len:, :, :]
                    pred_seq_rel = seq_rel[obs_len:, :, :]
                    self.__add_to_batch__(seq, seq_rel, obs_seq, obs_seq_rel, pred_seq, pred_seq_rel,
                                          num_seq_starts=2 if self.also_backwards else 1)
                    # the '* 2' is to include the relative trajectories
                    tensors_size = (seq.element_size() * seq.nelement()) * 2 / (1024 * 1024)  # size in MB
                    self.metadata_list.append(MetadataVariableLen(path, angle, size=tensors_size))
        # total number of sequences, or batches of sequences
        self.total_num_seqs = len(self.seq_list)
        self.average_peds_in_seq = np.mean(np.array(self.num_peds_in_seq))

    def __add_to_batch__(self, seq, seq_rel, obs_seq, obs_seq_rel, pred_seq, pred_seq_rel, num_seq_starts=1):
        """
        Group sequence of trajectories that have the same length into the an existing batch, until the specified batch
        size is reached, OR a new batch if no sequence of trajectories with that length has been created. The created
        sequence lists will be lists of tensors (cannot convert the whole data to a single tensor because dimensions
        won't match - side effect of dealing with variable trajectory lengths)
        :param seq: Tensor of shape (seq_len, batch, 2). The original full sequence of trajectories
        :param seq_rel: Tensor of shape (seq_len, batch, 2). The sequence of trajectories, but in terms of relative
        displacements (or velocities)
        :param obs_seq: Tensor of shape (obs_seq_len, batch, 2). The original past sequence of trajectories, that can be
        observed by models.
        :param obs_seq_rel: Tensor of shape (obs_seq_len, batch, 2). The past sequence of trajectories, that can be
        observed by models, but in terms of relative displacements (or velocities)
        :param pred_seq: Tensor of shape (pred_seq_len, batch, 2). The original past sequence of trajectories, that can
        be used to compare with the predictions made by models
        :param pred_seq_rel: Tensor of shape (pred_seq_len, batch, 2). The original past sequence of trajectories, that
        can be used to compare with the predictions made by models. This one is in terms of relative displacments (or
        velocities), so be careful when computing standard metrics like ADE or FDE with these values, since those are
        more specific to absolute positions. For that use pred_seq's.
        :return: nothing, the lists are updated inside this object
        """
        # NOTE: sequence lists will be lists of tensors (cannot convert the whole data to a single tensor because
        # dimensions won't match - side effect of dealing with variable trajectory lengths)
        seq_len = seq.shape[0]
        num_ped = seq.shape[1]
        curr_num_batches = len(self.seq_list)
        # The last batch of sequences for trajectories of the same length (will be the one with an incomplete batch)
        # unless the last batch is already complete too (has 'batch_size' sequences), if so will create a new batch.
        # Note that the list of sequences (or batches of sequences) is done in reverse to improve efficiency
        idx = next((curr_num_batches - idx - 1 for idx, b in enumerate(reversed(self.seq_list)) if
                    (b.shape[0] == seq_len)), None)
        if idx is None or self.seq_start_end_list[idx].shape[0] >= self.batch_size:
            # no new batch or last batch is full - new batch to be added to the list
            self.seq_list.append(seq)
            self.seq_list_rel.append(seq_rel)
            self.obs_seq_list.append(obs_seq)
            self.obs_seq_list_rel.append(obs_seq_rel)
            self.pred_seq_list.append(pred_seq)
            self.pred_seq_list_rel.append(pred_seq_rel)
            if num_seq_starts <= 1:
                seq_start_end = torch.LongTensor([[0, num_ped]])
            else:
                seq_start_end = torch.LongTensor([])
                for i in range(num_seq_starts):
                    seq_start_end = torch.cat((seq_start_end, torch.LongTensor(
                        [[int(num_ped / num_seq_starts * i), int(num_ped / num_seq_starts * (i + 1))]])), dim=0)
            self.seq_start_end_list.append(seq_start_end)
        else:
            # append data to that batch
            self.seq_list[idx] = torch.cat((self.seq_list[idx], seq), dim=1)
            self.seq_list_rel[idx] = torch.cat((self.seq_list_rel[idx], seq_rel), dim=1)
            self.obs_seq_list[idx] = torch.cat((self.obs_seq_list[idx], obs_seq), dim=1)
            self.obs_seq_list_rel[idx] = torch.cat((self.obs_seq_list_rel[idx], obs_seq_rel), dim=1)
            self.pred_seq_list[idx] = torch.cat((self.pred_seq_list[idx], pred_seq), dim=1)
            self.pred_seq_list_rel[idx] = torch.cat((self.pred_seq_list_rel[idx], pred_seq_rel), dim=1)
            new_start = self.seq_start_end_list[idx][-1, 1]  # end of last seq
            if num_seq_starts <= 1:
                seq_start_end = torch.LongTensor([[new_start, new_start + num_ped]])
            else:
                seq_start_end = torch.LongTensor([])
                for i in range(num_seq_starts):
                    seq_start_end = torch.cat((seq_start_end, torch.LongTensor(
                        [new_start + [int(num_ped / num_seq_starts * i),
                                      new_start + int(num_ped / num_seq_starts * (i + 1))]])), dim=0)
            self.seq_start_end_list[idx] = torch.cat((self.seq_start_end_list[idx], seq_start_end), dim=0)

    def __len__(self):
        return self.total_num_seqs

    def __getitem__(self, index):
        traj_len, num_peds = self.seq_list[index].shape[0], self.seq_list[index].shape[1]
        out = [
            self.obs_seq_list[index], self.pred_seq_list[index],
            self.obs_seq_list_rel[index], self.pred_seq_list_rel[index],
            self.metadata_list[index], torch.ones(num_peds, traj_len),
            self.seq_start_end_list[index]
        ]
        return out


def __get_pedestrians_for_same_seq__(pedestrians_in_frame, p, pedestrian_data):
    """
    Retrieve the pedestrians that are apart of the same sequence, having a certain 'primary' pedestrian that will
    always be apart of this sequence. Two pedestrians are apart of the same sequence if their trajectories are synced in
    time, i.e., if their trajectories belong to the same time frame, starting in the same instants and spanning across
    the same instants too (which also implies same trajectory length for all of them)
    :param pedestrians_in_frame: all pedestrains present in this frame, where trajectory for pedestrian 'p' starts
    :param p: actual 'primary' pedestrian, that will for sure belong to this sequence
    :param pedestrian_data: the list of trajectories for all pedestrians.
    :return:
    """
    pedestrians_in_same_seq = [p]
    for p_other in pedestrians_in_frame:
        if p_other == p:
            continue
        # NOTE - on biwi/crowds (ETH/UCY) datasets, there is no case where two trajectories, that start in the same
        # instant, have the same length but are not synced (correspond to the exact same frames). So the verification
        # of 'being' synced, isn't actually done here, to avoid an extra O(n) complexity step
        if pedestrian_data[p].shape[0] == pedestrian_data[p_other].shape[0] and pedestrian_data[p][0, 0] == \
                pedestrian_data[p_other][0, 0]:
            pedestrians_in_same_seq.append(p_other)
    return pedestrians_in_same_seq


def __convert_tensor_to_lstm_format__(numpy_array, device):
    #   Data format: batch, seq_len, input_size
    #   LSTM input format: seq_len, batch, input_size
    return torch.from_numpy(numpy_array).float().permute(1, 0, 2).to(device)


def seq_collate(data):
    """
    Collect the several fields of the (collate <-> collect and combine)
    :param data: the original data, not formatted
    :return: a tuple containing the several parts of the data:
        - obs_traj: Tensor of shape (obs_seq_len, batch, 2). The observed trajectory, in absolute coordinates
        - pred_traj: Tensor of shape (pred_seq_len, batch, 2). The predicted trajectory, in absolute coordinates
        - obs_traj_rel: Tensor of shape (obs_seq_len, batch, 2). The observed trajectory, in relative displacements
        - pred_traj_rel: Tensor of shape (pred_seq_len, batch, 2). The predicted trajectory, in relative displacements
        - non_linear_ped: Tensor of shape (batch). 1 if trajectory is non linear, 0 otherwise
        - loss_mask: Tensor of shape (traj_seq_len, batch). Loss mask to (optionally) use in training.
        - seq_start_end: Tensor containing the beginning and end (in terms of pedestrians in this batch) of a sequence.
            A sequence corresponds to a set of trajectories that belong to the same time frame. Note that the batch may
            have several sequences.
    """
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, metadata_list, loss_mask_list,
     seq_start_end) = zip(*data)

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0)
    pred_traj = torch.cat(pred_seq_list, dim=0)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.cat(seq_start_end, dim=0)
    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, metadata_list, loss_mask, seq_start_end]

    return tuple(out)


def data_loader_variable_len(args, device, path, _load_pred=False):
    """
    provide the data loader that can be used to iterate through the dataset
    :param args: arguments containing several dataset-related options
    :param device: torch.device to map the tensors to (like 'cpu', 'cuda')
    :param path: the path to the directory or file containing the data
    :param _load_pred: if meant to load predictions from file instead of actual data (not implemented)
    :return: the dataset, of type TrajectoryDatasetVariableLen, and the data loader, of type torch.utils.data.DataLoader
    """
    dataset = TrajectoryDatasetVariableLen(
        path, device,
        obs_percentage=args.obs_percentage,
        max_obs_len=args.obs_len,
        max_pred_len=args.pred_len,
        same_obs_len=args.use_same_obs_len,
        same_pred_len=args.use_same_pred_len,
        split_trajectories=args.split_trajectories,
        skip=args.skip,
        delim=args.delim,
        batch_size=args.batch_size,
        also_backwards=args.add_backwards_trajectories,
        random_rotation_angle_std=args.random_rotate_std,
        random_rotation_angle_threshold=args.random_rotate_thresh)
    # the batches will be provided by the dataset, since these are of variable length, and as such can't be controlled
    #   by the loader; the shuffle is to shuffle each of the batches (but each individual batch will not change)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=not args.do_not_shuffle if hasattr(args, 'do_not_shuffle') else False,
        collate_fn=seq_collate)
    return dataset, loader
