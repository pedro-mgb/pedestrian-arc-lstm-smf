"""
Created on June 28th 2021

WARNING: Experimental

Allows to load precomputed pooling data (occupancy and directional) from disk into memory.
For big datasets (such as the full Trajnet++ benchmark), this data would not fit in memory.
Since this is experimental, it should be used with smaller datasets, such as the datasets_in_trajnetpp21 folder.
"""
import argparse
from itertools import product
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.data.loaders import load_train_val_data
from models.utils.parser_options import add_parser_arguments_for_training, get_interaction_module_label
from models.utils.utils import tensor_size_bytes
from models.interaction_modules.shape_based import ShapeBasedPooling
from models.lstm.loaders import build_interaction_module


class ShapeDataset(Dataset):
    """

    """

    def __init__(self, args, trajectory_loaders, pooling_layer, device=torch.device('cpu'), path=None, log=True):
        self.pool_data, self.occ_data = torch.tensor([], device=device), torch.tensor([], device=device)
        self.seq_start_end = torch.tensor([], device=device, dtype=torch.long)
        self.seq_shape_start_end = torch.tensor([], device=device, dtype=torch.long)
        self.all_obs_traj, self.all_pred_traj_gt = torch.tensor([], device=device), torch.tensor([], device=device)
        self.all_prim_obs_traj, self.all_prim_pred_traj_gt = torch.tensor([], device=device), \
                                                             torch.tensor([], device=device)
        # THIS data will be initialized layer
        self.shape_type = self.shapes = self.pool_dims = self.pool_occ_dims = self.num_shapes = None
        self.total_num_seqs, self.total_num_batches = 0, 0
        self.write, is_directory = False, False  # meant to write, and write to directory or to file
        files = None  # to store files to write or to read from
        if path is not None:
            # a path was provided, it can be for reading, or for writing, if the path does not contain any files
            if os.path.isdir(path):
                is_directory = True
                files = os.listdir(path)
                if len(files) > 0:
                    # there is content - read it and use it
                    files = [os.path.join(path, _path) for _path in files]
                else:
                    # meant to write in a directory?
                    raise Exception('TODO not implemented')
                    # files = path
                    # self.write = True
            elif os.path.isfile(path):
                # it is a file
                files = [path]
            else:
                # it is a path to a file that does not exist
                files = path
                self.write = True
            if not self.write:
                total_size = np.sum([os.stat(f).st_size for f in files])
                if total_size > 2e9:  # larger than 2GB - will not read; SHOULD BE SPLIT IN PARTITIONS
                    raise Exception(f'ERROR! Data is too large to load - {total_size / 1e9} GB')
                # read the files content
                first_file = True
                for f in files:
                    file_data = torch.load(f, map_location=device)
                    self.all_obs_traj = torch.cat((self.all_obs_traj, file_data['obs']), dim=1)
                    self.all_pred_traj_gt = torch.cat((self.all_pred_traj_gt, file_data['pred']), dim=1)
                    self.all_prim_obs_traj = torch.cat((self.all_prim_obs_traj, file_data['obs_prim']), dim=1)
                    self.all_prim_pred_traj_gt = torch.cat((self.all_prim_pred_traj_gt, file_data['pred_prim']), dim=1)
                    self.pool_data = torch.cat((self.pool_data, file_data['pool_data']), dim=1)
                    self.occ_data = torch.cat((self.occ_data, file_data['occ_data']), dim=1)
                    self.total_num_seqs += file_data['num_seqs']
                    self.total_num_batches += file_data['batches']
                    if first_file:
                        self.seq_shape_start_end = torch.cat((self.seq_start_end, file_data['seq_shape_start_end']))
                        self.seq_start_end = torch.cat((self.seq_start_end, file_data['seq_start_end']))
                        self.shape_type, self.shapes, self.num_shapes = file_data['shape_type'], file_data['shapes'], \
                                                                        file_data['num_shapes']
                        self.pool_dims, self.pool_occ_dims = file_data['pool_dims'], file_data['occ_dims']
                    else:
                        new_start = self.seq_start_end[-1, 1]
                        self.seq_start_end = torch.cat((self.seq_start_end, file_data['seq_start_end'] + new_start))
                        new_shape_start = self.seq_shape_start_end[-1, 1]
                        self.seq_shape_start_end = torch.cat((self.seq_shape_start_end,
                                                              file_data['seq_shape_start_end'] + new_shape_start))
                self.__create_rel_trajectories__()
                return  # all data has been loaded
        assert hasattr(args, 'variable_shape') and args.variable_shape and isinstance(pooling_layer,
                                                                                      ShapeBasedPooling), \
            'THIS \'DATASET\' IS NOT AVAILABLE FOR MODELS THAT DO NOT HAVE VARIABLE SHAPE CONFIGURATION'
        assert pooling_layer.args.pooling_type.SOCIAL, \
            'The Social Pooling Configuration is not possible to build the shape pooling data apriori, because it ' \
            'requires use of an LSTM state, and, therefore, an LSTM network'
        print("No Shape Data to read. Will compute from scratch. Depending on dataset size, it can take some time")
        # get shape values used in training
        self.shape_type = args.pooling_shape
        if self.shape_type == 'arc':
            # results in tensor of shape [1, num_radii * num_angles]
            self.shapes = list(product(args.radius_values, args.angle_values))
            self.pool_dims = [pooling_layer.pooling_dim, pooling_layer.args.n_a, pooling_layer.args.n_r]
            self.pool_occ_dims = [1, pooling_layer.args.n_a, pooling_layer.args.n_r]
        else:
            self.shapes = [args.cell_side_values]  # results in tensor of shape [1, num_cell_sides]
            self.pool_dims = [pooling_layer.pooling_dim, pooling_layer.args.n, pooling_layer.args.n]
            self.pool_occ_dims = [1, pooling_layer.args.n, pooling_layer.args.n]
        self.num_shapes = len(self.shapes)
        self.total_num_batches = np.sum(np.array([len(t) for t in trajectory_loaders]))
        self.pool_dims = [pooling_layer.pooling_dim, pooling_layer.args.n_a, pooling_layer.args.n_r]
        curr_batch_num, num_sequences_skipped = 0, 0
        for loader in trajectory_loaders:
            for batch in loader:
                curr_batch_num += 1
                # only get the absolute trajectories and start and end of the sequence (or minibatch)
                # for social datasets, it is assumed that relative trajectories (displacements) are not necessary
                # IF NEEDED, this dataset/loader can be changed to also send that data
                (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, metadata, _, seq_start_end) = batch
                if not args.primary_ped_only:
                    raise Exception('NOT IMPLEMENTED; ALSO dis-advised due to high usage of memory')
                # only store values for primary pedestrians - SHOULD USE THIS
                full_traj = torch.cat((obs_traj, pred_traj_gt), dim=0)
                full_traj_rel = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
                # social methods do not use 1st position (because there is no displacement).
                # So seq_mask has shape [obs_len - 1 + pred_len, batch, 2]
                seq_mask = (torch.isnan(full_traj[:-1, :, 0]) + torch.isnan(full_traj[1:, :, 0]) +
                            torch.isnan(full_traj_rel[1:, :, 0])) == 0
                num_seqs = seq_start_end.shape[0]
                curr_seq_log_num = 0
                for (start, end) in seq_start_end:
                    curr_seq_log_num += 1
                    if end - start == 1:
                        # just one pedestrian - no social interaction - all pooling will yield the same
                        num_sequences_skipped += 1
                        continue
                    full_seq = full_traj[:, start:end, :]  # discard first instant (not used)
                    pool_data_seq, occ_data_seq = torch.tensor([], device=device), torch.tensor([], device=device)
                    for t in range(1, full_seq.shape[0]):
                        # for a specific time step, will include all shapes
                        pool_data_t, occ_data_t = torch.tensor([], device=device), torch.tensor([], device=device)
                        if log:
                            print(f'\rPROCESSING BATCH {curr_batch_num}/{self.total_num_batches}, '
                                  f'SEQ {curr_seq_log_num}/{num_seqs}, TIME {t}/{full_seq.shape[0] - 1}', end='')
                        past_pos_with_nans, curr_pos_with_nans = full_seq[t - 1], full_seq[t]
                        past_pos = past_pos_with_nans[seq_mask[t - 1, start:end], :]  # apply sequence mask
                        curr_pos = curr_pos_with_nans[seq_mask[t - 1, start:end], :]  # apply sequence mask
                        for s in self.shapes:
                            if self.shape_type == 'arc':
                                pooling_layer.shape_values.all_radius = torch.full((curr_pos.shape[0],), s[0],
                                                                                   device=device)
                                pooling_layer.shape_values.all_angles = torch.full((curr_pos.shape[0],),
                                                                                   s[1] * np.pi / 180, device=device)
                            else:
                                pooling_layer.shape_values.all_sides = torch.full((curr_pos.shape[0],), s, device=device)
                            pool_data_s_all_ped, occ_data_s_all_ped = pooling_layer(None, curr_pos, past_pos, False)
                            # for arc tensors of shape [1, pool_dim, n_r, n_a] and [1, 1, n_r, n_a]
                            # for grid: tensors of shape [1, pool_dim, n, n] and [1, 1, n, n]
                            pool_data_s, occ_data_s = pool_data_s_all_ped[0:1], occ_data_s_all_ped[0:1]
                            # include data from all shapes
                            pool_data_t = torch.cat((pool_data_t, pool_data_s))
                            occ_data_t = torch.cat((occ_data_t, occ_data_s))
                        pool_data_seq = torch.cat((pool_data_seq, pool_data_t.unsqueeze(0)))
                        occ_data_seq = torch.cat((occ_data_seq, occ_data_t.unsqueeze(0)))
                    if torch.nonzero(occ_data_seq).nelement() == 0:
                        # shape pooling data for this sequence contains just zeros - not used
                        num_sequences_skipped += 1
                        continue
                    self.total_num_seqs += 1  # increased here so seq_start_end doesn't account for discarded sequences
                    self.all_prim_obs_traj = torch.cat((self.all_prim_obs_traj, obs_traj[:, start:start + 1, :]), dim=1)
                    self.all_prim_pred_traj_gt = torch.cat((self.all_prim_pred_traj_gt,
                                                            pred_traj_gt[:, start:start + 1, :]), dim=1)
                    self.all_obs_traj = torch.cat((self.all_obs_traj, obs_traj[:, start:end, :]), dim=1)
                    self.all_pred_traj_gt = torch.cat((self.all_pred_traj_gt, pred_traj_gt[:, start:end, :]), dim=1)
                    new_start = 0 if self.seq_start_end.nelement() == 0 else self.seq_start_end[-1, -1]
                    self.seq_start_end = torch.cat((
                        self.seq_start_end, torch.tensor([[new_start, new_start - start + end]],
                                                         device=device, dtype=torch.long)))
                    # the start and end of a sequence for a primary pedestrian is given by the total number of shapes
                    self.seq_shape_start_end = torch.cat(
                        (self.seq_shape_start_end, torch.tensor([[self.num_shapes * (self.total_num_seqs - 1),
                                                                  self.num_shapes * self.total_num_seqs]],
                                                                device=device, dtype=torch.long)))
                    # for arc: tensor of shape [obs_len-1+pred_len, num_primary_ped * num_shapes, pool_dim, n_r, n_a]
                    self.pool_data = torch.cat((self.pool_data, pool_data_seq), dim=1)
                    self.occ_data = torch.cat((self.occ_data, occ_data_seq), dim=1)
        self.__create_rel_trajectories__()
        if log:
            print(f'{os.linesep}DONE! Skipped {num_sequences_skipped} sequences due to not having relevant shape '
                  f'pooling data (all zeroes) - meaning there was no social context for these sequences '
                  f'(using provided shapes)')
        if self.write:
            data_size = tensor_size_bytes(self.all_obs_traj, 'MB') + tensor_size_bytes(self.all_pred_traj_gt, 'MB') + \
                        tensor_size_bytes(self.all_prim_obs_traj, 'MB') + \
                        tensor_size_bytes(self.all_prim_pred_traj_gt, 'MB') + \
                        tensor_size_bytes(self.seq_start_end, 'MB') + tensor_size_bytes(self.pool_data, 'MB') + \
                        tensor_size_bytes(self.occ_data, 'MB')
            full_data = {'obs': self.all_obs_traj, 'pred': self.all_pred_traj_gt, 'obs_prim': self.all_prim_obs_traj,
                         'pred_prim': self.all_prim_pred_traj_gt, 'seq_start_end': self.seq_start_end,
                         'seq_shape_start_end': self.seq_shape_start_end, 'num_seqs': self.total_num_seqs,
                         'shape_type': self.shape_type, 'shapes': self.shapes, 'num_shapes': self.num_shapes,
                         'pool_dims': self.pool_dims, 'occ_dims': self.pool_occ_dims, 'pool_data': self.pool_data,
                         'occ_data': self.occ_data, 'batches': self.total_num_batches}
            torch.save(full_data, files)
            if log:
                print(f'Written {data_size:.3f}MB of data to {files}')

    def __create_rel_trajectories__(self):
        self.all_prim_pred_traj_gt_rel = torch.zeros_like(self.all_prim_pred_traj_gt)
        self.all_prim_pred_traj_gt_rel[0, :, :] = self.all_prim_pred_traj_gt[0] - self.all_prim_obs_traj[-1]
        self.all_prim_pred_traj_gt_rel[1:, :, :] = \
            self.all_prim_pred_traj_gt[1:] - self.all_prim_pred_traj_gt[:-1]

    def __len__(self):
        return self.total_num_seqs

    def __getitem__(self, index):
        start, end = self.seq_start_end[index][0], self.seq_start_end[index][1]
        obs, pred = self.all_obs_traj[:, start:end, :], self.all_pred_traj_gt[:, start:end, :]
        start_s, end_s = self.seq_shape_start_end[index][0], self.seq_shape_start_end[index][1]
        num_shape_seqs = end_s - start_s  # should be equal to num_shapes
        obs_prim, pred_prim, pred_prim_rel = self.all_prim_obs_traj[:, index, :], \
                                             self.all_prim_pred_traj_gt[:, index, :], \
                                             self.all_prim_pred_traj_gt_rel[:, index, :]
        obs_prim, pred_prim, pred_prim_rel = obs_prim.unsqueeze(1).repeat(1, num_shape_seqs, 1), \
                                             pred_prim.unsqueeze(1).repeat(1, num_shape_seqs, 1), \
                                             pred_prim_rel.unsqueeze(1).repeat(1, num_shape_seqs, 1)
        pool_data, occ_data = self.pool_data[:, start_s:end_s], self.occ_data[:, start_s:end_s]
        return [obs, pred, obs_prim, pred_prim, pred_prim_rel, pool_data, occ_data, self.shapes]


def seq_collate(data):
    """
    Collect the several fields of a sequence and combine them for direct use by models (collate <-> collect and combine)
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
    (obs, pred, obs_pp, pred_pp, pred_pp_rel, pool, occ, shape_list) = zip(*data)

    _len = [seq.shape[1] for seq in obs]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    _len_s = [seq.shape[1] for seq in obs_pp]
    cum_s_start_idx = [0] + np.cumsum(_len_s).tolist()
    seq_shape_start_end = [[start, end] for start, end in zip(cum_s_start_idx, cum_s_start_idx[1:])]

    obs_traj, pred_traj = torch.cat(obs, dim=1), torch.cat(pred, dim=1)
    obs_pp_traj, pred_pp_traj, pred_pp_traj_rel = torch.cat(obs_pp, dim=1), torch.cat(pred_pp, dim=1), \
                                                  torch.cat(pred_pp_rel, dim=1)
    pool_data, occ_data = torch.cat(pool, dim=1), torch.cat(occ, dim=1)
    seq_start_end = torch.tensor(seq_start_end, device=obs_traj.device, dtype=torch.long)
    seq_shape_start_end = torch.tensor(seq_shape_start_end, device=obs_traj.device, dtype=torch.long)

    # return single list if provided shapes are equal for all the items in this batch; else return full list
    shapes_return = shape_list[0] if shape_list.count(shape_list[0]) == len(shape_list) else shape_list
    out = [obs_traj, pred_traj, obs_pp_traj, pred_pp_traj, pred_pp_traj_rel, pool_data, occ_data, seq_start_end,
           seq_shape_start_end, shapes_return]

    return tuple(out)


def data_loader_shapes(args, trajectory_loaders, pool_layer, device):
    """
    provide the data loader that can be used to iterate through the dataset with Trajnet++ configuration
    :param args: arguments containing several dataset-related options.
    Also includes, (OPTIONAL) the path to the directory or file containing the data for the shape-based configuration.
    Note that this is particularly useful when this data is too big to fit in memory, or computation takes too long.
    Note however that this will mean that the data will be loaded FROM DISK everytime a batch is meant to be retrieved
    (increasing the number of processes won't have benefit IO-Bound operations)
    :param trajectory_loaders:
    :param pool_layer:
    :param device: torch.device to map the tensors to (like 'cpu', 'cuda')
    :return: the dataset, of type TrajnetppDataset, and the data loader, of type torch.utils.data.DataLoader
    """
    dataset = ShapeDataset(args, trajectory_loaders, pool_layer, device, args.shape_data_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.do_not_shuffle if hasattr(args, 'do_not_shuffle') else False,
        collate_fn=seq_collate)
    return dataset, loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_parser_arguments_for_training(parser)
    _args = parser.parse_args()
    if _args.use_gpu and not torch.cuda.is_available():
        _args.use_gpu = False
        print("WARNING: Use GPU was activated but CUDA is not available for this pytorch version. "
              "Will use CPU instead")
    t_device = torch.device('cuda') if _args.use_gpu else torch.device('cpu')
    # FORCE some arguments to specific values, that in this context should nor need to be used
    _args.do_not_shuffle = True  # to be in same order as original trajectories
    _args.val_dir = None  # no need to get validation data, meant only to be for a single path of data
    _args.primary_ped_only = True  # this implementation was SPECIFIC for Trajnet++ and only primary pedestrians.

    _trajectory_loaders, _, _ = load_train_val_data(_args, t_device, None)
    assert not _args.fixed_len and not _args.variable_len, \
        'ERROR - THIS IMPLEMENTATION IS CURRENTLY ONLY AVAILABLE FOR TRAJNET++ DATA CONFIGURATION'
    pool_type, shape_type = get_interaction_module_label(_args)
    shape_pooling_layer = build_interaction_module(_args, pool_type, shape_type)
    shape_pooling_layer.eval()  # no need for gradients here

    shape_dataset, shape_loader = data_loader_shapes(_args, _trajectory_loaders, shape_pooling_layer, t_device)
    if not shape_dataset.write:
        # data not being written to file
        for i, b in enumerate(shape_loader):
            # SOME STATISTICS ABOUT THE SHAPE DATA CAN GO HERE
            (obs_t, pred_t, obs_pp_t, pred_pp_t, pred_pp_t_rel, pool_d, occ_d, start_end, shape_start_end, l_shapes) = b
            print(f'Batch number {i + 1}')
