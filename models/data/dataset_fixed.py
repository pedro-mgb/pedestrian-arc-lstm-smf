"""
Created on 05/March/2021

File that contains processing of trajectory datasets, available in '../datasets(...)' folders, assuming fixed length
This is what is used by social GAN method. Some differences may be present, but they are minimal

@inproceedings{gupta2018social,
  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  number={CONF},
  year={2018}
}

Source: https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py
"""
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from models.data.common_dataset import read_file, poly_fit, Metadata


class MetadataFixedLen(Metadata):
    """
    Added in September 2021. Not present in SGAN repo. Contains extra information not 100% essential for training or
    testing the models. For instance, the number of non-linear trajectories of pedestrians.
    """
    def __init__(self, origin_label, non_linear_ped, size=0):
        """
        builds the Metadata object
        :param origin_label: label identifying the source of the trajectory (e.g. the file name)
        :param non_linear_ped: Tensor of shape (num_peds). Indicates if each trajectory is classified as linear (0) or
        non-linear (1)
        :param size: size (assumed MegaBytes/MB) in memory of the trajectories this metadata refers to
        """
        super(MetadataFixedLen, self).__init__(origin_label, size)
        self.non_linear_ped = non_linear_ped



def seq_collate(data):
    # September 2021 - Changed from original repo - non_linear_ped incorporated on metadata
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     metadata_list, loss_mask_list) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    # Data format: batch, input_size, seq_len
    # LSTM input format: seq_len, batch, input_size
    obs_traj = torch.cat(obs_seq_list, dim=0).permute(2, 0, 1)
    pred_traj = torch.cat(pred_seq_list, dim=0).permute(2, 0, 1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).permute(2, 0, 1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=0).permute(2, 0, 1)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, metadata_list,
        loss_mask, seq_start_end
    ]

    return tuple(out)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self, data_dir, device=torch.device('cpu'), obs_len=8, pred_len=12, skip=1, threshold=0.002,
            min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - device: torch.device to map tensors to
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        # 05-MARCH-2021 - CHANGED FROM ORIGINAL SOCIAL GAN - Support path to single file, instead of just directories
        if os.path.isdir(self.data_dir):
            all_files = sorted(os.listdir(self.data_dir))
            all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        else:
            # is a single file
            all_files = [self.data_dir]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        metadata_list = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                num_peds_considered = 0
                _metadata = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    # Linear vs Non-Linear Trajectory - Now added to Metadata object (September 2021)
                    non_linear = poly_fit(curr_ped_seq, pred_len, threshold)
                    tensor_size = curr_ped_seq.nbytes * 2  # '* 2' to include rel array too
                    _metadata.append(MetadataFixedLen(path, non_linear, tensor_size))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    metadata_list += _metadata
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float).to(device)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float).to(device)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float).to(device)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float).to(device)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float).to(device)
        self.metadata_list = metadata_list
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.metadata_list[index], self.loss_mask[start:end, :]
        ]
        return out


def data_loader_fixed_len(args, device, path, _load_pred=False):
    dset = TrajectoryDataset(
        path, device,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=not args.do_not_shuffle if hasattr(args, 'do_not_shuffle') else False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
