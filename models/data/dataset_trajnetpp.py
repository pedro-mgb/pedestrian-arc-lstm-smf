"""
Created on 27/May/2021

Contains processing of trajectories in Trajnet++ format. This data comes in ndjson file and has the following types of
data:
- A scene, identifying an interval of frames where there are one or more trajectories. The trajectory has one pedestrian
called the primary pedestrian ("p") - the most relevant pedestrian in the scene; in terms of computing metrics in that
scene, it will only be done for the primary pedestrian, the other pedestrians serve more as social context. It also
has a type of trajectory (I:static, II:linear, III:interaction, IV:non-interacting) and subtype for type III:interaction
(I:leader-follower, II:Collision-Avoidance, III:Group, IV:Other)
{“scene”: {“id”: 266, “p”: 254, “s”: 10238, “e”: 10358, “fps”: 2.5, “tag”: [2, []]}}
- A track, a certain position of a certain pedestrian. A track can belong to more than one scene, in the case that
scenes have overlapping intervals [s_scene1, e_scene1] and [s_scene2, e_scene2]
{“track”: {“f”: 10238, “p”: 248, “x”: 13.2, “y”: 5.85}}
    When a model predicts trajectories and outputs to a ndjson file, it has information like the prediction number
    (can be > 0 in case of multimodality), and the identifier of the scene the track belongs to
    {“track”: {“f”: 10238, “p”: 248, “x”: 13.2, “y”: 5.85, “pred_number”: 0, “scene_id”: 123}}


For more information about how these trajectories are made and structured, and also to learn more about Trajnet++,
here are a few links of interest:
- https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge
- https://thedebugger811.github.io/posts/2020/03/intro_trajnetpp/
- https://github.com/vita-epfl/trajnetplusplusdata/
- https://github.com/vita-epfl/trajnetplusplusdataset/
- https://github.com/vita-epfl/trajnetplusplustools/
- https://github.com/vita-epfl/trajnetplusplusbaselines/

CREDITS: The code is built from Trajnet++ repositories, so if you use this work, make sure you cite the work from
Trajnet++ authors:
@article{Kothari2020HumanTF,
  title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
  author={Parth Kothari and S. Kreiss and Alexandre Alahi},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.03639}
}
"""
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import trajnetplusplustools

from models.data.common_dataset import read_ndjson_file, Metadata
from models.utils.utils import random_rotation, center_scene


class MetadataTrajnetpp(Metadata):
    """
    Class to store metadata-related information regarding a trajectory or sequence of trajectories. This data is not
    100% essential for training / inference, but can provide additional information for auxiliary tasks
    """

    def __init__(self, origin_label, seq_id, rot_angle, center, prim_id, all_ids, size=0):
        """
        builds the Metadata object
        :param origin_label: label identifying the source of the trajectory (e.g. the file name)
        :param seq_id: ID that identifies the sequence of pedestrian trajectories (which is named scene in Trajnet++)
        :param rot_angle: The rotation angle of trajectories, in radius. Applies for cases where --normalize_scene or
        --augment_rotation command line arguments are supplied
        :param center: The displacement in xy coordinates. Applies for cases where --normalize_scene or
        :param prim_id: number identifying the primary pedestrian
        :param all_ids: list of numbers identifying all pedestrians, by order
        :param size: size (assumed MegaBytes/MB) in memory of the trajectories this metadata refers to
        """
        super(MetadataTrajnetpp, self).__init__(origin_label, size)
        self.id = seq_id
        self.primary_ped_id = prim_id
        self.all_ped_ids = all_ids
        self.angle = rot_angle
        self.center = center


class TrajnetppDataset(Dataset):
    """Dataset for the Trajectory datasets - Can handle Trajnetpp format of fixed length trajectories (although there
    is the possibility of having partial trajectories, for non-primary pedestrians).
    Each scene, which will be referred to as a sequence is said to be a sort of sub-batch
    (not to be confused with a batch returned at each step, that can contain multiple sequences in it), and comprises
    set of trajectories of the same length, synchronized in time.
    This idea of synchronized in time means that all those trajectories are present in the exact same time frame.
    For format of the trajectories, see the above file description comment, or go to the following link:
    https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge
    """

    def __init__(self, data_location, device=torch.device('cpu'), obs_len=9, pred_len=12, filter_tags=(),
                 filter_sub_tags=(), consider_partial_trajectories=True, normalize_scene=False, augment_rotation=False):
        """
        Initializes the dataset contents from the file(s)
        :param data_location: Location of the dataset files in the format (can be a directory or a single file).
        Assumed .ndjson format. See https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge
        :param device: device to map the tensors to. By default will go to cpu
        :param obs_len: Number of time-steps in input trajectories (to observe and process)
        :param pred_len: Number of time-steps in output trajectories (to perform prediction)
        :param filter_tags: list of tags to filter (i.e. to not consider). By default empty - consider all tags
        :param filter_sub_tags: list of sub-tags to filter. By default empty - consider all sub-tags
        :param consider_partial_trajectories: Use this flag if meant to include partial trajectories (of length smaller
        than obs_len+pred_len). For the instants without positions, will fill with NaN.
        :param
        """
        super(TrajnetppDataset, self).__init__()

        self.data_location = data_location
        self.filter_tags, self.filter_sub_tags = filter_tags, filter_sub_tags
        self.consider_partial_trajectories = consider_partial_trajectories
        self.normalize_scene, self.augment_rotation = normalize_scene, augment_rotation

        self.total_num_seqs = 0  # total number of sequences of trajectories spanned across time.

        if os.path.isdir(self.data_location):
            self.all_files = sorted(os.listdir(self.data_location))
            self.all_files = [os.path.join(self.data_location, _path) for _path in self.all_files]
        else:
            # is a single file
            self.all_files = [self.data_location]
        self.num_peds_in_seq = []
        self.obs_seq_list = []
        self.obs_seq_list_rel = []
        self.pred_seq_list = []
        self.pred_seq_list_rel = []
        self.seq_start_end_list = []
        self.metadata_list = []

        self.num_distinct_primary_pedestrians = 0
        self.seen_primary_ped = {}

        for path in self.all_files:
            data_reader = read_ndjson_file(path)
            self.seen_primary_ped[path] = []
            for scene_id, scene in data_reader.scenes():
                [main_tag, sub_tags_list] = data_reader.scenes_by_id[scene_id].tag
                if main_tag not in self.filter_tags or (
                        sub_tags_list and not any([s in self.filter_sub_tags for s in sub_tags_list])):
                    continue  # not in list of tags/sub-tags to filter
                ped_id_list = [t[0].pedestrian for t in scene]
                primary_ped_id = scene[0][0].pedestrian
                if primary_ped_id not in self.seen_primary_ped[path]:
                    self.num_distinct_primary_pedestrians += 1
                    self.seen_primary_ped[path].append(primary_ped_id)
                data = trajnetplusplustools.Reader.paths_to_xy(scene)
                # tensor of shape [obs_len+pred_len,num_peds,2]
                seq = torch.from_numpy(data).to(device).to(torch.float32)
                if seq.shape[0] != (obs_len + pred_len):
                    raise Exception(f'Got trajectory length {seq.shape[0]}, but expected length of '
                                    f'{obs_len}+{pred_len}={obs_len + pred_len}')
                if not self.consider_partial_trajectories:
                    # discard the trajectories that have nan values
                    seq = seq[:, torch.any(torch.all(~torch.isnan(seq), dim=0), dim=1), :]
                rot, center = None, None
                if normalize_scene:
                    seq, rot, center = center_scene(seq, obs_length=obs_len)
                elif augment_rotation:
                    seq, rot = random_rotation(seq)
                seq_rel = torch.zeros_like(seq)
                seq_rel[1:, :, :] = seq[1:, :, :] - seq[:-1, :, :]
                self.obs_seq_list.append(seq[:obs_len, :, :])
                self.obs_seq_list_rel.append(seq_rel[:obs_len, :, :])
                self.pred_seq_list.append(seq[obs_len:, :, :])
                self.pred_seq_list_rel.append(seq_rel[obs_len:, :, :])
                # primary pedestrian will always be the one with id 0 - the 'start' in seq_start_end
                # in trajnetpp repositories - seq_start_end <=> batch_split
                num_peds = seq.shape[1]
                if self.seq_start_end_list:
                    new_start = self.seq_start_end_list[-1][-1]
                    self.seq_start_end_list.append([new_start, new_start + num_peds])
                else:
                    self.seq_start_end_list.append([0, num_peds])
                # for the case where multiple files are loaded, one needs to distinguish between several scenes
                self.num_peds_in_seq.append(num_peds)
                self.total_num_seqs += 1
                tensors_size = seq.element_size() * seq.nelement() + seq_rel.element_size() * seq_rel.nelement()
                metadata = MetadataTrajnetpp(path, scene_id, rot, center, primary_ped_id, ped_id_list,
                                             size=tensors_size / (1024 * 1024))  # size in MB
                self.metadata_list.append(metadata)

    def __len__(self):
        return self.total_num_seqs

    def __getitem__(self, index):
        num_peds = self.obs_seq_list[index].shape[1]
        traj_len = self.obs_seq_list[index].shape[0] + self.pred_seq_list[index].shape[0]
        # non_linear_peds and loss_mask doesn't really apply to Trajnet++ format
        out = [
            self.obs_seq_list[index], self.pred_seq_list[index],
            self.obs_seq_list_rel[index], self.pred_seq_list_rel[index],
            self.metadata_list[index], torch.ones(num_peds, traj_len, device=self.obs_seq_list[index].device)
        ]
        return out


class TrajnetppPredictionsDataset(Dataset):
    """

    """

    def __init__(self, data_location, device=torch.device('cpu'), pred_len=12, consider_partial_trajectories=True,
                 num_samples=1):
        self.total_num_seqs = 0
        self.data_location = data_location
        self.consider_partial_trajectories = consider_partial_trajectories
        self.num_samples = num_samples

        self.total_num_seqs = 0  # total number of sequences of trajectories spanned across time.

        if os.path.isdir(self.data_location):
            self.all_files = sorted(os.listdir(self.data_location))
            self.all_files = [os.path.join(self.data_location, _path) for _path in self.all_files]
        else:
            # is a single file
            self.all_files = [self.data_location]
        self.num_peds_in_seq = []
        self.pred_seq_list = []
        self.seq_start_end_list = []
        self.metadata_list = []

        for path in self.all_files:
            data_reader = read_ndjson_file(path)
            for scene_id, scene in data_reader.scenes():
                ped_id_list = [t[0].pedestrian for t in scene]
                primary_ped_id = scene[0][0].pedestrian
                self.num_peds_in_seq.append(len(ped_id_list))
                num_pred = max([row.prediction_number for row in scene[0]]) + 1
                if num_pred < self.num_samples:
                    raise Exception(f'Some predictions in {path} lack the required number of samples '
                                    f'({self.num_samples})')
                # if there are more predictions than self.num_samples, those remaining will be discarded
                pred_list_for_scene = []
                for s in range(self.num_samples):
                    scene_sample_s = [[r for r in track if r.prediction_number == s and r.scene_id == scene_id]
                                      for track in scene]
                    # sometimes predictions from different scenes may be mixed, and the above line may cause "empty"
                    #   lists to be left in the variable, which causes the program to crash. The line below avoids it
                    scene_sample_s = [track for track in scene_sample_s if len(track) > 0]
                    data = trajnetplusplustools.Reader.paths_to_xy(scene_sample_s)
                    pred_sample = torch.from_numpy(data).to(device).to(torch.float32)
                    pred_list_for_scene.append(pred_sample)
                # tensor of shape [obs_len+pred_len,num_peds,2]
                if self.num_samples == 1:
                    pred = pred_list_for_scene[0]
                    if pred.shape[0] != pred_len:
                        raise Exception(f'Got trajectory length {pred.shape[0]}, but expected length {pred_len}')
                else:
                    pred = torch.cat([p.unsqueeze(0) for p in pred_list_for_scene])
                self.pred_seq_list.append(pred)
                new_start = self.seq_start_end_list[-1][-1] if self.seq_start_end_list else 0
                self.seq_start_end_list.append([new_start, new_start + len(ped_id_list)])
                tensors_size = pred.element_size() * pred.nelement()
                metadata = MetadataTrajnetpp(path, scene_id, None, None, primary_ped_id, ped_id_list,
                                             size=tensors_size / (1024 * 1024))  # size in MB
                self.metadata_list.append(metadata)
                self.total_num_seqs += 1

    def __getitem__(self, index):
        return self.pred_seq_list[index], self.metadata_list[index]

    def __len__(self):
        return self.total_num_seqs


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
    if len(data[0]) < 6:
        # It is not actual data, but predictions stored in files
        (pred_seq_list, metadata_list) = zip(*data)
        p_index = 1 if pred_seq_list[0].ndim < 4 else 2
        _len = [seq.shape[p_index] for seq in pred_seq_list]
        cum_start_idx = [0] + np.cumsum(_len).tolist()
        seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        seq_start_end = torch.tensor(seq_start_end, device=pred_seq_list[0].device, dtype=torch.long)
        return torch.cat(pred_seq_list, dim=p_index), seq_start_end, metadata_list

    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list, metadata_list, loss_mask_list) = zip(*data)

    _len = [seq.shape[1] for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_seq_list, dim=1)
    pred_traj = torch.cat(pred_seq_list, dim=1)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=1)
    pred_traj_rel = torch.cat(pred_seq_rel_list, dim=1)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.tensor(seq_start_end, device=obs_traj.device, dtype=torch.long)
    out = [obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, metadata_list, loss_mask, seq_start_end]

    return tuple(out)


def data_loader_trajnetpp(args, device, path, load_pred=False):
    """
    provide the data loader that can be used to iterate through the dataset with Trajnet++ configuration
    :param args: arguments containing several dataset-related options
    :param device: torch.device to map the tensors to (like 'cpu', 'cuda')
    :param path: the path to the directory or file containing the data
    :param load_pred: if meant to load predictions from file instead of actual data
    :return: the dataset, of type TrajnetppDataset, and the data loader, of type torch.utils.data.DataLoader
    """
    if load_pred:
        dataset = TrajnetppPredictionsDataset(
            path, device,
            pred_len=args.pred_len,
            consider_partial_trajectories=not args.no_partial_trajectories,
            num_samples=args.num_samples
        )
    else:
        dataset = TrajnetppDataset(
            path, device,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            consider_partial_trajectories=not args.no_partial_trajectories,
            normalize_scene=args.normalize_scene,
            augment_rotation=args.augment_rotation,
            filter_tags=args.filter_tags,
            filter_sub_tags=args.filter_sub_tags)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.do_not_shuffle if hasattr(args, 'do_not_shuffle') else False,
        collate_fn=seq_collate)
    return dataset, loader
