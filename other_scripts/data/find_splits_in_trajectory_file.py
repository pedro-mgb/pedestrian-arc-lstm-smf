"""
Created on March 14th 2021
Sample file to decide when a split in trajectory happens for a particular file.
The purpose of this is to provide reassurance that we can split a file of trajectories without splitting trajectories
in half and also not splitting interacting trajectories (common to same instants).
This split can be to divide a file in training and testing (or validation).
"""
import argparse
import os

import numpy as np

from models.utils.parser_options import *
from models.data.common_dataset import read_file

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_data(parser)
parser.add_argument('--data_location', default='datasets_sgan/raw/all_data/',
                    help='the relative path to the directory where the data files are, or relative path to a file')


def main(args):
    if os.path.isdir(args.data_location):
        all_files = os.listdir(args.data_location)
        all_files = [os.path.join(args.data_location, _path) for _path in all_files]
    else:
        # is a single file
        all_files = [args.data_location]

    print("NOTE: Each element of a split list (for each file) features", os.linesep, os.linesep)

    for path in all_files:
        split_list = []
        data = read_file(path, args.delim)
        frames = np.unique(data[:, 0])
        pedestrians = np.unique(data[:, 1])
        frame_data = [data[f == data[:, 0], :] for f in iter(frames)]
        pedestrian_data = [data[p == data[:, 1], :] for p in iter(pedestrians)]
        # pedestrian data is organized by the first frame where they appear
        num_frames = len(frame_data)
        num_pedestrians = len(pedestrian_data)
        frame_start, frame_end = frame_data[0][0, 0], frame_data[-1][-1, 0]
        frame_range = frame_end - frame_start
        max_frame_id = 0
        traj_lens = [len(traj) for traj in pedestrian_data]
        avg_traj_len = np.mean(traj_lens)
        sum_traj_len = np.sum(traj_lens)
        traj_len_acc = 0
        # decide about the splits
        for idx, person in enumerate(pedestrian_data):
            # get last position of pedestrian; see if next pedestrian starts after
            last_frame = person[-1, 0]
            traj_len_acc += person.shape[0]
            if last_frame > max_frame_id:
                max_frame_id = last_frame
            if idx + 1 >= len(pedestrian_data):
                # reached end of list
                break
            next_person = pedestrian_data[idx + 1]
            next_person_first_frame = next_person[0, 0]
            if next_person_first_frame > max_frame_id:
                percentage_frames_covered = "{0:.3f}".format(float(next_person_first_frame - frame_start) / frame_range)
                num_pedestrians_covered = "{0:.3f}".format(float(idx + 1) / num_pedestrians)
                traj_lens_covered = "{0:.3f}".format(float(traj_len_acc) / sum_traj_len)
                split_list.append({'frame_id': next_person_first_frame, 'frames': percentage_frames_covered,
                                   'peds': num_pedestrians_covered, 'traj_lens': traj_lens_covered})
        print("Splits for ", path)
        print(f"Frames from {frame_start} to {frame_end} - {frame_range} range; {num_frames} frames")
        print(f"Average trajectory length - {avg_traj_len}; Sum of lengths - {sum_traj_len}")
        print(f"Number of pedestrians - {num_pedestrians}")
        print(split_list)
        print(os.linesep)


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
