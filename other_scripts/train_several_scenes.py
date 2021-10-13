"""
Created on March 21st 2021
Script to train model with a leave-one-out approach (folder datasets_sgan)
or for each scene separately (folders datasets, datasets_in_trajnetpp21 and datasets_in_trajnetpp11).
Several models will be trained and validated leaving out data from one context.
The procedure will create S models, where S is the number of different contexts (or scenes) to train.
For standard leave-one-out, it should be 5 models. To train in each of the BIWI/Crowds scenes, it should be 4 models.
"""
import argparse
import os

from models.utils.parser_options import add_parser_arguments_for_training
from models.lstm.train import main as train

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_training(parser)

loo_option, biwi_crowds_option = 'leave_one_out', 'biwi_crowds'
parser.add_argument('--train_method', type=str, choices=[loo_option, biwi_crowds_option], default=biwi_crowds_option)

exception_dir = ['raw']  # for sgan_datasets; specific for --leave_one_out
biwi_crowds_scenes_dir = ['biwi_eth', 'biwi_hotel', 'crowds_univ', 'crowds_zara']


def main(args):
    training_dir = args.train_dir
    sub_dirs = []
    d_names = []

    if args.train_method == loo_option:
        for d in os.listdir(training_dir):
            path = os.path.join(training_dir, d)
            if d not in exception_dir and os.path.isdir(path):
                sub_dirs.append(path)
                d_names.append(d)
    elif args.train_method == biwi_crowds_option:
        for d in os.listdir(training_dir):
            path = os.path.join(training_dir, d)
            if d in biwi_crowds_scenes_dir and os.path.isdir(path):
                sub_dirs.append(path)
                d_names.append(d)
    model_base_name = args.save_to
    for dataset_name, directory in zip(d_names, sub_dirs):
        # print(name, dir)
        # we assume that the model path will have an extension (a .)
        extension_index = model_base_name.index('.')
        args.save_to = model_base_name[:extension_index] + '_' + dataset_name + model_base_name[extension_index:]
        args.train_dir = os.path.join(directory, 'train')
        args.val_dir = os.path.join(directory, 'val')
        if not os.path.exists(args.val_dir):
            print(f"WARNING - Validation directory nor present for {dataset_name}. Using train instead")
            args.val_dir = os.path.join(directory, 'train')
        # print(args.save_to, args.train_dir, args.val_dir)
        if args.train_method == 'leave_one_out':
            print(f'Training model with leave one out, with every dataset except {dataset_name}')
        elif args.train_method == 'biwi_crowds':
            print(f'Training model on data only from scene {dataset_name}')
        train(args)


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
