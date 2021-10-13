"""
Created on September 11th, 2021

Simple script to compute the size of a dataset (with one or multiple files) in memory (RAM).
Note that this measure will give an approximate approximate value

IMPORTANT NOTE:
The computation may stop with an error if the data does not fit in memory.
This makes training with such datasets impossible in our repository.
The easy alternative would be to use a subset of this data.
If that is not desirable, some form of lazy loading should be done. Sadly it is not implemented in this repository.
To implement such a procedure may take some time, and lazy loading can be quite slow, depending on the files.
For more information see: https://discuss.pytorch.org/t/how-to-use-dataset-larger-than-memory/37785
"""
import sys
import time
import argparse
import gc
import os

import psutil
import torch

from models.data.loaders import get_data_loader
from models.utils.parser_options import add_parser_arguments_for_data, add_parser_arguments_misc

parser = argparse.ArgumentParser()
parser = add_parser_arguments_for_data(parser)
parser = add_parser_arguments_misc(parser)
parser.add_argument('--data_location', default='datasets_in_trajnetpp21/train/',
                    help='the relative path to the directory where the f data files are, or relative path to a file')
parser.add_argument('--process_files_individually', action='store_true',
                    help='If supplied, will retrieve each file individually, and get their memory content. '
                         'Each file will be loaded sequentially - after the size of that file has been computed, it '
                         'will be removed from memory. Especially useful for datasets that do not fit in memory')


def main(args):
    if args.use_gpu:
        args.use_gpu = False
        print("WARNING: Use of GPU was de-activated since this script only supports CPU")
    # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    device = torch.device('cpu')
    data_location = os.path.relpath(args.data_location)

    print("Getting available memory...")
    v_memory = psutil.virtual_memory()
    free_memory = v_memory.free / (1024 * 1024)
    print(f"CPU: total of {v_memory.total / (1024 * 1024):.3f}MB;\t {v_memory.used / (1024 * 1024):.3f}MB is used")
    if torch.cuda.is_available():
        total_mem_gpu = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: Total of {total_mem_gpu / (1024 * 1024):.3f}MB memory reserved")

    if args.process_files_individually and os.path.isdir(data_location):
        print('Loading each file separately (avoids out-of-memory errors)')
        data_path_list = [os.path.join(data_location, _path) for _path in sorted(os.listdir(data_location))]
    else:
        print('Loading all data at the same time (may have issues if data does not fit in memory)')
        data_path_list = [data_location]
    print('')

    full_size, full_t_size = 0, 0  # expected unit: MegaBytes (MB) or 2^20 bytes
    for path in data_path_list:
        trajs_size = 0
        if args.process_files_individually:
            print(f"Reading {path}... ", end='')
        # get_data_loader returns a function object to retrieve the dataloader
        _, loader = (get_data_loader(args, path))(args, device, path)
        #  print(f"Dataset in {path}:{os.linesep}\tsize {sys.getsizeof(dataset)} + {sys.getsizeof(loader)}")
        num_batches = len(loader)
        loader_size = get_size(loader) / (1024 * 1024)
        print("Done!")
        # print(f'Dataset: {dataset_size / (1024 * 1024)}MB;\t Loader: {loader_size / (1024 * 1024)}MB')
        for i, batch in enumerate(loader):
            print(f"\r Batch: {i + 1}/{num_batches}", end='', flush=True)
            (_, _, _, _, metadata, _, _) = batch
            trajs_size += sum([m.size for m in metadata])
            time.sleep(0.0001)
        print('\r Clearing memory...', end='')
        dataset = loader = None
        gc.collect()  # explicitly free any unused memory
        print('\r', end='')
        if args.process_files_individually:
            percentage = trajs_size / loader_size * 100
            print(f"{path} with approximately {loader_size:.2f}MB in memory.{os.linesep}\t "
                  f"Of which {trajs_size:.2f}MB ({percentage:.2f}%) come directly from trajectories {os.linesep}")
        dataset = loader = None
        gc.collect()  # explicitly free any unused memory
        full_size += loader_size
        full_t_size += trajs_size

    percentage = full_t_size / full_size * 100
    print(f"{os.linesep}Data in {args.data_location} occupying approximately {full_size:.2f}MB in memory."
          f"{os.linesep}\t Of which {full_t_size:.2f}MB ({percentage:.2f}%) come directly from trajectories")

    if full_size * 1.1 > free_memory:  # the 1.1 is to account for additional memory allocated by training/testing
        print(F"ERROR: THE DATASET IN {args.data_location} most likely does not fit in memory for this machine")


def get_size(obj, seen=None):
    """Recursively finds size of objects
    Might not work 100%
    Credits: https://goshippo.com/blog/measure-real-size-any-python-object/"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif torch.is_tensor(obj):
        size += obj.element_size() * obj.nelement()
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == '__main__':
    arguments = parser.parse_args()
    main(arguments)
