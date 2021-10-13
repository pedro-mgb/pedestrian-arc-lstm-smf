# Folder: other_scripts

Contains scripts that can be useful to use along with the [model code](../models),
for tasks like visualization, dataset statistics, etc.

Make sure you call these scripts from the parent repository folder (`cd ..`).

## Relevant files

- [data/display_dataset_trajectories.py](data/display_dataset_trajectories.py):
Script to plot all trajectories associated to a data file.
Example for Biwi Hotel scene, train and test set:
    ```
    python data/display_dataset_trajectories.py --data_location datasets/biwi/biwi_hotel.txt --variable_len --distinguish_start_end --switch_x_y --invert_y --environment_location datasets_utils/environment/biwi_hotel.txt
    ```

- [data/stats_datasets.py](data/stats_datasets.py):
Script to compute some statistics regarding one or multiple data files.
Statistics include such information as pedestrian velocity, crowd density, among others.
Example for our Trajnet++ test set with trajectory length L=21:
    ```
    python data/stats_datasets.py --do_not_plot --data_location datasets_in_trajnetpp21/test --social_stats
    ```

- [plot_predictions_social.py](plot_predictions_social.py):
Plot trajectory predictions for situations with social interactions.
Specific to Trajnet++. Allows plotting of multiple model predictions.
They include plotting of neighbour motion/positions, and allow to plot a map of the environment
via --environment_location argument.
To find out more on how to run this script, see the [VISUALIZING README](../VISUALIZING.md).
The directory JSON args directory contains a list of arguments to help in visualizing trajectories.
Supply the path to the file via `--load_args_from_json` argument.

- [plot_single_predictions.py](plot_single_predictions.py):
Plot trajectory predictions for single trajectories.
Each plot contains only one GT trajectory, but can contain predictions from different models.
The script allows plotting of a map of the environment via --environment_location argument.
To find out more on how to run this script, see the [VISUALIZING README](../VISUALIZING.md).
The directory JSON args directory contains a list of arguments to help in visualizing trajectories.
Supply the path to the file via `--load_args_from_json` argument.

- [train_several_scenes.py](train_several_scenes.py):
Training script, to train several models from several scenes.,or contexts.
Supports, for instance, training the Arc-LSTM-SMF model in our Trajnet++ data
for the existing 4 scenes (eth, hotel, univ, zara).
For more information on training, see the [TRAINING README](../TRAINING.md)

