# Visualize model predictions

Visualization tools form an important way of evaluating the quality of the predictions, 
being a more interactive and user-friendly alternative than a quantitative evaluation
(via [evaluation script](models/evaluate.py)).

To run visualization tools, you require libraries like matplotlib to be installed.
When [setting up](README.md#getting-started), make sure you installed dependencies from
[requirements.txt](requirements.txt) and not from
[requirements_lightweight.txt](requirements_lightweight.txt).

This repository contains 2 scripts for visualization:

- __[plot_predictions_social.py](other_scripts/plot_predictions_social.py):__
Plot trajectory predictions for situations with social interactions.
Specific to Trajnet++. Allows plotting of multiple model predictions.
They include plotting of neighbour motion/positions, and allow to plot a map of the environment
via --environment_location argument.

- __[plot_single_predictions.py](other_scripts/plot_single_predictions.py):__
Plot trajectory predictions for single trajectories.
Each plot contains only one GT trajectory, but can contain predictions from different models.
The script allows plotting of a map of the environment via --environment_location argument.

Should you have any problem or difficulty running the scripts,
you can always post an Issue describing what is failing to work. 

## Introduction to running visualization scripts

There are several arguments that can be used to configure the plotting.
Supply the help flag to read about each one (WARNING - large output to command line):
```
python other_scripts/plot_predictions_social.py -h
```
```
python other_scripts/plot_single_predictions.py -h
```
Alternatively see the [parser_options.py file](models/utils/parser_options.py).

Some of the most important arguments are:

- `--test_dir`: the directory or file with the data to be plotted.
Ideally you should only plot data from one scene/file at a time, to avoid confusion between scenes.
- `--model_paths`: You must supply a list (at least one) of paths to the models you wish to see.
In case you want to also use pre-computed predictions, you can supply the path to the file(s) containing the
pre-computed predictions
- `--model_labels`: You must supply a list of labels to name and identify each of the models.
The number of labels supplied must be equal to the number of models.
- `--max_trajectories`: The maximum number of trajectory plots to perform.
By default a low number is supplied, so the screen isn't bombarded with a large number of plots
- `--environment_path`: Path to a map of the environment to draw along with the trajectories.
These maps are available in the [datasets_utils/environment](datasets_utils/environment) directory.
- `--animate`: Supply this flag for 
[other_scripts/plot_predictions_social.py](other_scripts/plot_predictions_social.py)
if you wish to view an animation (trajectory evolving through time).

Example for plotting some animations with Arc-LSTM-SMF and Social-LSTM predictions,
for the Zara scene, with trajectory lengths 21:
```
python other_scripts/plot_predictions_social.py --test_dir datasets_in_trajnetpp21/test/crowds_zara02.ndjson --model_paths saved_models/other_models_21/lstm_social_None_modes1/crowds_zara02.ndjson saved_models/arc_lstm_smf_21/zara_arc_lstm_smf_trajnetpp21.pt --model_labels S-LSTM Arc-LSTM-SMF --environment_location datasets_utils/environment/crowds_zara.txt --animate
```

## JSON Templates for configuring visualization

#### Available in the [other_scripts/json_args](other_scripts/json_args) directory.

To facilitate visualization without having to go through the several possible arguments,
we some template JSON files with pre-configured arguments.

The only thing you require is to pass the desired file name
via the --load_args_from_json command line argument.

Example for [other_scripts/plot_predictions_social.py](other_scripts/plot_predictions_social.py):
```
python other_scripts/plot_predictions_social.py --load_args_from_json other_scripts/json_args/plot_social_predictions_zara_state_of_the_art.py
```
Example for [other_scripts/plot_single_predictions.py](other_scripts/plot_single_predictions.py):
```
python other_scripts/plot_single_predictions.py --load_args_from_json other_scripts/json_args/plot_single_predictions_hotel_state_of_the_art.py
```
WARINING: the --max_trajectories is set to a high value.
If you wish to only a see a very small sample of figures, change the argument in the JSON file.


You may modify these templates as you wish to visualize trajectories for other models / data.


## Pre-generated animations and plots

If you do not wish / are not able to run the scripts, we have made publicly available
some extra figures to see some predictions of our models
when compared to state-of-the-art and simple baselines.
The plots/animations can be accessed
[in this Drive folder](https://drive.google.com/drive/folders/1enbIeKKh92AGdj5Xy3fGvtWAgIwoLV93?usp=sharing) 