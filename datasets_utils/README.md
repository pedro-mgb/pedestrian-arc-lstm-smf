# Datasets Extra Utilities folder

This contains other information that may be useful, besides the actual files with trajectories containing the data.

Each of the sub-folders is described below:

## 'environment' folder

Contains .txt files with information regarding the static environment of the several scenes being used.
This includes the presence of obstacles, or limits of unwalkable areas,
as well as the limits or bounds of the scene (in which all dataset trajectories are included)

The files contain presence of line segments (text line starts with 'l'),
and circles (text line starts with 'c'). These obstacles are in world coordinates.

The biwi_eth and biwi_hotel obstacles were obtained originally from 
(OpenTraj repository)[https://github.com/crowdbotp/OpenTraj], with some added obstacles.
The dataset from crowds_zara is not directly the one from there, since the trajectories were converted using a different homography matrix.
There are no obstacles for crowds_univ, only scene limits.

See file [models/data/environment.py](../models/data/environment.py) to know more on how this is used.

To cite OpenTraj, provider of this information, 
use:
```
@inproceedings{amirian2020opentraj,
      title={OpenTraj: Assessing Prediction Complexity in Human Trajectories Datasets}, 
      author={Javad Amirian and Bingqing Zhang and Francisco Valente Castro and Juan Jose Baldelomar and Jean-Bernard Hayet and Julien Pettre},
      booktitle={Asian Conference on Computer Vision (ACCV)},
      number={CONF},      
      year={2020},
      organization={Springer}
}
```

## 'homography' folder

Contains the homography matrices, to convert the trajectory data in image coordinates
(pixels in the original scene images) to metric units.
Each file contains 3 lines, and 3 values per line (separated by tabs),
representing the 3x3 homography.

Note of course that these homographies are always approximated, and so the conversion from
world coordinates to pixels (for this you must use the INVERSE of the provided homography) will never be exact.
However, for the sake of vizualizing trajectories in the original image (see 'scene_images' folder for such example).

**Credits:** The homography matrices from the biwi/crowds (aka ETH/UCY) datasets were taken
from [the Scene-LSTM work](https://github.com/trungmanhhuynh/Scene-LSTM/tree/master/data_utils/homography_matrix%20).
To cite Scene-LSTM, please use:

> Huynh, Manh, and Gita Alaghband. "Trajectory prediction by coupling scene-LSTM with human movement LSTM." International Symposium on Visual Computing. Springer, Cham, 2019.

## 'scene_images' folder

This folder contains images for several scenes apart of the datasets used for training and testing models
(e.g. biwi, aka ETH, and crowds, aka UCY, datasets).

These images are frames from the original videos, provided in the datasets.

Each file name indentifies the scene image, and also the instant it refers to (can be an approximation).
Note that the frame number may not actually be the one present in the dataset configuration in 'datasets' folder.
Our data is originally taken from [the social gan repository](https://github.com/agrimgupta92/sgan/),
and the interpolation used there is different.
Data here appears every 10 frame ids, but originally for biwi and crowds datasets, it was every 6 frame ids.
As such, do not trust 100% the frame id values for the file names.