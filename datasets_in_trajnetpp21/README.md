# DATA USED

Belonging to two different datasets: BIWI (ETH) and Crowds (UCY).
The structure of the trajectories follows the
[Trajnet++ configuration](https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge).
There are scenes (aka mini-batches or situations with one or more pedestrians),
each having a primary pedestrian, that is the focus of that scene.
It is the only pedestrian for which ADE/FDE or collision metrics are computed.

The actual data conversion was made using this [repository](https://github.com/vita-epfl/trajnetplusplusdataset), with the data from '../datasets' folder, using option `--chunk_stride 21` (for the repeated portions of a trajectory for a primary pedestrian)

This dataset configuration is meant to be used as input to sparse motion fields 
method in conjunction with LSTM networks
(but can be used as input to other methods too).

To cite the Trajnet++ benchmark, use:
```
@article{Kothari2020HumanTF,
  title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective},
  author={Parth Kothari and S. Kreiss and Alexandre Alahi},
  journal={ArXiv},
  year={2020},
  volume={abs/2007.03639}
}
```

To cite Sparse Motion Fields, use:

```
@article{BARATA2021107631,
title = {Sparse motion fields for trajectory prediction},
journal = {Pattern Recognition},
volume = {110},
pages = {107631},
year = {2021},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2020.107631},
url = {https://www.sciencedirect.com/science/article/pii/S0031320320304349},
author = {Catarina Barata and Jacinto C. Nascimento and João M. Lemos and Jorge S. Marques},
}
```

## Structure of folders

There are a total of 5 folders containing data, relevant to different contexts:

- ***test***: used for testing models, i.e., evaluating their performance. This is known as ***test_private*** in Trajnet++.
- ***test_no_pred***: used for testing models, but without access to the real future data for evaluation. This is known as ***test*** in Trajnet++.
- ***train***: used to train the model, when applicable. Expected the validation set (***val*** directory) to be used 
as well.
- ***train_and_val***: used to train the model, when applicable. Assumes that the validation set (***val*** directory) 
is not needed. This corresponds to the combination of the data in ***train*** and ***val*** directories.
- ***val***: used to evaluate the model at train time, or help choosing hyperparameters. Expected to be used with the
training set (***train*** directory)

Note that is information of each scene (find out more about the different scenes below) in both ***train*** and 
***test*** subsets.

This is particularly useful for methods that attempt to learn information regarding the scene. 
The spare motion fields method, and the use of it with LSTM are examples of this. With this, one can train the model(s) 
having available information of each scene, and also test those models on those several scenes (with unseen trajectory data).

## ETH 

Files 'biwi_hotel' 'biwi_eth', each belonging to an entirely different scene (one in hotel, one at entry of campus). Each file was split (around 50-50) and used for training / testing.

Original dataset available here (may be slightly different) - https://icu.ee.ethz.ch/research/datsets.html -> Search for "BIWI Walking Pedestrians dataset"

To cite, use:
```
@inproceedings{pellegrini2009you,
  title={You'll never walk alone: Modeling social behavior for multi-target tracking},
  author={Pellegrini, Stefano and Ess, Andreas and Schindler, Konrad and Van Gool, Luc},
  booktitle={2009 IEEE 12th International Conference on Computer Vision},
  pages={261--268},
  year={2009},
  organization={IEEE}
}
```

## UCY

Two different scenes:

1. univ (at university campus) - 3 files: 'students001', 'students003' and 'uni_examples'. These files may also appear as starting with 'crowds_', to identify the dataset to which they belong to (and to appear first than zara, alphabetically)
2. zara - 3 files: 'crowds_zara01', 'crowds_zara02' and 'crowds_zara03'

The content of each of these scenes was split (around 50-50) for training / testing

Original dataset **used to be available** here (may be slightly different) - https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data

To cite, use:
```
@inproceedings{lerner2007crowds,
  title={Crowds by example},
  author={Lerner, Alon and Chrysanthou, Yiorgos and Lischinski, Dani},
  booktitle={Computer graphics forum},
  volume={26},
  number={3},
  pages={655--664},
  year={2007},
  organization={Wiley Online Library}
}
```
