# DATA USED

The original data files, that were converted to Trajnet++ using
[publicly available code](https://github.com/vita-epfl/trajnetplusplusdataset).
The result of that conversion is in folders
[datasets_in_trajnetpp21](../datasets_in_trajnetpp21) and
[datasets_in_trajnetpp11](../datasets_in_trajnetpp11).  


This data belongs to two different datasets: ETH and UCY.
To use this data for training/evaluation, supply either --variable_len or --fixed_len.
With variable


## Structure of folders

There are a total of 4 folders containing data, relevant to different contexts:

- ***test***: used for testing models, i.e., evaluating their performance.
- ***train***: used to train the model, when applicable. Expected the validation set (***val*** directory) to be used 
as well.
- ***train_and_val***: used to train the model, when applicable. Assumes that the validation set (***val*** directory) 
is not needed. This corresponds to the combination of the data in ***train*** and ***val*** directories.
- ***val***: used to evaluate the model at train time, or help choosing hyperparameters. Expected to be used with the
training set (***train*** directory)

Note that is information of each scene (find out more about the different scenes below) in both ***train*** and 
***test*** subsets.

There are also 4 other folders, with test/train/val data specific to each of the 4 available scenes (biwi_eth, biwi_hotel, crowds_univ and crowds_zara).

This is particularly useful for methods that attempt to learn information regarding the scene. 
The spare motion fields method, and the use of it with LSTM are examples of this. With this, one can train the model(s) 
having available information of each scene, and also test those models on those several scenes.

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
