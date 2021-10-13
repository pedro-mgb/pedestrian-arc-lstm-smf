# DATA USED

Belonging to two different datasets: ETH and UCY.
To be used a Leave-One-Out (LOO) approach with a total of 5 different contexts:
ETH, Hotel, Univ, Zara1 and Zara2.

## Similar work that used these datasets

The following are links to public repositories of models that were trained and tested using this exact data:

- Social GAN: https://github.com/agrimgupta92/sgan

- Social-STGCNN: https://github.com/abduallahmohamed/Social-STGCNN

To use this data in the same fashion as Social GAN and Social-STGCNN works, use command line argument
"--fixed_len --obs_len 8 --pred_len 12".
To train 5 models for the 5 different contexts, you can see file
[train_several_scenes.py](../other_scripts/train_several_scenes.py).

## ETH 

Associated to folders 'eth' and 'hotel'.

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

Associated to folders 'univ', 'zara1' and 'zara2'.

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
