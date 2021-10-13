# (Sparse) Motion Fields for Trajectory Prediction

This folder contains code that utilizes motion fields, in particular Sparse Motion Fields work from Barata et. al to forecast pedestrian trajectories.

By default, the sparse motion fields are learned on data from a single scene, and evaluated from test data of that same scene.
These motion fields attempt to learn scene-specific information from trajectories, limiting the predictions to walkable areas and to avoid obstacles.

## Disclaimer

Note that the code to train the motion fields is not available, since it belongs to the authors.

One can only load an already trained model (usually in .mat format) 

## Credits

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
