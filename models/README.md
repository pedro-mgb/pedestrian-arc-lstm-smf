# models folder

Contains most of the code of this repository, such as the building of the Arc-LSTM-Model
and its standalone parts, as well as training and evaluation scripts.

For the case of scripts, make sure you call them from the parent repository folder (`cd ..`).

## Folders and Files

- [classical](classical): Folder with implementation of classical non-Neural Network methods for
trajectory forecast, such as [Constant Velocity (CV)](classical/constant_velocity.py)

- [data](data): Folder containing code to read and process trajectory data
([Trajnet++](data/dataset_trajnetpp.py) and other configurations) and also
[scene-specific environments](data/environment.py).

- [evaluate.py](evaluate.py): Script to evaluate the models, including LSTM-based,
Sparse Motion Fields (supply "--smf" flag), and Constant Velocity (supply "--cv" flag).

- [fields](fields): Folder with an evaluation-only implementation of sparse motion fields.
The training code is not publicly available.
Refer to [this README](../README.md#download-trained-models) to see how to obtain pre-trained
Sparse Motion Fields models (as well as other models).

- [interaction_modules](interaction_modules): Folder containing interaction
modules or layers to integrate with LSTM networks. The modules are
[Arc pooling, Directional-Grid pooling and Social-Grid pooling](interaction_modules/shape_based.py).
Also contains modules that vary the pooling shape dimensions, based on
[heuristics](interaction_modules/simple_shape_config.py) and [other LSTMs](interaction_modules/complex_shape_config.py).

- [losses_and_metrics.py](losses_and_metrics.py): File with the implementation of training losses
(L2-based and NLL), as well as metrics such as ADE, FDE, and collisions between pedestrians.

- [lstm](lstm): Folder containing implementation of LSTM architectures,
such as [base LSTM](lstm/lstm.py), [LSTM-SMF](lstm/lstm_fields.py),
[Arc-LSTM](lstm/lstm_interactions.py), and [Arc-LSTM-SMF](lstm/lstm_fields_interactions.py).
Also contains the [training script](lstm/train.py) for all of these models.
For more information on Training, see [TRAINING README](../TRAINING.md).

- [modules.py](modules.py): File containing implementation of miscellaneous modules
that can be integrated with LSTM.

- [utils](utils): Folder containing code with utilities for several contexts, such as
[evaluation](utils/evaluator.py), [command line arguments parsing](utils/parser_options.py),
and [visualization](utils/plotting.py).