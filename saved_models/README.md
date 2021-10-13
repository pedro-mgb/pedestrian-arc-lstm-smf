# saved_models direcory

This is the default directory where trained models will be saved to,
and where they can be used for evaluation (predicting trajectories in a test set).

The models should have a .pt extension, and can be loaded using
`torch.load(<path_to>/saved_models/<model_filename>.pt)`

### Already Trained models from this repo

Can be downloaded from the instructions in [this README](../README.md#download-trained-models).

Contains files `base_lstm_trajnetpp*` for base LSTM networks trained on Trajnet++ data
for length [L=21](../datasets_in_trajnetpp21) and [L=11](../datasets_in_trajnetpp11).
__Contains the following folders:__

- [arc_lstm_smf_11](arc_lstm_smf_11), [arc_lstm_smf_21](arc_lstm_smf_21):
The interaction and scene-aware [Arc-LSTM-SMF](../models/lstm/lstm_fields_interactions.py)
models, trained per each scene on Trajnet++ data for length
[L=11](../datasets_in_trajnetpp11) and [L=21](../datasets_in_trajnetpp21), respectively.

- [interaction_models11](interaction_models11), [interaction_models21](interaction_models21):
Interaction-aware models, such as Arc-LSTM, and its variants with variable shape dimensions.
There is 1 model trained on all scenes, for Trajnet++ data with length
[L=11](../datasets_in_trajnetpp11) and [L=21](../datasets_in_trajnetpp21), respectively.

- [lstm_smf_i_11](lstm_smf_i_11), [lstm_smf_i_21](lstm_smf_i_21):
The scene-aware [LSTM-SMF-I](../models/lstm/lstm_fields.py) (use of `--simple_fields` flag)
models, trained per each scene on Trajnet++ data for length
[L=11](../datasets_in_trajnetpp11) and [L=21](../datasets_in_trajnetpp21), respectively.

- [lstm_smf_ii_11](lstm_smf_ii_11), [lstm_smf_ii_21](lstm_smf_ii_21):
The scene-aware [LSTM-SMF-II](../models/lstm/lstm_fields.py) (use of `--feed_all_fields` flag)
models, trained per each scene on Trajnet++ data for length
[L=11](../datasets_in_trajnetpp11) and [L=21](../datasets_in_trajnetpp21), respectively.

- [sparse_motion_fields](sparse_motion_fields):
The scene-aware [Sparse Motion Fields (SMF)](../models/fields/sparse_motion_fields.py),
trained per each scene on the [original data trajectories](../datasets),
with variable length (--variable_len flag; training code is private).

### Pre-computed predictions from other repos

Can be downloaded from the instructions in [this README](../README.md#download-trained-models).

These predictions can be used in the [evaluation script](../models/evaluate.py).
For that, you must supply the the path to the predictions via the --model_pred_path command line argument
(search in [parser_options.py](../models/utils/parser_options.py)
or run `python models/evaluate.py -h` for more information).


__Contains the following folders:__

- [other_models_21](other_models_21):
pre-computed predictions for several state-of-the-art models,
trained on Trajnet++ data for length [L=21](../datasets_in_trajnetpp21).
Models trained using [original Trajnet++ code](https://github.com/pedro-mgb/trajnetplusplusbaselines).
These models are:
    - [Social-LSTM](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf) -
    predictions in [lstm_social_None_modes1](other_models_21/lstm_directional_None_modes1) directory.
    
    - [Directional-LSTM](https://arxiv.org/pdf/2007.03639.pdf) -
    predictions in [lstm_directional_None_modes1](other_models_21/lstm_social_None_modes1) directory.
    
    - [Social-GAN](https://arxiv.org/pdf/1803.10892.pdf) - 
    predictions in [sgan_hiddenstatemlp_deterministicl2_modes1](other_models_21/sgan_hiddenstatemlp_deterministicl2_modes1)
    directory for deterministic version, and 
    [sgan_hiddenstatemlp_lrs20_50it_k20l2_modes50](other_models_21/sgan_hiddenstatemlp_lrs20_50it_k20l2_modes50)
    for multimodal version (total of 50 samples).
    You can limit the number of samples at evaluation by regulating the "--num_samples" command line argument.
    
-  [other_models_21](other_models_11):
pre-computed predictions for several state-of-the-art models,
trained on Trajnet++ data for length [L=11](../datasets_in_trajnetpp11).
Models trained using [original Trajnet++ code](https://github.com/pedro-mgb/trajnetplusplusbaselines).
These models are:
    - [Social-LSTM](https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf) -
    predictions in [lstm_social_None_modes1](other_models_11/lstm_directional_None_modes1) directory.
    
    - [Directional-LSTM](https://arxiv.org/pdf/2007.03639.pdf) -
    predictions in [lstm_directional_None_modes1](other_models_11/lstm_social_None_modes1) directory.
    
    - [Social-GAN](https://arxiv.org/pdf/1803.10892.pdf) - 
    predictions in [sgan_hiddenstatemlp_deterministicl2_modes1](other_models_11/sgan_hiddenstatemlp_deterministicl2_modes1)
    directory for deterministic version.