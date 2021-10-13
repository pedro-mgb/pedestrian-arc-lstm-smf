# DATA USED

NOTE: To use this data in training/evaluation - You must supply "--obs_len 5 --pred_len 6"

Same configuration as folder ['../datasets_in_trajnetpp_21'](../datasets_in_trajnetpp21/), but with trajectories of length 11
(5 instants for observation, 6 for prediction; use of "--chunk_stride 11").
The main reason to have this smaller length is because some scenes
(ETH and Hotel from BIWI Walking Pedestrians dataset) have a lot of its trajectories be discarded, 
due to having length smaller than the original length proposed in Trajnet++ (21 instants). 

With this, one can use more trajectories for these scenes for training and evaluation.
It is worth noting that the other scenes (Zara and Univ from Crowds by example dataset)
will also have  an inflated number of trajectories,
larger than the actual number of distinct pedestrians.

To find out more about this configuration, see the
[README from datasets_in_trajnetpp21](../datasets_in_trajnetpp21/README.md).