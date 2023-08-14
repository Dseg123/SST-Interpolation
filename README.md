# SST-Interpolation
Summer 2023 HMEI Internship with University of Washington Ocean Dynamics Group -- Deep Learning for Sea Surface Temperature Interpolation

This repository contains a rough framework for deep learning models used in SST Interpolation during my internship with the University
of Washington's Ocean Dynamics Group through Princeton's High Meadows Environmental Institute.

More background and detail on the methods can be found in the final presentation at 
https://docs.google.com/presentation/d/1oZTKBqgCtWaPyxrfsxod-DkiYNJK7LbGKqcRYiqn5AI/edit?usp=sharing.

Below is a brief explanation of the important files and general architecture:

**Collecting the Data**
To collect the data one can run "get_all_data.py". Here a variety of parameters can be specified -- including satellite name, spatial
and temporal ranges, and the name of the output directory. Then the data is collected from the given satellite and placed in "raw_data".
The data is processed into tiles and blocks and placed in "full_data". The clouds are isolated from the full data and placed in 
"cloud_data". And the clouds are randomly distributed among tiles to create "masked_data". In the process, moment statistics for the
full and masked data are generated and stored in "full_stats.npy" and "masked_stats.npy" respectively. Also, the set
of training ids and validation ids (in the form "(block, tile, index within block)") are generated and stored in 
"train_ids.npy" and "val_ids.npy". Finally, all the params used in data collection are stored in "data_params.npy" for future reference.

No data are included within the repository because the data folders are too large to be stored remotely in Github. However, they can be
regenerated easily using get_all_data.py with the same parameters.

**Collecting the GAN Data**
The GAN portion of the architecture requires separate data collection to isolate cloudless samples of the desired shape. To collect
the GAN data one can run "gan/collect_gan_data.py". Here a variety of parameters can be specified -- including input data directory,
spatial size of samples, and the name of the output directory. Then the data is collected from the given input data directory and placed
in "samples". In the process, mean, std, and sample count are are calculated and stored in "stats.npy". Finally, all the params used in 
data collection are stored in "data_params.npy" for future reference.

No GAN data are included within the repository because the data folders are too large to be stored remotely in Github. However, they can be
regenerated easily using collect_gan_data.py with the same parameters.

**Creating a Dataset**
"get_src_dataset" in "src/generators.py" can be used to create a Tensorflow Dataset of the source data, and similarly "get_gan_dataset" in
"gan/gan_dataset.py" can be used to create a Tensorflow Dataset of the GAN data.

**Training a model**
Right now only two model architectures have been created: unseeded ConvLSTM and seeded ConvLSTM (which uses a random seed as a separate channel
to support GAN usage). Both of these models are included in "src/models.py". The two losses for training the models -- MSE loss and GAN loss --
are also included in "src/losses.py". There are three training loops in "train_model_manual.py" (for unseeded ConvLSTM with just MSE loss),
"train_model_unseeded.py" (for unseeded ConvLSTM with both losses), and "train_gan_model.py" (for seeded ConvLSTM with both losses). There
is also only one discriminator architecture which is included in "gan/gan_models.py". Discriminator and generator architectures can be trained
separately from the interpolation task using the training loop in "gan/gan_loop.py". 

**Testing a model**
A model can be loaded from one of the experiments in the "experiments" directory. Some code for evaluating models of different types can be found
in "evaluate_models.py", which also includes code for evaluating the baseline "kriging" method. Results can also be plotted using code in 
"final_visualizations.ipynb".

**Experiments**
The "experiments" directory contains trained models for the three different training procedures, labelled "experiment", "unseed_gan_experiment", and
"gan_experiment" respectively along with the timestamp at which they were trained. Each experiment contains the interpolator (and if applicable discriminator)
weights for each epoch, alongside csv files with the loss curves obtained during training. A new experiment will automatically be added when running
one of the training procedures.

**Other**
Many of the other jupyter notebooks were simply used for generating visualizations but may still contain useful code. The "subplots_animation.gif" files
contain example outputs from "final_visualizations.ipynb". The file "git_pusher.py" is useful for pushing experiments or data directories that may 
surpass the pack limit of a standard git commit.

**Future work**
Expect more models and perhaps additional pipelines to be added in the coming weeks! But the core functionality should remain the same. Contact me
at de7281@princeton.edu if you have any questions!


