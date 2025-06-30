# Unrolled-NMF
Deep unrolled NMF methods for Automatic Music Transcription:

Automatically transcribe any audio file using a pretrained unrolled NMF model

### Structure

**/src**:
    contains the implementation of the model and some functions for the training

**{piano, synth}-dataset**:
    Contain 7 audio/ MIDI files pair as small toy datasets (recordings from virtual piano and virtual synth)

**model**:
    contains pretrained models weights

**cluster**:
    contains all the necessary material to run a training on the cluster

### Steps to train:

In the **/cluster** folder, the `job_script.sh` file is the one containing the command to be ran on the cluster, you can modify here the `learning_rate`(LR), the number of `epochs` to be made (EPOCHS), the `batch_size` (BATCH), the size of the `subset` of the dataset to train on (SUBSET), as well as the train/ test `split` size (SPLIT) 

This command will run the `trainer.py` file that loads the dataset (*MAPS*), loads the model and trains it with the provided parameters.

The model is saved under **/models**.