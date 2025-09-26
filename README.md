# Unrolled-NMF
Deep unrolled NMF methods for Automatic Music Transcription:

This repository contains an unrolled version of the Multiplicative Updates algorithm to solve the NMF for Automatic Music Transcription, based on the work of Christophe Kervazo and Jérémy Cohen  [[1]](#1) [[2]](#2).

The main goal is to get the transcription of an audio file represented by its CQT spectrogram, a tensor $\boldsymbol{M}$, by solving the NMF: $\boldsymbol{M} \simeq \boldsymbol{WH}$ where:

- $\boldsymbol{W}$: dictionnary tensor containing the spectrum of each individual notes of the recording.

- $\boldsymbol{H}$: activation tensor containing the moments when each note is activated througout the recording.

<p align="center">
<img src=https://github.com/edabier/Unrolled-NMF/blob/a533e2583d1874db33e63e65dfe44665fd3db9d1/images/illustration-NMF.png width=500>
</p>

We propose to solve the NMF using the MU algorithm, with the $\beta$-divergence cost function, and we **unroll** this iterative algorithm, each iteration acting like a neural network layer, inside which we add **acceleration factors**, that are predicted by **CNNs** ($A_W$ and $A_H$):

$$ W \leftarrow W \odot \boldsymbol{A_W(W)} \odot \frac{[(WH)^{\beta - 2}.M]H^T}{[WH]^{\beta -1} H^T}$$
$$H \leftarrow H \odot \boldsymbol{A_H(H)} \odot \frac{W^T[(WH)^{\beta - 2}.M]}{W^T[WH]^{\beta -1}}$$ 

### Structure

**/src**:
    contains the implementation of the model and some functions for the training:

- `init.py`: contains the code to **initialize** the W and H matrix for the MU and unrolled MU models, as well as functions to convert the W and H tensors to a midi tensor.
- `models.py`: contains the **architecture of the unrolled model**.
- `spectrograms.py`: contains functions to **compute the CQT sepctrogram** of a signal and visualize it, as well as **midi** and **jams** conversion and visualization functions.
- `utils.py`: contains some **information retrieval** util functions, the definition of the **datasets**, functions to **compute metrics** and **test models** using mir_eval [[3]](#3), and finally functions to **train** the model.

**/{piano, synth}-dataset**:
    Contain 7 audio/ MIDI files pair as small toy datasets (recordings from virtual piano and virtual synth).

**/models**:
    contains pretrained models weights.

**/test_data**:
    Contains small audio/ midi pairs example for some tests.

### Steps to train:

- The `training_ralmu.sh` file contains the command to be launched on the cluster to train the model, you can modify here the amount of unrolled iterations that the model has (`ITER`), the learning rate (`LR`), the number of epochs to be made (`EPOCHS`), the batch size (`BATCH`), the size of the subset of the dataset to train on (`SUBSET`), as well as the train/ test split size (`SPLIT`) and whether to clip the H matrix in every unrolled iterations (`CLIP`).

This command will run the `trainer.py` file that loads the dataset (*MAPS*), loads the model and trains it with the provided parameters.

- The `warmup_train_ralmu.sh` file contains the command to be launched on the cluster to warmup train the model. The same parameters as in `training_ralmu.sh` can be modified. 

This command will run the `warmup_trainer.py` file that loads the dataset (*MAPS*), loads the model and trains it with the provided parameters.

The models are saved under **/models**.

- The `test_models.sh` file launches the function to test the models (*compute some metrics on the provided test datasets*) on the cluster.

- The `maps_to_cqt.sh` and `guitarset_to_cqt.sh` files launch the python script saving locally the MAPS [[4]](#4) and Guitarset [[5]](#5) datasets.

## References
<a id="1">[1]</a> 
Christophe Kervazo, Abdelkhalak Chetoui et Jérémy Cohen. “Deep unrolling of the multiplicative updates algorithm for blind source separation, with application to hyperspectral unmixing”. In : EUSIPCO 2024 - 24th European Signal Processing Conference. Lyon, France, août 2024. 
url : https://hal.science/hal-04736884

<a id="2">[2]</a> 
Christophe Kervazo et Jérémy Cohen. “Unrolled Multiplicative Updates for Nonnegative Matrix Factorization applied to Hyperspectral Unmixing”. In : In prep.

<a id="3">[3]</a> 
Colin Raffel et al. “mir_eval : A Transparent Implementation of Common MIR Metrics”. In : Proceedings of the 15th International Society for Music Information Retrieval Conference (ISMIR). 2014

<a id="4">[4]</a> 
Valentin Emiya, Roland Badeau et Bertrand David. “Multipitch Estimation of Piano Sounds Using a New Probabilistic Spectral Smoothness Principle”. In : IEEE Transactions on Audio, Speech, and Language Processing 18.6 (2010)

<a id="5">[5]</a> 
Qiyang Xi et al. “GuitarSet : A Dataset for Guitar Transcription”. In : Proceedings of the 19th International Society for Music Information Retrieval Conference (ISMIR). Paris, France, 2018
