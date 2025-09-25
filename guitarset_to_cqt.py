from adasp_data_management import music
import numpy as np
import pandas as pd
import memory_profiler
import torchaudio
import torch
from tqdm import trange
from tqdm import tqdm
import argparse

import src.spectrograms as spec 
import src.init as init 
import src.utils as utils

"""
This code saves the Guitarset dataset locally as torch tensors and creates a csv file with the location of every file

The metadata.csv file contains:
    - The paths to audio CQT spectrogram (matrix M)
    - The paths to corresponding Midi ground truth file
    - The paths to ground truth H matrix corresponding to the midi file
    - The paths to ground truth onsets and offsets matrix files
    - The duration of each audio (in seconds)
    - The amount of time steps of the file
"""

def compute_lengths(metadata, hop_length=128):
    """
    Computes the duration of every audio file in the metadata csv in seconds and in time steps
    """
    durations = []
    time_steps = []
    
    for i, row in tqdm(metadata.iterrows()):
        waveform, sr = torchaudio.load(row.file_path)
        
        if i%5 == 0:
            gpu_info = utils.get_gpu_info()
            utils.log_gpu_info(gpu_info, filename="AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        _, times_cqt, _ = spec.cqt_spec(waveform, sr, hop_length)
        
        durations.append(waveform.shape[1]/sr)
        time_steps.append(times_cqt.shape[0])
        
    return np.array(durations), time_steps

def save_audio(metadata_list, row, save_path, dtype=None):
    """
    Opens the audio and jams files to save the M, H, Midi, Onset and Offset tensors locally
    """
    file_name = row['file_path'].split('audio_mono-mic/')[1][:-4] + ".pt"
    try:
        waveform, sr = torchaudio.load(row['file_path'])
            
        M, times_cqt, _ = spec.cqt_spec(waveform, sr, dtype=dtype)
        midi, onset, offset, _ = spec.jams_to_pianoroll(row['jams_path'], times_cqt, sr, dtype=dtype)

        active_midi = [i for i in range(88) if (midi[i, :] > 0).any().item()]
        H = init.MIDI_to_H(midi, active_midi, onset, offset)
        
        metadata_list.append({
            "file_path": f"{save_path}/M/{file_name}",
            "H_path": f"{save_path}/H/H/{file_name}",
            "midi_path": f"{save_path}/H/midi/{file_name}",
            "onset_path": f"{save_path}/H/onsets/{file_name}",
            "offset_path": f"{save_path}/H/offsets/{file_name}",
            "duration": waveform.shape[1]/sr,
            "time_steps": times_cqt.shape[0]
        })
        
        torch.save(M, f"{save_path}/M/{file_name}")
        torch.save(H, f"{save_path}/H/H/{file_name}")
        torch.save(midi, f"{save_path}/H/midi/{file_name}")
        torch.save(onset, f"{save_path}/H/onsets/{file_name}")
        torch.save(offset, f"{save_path}/H/offsets/{file_name}")
    except Exception as e:
        print(f"Skipping {file_name} due to error: {e}")
        return

if __name__ == "__main__":

    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        print(f"{torch.cuda.is_available()}")
        dev = "cpu"

    print(f"Start of the script, device = {dev}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default=1, type=float)
    args = parser.parse_args()
    
    subset = args.subset
    dtype  = torch.float16

    #########################################################

    print("Loading the dataset...")
    metadata = pd.read_csv("/home/ids/edabier/AMT/Unrolled-NMF/Guitarset/metadata.csv")
    
    if subset is not None:
        num_files = int(len(metadata) * subset)
        metadata = metadata.iloc[:num_files]
        
    print("Computing length of files...")
    durations, time_steps = compute_lengths(metadata)
    
    print("Sorting by length and filtering...")
    metadata = metadata.copy()
    metadata.loc[:, 'duration'] = durations
    metadata.loc[:, 'time_steps'] = time_steps
    metadata = metadata.sort_values(by='duration')
    
    # filter_condition = metadata['duration'] > 60
    indices = metadata.index.to_list()
    
    # metadata = metadata[filter_condition].reset_index(drop=True)
    
    print("Computing and saving the CQT...", len(metadata))
    save_path = 'AMT/Unrolled-NMF/Guitarset'
    metadata_list = []
    for idx in tqdm(range(len(metadata))):
        row = metadata.iloc[idx]
        if idx%5 == 0:
            gpu_info = utils.get_gpu_info()
            utils.log_gpu_info(gpu_info, filename="AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        save_audio(metadata_list, row, save_path, dtype)
    
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(f'{save_path}/metadata_full.csv', index=False)
    