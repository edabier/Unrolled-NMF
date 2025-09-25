import torch
import torchaudio
import os
import librosa
import numpy as np

import src.spectrograms as spec
import src.utils as utils
  
"""
Initialisation NMF
"""
def init_W(folder_path=None, hop_length=128, bins_per_octave=36, n_bins=288, normalize_thresh=None, dtype=None):
    """
    Create a W matrix from all audio files contained in the input path
    takes the column of highest energy of the CQT
    If no folder is provided, the W matrix is initialized with 88 columns, with all zeros and only 1s for each column's fundamental frequency
    
    Returns:
        W: the W tensor 
        freqs: the fundamental frequencies corresponding to every column of W
        sr: the sample rate of the files in folder_path
        true_freqs: the fundamental frequencies from A0 to C8
    
    Args:
        folder_path (str, optional): if provided, the path of the folder containing the recordings of single notes. Otherwise, initialize with synthetic data (default: None)
        hop_length (int, optional): value to compute the CQT (default: 128)
        bins_per_octave (int, optional): value to compute the CQT (default: 36)
        n_bins (int, optional): value to compute the CQT (default: 288)
        normalize_thresh (float, optional): if set, the columns are normalized by their L1 sum if this sum is above the threshold (default: None)
        dtype: data type of the resulting W tensor
    """
    if dtype is not None:
        eps = torch.finfo(type=dtype).min
    else:
        eps = torch.finfo().min
    
    if folder_path is not None:
        templates = []
        true_freqs     = []
        file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.wav'))])
        
        note_to_midi = {
            'C': 0, 'C#': 1,'D': 2, 'D#': 3, 
            'E': 4, 'F': 5, 'F#': 6, 'G': 7, 
            'G#': 8, 'A': 9, 'A#': 10, 'B': 11
        }

        for fname in file_list:
            path = os.path.join(folder_path, fname)
            y, sr = torchaudio.load(path)
            
            duration = y.shape[1] / sr
            min_duration = (n_bins * hop_length) / sr
            assert duration >= min_duration, f"Audio file {fname} is too short. Duration: {duration:.2f}s, Required: {min_duration:.2f}s"
            
            spec_cqt, _, freq = spec.cqt_spec(y, sample_rate=sr, hop_length=hop_length,
                                    bins_per_octave=bins_per_octave, n_bins=n_bins, normalize_thresh=normalize_thresh, dtype=dtype)
            
            if len(fname) == 7:
                note, octave = fname[0:2], int(fname[2])
            else:
                note, octave = fname[0], int(fname[1])
                
            midi_note = note_to_midi[note] + (octave + 2) * 12  - 12 # MIDI note number
            expected_freq = midi_to_hz(torch.tensor(midi_note, dtype=torch.float32))
            true_freqs.append(expected_freq)
            
            # Choose frame with max energy (sum across frequencies)
            energy_per_frame = spec_cqt.sum(axis=0)
            best_frame_idx = torch.argmax(energy_per_frame)
            template = spec_cqt[:,best_frame_idx]

            templates.append(template)

        sorted_indices = sorted(range(len(true_freqs)), key=lambda k: true_freqs[k])
        templates = [templates[i] for i in sorted_indices]
        freqs = [true_freqs[i] for i in sorted_indices]
        
        # W of shape f * (88*n)
        W = torch.stack(templates, axis=1)

    else:
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=librosa.note_to_hz("A0"), bins_per_octave=bins_per_octave)
        true_freqs = freqs
        midi_notes = range(21, 109)
        note_frequencies = librosa.midi_to_hz(midi_notes)
        bin_indices = np.searchsorted(freqs, note_frequencies)
        W = torch.zeros((n_bins, len(midi_notes)))
        for col, bin_idx in enumerate(bin_indices):
            W[bin_idx, col] = 1
        sr = None
        print("Initialized W with synthetic data")
    return W, freqs, sr, true_freqs

def init_H(W, M, n_init_steps, beta=1, device=None, dtype=None):
    """
    Initializes the H tensor with `n_init_steps` of MU iterations with the β-divergence from a random initialization
    
    Args:
        W (torch.tensor): the W tensor used to update H
        M (torch.tensor): the M tensor used to update H
        n_init_steps (int): the number of MU iterations to do
        beta (int, optional): the value of β for the β-divergence (default: 1)
        device: the device on which to compute
        dtype (optional): the data type of the tensor 
    """
    if dtype is not None:
        eps = torch.finfo(type=dtype).min
    else:
        eps = torch.finfo().min
    
    if len(M.shape) > 2:
        batch_size = W.shape[0]
        l = W.shape[2]
        t = M.shape[2]
        H = torch.rand(batch_size, l, t, dtype=dtype)
    else:
        l = W.shape[1]
        t = M.shape[1]
        H = torch.rand(l, t, dtype=dtype)
        
    H = torch.clamp(H, min=eps)
        
    if device is not None:
        H = H.to(device)
        
    # create H with n iterations of MU
    for _ in range(n_init_steps):
        
        Wh = W @ H
        Wh = torch.clamp(Wh, min=eps)
        Wh_beta_minus_2 = Wh ** (beta - 2)
        Wh_beta_minus_1 = Wh ** (beta - 1)
        
        if batch_size is not None:
            Wt = W.transpose(1, 2)
        else:
            Wt = W.T

        numerator = Wt @ (Wh_beta_minus_2 * M)
        denominator = Wt @ Wh_beta_minus_1
        denominator = torch.clamp(denominator, min=eps)

        H = H * (numerator / denominator)
    H = torch.clamp(H, min=eps, max=min(1, H.max()))
    
    return H

def scale_W(M, W, H, return_lambda=False):
    """
    Computes the lambda that minimizes the KL between M and WH
    And multiplies W by this lambda
    
    min_lambda KL(M, lambda WH)
    """
    is_batched = M.dim() == 3

    if is_batched:
        wh = W @ H
        l1_M = torch.norm(M, p=1, dim=(1, 2))
        l1_WH = torch.norm(wh, p=1, dim=(1, 2))
        lambda_factor = l1_M / l1_WH
        lambda_factor = lambda_factor.view(-1, 1, 1)
        W = W * lambda_factor
    else:
        wh = W @ H
        l1_M = torch.norm(M, p=1)
        l1_WH = torch.norm(wh, p=1)
        lambda_factor = l1_M / l1_WH
        W = W * lambda_factor

    if return_lambda:
        return W, lambda_factor
    else:
        return W


"""
WH -> MIDI
"""
def hz_to_midi(frequency):
    return 69 + 12 * torch.log2(frequency / 440)

def midi_to_hz(midi_note):
    return 440 * (2 ** ((midi_note - 69) / 12))

def frequency_to_note(frequency, thresh):
    """
    Map a frequency to its corresponding musical note.
    
    Args:
        thresh (float): Threshold in semitones to account for small variations in the frequency
    """
    
    frequency = torch.tensor(frequency, dtype=torch.float32)
    midi_note = hz_to_midi(frequency)
    note_frequency = midi_to_hz(torch.round(midi_note))
    semitone_diff = torch.abs(midi_note - hz_to_midi(note_frequency))

    # Check if the frequency is within the threshold
    if semitone_diff <= thresh:
        return torch.round(midi_note)
    else:
        return torch.tensor(0, dtype=torch.float32)

def W_to_pitch(W, true_freqs=None, thresh=0.4, H=None, use_max=False, sort=False):
    """
    Assign a pitch to every column of W by using the frequency of max energy or by refering to the true_freqs
    
    Args:
        W (torch.tensor): The W tensor (notes' spectrograms dictionnary)
        true_freqs (list): The true frequency correspondence of every W column's spectrogram
        thresh (float, optional): Threshold for the conversion from frequency to note (default: ``0.4``)
        us_max (bool, optional): Whether to take the maximum value of each CQT column as the fundamental frequency or select the expected frequency instead
        sort (bool, optional): Whether to sort the frequencies and W columns by increasing order or not
    """  
    frequencies = torch.empty(W.shape[1], dtype=torch.float32)
    notes = torch.empty(W.shape[1])
    freq_range = librosa.cqt_frequencies(W.shape[0], fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
    
    if true_freqs is not None:
        true_freqs.sort()  
    else:
        true_freqs = freq_range
    
    for i in range(W.shape[1]):
        if use_max:
            _, max_id = torch.max(W[:,i], dim=0)
            freq = freq_range[max_id] # Use freq with max amplitude in the CQT
        else:
            freq = true_freqs[i]  # Use the known frequency for the note
        
        if freq >= 27.5 and freq < 4187:
            if sort:
                note_true = frequency_to_note(true_freqs[i], thresh) - 21
                note = frequency_to_note(freq, thresh) - 21
                if torch.abs(note - note_true) == 12:
                    frequencies[i] = true_freqs[i]
                    notes[i] = note_true
                else:    
                    frequencies[i] = freq
                    notes[i] = note
            else:
                frequencies[i] = freq
                notes[i] = frequency_to_note(freq, thresh) - 21

    if sort:
        sorted_indices = torch.argsort(frequencies)
        sorted_frequencies = frequencies[sorted_indices]
        sorted_notes = notes[sorted_indices]
        sorted_W = W[:, sorted_indices]
        
        if H is not None:
            sorted_H = H[sorted_indices, :]
            return sorted_frequencies, sorted_notes, sorted_W, sorted_H
        else:
            return sorted_frequencies, sorted_notes, sorted_W
    else:
        return frequencies, notes
  
def WH_to_MIDI(W, H, notes, threshold=0.02, smoothing_window=None, normalize=False, min_note_length=5, sr=48000):
    """
    Form a MIDI format tensor from W and H
    
    Args:
        W (torch.tensor): The W tensor (notes' spectrograms dictionnary)
        H (torch.tensor): The H tensor (notes activations)
        notes (torch.tensor): the amount of distinct single notes to transcribe
        threshold (float, optional): The minimum value under which to discard a potential note (default: ``0.02``)
        smoothing_window (float, optional): The size of the window to take the average activation of to adapt the threshold (default: ``None``)
        min_note_length (int, optional): the minimum duration of a note, we merge small notes together until we reach this length (default: ``5``)
        
        ==> The code checks each segment's length and merges it with the next segment if it is shorter than min_note_length and the gap between segments is also smaller than min_note_length.
        The intensity is then set to 1, constant for the duration of the note.
    """    
    if normalize:
        H_max = torch.norm(H, 'fro')
    else:
        H_max = 1
    
    l, t = H.shape
    
    midi = torch.zeros((l, t), dtype=torch.float32)
    activations = {i: torch.zeros(t, dtype=torch.float32) for i in range(0, l)}

    # Set each activation to the corresponding rows of H
    for i in range(l): # for each note
        midi_note = int(notes[i].item())  # Get the MIDI note
        activations[midi_note] += H[i, :]/ H_max
    
    for midi_note, activation in activations.items():
        if midi_note <= 108:
            if smoothing_window is not None:
                dynamic_threshold = threshold + torch.mean(activation[:smoothing_window])
                active_indices = activation > dynamic_threshold
            else:
                active_indices = activation > threshold
            midi[midi_note, active_indices] = activation[active_indices]
            
            diff = torch.diff(active_indices.float())
            start_indices = torch.where(diff == 1)[0] + 1
            end_indices = torch.where(diff == -1)[0] + 1

            # Ensure the first and last segments are included
            if active_indices[0]:
                start_indices = torch.cat((torch.tensor([0]), start_indices))
            if active_indices[-1]:
                end_indices = torch.cat((end_indices, torch.tensor([len(active_indices)])))

            # Merge segments that are too short
            merged_start_indices = []
            merged_end_indices = []
            i = 0
            while i < len(start_indices):
                start = start_indices[i]
                end = end_indices[i]

                # Merge with next segments if they are too close
                while i + 1 < len(start_indices) and start_indices[i + 1] - end_indices[i] - 1 < min_note_length:
                    i += 1
                    end = end_indices[i]

                merged_start_indices.append(start)
                merged_end_indices.append(end)
                i += 1

            # Calculate average intensity and set constant intensity for each merged segment
            for start, end in zip(merged_start_indices, merged_end_indices):
                midi[midi_note, start:end + 1] = 1
    
    active_midi = [i for i in range(88) if (midi[i,:]>0).any().item()]
    
    return midi, active_midi

def MIDI_to_H(midi, active_midi, onsets, offsets, values_range=127):
    """
    Generates a H tensor from a MIDI by setting linearly decreasing values to H rows from onsets to offsets.
    
    Args:
        midi (torch.tensor): the midi tensor to create H from
        active_midi (list): the rows of the midi file with active notes
        onsets (torch.tensor): the tensor containing the time steps with notes turning on (onsets)
        offsets (torch.tensor): the tensor containing the time steps with notes turning off (offsets)
        valus_range (int, optional): the max value from which the H row linearly decreases (default: 127)
    """
    H = torch.zeros_like(midi, dtype=torch.float)

    for note in active_midi:
        # Get indices of onsets and offsets
        onset_indices = torch.nonzero(onsets[note, :], as_tuple=False).squeeze()
        offset_indices = torch.nonzero(offsets[note, :], as_tuple=False).squeeze()
        if onset_indices.dim() == 0:
            onset_indices = onset_indices.unsqueeze(0)
        if offset_indices.dim() == 0:
            offset_indices = offset_indices.unsqueeze(0)

        if len(offset_indices) < len(onset_indices):
            # Handle the case where there are more onsets than offsets
            print("There are more onsets than offsets!")
            offset_indices = torch.cat([offset_indices, torch.tensor([midi.shape[1] - 1])])

        for onset_idx, offset_idx in zip(onset_indices, offset_indices):
            onset_idx = onset_idx.item()
            offset_idx = offset_idx.item()

            if onset_idx < offset_idx:
                decay_length = offset_idx - onset_idx
                # Linear decay from 127 to 0
                decay = torch.linspace(values_range, 0, steps=decay_length)
                H[note, onset_idx:offset_idx] = decay
            else:
                print(f"Warning: onset {onset_idx} is not before offset {offset_idx}")

    return H
