import torch
import torchaudio
import os
import librosa
import numpy as np
import src.spectrograms as spec
  
"""
Initialisation NMF
"""
def init_W(folder_path, hop_length=128, bins_per_octave=36, n_bins=288):
    """
    Creates a W matrix from all audio files contained in the input path
    By taking the column of highest energy of the CQT
    """
    templates = []
    freqs     = []
    file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.wav'))])

    for fname in file_list:
        path = os.path.join(folder_path, fname)
        y, sr = torchaudio.load(path)
        duration = y.shape[1] / sr
        min_duration = (n_bins * hop_length) / sr
        assert duration >= min_duration, f"Audio file {fname} is too short. Duration: {duration:.2f}s, Required: {min_duration:.2f}s"
        
        spec_db, _, freq = spec.cqt_spec(y, sample_rate=sr, hop_length=hop_length,
                                 bins_per_octave=bins_per_octave, n_bins=n_bins)
        freqs.append(freq)
        
        # Choose frame with max energy (sum across frequencies)
        energy_per_frame = np.sum(spec_db, axis=0)
        best_frame_idx = np.argmax(energy_per_frame)
        template = spec_db[:, best_frame_idx]

        # Convert from dB to linear for multiplication use
        template_lin = librosa.db_to_amplitude(template)

        # Normalize
        template_lin /= np.linalg.norm(template_lin) + 1e-8
        
        templates.append(template_lin)

    # W of shape f * (88*4)
    W = np.stack(templates, axis=1)
    
    return torch.tensor(W, dtype=torch.float32), freqs

def init_H(l, t, W, M, n_init_steps, beta=1):
    eps = 1e-8
    H = torch.rand(l, t) + eps
    
    # create H with n iterations of MU
    for i in range(n_init_steps):
        Wh = W @ H
        Wh_beta_minus_2 = Wh ** (beta - 2)
        Wh_beta_minus_1 = Wh ** (beta - 1)

        numerator = W.T @ (Wh_beta_minus_2 * M)
        denominator = W.T @ Wh_beta_minus_1 + eps

        H = H * (numerator / denominator)
    
    return H


"""
WH -> MIDI
"""
def hz_to_midi(frequency):
    return 69 + 12 * torch.log2(frequency / 440)

def midi_to_hz(midi_note):
    return 440 * (2 ** ((midi_note - 69) / 12))

def frequency_to_note(frequency, thresh):
    """
    Maps a frequency to its corresponding musical note.
    We add a semitones thresholding to account for small variations in the frequency
    """
    
    midi_note = hz_to_midi(frequency)
    note_frequency = midi_to_hz(torch.round(midi_note))
    semitone_diff = torch.abs(midi_note - hz_to_midi(note_frequency))

    # Check if the frequency is within the threshold
    if semitone_diff <= thresh:
        return torch.round(midi_note)
    else:
        return torch.tensor(0, dtype=torch.float32)
  
def W_to_pitch(W, H, freqs, thresh=0.4):
    """
    Assign a pitch to every column of W
    freqs being the frequency correspondance of every column's sample
    """
    pitches = torch.empty(W.shape[1], dtype=torch.float32)
    notes   = torch.empty(W.shape[1])
    for i in range(W.shape[1]):
        freq        = freqs[i]
        y           = W[:,i]
        y           = y.squeeze() # Ensure y is a 1D tensor
        max_idx     = torch.argmax(y)
        pitch       = torch.as_tensor(freq[max_idx.item()], dtype=torch.float32)
        pitches[i]  = pitch
        notes[i]    = frequency_to_note(pitch, thresh) 
    
    sorted_indices = torch.argsort(pitches)
    sorted_pitches = pitches[sorted_indices]
    sorted_notes = notes[sorted_indices]
    sorted_W = W[:, sorted_indices] 
    sorted_H = H[sorted_indices, :]     
    
    return  sorted_pitches, sorted_notes, sorted_W, sorted_H
  
def WH_to_MIDI(W, notes, H, normalize=False, threshold=0.01, smoothing_window=5, adaptative=True):
    """
    Form a MIDI format tensor from W and H
    """
    midi = torch.zeros((88, H.shape[1]), dtype=torch.float32)
    
    if normalize:
        H_max = torch.norm(H, 'fro')
    else:
        H_max = 1
        
    activations = {i: torch.zeros(H.shape[1], dtype=torch.float32) for i in range(0, 88)}

    # Sum the activation rows of the same note
    for i in range(W.shape[1]):
        midi_note = int(notes[i].item())  # Get the MIDI note
        activations[midi_note] += H[i, :]/ H_max
    
    for midi_note, activation in activations.items():
        if midi_note <= 109:
            if adaptative:
                dynamic_threshold = threshold + torch.mean(activation[:smoothing_window])
                active_indices = activation > dynamic_threshold
            else:
                active_indices = activation > threshold
            midi[midi_note-21, active_indices] = activation[active_indices]
    
    active_midi = [i for i in range(88) if (midi[i,:]>0).any().item()]
    
    return midi, active_midi