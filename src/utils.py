import torch.nn as nn
import torch.nn.functional as F
from torchbd.loss import BetaDivLoss
import torch
import torchaudio
import torchyin
import librosa
from torch.utils.data import Dataset
import glob, os
import numpy as np
import src.spectrograms as spec

class MaestroNMFDataset(Dataset):
    
    def __init__(self, audio_dir, midi_dir, n_fft=4096, hop_length=128):
        self.audio_files    = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        self.midi_dir       = midi_dir
        self.n_fft          = n_fft
        self.hop_length     = hop_length

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path  = self.audio_files[idx]
        filename    = os.path.basename(audio_path).replace(".wav", ".mid")
        midi_path   = os.path.join(self.midi_dir, filename)

        print(f"Loading audio file: {audio_path}")
        print(f"Loading MIDI file: {midi_path}")

        waveform, sr = torchaudio.load(audio_path)
        # if waveform.shape[0] > 1:
        #     waveform = waveform.mean(dim=0)

        spec_db, times_cqt, freq_cqt = spec.cqt_spec(waveform, sr, self.hop_length)
        M = spec_db#[:250,:900]

        midi, times_midi = spec.midi_to_pianoroll(midi_path, waveform, M.shape[1], self.hop_length, sr)

        return M, midi
  
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
        activations[midi_note] += H[i, :]#/ H_max
    
    for midi_note, activation in activations.items():
        if midi_note <= 109:
            if adaptative:
                dynamic_threshold = threshold + torch.mean(activation[:smoothing_window])
                active_indices = activation > dynamic_threshold
            else:
                active_indices = activation > threshold
            midi[midi_note-21, active_indices] = activation[active_indices]
    
    return midi

"""
Loss
"""
def detect_onsets_offsets(midi):
    """
    Detects onsets and offsets
    Returns one tensor of shape midi with 1s at onsets and offsets, 0 elsewhere
    """
    onsets = torch.zeros_like(midi)
    offsets = torch.zeros_like(midi)
    
    # For every time step
    for t in range(1, midi.shape[1]):
        # Onset: note active at time t and not t-1
        onsets[:, t] = ((midi[:, t] > 0) & (midi[:, t-1] == 0)).float()
        
        # Offset: note not active at time t and active at t-1
        offsets[:, t] = ((midi[:, t] == 0) & (midi[:, t-1] > 0)).float()

    return onsets, offsets

def midi_loss(midi_hat, midi_gt):
    """
    Computes the pitch distance loss between the predicted and ground truth MIDI tensors.
    Loss increase with distance of pitch, except for octave distance
    Adds the MSE loss of onsets and offsets
    
    L = L_pitch + L_onset + L_offset + L_velocity
    """
    assert midi_hat.shape == midi_gt.shape, "Predicted and ground truth MIDI tensors must have the same shape."
    
    loss = 0
    miss_loss = 11 # Missing a note: equivalent to 11 semitones mistake
    
    # Detect onsets and offsets
    pred_onsets, pred_offsets = detect_onsets_offsets(midi_hat)
    gt_onsets, gt_offsets = detect_onsets_offsets(midi_gt)
    
    # Pitch distance for every time step
    for t in range(midi_hat.shape[1]):
        # Get notes activated at time t
        pred_notes = torch.nonzero(midi_hat[:, t]).squeeze(1)
        gt_notes = torch.nonzero(midi_gt[:, t]).squeeze(1)

        # Compare notes
        if len(pred_notes) > 0 and len(gt_notes) > 0:
            for pred_note in pred_notes:
                min_distance = float('inf')
                # Compute the absolute distance between every pair of notes
                # We also consider the predicted notes 1 octave above and bellow
                for gt_note in gt_notes:
                    distance = abs(pred_note - gt_note)
                    distance_oct_sup = abs(pred_note + 12 - gt_note)
                    distance_oct_inf = abs(pred_note - 12 - gt_note)
                    min_distance = min([min_distance, distance, distance_oct_sup, distance_oct_inf])
                loss += min_distance #* (1 - octave_weight * (min_distance % 12 == 0))

        # Predicted a note that is not present
        elif len(pred_notes)>0 and len(gt_notes) == 0:
            loss += miss_loss
            
        # Did not predict the note
        elif len(pred_notes)==0 and len(gt_notes)>0:
            loss += miss_loss
    
    # Onset/ Offset loss   
    loss += F.mse_loss(pred_onsets, gt_onsets)
    loss += F.mse_loss(pred_offsets, gt_offsets)
        
    # We normalize the loss by the size of the midi so that different length MIDI files can be compared
    normalized_loss = loss / (midi_hat.shape[1] * midi_hat.shape[0])
    
    return normalized_loss
    
def compute_loss(M, M_hat, midi, midi_hat, H_hat, lambda_rec=0.1, lambda_sparsity=0.01):
        
    # Reconstruction loss (KL)
    beta = 1 
    loss = BetaDivLoss(beta=beta)
    loss_rec = loss(M_hat, M)

    # Sparsity loss on H (L1)
    loss_sparsity = torch.sum(torch.abs(H_hat))
    
    midi_loss = midi_loss(midi_hat, midi)

    # Total loss
    total_loss = midi_loss + lambda_rec * loss_rec + lambda_sparsity * loss_sparsity
    
    return total_loss
   
   
"""
Train the network
"""
def train(n_epochs, model, optimizer, loader, device):
    
    model.train()
    model.to(device=device)
    losses = []
    
    for epoch in range(n_epochs):
        for M, midi in loader:

            W_hat, H_hat, M_hat = model(M)
            midi_hat = WH_to_MIDI(W_hat, H_hat)
            
            # if epoch == 1:
            #     print("W0 after first forward pass:")
            #     print(model.W0.min(), model.W0.max())
            #     print("H0 after first forward pass:")
            #     print(model.H0.min(), model.H0.max())
            
            loss = compute_loss(M, M_hat, midi, midi_hat, H_hat)
            losses.append(loss.detach().numpy())
            
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for param in model.parameters():
                if param.grad is not None:
                    print(f"Grad norm: {param.grad.norm()}")

            optimizer.zero_grad()
            loss.backward()#retain_graph=True)
                
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return losses