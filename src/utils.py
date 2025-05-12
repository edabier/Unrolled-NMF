import torch.nn as nn
import torch.nn.functional as F
from torchbd.loss import BetaDivLoss
import torch
import torchaudio
from torch.utils.data import Dataset
import librosa
import numpy as np
import glob, os
import src.spectrograms as spec
import src.init as init

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

        # print(f"Loading audio file: {audio_path}")
        # print(f"Loading MIDI file: {midi_path}")

        waveform, sr = torchaudio.load(audio_path)

        spec_db, times_cqt, freq_cqt = spec.cqt_spec(waveform, sr, self.hop_length)
        midi, times_midi = spec.midi_to_pianoroll(midi_path, waveform, times_cqt, self.hop_length, sr)

        return spec_db, midi

"""
Loss (no  batch_size)
"""
def gaussian_kernel(kernel_size=3, sigma=2):
    """
    Creates a 1D Gaussian kernel.
    """
    kernel = np.exp(-np.linspace(-kernel_size / 2, kernel_size / 2, kernel_size)**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

def filter1d_tensor(tensor, kernel, axis=0):
    """
    Applies a 1D Gaussian filter to a 2D tensor along a specified axis.
    """
    if axis == 0:
        # Apply the kernel to each column
        tensor = tensor.T.unsqueeze(0).unsqueeze(0)
        filtered_tensor = F.conv2d(tensor, kernel, padding=(kernel.size(-2) // 2, 0)).squeeze(0).squeeze(0).T
    elif axis == 1:
        # Apply the kernel to each row
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        filtered_tensor = F.conv2d(tensor, kernel, padding=(kernel.size(-2) // 2, 0)).squeeze(0).squeeze(0)
    else:
        raise ValueError("Axis must be 0 or 1")

    return filtered_tensor

def detect_onset_offset(midi, filter=False):
    """
    Detects onsets and offsets
    Returns one tensor of shape midi with 1s at onsets and offsets, 0 elsewhere
    If filter=True, applies a gaussian filters to the matrix along the temporal axis
    """
    onset = torch.zeros_like(midi)
    offset = torch.zeros_like(midi)
    
    # For every time step
    for time in range(1, midi.shape[1]):
        # Onset: note active at time t and not t-1
        onset[:, time] = ((midi[:, time] > 0) & (midi[:, time-1] == 0)).float()
        
        # Offset: note not active at time t and active at t-1
        offset[:, time] = ((midi[:, time] == 0) & (midi[:, time-1] > 0)).float()
    
    if filter:
        kernel = gaussian_kernel(15, 5)
        filtered_onset = filter1d_tensor(onset, kernel, axis=0)
        filtered_offset = filter1d_tensor(offset, kernel, axis=0)
        return filtered_onset, filtered_offset
    else:
        return onset, offset

def precision_recall_f1(pred, target, epsilon=1e-7):
    """
    Computes precision, recall, and F1 score.
    """
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision, recall, f1

def pitch_mask(input_column, active_notes, octave_weight, note_weight):
    """
    Adds weights to the current midi column's notes
    Notes that are different from the current note get an increasingly high weight
    Notes that are octaves apart from the note get a lower weight
    """
    mask = torch.zeros_like(input_column)
    for active_note in active_notes:
        for i in range(mask.size(0)):
            mask[i] = note_weight * abs(i - active_note) % 12 + octave_weight * abs(i - active_note) // 12
    return mask

def loss_midi(midi_hat, midi_gt, window_size=5):
    
    assert midi_hat.shape == midi_gt.shape, "Predicted and ground truth MIDI tensors must have the same shape."
    
    onset_hat, offset_hat = detect_onset_offset(midi_hat)
    onset_gt, offset_gt = detect_onset_offset(midi_gt)
    
    # Aggregate onsets and offsets across all pitches
    onset_hat_agg = onset_hat.sum(dim=0) / midi_hat.shape[0]
    onset_gt_agg = onset_gt.sum(dim=0) / midi_gt.shape[0]
    offset_hat_agg = offset_hat.sum(dim=0) / midi_hat.shape[0]
    offset_gt_agg = offset_gt.sum(dim=0) / midi_gt.shape[0]
    
    bce_loss = nn.BCELoss()
    onset_loss = bce_loss(onset_hat_agg, onset_gt_agg)
    offset_loss = bce_loss(offset_hat_agg, offset_gt_agg)
    
    # onset_loss = F.mse_loss(onset_hat_agg, onset_gt_agg)
    # offset_loss = F.mse_loss(offset_hat_agg, offset_gt_agg)
    
    ce_loss = nn.CrossEntropyLoss()
    pitch_loss = 0
    
    for t in range(midi_hat.shape[1]-window_size):
        midi_hat_agg = midi_hat[:, t:t+window_size].sum(dim=1)
        midi_gt_agg = midi_gt[:, t:t+window_size].sum(dim=1)
        pitch_classes_hat = midi_hat_agg % 12
        pitch_classes_gt = midi_gt_agg % 12
        pitch_loss += ce_loss(midi_hat_agg, midi_gt_agg.argmax(dim=0))
    
    loss = onset_loss + offset_loss + pitch_loss
    
    return loss

def compute_midi_loss(midi_hat, midi_gt, active_midi, octave_weight, note_weight, sparse_factor):
    """
    Computes the pitch distance loss between the predicted and ground truth MIDI tensors.
    Loss increase with distance of pitch, except for octave distance
    Adds the MSE loss of onsets and offsets (MSE for rows with active midi only)
    
    L = L_pitch + L_onset + L_offset (+ L_velocity)
    """
    assert midi_hat.shape == midi_gt.shape, "Predicted and ground truth MIDI tensors must have the same shape."
    
    l_pitch     = 0
    miss_loss   = 11 # Missing a note: equivalent to 11 semitones mistake
    
    # Detect onsets and offsets
    pred_onsets, pred_offsets = detect_onset_offset(midi_hat, filter=True)
    gt_onsets, gt_offsets = detect_onset_offset(midi_gt, filter=True)
    
    # Pitch distance for every time step
    for t in range(midi_hat.shape[1]):
        # Get notes activated at time t
        pred_column = midi_hat[:, t]
        gt_column   = midi_gt[:, t]
        pred_notes  = torch.nonzero(pred_column).squeeze(1)
        gt_notes    = torch.nonzero(gt_column).squeeze(1)
        active_indices = torch.nonzero(gt_column, as_tuple=False).squeeze(1)
        
        if active_indices.numel() > 0:
            mask    = pitch_mask(gt_column, active_indices, octave_weight, note_weight)
            l_pitch += F.l1_loss(pred_column * mask, gt_column * mask)

        # Predicted a note that is not present
        if len(pred_notes)>0 and len(gt_notes) == 0:
            l_pitch += miss_loss
            
        # Did not predict the note
        elif len(pred_notes)==0 and len(gt_notes)>0:
            l_pitch += miss_loss
    
    l_pitch /= midi_hat.shape[1]
    
    # Onset/ Offset loss
    l_onset  = sparse_factor * F.mse_loss(pred_onsets[active_midi,:], gt_onsets[active_midi,:])
    l_offset = sparse_factor * F.mse_loss(pred_offsets[active_midi,:], gt_offsets[active_midi,:])
    loss     = torch.sum(torch.stack([l_pitch, l_onset, l_offset], dim=0))
        
    # We normalize the loss by the size of the midi so that different length MIDI files can be compared
    normalized_loss = loss / 1#(midi_hat.shape[1] * midi_hat.shape[0])
    # print(f"Losses: onset= {l_onset}, offset= {l_offset}, pitch= {l_pitch}, total= {loss}, normalized= {normalized_loss}")
    
    return normalized_loss
    
def compute_loss(M, M_hat, midi, midi_hat, H_hat):
        
    # Reconstruction loss (KL)
    beta = 1 
    betaloss = BetaDivLoss(beta=beta)
    loss_reconstruct = betaloss(input=M_hat, target=M)
    

    # Sparsity loss on H (L1)
    loss_sparsity = torch.sum(torch.abs(H_hat))
    
    # octave_weight, note_weight, sparse_factor = 1, 10, 1e4
    # active_midi = [i for i in range(88) if (midi[i,:]>0).any().item()]
    # midi_loss = compute_midi_loss(midi_hat, midi, active_midi, octave_weight, note_weight, sparse_factor)
    midi_loss = loss_midi(midi_hat, midi)
    print(f"midi_loss: {midi_loss}")
    
    return midi_loss, loss_reconstruct, loss_sparsity
   
   
"""
Loss
"""
def gaussian_kernel_batch(kernel_size=3, sigma=2):
    """
    Creates a 1D Gaussian kernel.
    """
    kernel = np.exp(-np.linspace(-kernel_size / 2, kernel_size / 2, kernel_size)**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

def filter1d_batch(tensor, kernel, axis=2):
    """
    Applies a 1D Gaussian filter to a 3D tensor along a specified axis.
    """
    if axis == 2:
        # Apply the kernel to each time step
        tensor = tensor.permute(0, 2, 1).unsqueeze(1)  # Shape: (batch_size, 1, times, 88)
        filtered_tensor = F.conv2d(tensor, kernel, padding=(kernel.size(-2) // 2, 0), groups=tensor.size(0)).squeeze(1).permute(0, 2, 1)  # Shape: (batch_size, 88, times)
    else:
        raise ValueError("Axis must be 2 for 3D tensor")

    return filtered_tensor

def detect_onset_offset_batch(midi, filter=False):
    """
    Detects onsets and offsets
    Returns one tensor of shape midi with 1s at onsets and offsets, 0 elsewhere
    If filter=True, applies a gaussian filters to the matrix along the temporal axis
    """
    batch_size, _, t = midi.shape
    onset = torch.zeros_like(midi)
    offset = torch.zeros_like(midi)
    
    # For every time step
    for time in range(1, t):
        # Onset: note active at time t and not t-1
        onset[:, :, time] = ((midi[:, :, time] > 0) & (midi[:, :, time-1] == 0)).float()
        
        # Offset: note not active at time t and active at t-1
        offset[:, :, time] = ((midi[:, :, time] == 0) & (midi[:, :, time-1] > 0)).float()
    
    if filter:
        kernel = gaussian_kernel_batch(15, 5)
        filtered_onset = filter1d_batch(onset, kernel, axis=2)
        filtered_offset = filter1d_batch(offset, kernel, axis=2)
        return filtered_onset, filtered_offset
    else:
        return onset, offset
   
def compute_midi_loss_batch(midi_hat, midi_gt, active_midi, octave_weight, note_weight, sparse_factor):
    """
    Computes the pitch distance loss between the predicted and ground truth MIDI tensors.
    Loss increase with distance of pitch, except for octave distance
    Adds the MSE loss of onsets and offsets (MSE for rows with active midi only)
    
    L = L_pitch + L_onset + L_offset (+ L_velocity)
    """
    assert midi_hat.shape == midi_gt.shape, "Predicted and ground truth MIDI tensors must have the same shape."
    
    batch_size, _, t = midi_hat.shape
    l_pitch     = 0
    miss_loss   = 11 # Missing a note: equivalent to 11 semitones mistake
    
    # Detect onsets and offsets
    pred_onsets, pred_offsets = detect_onset_offset_batch(midi_hat, filter=True)
    gt_onsets, gt_offsets = detect_onset_offset_batch(midi_gt, filter=True)
    
    # Pitch distance for every time step
    for time in range(t):
        # Get notes activated at time t
        pred_column = midi_hat[:, :, time]
        gt_column   = midi_gt[:, :, time]
        
        
        # # Compare notes
        # if len(pred_notes) > 0 and len(gt_notes) > 0:
        #     for pred_note in pred_notes:
        #         min_distance = float('inf')
        #         # Compute the absolute distance between every pair of notes
        #         # We also consider the predicted notes 1 octave above and bellow
        #         for gt_note in gt_notes:
        #             distance = abs(pred_note - gt_note)
        #             distance_oct_sup = abs(pred_note + 12 - gt_note)
        #             distance_oct_inf = abs(pred_note - 12 - gt_note)
        #             min_distance = min([min_distance, distance, distance_oct_sup, distance_oct_inf])
        #         l_pitch += min_distance
        
        for b in range(batch_size):
            pred_notes  = torch.nonzero(pred_column[b]).squeeze(1)
            gt_notes    = torch.nonzero(gt_column[b]).squeeze(1)
            active_indices = torch.nonzero(gt_column[b], as_tuple=False).squeeze(1)
            
            if active_indices.numel() > 0:
                mask    = pitch_mask(gt_column[b], active_indices, octave_weight, note_weight)
                l_pitch += F.l1_loss(pred_column[b] * mask, gt_column[b] * mask)

            # Predicted a note that is not present
            if len(pred_notes)>0 and len(gt_notes) == 0:
                l_pitch += miss_loss
                
            # Did not predict the note
            elif len(pred_notes)==0 and len(gt_notes)>0:
                l_pitch += miss_loss
    
    l_pitch /= t
    
    # Onset/ Offset loss
    l_onset  = sparse_factor * F.mse_loss(pred_onsets[:, active_midi, :], gt_onsets[:, active_midi, :])
    l_offset = sparse_factor * F.mse_loss(pred_offsets[:, active_midi, :], gt_offsets[:, active_midi, :])
    loss     = torch.sum(torch.stack([l_pitch, l_onset, l_offset], dim=0))
        
    # We normalize the loss by the size of the midi so that different length MIDI files can be compared
    normalized_loss = loss / (t * 88)
    print(f"Losses: onset= {l_onset}, offset= {l_offset}, pitch= {l_pitch}, total= {loss}, normalized= {normalized_loss}")
    
    return normalized_loss
    
def compute_loss_batch(M, M_hat, midi, midi_hat, H_hat, lambda_rec=0.1, lambda_sparsity=0.01):
        
    # Reconstruction loss (KL)
    beta = 1 
    loss = BetaDivLoss(beta=beta)
    loss_rec = loss(M_hat, M)
    
    active_midi = [i for i in range(88) if (midi[0, i,:]>0).any().item()]

    # Sparsity loss on H (L1)
    loss_sparsity = torch.sum(torch.abs(H_hat))
    
    octave_weight, note_weight, sparse_factor = 1, 10, 1e4
    midi_loss = compute_midi_loss_batch(midi_hat, midi, active_midi, octave_weight, note_weight, sparse_factor)

    # Total loss
    total_loss = midi_loss + lambda_rec * loss_rec + lambda_sparsity * loss_sparsity
    
    return total_loss   
   
   
"""
Train the network
"""
def train(n_epochs, model, optimizer, loader, device, criterion):
    
    model.train()
    model.to(device=device)
    losses = []
    # monitor_reconstruct = []
    # monitor_sparsity    = []
    
    model.W_cache = {}
    model.H_cache = {}
    
    for epoch in range(n_epochs):
        for M, midi in loader:
            
            model.init_WH(M)
            M = M.to(device)
            midi = midi.to(device)

            W_hat, H_hat, M_hat = model(M)
            freqs = librosa.cqt_frequencies(n_bins=288, fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
            _, notes, _, _ = init.W_to_pitch(W_hat, freqs, H=H_hat)
            midi_hat, active_midi = init.WH_to_MIDI(W_hat, H_hat, notes)
            M = M.squeeze(0)
            midi = midi.squeeze(0)
            
            # loss, monitor_loss1, monitor_loss2 = compute_loss(M, M_hat, midi, midi_hat, H_hat)
            # losses.append(loss.detach().numpy())
            # monitor_reconstruct.append(monitor_loss1)
            # monitor_sparsity.append(monitor_loss2)
                    
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            weight_mask = (midi > 0).float()
            weight_mask = weight_mask / weight_mask.sum() * midi.numel()

            optimizer.zero_grad()
            
            loss = 100*(criterion(midi_hat, midi) * weight_mask).mean()
            
            loss.backward()

            for param in model.parameters():
                if param.grad is not None:
                    print(f"Grad norm: {param.grad.norm().item()}")
                else:
                    print(f"Grad is None for {param}")
                
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return losses#, monitor_reconstruct, monitor_sparsity

def transribe(model, M, device):
    model.eval()
    model.to(device=device)

    # Initialize W and H for each input tensor
    model.init_WH(M)

    with torch.no_grad():
        W_hat, H_hat, M_hat = model(M)
        freqs = librosa.cqt_frequencies(n_bins=288, fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
        pitches, notes, W_hat, H_hat = init.W_to_pitch(W_hat, freqs, H=H_hat)
        midi_hat, active_midi = init.WH_to_MIDI(W_hat, H_hat, notes)

    return W_hat, H_hat, M_hat, midi_hat, active_midi
