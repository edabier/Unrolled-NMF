import torch.nn as nn
import torch.nn.functional as F
from torchbd.loss import BetaDivLoss
import torch
import torchaudio
from torch.utils.data import Dataset
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
            midi_hat = init.WH_to_MIDI(W_hat, H_hat)
            
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