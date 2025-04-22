import torch.nn as nn
import torch.nn.functional as F
from torchbd.loss import BetaDivLoss
import torch
import torchaudio
from torch.utils.data import Dataset
import glob, os
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
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        spec_db, times_cqt, freq_cqt = spec.cqt_spec(waveform, sr, self.hop_length)
        M = spec_db#[:250,:900]

        H_target, times_midi = spec.midi_to_pianoroll(midi_path, waveform, M.shape[1], self.hop_length, sr)

        return M, H_target
    
def compute_loss(M, M_hat, H, H_hat, lambda_rec=0.1, lambda_sparsity=0.01):
        
    # Reconstruction loss (KL)
    beta = 1 
    loss = BetaDivLoss(beta=beta)
    loss_rec = loss(M_hat, M)

    # Activation loss (binary cross-entropy)
    loss_H = F.binary_cross_entropy_with_logits(H_hat, H)

    # Sparsity loss on H (L1)
    loss_sparsity = torch.sum(torch.abs(H_hat))

    # Total loss
    total_loss = loss_H + lambda_rec * loss_rec + lambda_sparsity * loss_sparsity
    
    return total_loss
   
def train(n_epochs, model, optimizer, loader, device):
    
    model.train()
    model.to(device=device)
    losses = []
    
    for epoch in range(n_epochs):
        for M, H in loader:

            W_hat, H_hat, M_hat = model(M)
            
            # if epoch == 1:
            #     print("W0 after first forward pass:")
            #     print(model.W0.min(), model.W0.max())
            #     print("H0 after first forward pass:")
            #     print(model.H0.min(), model.H0.max())
            
            loss = compute_loss(M, M_hat, H, H_hat)
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