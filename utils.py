import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import torchaudio.transforms as T
import torch.nn as nn
import torch.nn.functional as F
from torchbd.loss import BetaDivLoss
import torch
import torchaudio
import librosa
from torch.utils.data import Dataset
import glob, os

def spectrogram(signal, sample_rate, n_fft, hop_length, min_mag):
    
    """
    Computes the STFT spectrogram
    n_fft :         number of samples per window
    hop_length :    number of samples the next window will move
    min_mag :       min value of stft magnitude that we consider (0 if under)
    """
    
    window      = torch.hann_window(n_fft)
    stft        = torch.stft(signal, n_fft, hop_length, window=window, center=False, return_complex=True)
    stft_mag    = stft.abs()
    stft_mag    = torch.clamp(stft_mag, min=min_mag)

    # Convert to decibels
    spec_db     = T.AmplitudeToDB(stype='power', top_db=60)(stft_mag)
    spec_np     = spec_db.numpy()

    # Convert number of frames into time in s
    num_frames  = spec_db.shape[-1]
    frame_time  = hop_length / sample_rate
    times       = np.arange(num_frames) * frame_time

    # Convert the frequency bins to frequencies in Hz
    frequencies = np.fft.rfftfreq(n_fft, d=1/sample_rate)
    
    return spec_db, spec_np, times, frequencies

def cqt_spec(signal, sample_rate, hop_length, fmin=librosa.note_to_hz('A0'), bins_per_octave=36, n_bins=252, top_db=60):
    """
    Computes the CQT spectrogram
    """
    signal  = signal.squeeze().numpy()
    cqt     = librosa.cqt(y=signal, sr=sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    # Convert magnitude to decibels
    spec_db = librosa.amplitude_to_db(np.abs(cqt), top_db=top_db)
    
    # Time axis
    num_frames  = spec_db.shape[1]
    frame_time  = hop_length / sample_rate
    times       = np.arange(num_frames) * frame_time
    
    # Frequency axis
    frequencies = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    
    return spec_db, times, frequencies

def vis_cqt_spectrogram(spec_np, times, frequencies, start, stop, min_freq, max_freq):
    start_idx       = np.searchsorted(times, start)
    stop_idx        = np.searchsorted(times, stop)
    freq_start_idx  = np.searchsorted(frequencies, min_freq)
    freq_stop_idx   = np.searchsorted(frequencies, max_freq)

    spec_slice  = spec_np[freq_start_idx:freq_stop_idx, start_idx:stop_idx]
    freq_slice  = frequencies[freq_start_idx:freq_stop_idx]
    time_slice  = times[start_idx:stop_idx]

    # Convert frequencies to note labels
    note_labels = librosa.hz_to_note(freq_slice, octave=True, unicode=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(spec_slice, origin='lower', aspect='auto',
               extent=[time_slice[0], time_slice[-1],
                       0, len(freq_slice)])

    plt.title("CQT Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Notes")

    # Set y-ticks to note names
    step = max(1, len(note_labels) // 20)
    plt.yticks(ticks=np.arange(0, len(note_labels), step),
               labels=note_labels[::step])

    plt.tight_layout()
    plt.show()
    return

def vis_spectrogram(spec_np, times, frequencies, start, stop, min_freq, max_freq):
    start_idx   = np.searchsorted(times, start)
    stop_idx    = np.searchsorted(times, stop)
    freq_start_idx  = np.searchsorted(frequencies, min_freq)
    freq_stop_idx   = np.searchsorted(frequencies, max_freq)
    spec_slice = spec_np[freq_start_idx:freq_stop_idx, start_idx:stop_idx]
    freq_slice = frequencies[freq_start_idx:freq_stop_idx]
    time_slice = times[start_idx:stop_idx]
    
    plt.figure(figsize=(10, 5))
    plt.imshow(spec_slice, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1],
                       freq_slice[0]/1000, freq_slice[-1]/1000])
    plt.title("Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.show()
    return 

def midi_to_pianoroll(midi_path, n_time_steps, hop_length, sr=16000, n_keys=88):
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Keep only piano keys (A0-C8)
    note_start  = 21
    note_end    = 109

    piano_roll = midi.get_piano_roll(fs=sr / hop_length)[note_start:note_end]

    # Downsample/clip to match spectrogram time resolution
    if piano_roll.shape[1] > n_time_steps:
        piano_roll = piano_roll[:, :n_time_steps]
    elif piano_roll.shape[1] < n_time_steps:
        pad = n_time_steps - piano_roll.shape[1]
        piano_roll = np.pad(piano_roll, ((0, 0), (0, pad)))

    piano_roll = (piano_roll > 0).astype(np.float32)
    return torch.from_numpy(piano_roll)

def vis_midi(midi_mat, times, start, stop):
    start_idx   = np.searchsorted(times, start)
    stop_idx    = np.searchsorted(times, stop)
    midi_slice  = midi_mat[:, start_idx:stop_idx]
    time_slice  = times[start_idx:stop_idx]
    
    plt.figure(figsize=(10, 5))
    plt.imshow(midi_mat, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1],
                       0/1000, 88/1000])
    plt.title("Midi file")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    plt.tight_layout()
    plt.show()
    return 

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

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)

        spec_db, _, _, _ = spectrogram(waveform, sr, self.n_fft, self.hop_length, 1e-5)
        M = spec_db[:250,:900]

        H_target = midi_to_pianoroll(midi_path, M.shape[1], self.hop_length, sr)

        return M, H_target
    
    
def compute_loss(M, M_hat, H, H_hat, lambda_rec=0.1, lambda_sparsity=0.01):
        
    # Reconstruction loss (KL)
    beta = 1 
    loss = BetaDivLoss(beta=beta)
    loss_rec = loss(M_hat, M)

    # MIDI loss (binary cross-entropy)
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
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for param in model.parameters():
                if param.grad is not None:
                    print(f"Grad norm: {param.grad.norm()}")

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
                
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    return losses