import torch
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import torchaudio.transforms as T
import torch.nn.functional as F
import librosa
import nnAudio.features.cqt as nn_cqt
import time
import warnings
from scipy.interpolate import interp1d

# Problem in the pretty_midi library, some files are not loaded correctly, this patches the problem
pretty_midi.pretty_midi.MAX_TICK = 1e10

"""
SFT Spectrogram
"""
def stft_spec(signal, sample_rate, n_fft, hop_length, min_mag=1e-5):
    
    """
    Computes the STFT spectrogram
    n_fft :         number of samples per window
    hop_length :    number of samples the next window will move
    min_mag :       min value of stft magnitude that we consider (0 if under this value)
    """
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
        signal = signal.squeeze(0)
        
    window      = torch.hann_window(n_fft)
    stft        = torch.stft(signal, n_fft, hop_length, window=window, center=False, return_complex=True)
    stft_mag    = stft.abs()
    stft_mag    = torch.clamp(stft_mag, min=min_mag)
    spec        = stft_mag
    spec_np     = spec.numpy()

    # Convert number of frames into time in s
    num_frames  = spec.shape[-1]
    frame_time  = hop_length / sample_rate
    times       = np.arange(num_frames) * frame_time

    # Convert the frequency bins to frequencies in Hz
    frequencies = np.fft.rfftfreq(n_fft, d=1/sample_rate)
    
    return spec, spec_np, times, frequencies

def vis_spectrogram(spec, times, frequencies, start, stop, min_freq, max_freq):
    start_idx       = np.searchsorted(times, start)
    stop_idx        = np.searchsorted(times, stop)
    freq_start_idx  = np.searchsorted(frequencies, min_freq)
    freq_stop_idx   = np.searchsorted(frequencies, max_freq)
    spec_slice = spec[freq_start_idx:freq_stop_idx, start_idx:stop_idx]
    freq_slice = frequencies[freq_start_idx:freq_stop_idx]
    time_slice = times[start_idx:stop_idx]
    
    plt.figure(figsize=(10, 5))
    plt.imshow(spec_slice, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1],
                       freq_slice[0]/1000, freq_slice[-1]/1000], cmap='magma')
    plt.title("Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.show()
    return 

"""
CQT Spectrogram
"""
def cqt_spec(signal, sample_rate, hop_length=128, fmin=librosa.note_to_hz('A0'), bins_per_octave=36, n_bins=288, normalize_thresh=None, dtype=None):
    """
    Computes the CQT spectrogram
    """
    if type(signal) == torch.Tensor:
        # Convert to mono if stereo
        signal = signal.squeeze()
        if signal.dim() > 1 and signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0)
            
        warnings.filterwarnings("ignore", message="n_fft=.* is too large for input signal of length=")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        signal = signal.to(device)
        
        cqt_transform = nn_cqt.CQT2010v2(sr=sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave, verbose=False, output_format='Magnitude')
        
        with torch.no_grad():
            cqt = cqt_transform(signal).to(dtype)
            cqt = cqt.squeeze(0)
            
        if normalize_thresh is not None:
            cqt = l1_norm(cqt, threshold=normalize_thresh, set_to_zero=True)
            # cqt = cqt / torch.sum(torch.abs(cqt), dim=0, keepdim=True)
            
        # Convert magnitude to decibels
        # cqt = librosa.cqt(y=signal.cpu().numpy(), sr=sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
        # cqt = torch.from_numpy(cqt)
        
    else:
        signal  = signal.squeeze().numpy()
        if signal.shape[0] > 1: # Convert to mono if stereo
            signal = np.mean(signal, axis=0)
 
        cqt = librosa.cqt(y=signal, sr=sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
        if normalize_thresh is not None:
            cqt = np.apply_along_axis(lambda x: x / np.sum(np.abs(x)), axis=0, arr=cqt)

        # Convert magnitude to decibels
        cqt = np.abs(cqt)
    
    # Time axis
    num_frames  = cqt.shape[1]
    frame_time  = hop_length / sample_rate
    times       = np.arange(num_frames) * frame_time
    
    # Frequency axis
    frequencies = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    
    if type(signal) == torch.Tensor:
        return cqt, times, frequencies
    else:
        return torch.from_numpy(cqt), times, frequencies

def l1_norm(tensor, threshold, set_to_zero=True):
    l1_norm = torch.sum(torch.abs(tensor), dim=0, keepdim=True)
    mean_norm = torch.mean(l1_norm)
    
    mask = l1_norm >= threshold * mean_norm
    normalized_tensor = tensor / l1_norm

    if set_to_zero:
        tensor = normalized_tensor * mask
    else:
        tensor = torch.where(mask, normalized_tensor, tensor)

    return tensor

def vis_cqt_spectrogram(spec, times=None, frequencies=None, start=None, stop=None, set_note_label=False, add_C8=False, cmap="magma", title=None, x_axis=None):
    
    if times is None:
        times = np.arange(spec.shape[1])
    
    if frequencies is None:
        frequencies = np.arange(spec.shape[0])
    
    if start is None:
        start = 0
        end = spec.shape[1]    
    
    start_idx       = np.searchsorted(times, start)
    stop_idx        = np.searchsorted(times, stop)

    spec_slice  = spec[:, start_idx:stop_idx]
    time_slice  = times[start_idx:stop_idx]

    # Convert frequencies to note labels
    # note_labels = librosa.hz_to_note(frequencies, octave=True, unicode=False)
    note_labels = []
    for freq in frequencies:
        if freq == 0:
            note_labels.append("0")
        else:
            note_labels.append(librosa.hz_to_note(freq, octave=True, unicode=False))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(spec_slice, origin='lower', aspect='auto',
               extent=[time_slice[0], time_slice[-1],
                       0, len(frequencies)], cmap=cmap)
    if add_C8:
        plt.plot(np.arange(time_slice[-1]), [261]*time_slice[-1], color='g', label='4186Hz (C8)')
        plt.legend()

    if title is not None:
        plt.title(title)
    else:
        plt.title("CQT Spectrogram")
    if x_axis is not None:
        plt.xlabel(x_axis)
    else:
        plt.xlabel("Time (s)")
    plt.ylabel("Notes" if set_note_label else "Frequency")

    # Set y-ticks to note names
    if set_note_label: 
        step = max(1, len(note_labels) // 20)
        plt.yticks(ticks=np.arange(0, len(note_labels), step),
                labels=note_labels[::step])
    else: 
        labels = []
        for f in frequencies:
            if f <= 1000:
                labels.append(f"{f.item():.0f} Hz")  # Display Hz for low frequencies
            else:
                labels.append(f"{f.item() / 1000:.1f} kHz")  # Display kHz for high frequencies

        # Set y-ticks for Hz and kHz
        step = max(1, len(labels) // 20)
        plt.yticks(ticks=np.arange(0, len(labels), step),
                   labels=labels[::step])
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    return

"""
ERB Spectrogram
"""
def erb_freq(f):
    return 9.26 * np.log(0.00437 * f + 1)

def erb_filterbank(sample_rate, K_max, f_min, f_max):
    """
    Generate the ERB filterbank from f_min to f_max.
    """
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), num=K_max)
    
    erb_filters = []
    for f_center in freqs:
        # Create the filter for each center frequency (Gaussian band-pass filter)
        erb_filter = np.exp(-0.5 * ((np.arange(sample_rate // 2) - f_center) / (f_center / 2)) ** 2)
        erb_filter = erb_filter / np.sum(erb_filter)  # Normalize the filter
        erb_filters.append(erb_filter)
    
    erb_filters = np.array(erb_filters)
    return freqs, erb_filters

def erb_spec(signal, sample_rate, hop_length, K_max=40, f_min=20, f_max=8000):
    freqs, erb_filters = erb_filterbank(sample_rate, K_max, f_min, f_max)
    
    signal = signal.unsqueeze(0) if signal.ndimension() == 1 else signal
    
    stft = torch.stft(signal, n_fft=2048, hop_length=hop_length, center=False, return_complex=True)
    mag_stft = torch.abs(stft)
    erb_filters = torch.tensor(erb_filters, dtype=torch.float32)
    erb_filters = erb_filters[:, :mag_stft.shape[1]]
    
    spec = []
    for i in range(mag_stft.shape[-1]):
        frame = mag_stft[:, :, i].squeeze().numpy()
        band_energy = np.dot(erb_filters.numpy(), frame)
        spec.append(band_energy)
    
    spec = np.array(spec).T
    num_frames = spec.shape[1]
    frame_time = hop_length / sample_rate
    times = np.arange(num_frames) * frame_time
    
    return spec, times, freqs

def vis_erb_spectrogram(spec, freqs, times, start, stop):
    start_idx = np.searchsorted(times, start)
    stop_idx = np.searchsorted(times, stop)
    
    spec_slice = spec[:, start_idx:stop_idx]
    spec_slice = 20 * np.log10(spec_slice + 1e-10) 
    time_slice = times[start_idx:stop_idx]
    
    plt.figure(figsize=(10, 5))
    plt.imshow(spec_slice, origin='lower', aspect='auto',
               extent=[time_slice[0], time_slice[-1], freqs[0], freqs[-1]])
    plt.title("ERB Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

"""
MIDI processing
"""
def midi_to_pianoroll(midi_path, waveform, times, hop_length, sr=16000, dtype=None):
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    n_tracks = len(midi.instruments)

    # Keep only piano keys (A0-C8)
    note_start = 21
    note_end = 109

    # Get the piano roll from PrettyMIDI
    piano_roll = midi.get_piano_roll(fs=sr / hop_length)[note_start:note_end]

    # Generate time axis for the MIDI file
    num_samples = waveform.shape[0]
    duration = num_samples / sr
    # times = np.linspace(0, duration, n_time_steps)
    
    original_times = np.linspace(0, times[-1], piano_roll.shape[1])
    interp_func = interp1d(original_times, piano_roll, axis=1, kind='nearest', fill_value=0, bounds_error=False)
    piano_roll = interp_func(times)
    
    piano_roll = (piano_roll > 0).astype(np.float32)
    
    onset_matrix = np.zeros_like(piano_roll)
    offset_matrix = np.zeros_like(piano_roll)

    # Fill onset and offset matrices
    for instrument in midi.instruments:
        for note in instrument.notes:
            note_idx = note.pitch - note_start
            onset_idx = np.argmin(np.abs(times - note.start))
            offset_idx = np.argmin(np.abs(times - note.end))
            if onset_idx < onset_matrix.shape[1]:
                onset_matrix[note_idx, onset_idx] = 1
            if offset_idx < offset_matrix.shape[1]:
                offset_matrix[note_idx, offset_idx] = 1

    # Convert to tensors
    onsets_tensor = torch.from_numpy(onset_matrix).to(dtype)
    offsets_tensor = torch.from_numpy(offset_matrix).to(dtype)

    return torch.from_numpy(piano_roll).to(dtype), onsets_tensor, offsets_tensor, times

def cut_midi_segment(midi, onset, offset, start_idx, end_idx):
    # Ensure notes are turned off at the end of the segment
    active_notes = midi[:, end_idx - 1] > 0
    offset[:, end_idx - 1] = active_notes

    # Ensure notes are turned on at the beginning of the next segment if they are still active
    next_segment_onset = torch.zeros(midi.shape[0])
    next_segment_onset[active_notes] = 1
    onset[:, start_idx] = next_segment_onset

    # Cut the segment
    midi_segment = midi[:, start_idx:end_idx]
    onset_segment = onset[:, start_idx:end_idx]
    offset_segment = offset[:, start_idx:end_idx]

    return midi_segment, onset_segment, offset_segment

def vis_midi(midi_mat, times, start, stop):
    start_idx = np.searchsorted(times, start)
    stop_idx = np.searchsorted(times, stop)
    time_slice = times[start_idx:stop_idx]
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(midi_mat, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1], 21, 109], cmap='Greens')  # pitch range A0-C8
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    plt.tight_layout()
    plt.show()
    return

def compare_midi(midi_gt, midi_hat, times, start, stop, midi_2=None):
    start_idx = np.searchsorted(times, start)
    stop_idx = np.searchsorted(times, stop)
    time_slice = times[start_idx:stop_idx]
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(midi_hat, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1], 21, 109], alpha=0.9, cmap='Reds')  # pitch range A0-C8
    plt.imshow(midi_gt, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1], 21, 109], alpha=0.5, cmap="Greens")  # pitch range A0-C8
    if midi_2 is not None:
        plt.imshow(midi_2, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1], 21, 109], alpha=0.5, cmap="Blues")  # pitch range A0-C8
    plt.title("Predicted vs. Ground truth MIDI Files")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    # plt.fill_between([start,stop-0.01],y1=0, y2=20, color="black", edgecolor='grey', hatch="/", label="Not valid MIDI notes")
    # plt.legend()
    plt.tight_layout()
    plt.show()
    return
