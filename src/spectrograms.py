import torch
import numpy as np
import matplotlib.pyplot as plt
import pretty_midi
import torchaudio.transforms as T
import librosa


# SFT Spectrogram:
def stft_spec(signal, sample_rate, n_fft, hop_length, min_mag):
    
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
    spec        = T.AmplitudeToDB(stype='power', top_db=60)(stft_mag)
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
                       freq_slice[0]/1000, freq_slice[-1]/1000])
    plt.title("Spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.tight_layout()
    plt.show()
    return 

# CQT Spectrogram:
def cqt_spec(signal, sample_rate, hop_length, fmin=librosa.note_to_hz('A0'), bins_per_octave=36, n_bins=288, top_db=60):
    """
    Computes the CQT spectrogram
    """
    signal  = signal.squeeze().numpy()
    cqt     = librosa.cqt(y=signal, sr=sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins, bins_per_octave=bins_per_octave)
    
    # Convert magnitude to decibels
    spec = librosa.amplitude_to_db(np.abs(cqt), top_db=top_db)
    
    # Time axis
    num_frames  = spec.shape[1]
    frame_time  = hop_length / sample_rate
    times       = np.arange(num_frames) * frame_time
    
    # Frequency axis
    frequencies = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    
    return spec, times, frequencies

def vis_cqt_spectrogram(spec, times, frequencies, start, stop, set_note_label=False):
    start_idx       = np.searchsorted(times, start)
    stop_idx        = np.searchsorted(times, stop)

    spec_slice  = spec[:, start_idx:stop_idx]
    time_slice  = times[start_idx:stop_idx]

    # Convert frequencies to note labels
    note_labels = librosa.hz_to_note(frequencies, octave=True, unicode=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(spec_slice, origin='lower', aspect='auto',
               extent=[time_slice[0], time_slice[-1],
                       0, len(frequencies)])

    plt.title("CQT Spectrogram (dB)")
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
                labels.append(f"{f:.0f} Hz")  # Display Hz for low frequencies
            else:
                labels.append(f"{f / 1000:.1f} kHz")  # Display kHz for high frequencies

        # Set y-ticks for Hz and kHz
        step = max(1, len(labels) // 20)
        plt.yticks(ticks=np.arange(0, len(labels), step),
                   labels=labels[::step])

    plt.tight_layout()
    plt.show()
    return

# ERB Spectrogram:
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

# MIDI Processing
def midi_to_pianoroll(midi_path, waveform, n_time_steps, hop_length, sr=16000):
    midi = pretty_midi.PrettyMIDI(midi_path)

    # Keep only piano keys (A0-C8)
    note_start = 21
    note_end = 109

    # Get the piano roll from PrettyMIDI
    piano_roll = midi.get_piano_roll(fs=sr / hop_length)[note_start:note_end]

    # Generate time axis for the MIDI file
    print(waveform)
    num_samples = waveform.shape[0]
    duration = num_samples / sr
    times = np.linspace(0, duration, n_time_steps)
    piano_roll = (piano_roll > 0).astype(np.float32)

    return torch.from_numpy(piano_roll), times

def vis_midi(midi_mat, times, start, stop):
    start_idx = np.searchsorted(times, start)
    stop_idx = np.searchsorted(times, stop)
    time_slice = times[start_idx:stop_idx]
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(midi_mat, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1], 0, 88])  # pitch range A0-C8
    plt.title("MIDI File")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    plt.tight_layout()
    plt.show()
    return
