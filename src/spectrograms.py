import torch
import numpy as np
import jams
import matplotlib.pyplot as plt
import pretty_midi
import librosa
import nnAudio.features.cqt as nn_cqt
from io import BytesIO
from PIL import Image
import warnings
from scipy.interpolate import interp1d
import wandb

# Problem in the pretty_midi library, some files are not loaded correctly, this patches the problem
pretty_midi.pretty_midi.MAX_TICK = 1e10

"""
CQT Spectrogram
"""
def cqt_spec(signal, sample_rate, hop_length=128, fmin=librosa.note_to_hz('A0'), bins_per_octave=36, n_bins=288, normalize_thresh=None, dtype=None):
    """
    Computes the CQT spectrogram
    """
    if dtype is not None:
        eps = torch.finfo(type=dtype).min
    else:
        eps = torch.finfo().min
        dtype = signal.dtype
    
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
            cqt = cqt_transform(signal)
            cqt = cqt.to(dtype)
            cqt = cqt.squeeze(0)
            
        if normalize_thresh is not None:
            cqt, _ = l1_norm(cqt, threshold=normalize_thresh, set_min=1e-6)
            cqt = torch.clamp(cqt, min=eps)
            # cqt = cqt / torch.sum(torch.abs(cqt), dim=0, keepdim=True)
        
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
    
def max_columns(W):
    max_values, _ = torch.max(W, dim=0)
    W_max = torch.zeros_like(W)
    _, max_indices = torch.max(W, dim=0)
    for col in range(W.shape[1]):
        W_max[max_indices[col], col] = max_values[col]

    return W_max

def l1_norm(tensor, threshold, set_min=None):
    """
    Computes the sum of the L1 norm of each column of the input vector
    We normalize each column by this sum if it is below the threshold, otherwise, we set this column to the min_value
    
    Args:
        tensor (torch.tensor): The input tensor to normalize
        threshold (float): The threshold below which to not normalize the column
        set_min (bool, optional): Whether to overwrite the values of the columns with norm < threshold with a min_value (default: ``None``)
    """
    if len(tensor.shape) ==2: 
        l1_norm = torch.sum(torch.abs(tensor), dim=0, keepdim=True)
    else:
        l1_norm = torch.sum(torch.abs(tensor), dim=1, keepdim=True)
    mean_norm = torch.mean(l1_norm)
    
    mask = l1_norm >= threshold * mean_norm
    mask = mask.expand_as(tensor)
    normalized_tensor = tensor / l1_norm

    if set_min is not None:
        tensor = torch.where(mask, normalized_tensor, set_min)
    else:
        tensor = torch.where(mask, normalized_tensor, tensor)

    return tensor, l1_norm

def vis_cqt_spectrogram(spec, times=None, noFreq=False, frequencies=None, start=None, stop=None, set_note_label=False, add_C8=False, cmap="magma", title=None, x_axis=None, font_size=None, max_ticks=None, use_wandb=False):
    
    if times is None:
        times = np.arange(spec.shape[1])
    
    if frequencies is None:
        frequencies = np.arange(spec.shape[0])
    
    if start is None:
        start = 0
        stop = spec.shape[1]
        
    start_idx       = np.searchsorted(times, start)
    stop_idx        = np.searchsorted(times, stop)

    spec_slice  = spec[:, start_idx:stop_idx]
    time_slice  = times[start_idx:stop_idx]

    # Convert frequencies to note labels
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
    # plt.ylabel("Notes" if set_note_label else "Frequency")

    # Set y-ticks to note names
    if set_note_label: 
        if max_ticks is not None:
            step = max_ticks
        else:
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
        if max_ticks is not None:
            step = max_ticks
        else:
            step = max(1, len(labels) // 20)
        
        if noFreq:
            plt.yticks([])
        else:
            plt.yticks(ticks=np.arange(0, len(labels), step),
                    labels=labels[::step])
            
    cbar = plt.colorbar()
    
    if font_size is not None:
        plt.tick_params(axis='y', labelsize=font_size)
        cbar.ax.tick_params(labelsize=font_size)
        
    plt.tight_layout()
    
    if use_wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)

        # Log the image to wandb
        wandb.log({"CQT Spectrogram": wandb.Image(img_array)})
    
    plt.show()
    return

"""
MIDI processing
"""
def midi_to_pianoroll_tensor(midi_path, waveform, times, hop_length, sr=16000, dtype=None):
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
    # times = torch.from_numpy(times)
    
    original_times = np.linspace(0, times[-1], piano_roll.shape[1])
    interp_func = interp1d(original_times, piano_roll, axis=1, kind='nearest', fill_value=0, bounds_error=False)
    piano_roll = interp_func(times)
    
    piano_roll = torch.from_numpy(piano_roll > 0).to(dtype)
    
    onsets = torch.zeros_like(piano_roll, dtype=dtype)
    offsets = torch.zeros_like(piano_roll, dtype=dtype)

    # Fill onset and offset matrices
    for instrument in midi.instruments:
        for note in instrument.notes:
            note_idx = note.pitch - note_start
            onset_idx = torch.argmin(torch.abs(times - note.start))
            offset_idx = torch.argmin(torch.abs(times - note.end))
            onsets[note_idx, onset_idx] = 1
            offsets[note_idx, offset_idx] = 1

    return piano_roll, onsets, offsets, times

def jams_to_pianoroll(jams_path, times, hop_length, sr=16000, default_velocity=100, dtype=None):
    # Load the JAMS file
    jam = jams.load(jams_path)

    # Extract note annotations (assuming the first annotation is note-based)
    note_ann = jam.search(namespace='note_midi')

    # Create a PrettyMIDI object and populate it with the notes from JAMS
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Default to piano for simplicity

    for i in range(len(note_ann)):
        for note in note_ann[i].data:
            pitch = int(round(note.value))
            midi_note = pretty_midi.Note(
                velocity=default_velocity,
                pitch=pitch,
                start=note.time,
                end=note.time + note.duration
            )
            instrument.notes.append(midi_note)

    midi.instruments.append(instrument)

    # Now use your existing function logic to create the piano roll
    note_start = 21
    note_end = 109
    fs = sr / hop_length
    piano_roll = midi.get_piano_roll(fs=fs)

    # Ensure the piano roll is sliced with integer indices
    piano_roll = piano_roll[note_start:note_end, :]

    # Interpolate to match the desired time axis
    original_times = np.linspace(0, times[-1], piano_roll.shape[1])
    interp_func = interp1d(original_times, piano_roll, axis=1, kind='nearest', fill_value=0, bounds_error=False)
    piano_roll = interp_func(times)

    piano_roll = (piano_roll > 0).astype(np.float32)

    onset_matrix = np.zeros_like(piano_roll)
    offset_matrix = np.zeros_like(piano_roll)

    for note in instrument.notes:
        note_idx = note.pitch - note_start
        onset_idx = np.argmin(np.abs(times - note.start))
        offset_idx = np.argmin(np.abs(times - note.end))
        if onset_idx < onset_matrix.shape[1]:
            onset_matrix[note_idx, onset_idx] = 1
        if offset_idx < offset_matrix.shape[1]:
            offset_matrix[note_idx, offset_idx] = 1

    onsets_tensor = torch.from_numpy(onset_matrix).to(dtype)
    offsets_tensor = torch.from_numpy(offset_matrix).to(dtype)

    return (
        torch.from_numpy(piano_roll).to(dtype),
        onsets_tensor,
        offsets_tensor,
        times
    )

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

def cut_midi_segment(midi, onset, offset, start_idx, end_idx, cut_start_notes=False):
    """
    Cuts the midi, onset and offset matrix between start_idx and end_idx 
    by making sure the notes are cut at the end of the segment, and started again at the beginning of the next segment
    
    cut_start_notes allows to zero out the notes that are active at time step 0
    """
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
    
    if cut_start_notes:
        active_rows = midi_segment[:, 0] == 1
        offset_segment[offset_segment == 0] = midi_segment.size(1)  # If never 0, set to last column

        mask = offset_segment == 1
        first_offset = torch.argmax(mask.int(), dim=1)
        first_offset[~mask.any(dim=1)] = offset_segment.size(1)
        
        for row in torch.where(active_rows)[0]:
            midi_segment[row, :first_offset[row]] = 0

    return midi_segment, onset_segment, offset_segment

def pianoroll_to_midi(piano_roll, midi_path, times, program=0):
    """
    Convert a piano roll tensor to a MIDI file.

    Args:
        piano_roll (torch.Tensor): Binary piano roll tensor (notes x time).
        times (torch.Tensor): Time array corresponding to the columns of the piano roll.
        midi_path (str): Path to save the output MIDI file.
        note_start (int): MIDI note number for the first row of the piano roll.
        note_end (int): MIDI note number for the last row of the piano roll.
        program (int): MIDI program number for the instrument.
    """
    note_start, note_end = 21, 109
    times = torch.tensor(times)
        
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # Pad piano_roll with a zero at the start and end to detect edge transitions
    padded_piano_roll = torch.cat([
        torch.zeros((piano_roll.shape[0], 1), dtype=piano_roll.dtype),
        piano_roll,
        torch.zeros((piano_roll.shape[0], 1), dtype=piano_roll.dtype)
    ], dim=1)
    padded_times = torch.cat([
        torch.tensor([0.0]),
        times,
        torch.tensor([times[-1] + (times[-1] - times[-2])])
    ])

    # Iterate over each note (row)
    for note_idx in range(padded_piano_roll.shape[0]):
        note_number = note_start + note_idx
        note_row = padded_piano_roll[note_idx, :]
        # Find where the note starts (0->1) and ends (1->0)
        diff = torch.diff(note_row.float())
        onsets = (diff == 1).nonzero(as_tuple=True)[0]
        offsets = (diff == -1).nonzero(as_tuple=True)[0]

        # Pair onsets and offsets
        for onset, offset in zip(onsets, offsets):
            if onset < offset:
                start_time = padded_times[onset].item()
                end_time = padded_times[offset].item()
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_number,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(midi_path)

def vis_midi(midi_mat, times, start=None, stop=None, title=None):
    if start is None:
        start = 0
        stop = times[-1]
    
    start_idx = np.searchsorted(times, start)
    stop_idx = np.searchsorted(times, stop)
    time_slice = times[start_idx:stop_idx]
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.imshow(midi_mat, origin='lower', aspect='auto',
                extent=[time_slice[0], time_slice[-1], 21, 109], cmap='Greens')  # pitch range A0-C8
    if title is not None:
        plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    plt.tight_layout()
    plt.show()
    return

def compare_midi(midi_gt, midi_hat, times=None, start=None, stop=None, midi_2=None, title=None, use_wandb=False):
    
    if times is None:
        times = np.arange(midi_gt.shape[1])
    
    if start is None:
        start = 0
        stop = midi_gt.shape[1]
        
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
    if title is not None:
        plt.title(title)
    else:
        plt.title("Predicted vs. Ground truth MIDI Files")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch")
    # plt.fill_between([start,stop-0.01],y1=0, y2=20, color="black", edgecolor='grey', hatch="/", label="Not valid MIDI notes")
    # plt.legend()
    plt.tight_layout()
    
    if use_wandb:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        
        img = Image.open(buf)
        img_array = np.array(img)

        # Log the image to wandb
        wandb.log({"MIDI GT vs MIDI Predicted": wandb.Image(img_array)})
    
    plt.show()
    return
