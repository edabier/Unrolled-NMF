import torch.nn as nn
import torch.nn.functional as F
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler
import librosa
import numpy as np
import wandb
import matplotlib.pyplot as plt
import glob, os, warnings, operator
import mir_eval
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import src.spectrograms as spec
import src.init as init

def model_infos(model, names=False):
    """
    Displays the number of parameters of the model
    and the names of the layers if names is set to True
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if names:
        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()}")
    print(f"The model has {total_params} parameters")

class NMFDataset(Dataset):
    """
    creates a Dataset object from a folder of midi and audio files
    Args:
        audio_dir (str): the directory for audio (.wav) files
        midi_dir (str): the directory for midi (.mid) files
        hop_length (int, optional): define the hop length for the CQT spectrogram (default: ``128``)
        use_h (bool, optional): whether to use pianoroll midi or H matrix as dataset's item (default: ``False``)
        fixed_length (bool, optional): whether to have a constant duration for audio/ midi items (default: ``True``)
        num_files (int, optional): The number of files to include in the dataset. If None, all files are used (default: ``None``)
    """
    
    def __init__(self, audio_dir, midi_dir, hop_length=128, use_H=False, fixed_length=True, num_files=None):
        assert os.path.isdir(audio_dir) or os.path.isdir(midi_dir), f"The directory '{audio_dir} or {midi_dir}' does not exist"
        self.audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        self.midi_dir = midi_dir
        self.hop_length = hop_length
        self.use_H = use_H
        self.fixed_length = fixed_length
        
        self.num_files = num_files
        if num_files is not None and num_files < len(self.audio_files):
            self.audio_files = self.audio_files[:num_files]
            
        self.min_length = self._determine_min_length() if fixed_length else None
        self.segment_indices = self._precompute_segment_indices() if fixed_length else None

    def __len__(self):
        if self.fixed_length:
            return sum(self.segment_indices)
        else:
            return len(self.audio_files)

    def _determine_min_length(self):
        min_length = float('inf')
        for audio_path in self.audio_files:
            waveform, sr = torchaudio.load(audio_path)
            _, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
            length = times_cqt.shape[0]
            if length < min_length:
                min_length = length
        return min_length

    def _precompute_segment_indices(self):
        segment_indices = []
        for audio_path in self.audio_files:
            waveform, sr = torchaudio.load(audio_path)
            _, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
            length = times_cqt.shape[0]
            segment_indices.append(length // self.min_length)
        return segment_indices

    def __getitem__(self, idx):
        if self.fixed_length:
            # Find which audio file the idx corresponds to
            audio_idx = 0
            while idx >= self.segment_indices[audio_idx]:
                idx -= self.segment_indices[audio_idx]
                audio_idx += 1

            audio_path = self.audio_files[audio_idx]
            waveform, sr = torchaudio.load(audio_path)
            M, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
            midi_path = os.path.join(self.midi_dir, os.path.basename(audio_path).replace(".wav", ".mid"))
            midi, onset, offset, _ = spec.midi_to_pianoroll(midi_path, waveform, times_cqt, self.hop_length)

            if self.use_H:
                active_midi = [i for i in range(88) if (midi[i, :] > 0).any().item()]
                midi = init.MIDI_to_H(midi, active_midi, onset, offset)

            start_idx = idx * self.min_length
            end_idx = start_idx + self.min_length

            M_segment = M[:, start_idx:end_idx]
            midi_segment = midi[:, start_idx:end_idx]

            return M_segment, midi_segment
        else:
            audio_path = self.audio_files[idx]
            waveform, sr = torchaudio.load(audio_path)
            M, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
            midi_path = os.path.join(self.midi_dir, os.path.basename(audio_path).replace(".wav", ".mid"))
            midi, onset, offset, _ = spec.midi_to_pianoroll(midi_path, waveform, times_cqt, self.hop_length, sr)

            if self.use_H:
                active_midi = [i for i in range(88) if (midi[i, :] > 0).any().item()]
                midi = init.MIDI_to_H(midi, active_midi, onset, offset)

            return M, midi
        
class SequentialBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Split indices into batches
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        # Flatten the list of batches
        return iter(batches)

    def __len__(self):
        return len(self.data_source) // self.batch_size
    
def pad_by_repeating(tensor, max_length):
    if tensor.size(1) < max_length:
        repeat_times = int(np.ceil(max_length / tensor.size(1)))
        repeated_tensor = tensor.repeat(1, repeat_times)[:, :max_length]
        return repeated_tensor
    return tensor
    
def collate_fn(batch):
    # Separate audio and MIDI files
    audio_files = [item[0] for item in batch]
    midi_files = [item[1] for item in batch]
    
    max_audio_length = max(audio.size(1) for audio in audio_files)
    max_midi_length = max(midi.size(1) for midi in midi_files)

    # Pad audio and MIDI files by repeating their data
    audio_files_padded = torch.stack([pad_by_repeating(audio, max_audio_length) for audio in audio_files])
    midi_files_padded = torch.stack([pad_by_repeating(midi, max_midi_length) for midi in midi_files])

    # Return them as lists to avoid stacking
    return audio_files_padded, midi_files_padded      

class MapsDataset(Dataset):
    """
    Creates a Dataset object from Maps' metadata

    Args:
        metadata (pd.DataFrame): DataFrame containing 'file_path' and 'midi_path' columns.
        hop_length (int, optional): Define the hop length for the CQT spectrogram (default: 128).
        use_H (bool, optional): Whether to use pianoroll MIDI or H matrix as dataset's item (default: False).
        fixed_length (bool, optional): Whether to have a constant duration for audio/MIDI items (default: True).
        subset (float, optional): The subset of files to include in the dataset. If None, all files are used (default: None).
        mean_filter (bool, optional): Whether to filter the dataset to keep only files with length > mean(length) (default: `False`)
    """
    def __init__(self, metadata, hop_length=128, use_H=False, fixed_length=None, subset=None, sort=True, filter=False, verbose=False):
        assert 'file_path' in metadata.columns and 'midi_path' in metadata.columns, "Metadata should contain 'file_path' and 'midi_path' columns."

        self.metadata = metadata.copy()
        self.hop_length = hop_length
        self.use_H = use_H
        self.fixed_length = fixed_length
        self.sort = sort

        if subset is not None:
            num_files = int(len(metadata) * subset)
            self.metadata = self.metadata.iloc[:num_files]
        
        if self.sort:
            if verbose:
                print("Computing the length of files...")
            durations, self.segment_indices, time_steps = self._compute_lengths()
            self.metadata.loc[:, 'duration'] = durations
            self.metadata.loc[:, 'time_steps'] = time_steps
            self.metadata = self.metadata.sort_values(by='duration')
            
            if filter:
                self.metadata = self.metadata[self.metadata['time_steps'] > self.fixed_length].reset_index(drop=True)
        
            self.min_length = fixed_length
    
    def _compute_lengths(self):
        durations = []
        segment_indices = []
        time_steps = []
        for _, row in tqdm(self.metadata.iterrows()):
            waveform, _ = torchaudio.load(row['file_path'])
            sr = row["sampling_rate"]
            _, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
            durations.append(waveform.shape[1]/sr)
            segment_indices.append(times_cqt.shape[0])
            time_steps.append(times_cqt.shape[0])
            
        self.fixed_length = int((self.fixed_length * sr) / self.hop_length)
        return np.array(durations), segment_indices, time_steps

    def __len__(self):
        if self.fixed_length:
            return sum(self.segment_indices)
        else:
            return len(self.metadata)

    def __getitem__(self, idx):
        if self.fixed_length is not None:
            assert self.sort, "The dataset needs to be sorted to be able to use a fixed length, set sort to True"
            audio_idx = 0
            
            while idx >= self.segment_indices[audio_idx]:
                idx -= self.segment_indices[audio_idx]
                audio_idx += 1

            row = self.metadata.iloc[audio_idx]
            try:
                waveform, sr = torchaudio.load(row['file_path'])
                M, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
                midi, onset, offset, _ = spec.midi_to_pianoroll(row['midi_path'], waveform, times_cqt, self.hop_length)

                if self.use_H:
                    active_midi = [i for i in range(88) if (midi[i, :] > 0).any().item()]
                    midi = init.MIDI_to_H(midi, active_midi, onset, offset)

                start_idx = idx * self.fixed_length
                end_idx = start_idx + self.fixed_length

                M_segment = M[:, start_idx:end_idx]
                midi_segment = midi[:, start_idx:end_idx]

                return M_segment, midi_segment
            except Exception as e:
                print(f"Skipping {row['midi_path']} due to error: {e}")
                return self.__getitem__((idx + 1) % len(self))
        else:
            row = self.metadata.iloc[idx]
            try:
                waveform, sr = torchaudio.load(row['file_path'])
                M, times_cqt, _ = spec.cqt_spec(waveform, sr, self.hop_length)
                midi, onset, offset, _ = spec.midi_to_pianoroll(row['midi_path'], waveform, times_cqt, self.hop_length, sr)

                if self.use_H:
                    active_midi = [i for i in range(88) if (midi[i, :] > 0).any().item()]
                    midi = init.MIDI_to_H(midi, active_midi, onset, offset)

                return M, midi
            except Exception as e:
                print(f"Skipping {row['midi_path']} due to error: {e}")
                return self.__getitem__((idx + 1) % len(self))

"""
Metrics
"""
def compute_metrics(prediction, ground_truth, time_tolerance=0.05, threshold=0):
    """
    Compute the precision, recall, F-measure and the accuracy of the transcription using mir_eval
    """
    if prediction == []:
        return 0, 0, 0, 0
    
    gt_notes = extract_note_events(ground_truth)
    pred_notes = extract_note_events(prediction, threshold=threshold)
    
    if len(gt_notes) == 0 or len(pred_notes) == 0:
        return 0, 0, 0, 0
    
    gt_times = gt_notes[:,0:2]
    gt_pitch = gt_notes[:,2]
    
    pred_times = pred_notes[:,0:2]
    pred_pitch = pred_notes[:,2]
    
    if np.any(gt_pitch < 21) or np.any(pred_pitch < 21):
        raise ValueError("Pitch values must be valid MIDI notes (>= 21).")

    prec, rec, f_mes, _ = mir_eval.transcription.precision_recall_f1_overlap(
        gt_times, gt_pitch, pred_times, pred_pitch, 
        offset_ratio = None, onset_tolerance = time_tolerance, pitch_tolerance = 0.1)

    accuracy = accuracy_from_recall(rec, len(gt_times), len(gt_pitch))

    return prec, rec, f_mes, accuracy#, gt_notes, pred_notes

def extract_note_events(piano_roll, threshold=0):
    """
    Creates a note_event object from a piano_roll tensor
    The note_event is a list of notes (start, end, pitch)
    """
    # Pad the tensor to handle edge cases
    padded_tensor = torch.zeros(piano_roll.shape[0], piano_roll.shape[1] + 2, device=piano_roll.device)
    padded_tensor[:, 1:-1] = piano_roll

    note_events = []
    note_starts = ((padded_tensor[:, :-1] <= threshold) & (padded_tensor[:, 1:] > threshold)).nonzero(as_tuple=True)
    note_ends = ((padded_tensor[:, :-1] > threshold) & (padded_tensor[:, 1:] <= threshold)).nonzero(as_tuple=True)

    # Iterate over each pitch and pair starts with ends
    for pitch in range(padded_tensor.shape[0]):
        starts = note_starts[1][note_starts[0] == pitch]
        ends = note_ends[1][note_ends[0] == pitch]

        # Pair each start with an end
        for start, end in zip(starts, ends):
            note_events.append([start.item(), end.item(), pitch+21])

    return np.array(note_events, dtype=np.int32).reshape(-1, 3)

def accuracy_from_recall(recall, N_gt, N_est):
    """
    Compute the accuracy from recall, number of samples in ground truth (N_gt), and number of samples in estimation (N_est).
    """
    tp = int(recall * N_gt)
    fn = int(N_gt - tp)
    fp = int(N_est - tp)
    try:
        return tp / (tp + fp + fn)
    except ZeroDivisionError:
        return 0


"""
Loss (no  batch_size)
"""
def gaussian_kernel(kernel_size=3, sigma=2, is_2d=False):
    """
    Creates a 1D or 2D Gaussian kernel.
    """
    if is_2d:
        ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    else:    
        kernel = np.exp(-np.linspace(-kernel_size / 2, kernel_size / 2, kernel_size)**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    return kernel

def filter1d_tensor(tensor, kernel, axis=0, is_2d=False):
    """
    Applies a 1D or 2D Gaussian filter to a 2D tensor along a specified axis.
    """
    if is_2d:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        filtered_tensor = F.conv2d(tensor, kernel, padding=kernel.size(-1) // 2).squeeze(0).squeeze(0)
    else:
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
    
    onset[:, 0] = (midi[:, 0] > 0).float()
    offset[:, -1] = (midi[:, -1] > 0).float()
    # For every time step
    for time in range(1, midi.shape[1]-1):
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
   
"""
Train the network
"""
def permutation_match(W_new, W_init, rows=False):
    if rows:
        # Transpose the matrices to work with rows instead of columns
        W_init = W_init.T
        W_new = W_new.T
    
    cost_matrix = torch.zeros((W_init.shape[1], W_new.shape[1]))
    for i in range(W_init.shape[1]):
        for j in range(W_new.shape[1]):
            # Use Euclidean distance as the cost metric
            cost_matrix[i, j] = torch.norm(W_init[:, i] - W_new[:, j])
            
    cost_matrix_np = cost_matrix.numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_np)
    
    if rows:
        W_new_rearranged = W_new[:, col_ind].T
    else:
        W_new_rearranged = W_new[:, col_ind]
        
    return W_new_rearranged

def soft_permutation_match(tensor_new, tensor_init, rows=False):
    
    assert tensor_new.shape == tensor_init.shape, "Init and new tensors must have the same shape"
    warnings.filterwarnings("ignore", message="Using a target size")
    
    batched = (len(tensor_new.shape)==3 and len(tensor_init.shape)==3)
    
    if batched:
        batch_size = tensor_new.shape[0]
        tensor_new_rearranged = torch.zeros_like(tensor_new)

        for i in range(batch_size):
            tensor_new_rearranged[i] = soft_permutation_match(tensor_new[i].clone(), tensor_init[i].clone(), rows)
        return tensor_new_rearranged
    
    else:
        if rows:
            tensor_init = tensor_init.T
            tensor_new = tensor_new.T
        
        assert tensor_init.dim() == 2 and tensor_new.dim() == 2, "Input tensors must be 2-dimensional"

        # Compute the cost matrix using Euclidean distance
        cost_matrix = torch.cdist(tensor_init.T.unsqueeze(0), tensor_new.T.unsqueeze(0), p=2).squeeze(0)

        # Use the Hungarian algorithm to find the optimal permutation
        row_idx, col_idx = linear_sum_assignment(cost_matrix.detach().cpu())
        tensor_new_rearranged = tensor_new[:, col_idx]

        if rows:
            tensor_new_rearranged = tensor_new_rearranged.T

        return tensor_new_rearranged
    
def spectral_flatness(spec, log_val=1):
    geometric_mean = torch.exp(torch.mean(torch.log(log_val + spec), dim=0))
    arithmetic_mean = torch.mean(spec, dim=0)
    return geometric_mean / arithmetic_mean

def warmup_train(model, n_epochs, loader, optimizer, device, debug=False):
    losses = []
    H1 = None
    for i in range(n_epochs):
        train_loss = 0
        for idx, (M, _) in enumerate(loader):
            model.init_H(M[0])
            if i == 0 and idx==6:
                H1 = model.H0
            M = M.to(device)
            W_hat, H_hat, _ = model(M)
            
            W_hat_r = soft_permutation_match(W_hat, model.W0)
            H_hat_r = soft_permutation_match(H_hat, model.H0, rows=True)
            
            ground_truth_norm = torch.norm(model.W0) + torch.norm(model.H0)
            train_loss += torch.norm(model.W0 - W_hat_r) + torch.norm(model.H0 - H_hat_r)
            
            optimizer.zero_grad()
            train_loss.backward()
                    
            optimizer.step()
            train_loss = train_loss.item() / ground_truth_norm * 100
        losses.append(train_loss/ len(loader))
        print(f"------- epoch {i}, loss = {losses[i]:.3f} -------")
    if debug:
        plt.plot(losses, label='Reconstruction of W + H loss over epochs')
        plt.xlabel('epochs')
        plt.show()
        
    spec.vis_cqt_spectrogram(W_hat.detach(), np.arange(W_hat.shape[1]), np.arange(W_hat.shape[0]), 0, W_hat.shape[1], title="Aw(W) with line permuted W")
    spec.vis_cqt_spectrogram(W_hat_r.detach(), np.arange(W_hat_r.shape[1]), np.arange(W_hat_r.shape[0]), 0, W_hat_r.shape[1], title="rearranged Aw(W)")
    
    return losses, W_hat, H_hat, H1    
 
def train(model, loader, optimizer, criterion, device, epochs, valid_loader=None, wandb=False):
    if wandb:
        run = wandb.init(
            project=f"{model.__class__.__name__}_train",
            config={
                "learning_rate": optimizer.param_groups[-1]['lr'],
                "batch_size": loader.batch_size,
                "epochs": epochs,
            },
        )
    
    train_losses, valid_losses = [], []
    valid_loss_min = np.inf
    
    for epoch in range(epochs):
        
        train_loss, valid_loss = 0, 0
        
        model.train()
        for M, H in loader:
            
            M = M.to(device)
            H = H.to(device)
            
            W_hat, H_hat, _ = model(M, device=device)
            
            # sfm = spectral_flatness(W_hat, log_val=1e-5)
            # harmonicity = torch.mean(sfm)
            # print(f"Harmonicity score: {harmonicity}")
            
            # H_hat_r = soft_permutation_match(H_hat, model.H0, rows=True)
            
            optimizer.zero_grad()
            loss = criterion(H_hat, H)
            # loss = criterion(H_hat_r, H[0])
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(loader)
        train_losses.append(train_loss)
        print(f"epoch {epoch}, loss = {train_loss:5f}")
        
        if wandb:
            wandb.log({"training loss": train_loss})
    
        if valid_loader is not None:
        
            model.eval()
            for M, H in valid_loader:
                
                M = M.to(device)
                H = H.to(device)
                
                with torch.no_grad():
                    W_hat, H_hat, _ = model(M)
                
                # H_hat_r = soft_permutation_match(H_hat, model.H0, rows=True)
                # loss = criterion(H_hat_r, H[0])
                loss = criterion(H_hat, H)
                valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

            if wandb:
                wandb.log({"valid loss": valid_loss})
            
        if valid_loss <= valid_loss_min:
          print('validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
          valid_loss_min,
          valid_loss))
          torch.save(model.state_dict(), f'/home/ids/edabier/AMT/Unrolled-NMF/models/{model.__class__.__name__}.pt')
          valid_loss_min = valid_loss
    
    return train_losses, valid_losses, W_hat, H_hat
    
def midi_train(model, loader, optimizer, criterion, device, epochs):
    losses = []
    
    for epoch in range(epochs):
        
        epoch_loss = 0
        
        for M, midi_gt in loader:
            
            M = M.to(device)
            model.init_H(M[0], device=device)
            midi_gt = midi_gt[0].to(device)
            
            W_hat, H_hat, _ = model(M)
            
            _, notes_pred = init.W_to_pitch(W_hat, model.freqs, use_max=True)
            midi_pred, active_pred = init.WH_to_MIDI(W_hat, H_hat, notes_pred, threshold=0.05)

            active_gt = [i for i in range(88) if (midi_gt[i,:] > 0).any().item()]
            active = list(set(active_gt + active_pred))
            
            optimizer.zero_grad()
            loss = criterion(midi_pred[active,:], midi_gt[active,:])
    
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.detach()
        losses.append(epoch_loss/ loader.__len__())
        print(f"epoch {epoch}, loss = {losses[epoch]:5f}")
    
    return losses, W_hat, H_hat    
    
def transribe(model, M, device):
    model.eval()
    model.to(device=device)

    # Initialize H for each input tensor
    model.init_H(M)

    with torch.no_grad():
        W_hat, H_hat, M_hat = model(M)
        freqs = librosa.cqt_frequencies(n_bins=288, fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
        _, notes = init.W_to_pitch(W_hat, freqs, use_max=True)
        midi_hat, active_midi = init.WH_to_MIDI(W_hat, H_hat, notes, threshold=0.05)

    return W_hat, H_hat, M_hat, midi_hat, active_midi
