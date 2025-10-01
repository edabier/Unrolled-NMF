import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, Sampler
import pandas as pd
import librosa
import numpy as np
import wandb
import matplotlib.pyplot as plt
import os, warnings, math
import mir_eval
from tqdm import tqdm
import subprocess, csv, time
from datetime import datetime
from scipy.optimize import linear_sum_assignment
import fcntl

import src.spectrograms as spec
import src.init as init

"""
Model and gpu infos 
"""
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
    
def save_model(model, optimizer, directory, epoch=None, is_permanent=False, name='model_epoch'):
    """
    Overwrite the previous checkpoint save if not is_permanent, otherwise, saves a new version of the model. 
    We can provide the epoch to save the model and restart the training later on
    """
    if is_permanent:
        if epoch is not None:
            # Save a permanent copy of the model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(directory, f'{name}_{epoch}.pt'))
            print(f"Saved permanent model {name}_{epoch}.pt")
        else:
            # Save a permanent copy of the model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(directory, f'{name}.pt'))
            print(f"Saved permanent model {name}.pt")
    else:
        # Overwrite the temporary model save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(directory, 'checkpoint.pt'))
        print("Saved checkpoint model")
        
def load_checkpoint(path, model, optimizer):
    """
    Loads the last training checkpoint of the model
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")
    return start_epoch

def get_gpu_info():
    """
    When running a gpu process, gets the current value of the nvidia-smi command (gpu name, current power use, current memory use)
    """
    try:
        # Run the nvidia-smi command to get GPU name and power draw
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,power.draw,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Extract the GPU name and power draw values
        lines = result.stdout.strip().split('\n')
        gpu_info = [line.split(', ') for line in lines]
        return gpu_info
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return None

def log_gpu_info(gpu_info, filename='AMT/Unrolled-NMF/logs/gpu_info_log.csv'):
    """
    Writes the gpu information retrieved from nvidia-smi to a csv file
    """
    
    # If there is no csv log file, create one
    if not os.path.isfile(filename):
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time", "partition_name", "power_usage", "memory_usage"])
            
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for info in gpu_info:
            if len(info) == 3:  # Ensure name, power draw and memory are captured
                writer.writerow([timestamp, info[0], info[1], info[2]])

def compute_emissions(file_path, emission_per_kwh=10, emission_per_hour=3.67, start=None):
    """
    Computes the CO2e emissions of the energy consumed from a csv file containing the data of power usage and time.
    Emissions are calculated both based on energy consumption and usage time.
    By default, we use EDF's value of 10 gCO2e/kWh in France in 2024 and a value of 3.67 gCO2e/hour for usage time.
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])

    if start is not None:
        df = df[df['time'] >= pd.to_datetime(start)]

    # Compute the time difference between consecutive measures (in seconds)
    time_diffs = df['time'].diff().dt.total_seconds().fillna(0).values
    time_diffs_tensor = torch.tensor(time_diffs, dtype=torch.float32)

    # Detect larger differences to distinguish runs
    time_diff_diffs = torch.abs(time_diffs_tensor[1:] - time_diffs_tensor[:-1])
    time_diff_diffs = torch.cat([torch.tensor([0.0]), time_diff_diffs])
    threshold = torch.quantile(time_diff_diffs, 0.95)  # Use a 95% quantile threshold
    launch_ends = time_diff_diffs > threshold

    # Identify the start and end indices of each run
    run_starts = [0] + (torch.where(launch_ends)[0] + 1).tolist()
    run_ends = (torch.where(launch_ends)[0]).tolist() + [len(df) - 1]

    # Calculate the duration of each run (in hours)
    run_durations = []
    for start_idx, end_idx in zip(run_starts, run_ends):
        if start_idx <= end_idx:
            duration = (df.iloc[end_idx]['time'] - df.iloc[start_idx]['time']).total_seconds() / 3600
            run_durations.append(duration)

    total_active_time_hours = sum(run_durations)

    # Compute the consumed energy
    power_usage = torch.tensor(df['power_usage'].values, dtype=torch.float32)
    energy = power_usage * (time_diffs_tensor / 3600)  # Convert to kWh
    energy[launch_ends] = 0  # Don't count energy between launches
    total_energy_kwh = torch.sum(energy) / 1000  # Convert to kWh

    # Total CO2e emissions in kg of CO2e
    total_emissions_kwh = total_energy_kwh * emission_per_kwh / 1000
    total_emissions_hour = total_active_time_hours * emission_per_hour / 1000

    return total_energy_kwh, total_emissions_kwh.item(), total_emissions_hour

"""
Dataset class and utils functions
"""     
class SequentialBatchSampler(Sampler):
    """
    
    """
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        # Split indices into batches
        self.batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        # Flatten the list of batches
        return iter(self.batches)

    def __len__(self):
        return math.ceil(len(self.data_source) / self.batch_size)
    
def pad_by_repeating(tensor, max_length):
    """
    Pads files that are shorter than the desired length by repeating the content until we match the length
    """
    current_length = tensor.size(1)
    if current_length == 0:
        raise ValueError("Tensor length is zero, cannot pad.")
    
    if tensor.size(1) < max_length:
        repeat_times = (max_length + current_length - 1) // current_length
        repeated_tensor = tensor.repeat(1, repeat_times)[:, :max_length]
        return repeated_tensor
    return tensor
    
def create_collate_fn(use_midi=False):
    """
    Creates a custom collate function
    
    If we use midi, the collate function will return M, H_gt and MIDI_gt
    otherwise, we just return M and H_gt
    """
    def collate_fn(batch):
        if use_midi:
            # Separate audio and MIDI files
            M_files = [item[0] for item in batch if item[0].size(1) > 0]
            H_files = [item[1] for item in batch if item[1].size(1) > 0]
            midi_files = [item[2] for item in batch if item[2].size(1) > 0]
            
            if not M_files or not H_files or not midi_files:
                return torch.tensor([]), torch.tensor([]), torch.tensor([])
            
            max_M_length = max(M.size(1) for M in M_files)
            max_H_length = max(H.size(1) for H in H_files)
            max_midi_length = max(midi.size(1) for midi in midi_files)

            # Pad audio and MIDI files by repeating their data
            M_files_padded = torch.stack([pad_by_repeating(M, max_M_length) for M in M_files])
            H_files_padded = torch.stack([pad_by_repeating(H, max_H_length) for H in H_files])
            midi_files_padded = torch.stack([pad_by_repeating(midi, max_midi_length) for midi in midi_files])

            # Return them as lists to avoid stacking
            return M_files_padded, H_files_padded, midi_files_padded 
        else:
            # Separate audio and MIDI files
            M_files = [item[0] for item in batch if item[0].size(1) > 0]
            H_files = [item[1] for item in batch if item[1].size(1) > 0]
            
            if not M_files or not H_files:
                return torch.tensor([]), torch.tensor([])
            
            max_M_length = max(M.size(1) for M in M_files)
            max_H_length = max(H.size(1) for H in H_files)

            # Pad audio and MIDI files by repeating their data
            M_files_padded = torch.stack([pad_by_repeating(audio, max_M_length) for audio in M_files])
            H_files_padded = torch.stack([pad_by_repeating(midi, max_H_length) for midi in midi_files])

            # Return them as lists to avoid stacking
            return M_files_padded, H_files_padded 

class LocalDataset(Dataset):
    """
    Creates a Dataset object from localy saved CQT + H + midi

    Args:
        metadata (pd.DataFrame): DataFrame containing 'file_path', 'midi_path', 'onset_path' and 'offset_path' columns.
        use_midi (bool, optional): Whether to return a midi gt file or not as a point of the dataset.
        dev: The device on which to load the dataset (default: None).
        sr (int, optional): The sample rate of the recordings to convert the fixed_length in cqt steps (default: 22050).
        hop_length (int, optional): The hop_length to convert the fixed_length in cqt steps (default: 128).
        fixed_length (bool, optional): Whether to have a constant duration for audio/MIDI items (default: True).
        cut_start_notes (bool, optional): Whether to cut out the notes active at time step 0 or not (default: False).
        subset (float, optional): The subset of files to include in the dataset. If None, all files are used (default: None).
    """
    def __init__(self, metadata, use_midi=False, dev=None, sr=22050, hop_length=128, fixed_length=None, cut_start_notes=False, subset=None, dtype=None):

        self.metadata       = metadata
        self.use_midi       = use_midi
        self.dev            = dev if dev is not None else torch.device('cpu')
        self.sr             = sr
        self.hop_length     = hop_length
        self.fixed_length   = fixed_length
        self.cut_start_notes = cut_start_notes
        self.dtype          = dtype
        
        if subset is not None:
            num_files = int(len(self.metadata) * subset)
            self.metadata = self.metadata[:num_files]

        if self.fixed_length is not None:
            self.metadata.loc[:,'segment_indices'] = self.compute_length()
            
    def compute_length(self):
        """
        Computes the duration in seconds of every audio file M of the dataset and returns it as a list
        """
        self.fixed_length = math.ceil((self.fixed_length * self.sr) / self.hop_length)
        segment_indices = []
        for _, row in tqdm(self.metadata.iterrows()):
            M = torch.load(f"/home/ids/edabier/{row.file_path}", map_location=self.dev)
            segment_indices.append(-(M.shape[1] // -self.fixed_length))
        return segment_indices
    
    def __len__(self):
        """
        Returns the length of the dataset (if we use a fixed length, and thus cut the long files, we sum the number of resulting files)
        """
        if self.fixed_length is not None:
            return sum(self.metadata.segment_indices)
        else:
            return len(self.metadata)
    
    def safe_load(self, path):
        """
        Helper function to load the torch tensors
        """
        path = "/home/ids/edabier/" + path
        with open(path, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = torch.load(f, map_location=self.dev)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    
    def __getitem__(self, idx):
        """
        Actual function loading the data points of the dataset
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        if self.fixed_length is not None:
            cumulative_indices = np.cumsum(self.metadata.segment_indices)
            file_idx = np.searchsorted(cumulative_indices, idx, side='right')
            audio_idx = idx - cumulative_indices[file_idx - 1] if file_idx > 0 else idx
            
            row = self.metadata.iloc[file_idx]
            
            M = self.safe_load(row.file_path).to(self.dtype)
            
            if self.use_midi:
                H = self.safe_load(row.H_path).to(self.dtype)
                midi = self.safe_load(row.midi_path).to(self.dtype)
            else:
                H = self.safe_load(row.H_path).to(self.dtype)
                
            onset = self.safe_load(row.onset_path).to(self.dtype)
            offset = self.safe_load(row.offset_path).to(self.dtype)
            
            start_idx = audio_idx * self.fixed_length
            end_idx = start_idx + self.fixed_length
            
            # Ensure that the end index does not exceed the length of the data
            if end_idx > M.shape[1]:
                end_idx = M.shape[1]

            M_segment = M[:, start_idx:end_idx]
            
            if self.use_midi:
                midi_segment, _, _ = spec.cut_midi_segment(midi, onset, offset, start_idx, end_idx, self.cut_start_notes)
                H_segment = H[:, start_idx:end_idx]
            else:
                H_segment = H[:, start_idx:end_idx]
            
            # Pad the last segment if necessary
            if M_segment.shape[1] < self.fixed_length:
                M_segment = pad_by_repeating(M_segment, self.fixed_length)
                H_segment = pad_by_repeating(H_segment, self.fixed_length)
                if self.use_midi:
                    midi_segment = pad_by_repeating(midi_segment, self.fixed_length)

            if self.use_midi:
                return M_segment, H_segment, midi_segment
            else:
                return M_segment, H_segment
        
        else:
            row = self.metadata.iloc[idx]
            M = self.safe_load(row.file_path).to(self.dtype)
            
            if self.use_midi:
                H = self.safe_load(row.H_path).to(self.dtype)
                midi = self.safe_load(row.midi_path).to(self.dtype)
            else:
                H = self.safe_load(row.H_path).to(self.dtype)
            
            if self.use_midi:
                return M, H, midi
            else:
                return M, H


"""
Metrics
"""
def compute_metrics(prediction, ground_truth, time_tolerance=100, threshold=0):
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

    return prec, rec, f_mes, accuracy

def extract_note_events(piano_roll, threshold=0):
    """
    Creates a note_event object from a piano_roll tensor
    The note_event is a list of notes (start, end, pitch)
    Needed to compute the metrics using mir_eval
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
   
def evaluate_model(model, file, device=None):
    """
    Computes the accuracy, recall, precision and f_mesure of the model's prediction on the provided file
    """
    y, sr = torchaudio.load(f"test-data/synth-dataset/audios/{file}")
    M, times, _ = spec.cqt_spec(y, sr, normalize_thresh=0.1)
    single_note = 'test-data/synth-single-notes'
    W, _, _, true_freqs = init.init_W(single_note, normalize_thresh=0.1)
    midi, _, _, _ = spec.midi_to_pianoroll(f"test-data/synth-dataset/midis/{file}", y, times,128,sr)

    model.eval()
    with torch.no_grad():
        W_hat, H_hat, M_hat, _ = model.forward(M, device=device)
        M_hat = M_hat.detach()  
    
    _, notes_hat = init.W_to_pitch(W_hat, true_freqs=true_freqs, use_max=True)
    midi_hat, _ = init.WH_to_MIDI(W_hat, H_hat, notes_hat, normalize=False, threshold=0.5, smoothing_window=5)
    
    return compute_metrics(midi, midi_hat) 

def test_model(model, test_loader, device, criterion=None, valid_loader=None, sr=44100):
    """
    Evaluates the passed model on the passed test dataset on the following metrics (if no criterion is passed):
    Precision, Accuracy, Recall, F-mesure, Inference time
    The computation of the metrics is based on the comparison of a ground truth midi and a predicted midi
    => We need to convert the W and H matrix predicted by the model to a midi tensor
    The conversion depends on the choice of a threshold, we test 100 different thresholds and use the one yileding the best f_mes
    """
    
    eps = 1e-6
    threshs = torch.linspace(0.01, 10, 100)
    
    if criterion is not None:
        test_metrics = {"loss": [], "inference_time": []}
    else:
        test_metrics = {"precision": [], "accuracy": [], "recall": [], "f_mesure": [], "inference_time": []}
    model.eval()
    
    print(f"Starting the testing on {len(test_loader)} files")
    
    for i, (M, H, midi) in enumerate(test_loader):
        with torch.no_grad():
            M = M.squeeze(0)
            H = H.squeeze(0)
            midi = midi.squeeze(0)
            M = torch.clamp(M, min=eps)
            M = M.to(device)
            H = H.to(device)
            M = M/torch.max(M)
            midi = midi.to(device)
            
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            start = time.time()
            W_hat, H_hat, M_hat = model(M, device=device)
            stop = time.time()
            inf_time = stop - start
            
            if criterion is not None:
                loss = criterion(H_hat, H)
                test_metrics["loss"].append(loss.item())
                test_metrics["inference_time"].append(inf_time)
            else:
                # Tracking gpu usage
                gpu_info = get_gpu_info()
                log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
                try:
                    _, notes_hat = init.W_to_pitch(W_hat, true_freqs=None, use_max=True)
                    best_f = 0
                    best_thresh = threshs[0]
                    
                    for thresh in threshs:
                        midi_hat, _ = init.WH_to_MIDI(W_hat, H_hat, notes_hat, normalize=False, threshold=thresh, smoothing_window=10, min_note_length=30, sr=sr)
                        prec, rec, f_mes, accuracy = compute_metrics(midi, midi_hat, time_tolerance=200)
                        if f_mes > best_f:
                            best_f = f_mes
                            best_thresh = thresh
                            
                    midi_hat, _ = init.WH_to_MIDI(W_hat, H_hat, notes_hat, normalize=False, threshold=best_thresh, smoothing_window=10, min_note_length=30, sr=sr)
                    prec, rec, f_mes, accuracy = compute_metrics(midi, midi_hat, time_tolerance=200)
                    
                    test_metrics["precision"].append(prec) 
                    test_metrics["recall"].append(rec)
                    test_metrics["f_mesure"].append(f_mes)
                    test_metrics["accuracy"].append(accuracy)
                    test_metrics["inference_time"].append(inf_time)
                except Exception as e:
                    print(f"Skipping file {i} due to error: {e}")
                    continue
                
                if i%(len(test_loader)%100)==0:
                    print(f"Tested {i} files...")
                    print(f'current metrics: prec={np.mean(test_metrics["precision"])}, acc={np.mean(test_metrics["accuracy"])}, rec={np.mean(test_metrics["recall"])}, f={np.mean(test_metrics["f_mesure"])}, inf={np.mean(test_metrics["inference_time"])}')
        
    if valid_loader is not None:
        if criterion is not None:
            valid_metrics = {"los": [], "inference_time": []}
        else:
            valid_metrics = {"precision": [], "accuracy": [], "recall": [], "f_mesure": [], "inference_time": []}
        
        for i, (M, H, midi) in enumerate(valid_loader):
            with torch.no_grad():
                M = M.squeeze(0)
                H = H.squeeze(0)
                midi = midi.squeeze(0)
                M = torch.clamp(M, min=eps)
                M = M/ torch.max(M)
                M = M.to(device)
                H = H.to(device)
                midi = midi.to(device)
                
                gpu_info = get_gpu_info()
                log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
                start = time.time()
                W_hat, H_hat, M_hat = model(M, device=device)
                stop = time.time()
                inf_time = stop - start
            
                if criterion is not None:
                    loss = criterion(H_hat, H)
                    valid_metrics["loss"].append(loss.item())
                    valid_metrics["inference_time"].append(inf_time)
                else:
                    gpu_info = get_gpu_info()
                    log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                            
                    try:
                        _, notes_hat = init.W_to_pitch(W_hat.cpu(), true_freqs=None, use_max=True)
                        best_f = 0
                        best_thresh = threshs[0]
                        
                        for thresh in threshs:
                            midi_hat, _ = init.WH_to_MIDI(W_hat, H_hat, notes_hat, normalize=False, threshold=thresh, smoothing_window=10, min_note_length=30, sr=44100)
                            prec, rec, f_mes, accuracy = compute_metrics(midi, midi_hat, time_tolerance=200)
                            if f_mes > best_f:
                                best_f = f_mes
                                best_thresh = thresh
                        midi_hat, _ = init.WH_to_MIDI(W_hat, H_hat, notes_hat, normalize=False, threshold=best_thresh, smoothing_window=10, min_note_length=30, sr=44100)
                        prec, rec, f_mes, accuracy = compute_metrics(midi, midi_hat, time_tolerance=200)
                        
                        valid_metrics["precision"].append(prec) 
                        valid_metrics["recall"].append(rec)
                        valid_metrics["f_mesure"].append(f_mes)
                        valid_metrics["accuracy"].append(accuracy)
                        valid_metrics["inference_time"].append(inf_time)
                    except Exception as e:
                        print(f"Skipping file {i} due to error: {e}")
                        continue            
                    
                    if i%(len(test_loader)%100)==0:
                        print(f"Validated {i} files...")
                        print(f'current metrics: prec={np.mean(valid_metrics["precision"])}, acc={np.mean(valid_metrics["accuracy"])}, rec={np.mean(valid_metrics["recall"])}, f={np.mean(valid_metrics["f_mesure"])}, inf={np.mean(valid_metrics["inference_time"])}')
                        
        return test_metrics, valid_metrics
    else:
        return test_metrics
   
"""
Train the network
"""
def permutation_match(W_new, W_init, rows=False):
    """
    /!\ Not used
    
    Use the linear_sum_assignment permutation algorithm to find the best column permutation to match W_new and W_init
    """
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
    """
    /!\ Not used
    
    A differentiable version of the previous permutation algorithm to find the best column permutation to match W_new and W_init
    """
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
    """
    Computes the spectral flatness score
    """
    geometric_mean = torch.exp(torch.mean(torch.log(log_val + spec), dim=0))
    arithmetic_mean = torch.mean(spec, dim=0)
    return geometric_mean / arithmetic_mean

def warmup_loss(X, X_hat):
    """
    Loss for the warmup training: mse(X, X_hat) + |mean(X)-1|
    """
    mse = nn.MSELoss()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    # X_hat_flat = X_hat.reshape(X_hat.shape[0], -1)
    # X_flat = (1 - X).reshape(X.shape[0], -1)
    
    loss_mse = mse(X, torch.ones(X.shape))
    loss_mean = torch.norm(X_hat.mean() - 1)
    
    return loss_mse + loss_mean

def warmup_train(model, n_epochs, loader, optimizer, device, train_layers=False, save_name="warmup", load_check=None, use_wandb=False):
    """
    Setup to warmup train the model on a train set
    The training loss is a sum of the warmup_loss computed between the predicted W and the init W, and a MSE between the predicted H and the init H
    
    Args:
        model: model to train
        n_epochs: number of training epochs
        loader: the training loader containing the training data
        optimizer: the optimizer to use
        device: the device on which to compute
        train_layers (bool, optional): whether to compute the loss on the output of every layer or just the final one (default: False)
        save_name (str, optional): the name with which to save the trained model (default: "warmup")
        load_check (str, optional): the path of the checkpoint model from which to restart the training (default: None)
        use_wandb (bool, optional): whether to upload the training data to wandb or not (default: False)
    """
    if use_wandb:
        run = wandb.init(
            project=f"{model.__class__.__name__}_warmup_train",
            config={
                "learning_rate": optimizer.param_groups[-1]['lr'],
                "batch_size": loader.batch_sampler.batch_size,
                "epochs": n_epochs,
            },
        )
    
    if load_check:
        start_epoch = load_checkpoint(load_check, model, optimizer)
    else:
        start_epoch = 0
    
    mse = nn.MSELoss()
    losses = []
    for i in range(start_epoch, n_epochs):
        train_loss = 0
        for idx, (M, H) in enumerate(loader):
            M = M.to(device)
            H = H.to(device)
            M = torch.clamp(M, min=1e-6)
            M = M/ torch.max(M)
            
            if M.std() == 0:
                continue
            
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            W_hat, H_hat, M_hat, W0, H0, accel_W, accel_H = model(M, device)
        
            if torch.sum(torch.nonzero(torch.cat([torch.isnan(W_hat[0].view(-1))], 0))) != 0:
                spec.vis_cqt_spectrogram(W_hat[0].detach().cpu(), title=f"W_hat id: {idx}", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(H_hat[0].detach().cpu(), title=f"H_hat id: {idx}", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(W0[0].detach().cpu(), title=f"W0 id: {idx}", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(H0[0].detach().cpu(), title=f"H0 id: {idx}", use_wandb=use_wandb)
            
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            mse = nn.MSELoss()
            if train_layers:
                loss = 0
                for k in range(len(W_hat)):
                    loss += warmup_loss(W0, accel_W[k]).mean() + mse(accel_H[k], torch.ones(accel_H[k].shape))
            else:
                loss = warmup_loss(W0, accel_W).mean() + mse(accel_H, torch.ones(accel_H.shape))
                # loss = torch.norm(accel_W - torch.ones(accel_W.shape)) + torch.norm(accel_H - torch.ones(accel_H.shape))
                # loss = torch.norm(W0 - W_hat)/ torch.norm(W0) + torch.norm(H0 - H_hat)/ torch.norm(H0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(loader)
        losses.append(train_loss)
            
        if i%5 == 0:
            spec.vis_cqt_spectrogram(W0[0].detach().cpu(), title="init W", use_wandb=use_wandb)
            spec.vis_cqt_spectrogram(H0[0].detach().cpu(), title="init H", use_wandb=use_wandb)
            spec.vis_cqt_spectrogram(W_hat[-1][0].detach().cpu(), title="recreated W", use_wandb=use_wandb)
            spec.vis_cqt_spectrogram(H_hat[-1][0].detach().cpu(), title="recreated H", use_wandb=use_wandb)
        
        if i%5 == 0 and i > 0:
            save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", epoch=i, is_permanent=True, name=save_name)
        
        if use_wandb:
            wandb.log({"loss": train_loss})
        
        print(f"epoch {i}, loss = {train_loss:3f}")
        save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", epoch=i)
    
    save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", epoch=i, is_permanent=True, name=save_name)
            
    return losses, W_hat, H_hat
 
def train(model, train_loader, valid_loader, optimizer, criterion, device, epochs, W0=None, save_name=None, load_check=None, use_wandb=False):
    """
    Setup to train the model on a train set
    We also compute the f_mesure based on the comparison of a ground truth midi and a predicted midi
    => We need to convert the W and H matrix predicted by the model to a midi tensor
    The conversion depends on the choice of a threshold, we test 100 different thresholds and use the one yileding the best f_mes    
    
    Args:
        model: model to train
        train_loader: the training loader containing the training data
        valid_loader: the validation loader containing the validation data
        optimizer: the optimizer to use
        criterion: the criterion to be optimized (loss function)
        device: the device on which to compute
        epochs: number of training epochs
        W0: Whether to use the init W to normalize the W part of the loss or not (default: None)
        save_name (str, optional): the name with which to save the trained model (default: "warmup")
        load_check (str, optional): the path of the checkpoint model from which to restart the training (default: None)
        use_wandb (bool, optional): whether to upload the training data to wandb or not (default: False)
    """
    eps = 1e-6
    if use_wandb:
        run = wandb.init(
            project=f"{model.__class__.__name__}_train",
            config={
                "learning_rate": optimizer.param_groups[-1]['lr'],
                "batch_size": train_loader.batch_sampler.batch_size,
                "epochs": epochs,
            },
        )
    
    train_losses, valid_losses, train_fs, valid_fs = [], [], [], []
    valid_loss_min = np.inf
    threshs = torch.linspace(0.01, 10, 100)
    sr = 44100
    
    if load_check is not None: 
        start_epoch = load_checkpoint(load_check, model, optimizer)
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        
        train_loss, valid_loss, train_f, valid_f = 0, 0, 0, 0
        
        model.train()
        for n_item, (M, H, midi) in enumerate(train_loader):
            
            M = torch.clamp(M, min=eps)
            M = M.to(device)
            H = H.to(device)
            midi = midi.to(device)
            
            M = M/ torch.max(M)
            H = H/ torch.max(H)
            
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            W_hat, H_hat, M_hat = model(M, device=device)
        
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            optimizer.zero_grad()
            
            if W0 is not None:
                loss_H = criterion(H_hat, H)/torch.linalg.norm(H)
                loss_W = torch.linalg.norm(W_hat)/ torch.linalg.norm(W0)
                loss = loss_H + loss_W 
            else:
                loss = criterion(H_hat, H)
                
            try:
                notes_hats = [init.W_to_pitch(W_hat[i], true_freqs=None, use_max=True)[1] for i in range(W_hat.shape[0])]
                best_f = 0
                best_thresh = threshs[0]
                
                for thresh in threshs:
                    midi_hats = [init.WH_to_MIDI(W_hat[i], H_hat[i], notes_hats[i], normalize=False, threshold=thresh, smoothing_window=10, min_note_length=30, sr=sr)[0] for i in range(W_hat.shape[0])]
                    f_mess = [compute_metrics(midi[i], midi_hats[i], time_tolerance=200)[2] for i in range(W_hat.shape[0])]
                    f_mes = np.mean(f_mess)
                    if f_mes > best_f:
                        best_f = f_mes
                        best_thresh = thresh
                
                midi_hats = [init.WH_to_MIDI(W_hat[i], H_hat[i], notes_hats[i], normalize=False, threshold=best_thresh, smoothing_window=10, min_note_length=30, sr=sr)[0] for i in range(W_hat.shape[0])]
                f_mess = [compute_metrics(midi[i], midi_hats[i], time_tolerance=200)[2] for i in range(W_hat.shape[0])]
                f_mes = np.mean(f_mess)
            except Exception as e:
                print(f"Skipping file {n_item} due to error: {e}")
            
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            loss.backward()

            if sum([torch.isnan(param.grad).sum().item() for param in model.parameters() if param.grad is not None]) != 0:
                print(f"Bacward grads nans: {torch.sum(torch.nonzero(torch.cat([torch.isnan(param.grad.view(-1)) if param.grad is not None else torch.nan for param in model.parameters()], 0)))}")
            
            optimizer.step()
            
            if torch.sum(torch.cat([torch.nonzero(torch.isnan(param.data).view(-1)) for param in model.parameters()], 0)) != 0:
                print(f"Step param nans: {torch.sum(torch.cat([torch.nonzero(torch.isnan(param.data).view(-1)) for param in model.parameters()], 0))}")
            
            train_loss += loss.item()
            train_f += f_mes
        
        if epoch % 5 ==0: # Display the evolution of learned NMF
            if use_wandb:
                spec.vis_cqt_spectrogram(H[0].detach().cpu(), title="original H", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(H_hat[0].detach().cpu(), title="recreated H", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(W_hat[0].detach().cpu(), title="recreated W", use_wandb=use_wandb)
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        train_f /= len(train_loader)
        train_fs.append(train_f)
        print(f"epoch {epoch}, loss = {train_loss:5f}, f_mes = {train_f:5f}")
        
        if use_wandb:
            wandb.log({"training loss": train_loss, "train f": train_f})
    
        model.eval()
        for n_item, (M, H, midi) in enumerate(valid_loader):
            
            with torch.no_grad():
                M = torch.clamp(M, min=eps)
                M = M.to(device)
                H = H.to(device)
                midi = midi.to(device)
                
                M = M/ torch.max(M)
                H = H/ torch.max(H)
                
                # Tracking gpu usage
                gpu_info = get_gpu_info()
                log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
                W_hat, H_hat, M_hat = model(M, device=device)
                
                # Tracking gpu usage
                gpu_info = get_gpu_info()
                log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
                if W0 is not None:
                    loss_H = criterion(H_hat, H)/torch.linalg.norm(H)
                    loss_W = torch.linalg.norm(W_hat)/ torch.linalg.norm(W0)
                    loss = loss_H + loss_W 
                else:
                    loss = criterion(H_hat, H)
                
                try:
                    notes_hats = [init.W_to_pitch(W_hat[i], true_freqs=None, use_max=True)[1] for i in range(W_hat.shape[0])]
                    best_f = 0
                    best_thresh = threshs[0]
                    
                    for thresh in threshs:
                        midi_hats = [init.WH_to_MIDI(W_hat[i], H_hat[i], notes_hats[i], normalize=False, threshold=thresh, smoothing_window=10, min_note_length=30, sr=sr)[0] for i in range(W_hat.shape[0])]
                        f_mess = [compute_metrics(midi[i], midi_hats[i], time_tolerance=200)[2] for i in range(W_hat.shape[0])]
                        f_mes = np.mean(f_mess)
                        if f_mes > best_f:
                            best_f = f_mes
                            best_thresh = thresh
                            
                    midi_hats = [init.WH_to_MIDI(W_hat[i], H_hat[i], notes_hats[i], normalize=False, threshold=best_thresh, smoothing_window=10, min_note_length=30, sr=sr)[0] for i in range(W_hat.shape[0])]
                    f_mess = [compute_metrics(midi[i], midi_hats[i], time_tolerance=200)[2] for i in range(W_hat.shape[0])]
                    f_mes = np.mean(f_mess)
                    
                except Exception as e:
                    print(f"Skipping file {n_item} due to error: {e}")
                    # continue
                
                valid_loss += loss.item()
                valid_f += f_mes
        
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)
        
        valid_f /= len(valid_loader)
        valid_fs.append(valid_f)

        if use_wandb:
            wandb.log({"valid loss": valid_loss, "valid f": valid_f})
            
        save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", epoch=epoch)
            
        if valid_loss <= valid_loss_min:
            print('validation loss decreased ({:.6f} --> {:.6f})'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            if save_name is not None:
                # save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", epoch=epoch, is_permanent=True, name=save_name)
                save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", is_permanent=True, name=save_name)
            else:
                # save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", epoch=epoch, is_permanent=True, name="permanent_model")
                save_model(model, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", is_permanent=True, name="permanent_model")
          
    return train_losses, valid_losses, W_hat, H_hat
    
def midi_train(model, loader, optimizer, criterion, device, epochs):
    """
    Trains the model by computing the loss (criterion) between the predicted midi and the ground truth
    The predicted midi is obtained with a fixed threshold to do the conversion from WH to MIDI
    """
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
    
def transribe(model, M, device, threshold=0.05):
    """
    Computes the midi tensor of the input spectrogram M predicted by the model with the given threshold
    """
    model.eval()
    model.to(device=device)

    with torch.no_grad():
        W_hat, H_hat, M_hat, norm = model(M)
        freqs = librosa.cqt_frequencies(n_bins=288, fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
        _, notes = init.W_to_pitch(W_hat, freqs, use_max=True)
        H_hat = H_hat * norm
        midi_hat, active_midi = init.WH_to_MIDI(W_hat, H_hat, notes, threshold=threshold)

    return W_hat, H_hat, M_hat, midi_hat, active_midi
