import torch
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
import subprocess, csv
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
    
def save_model(model, epoch, optimizer, directory, is_permanent=False, name='model_epoch'):
    """
    Overwrite the previous model save if not is_permanent, otherwise, saves a new version of the model
    """
    if is_permanent:
        # Save a permanent copy of the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # Add any other information you want to save
        }, os.path.join(directory, f'{name}_{epoch}.pt'))
    else:
        # Overwrite the temporary model save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(directory, 'checkpoint.pt'))
        
def load_checkpoint(path, model, optimizer):
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
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for info in gpu_info:
            if len(info) == 3:  # Ensure name, power draw and memory are captured
                writer.writerow([timestamp, info[0], info[1], info[2]])

def compute_emissions(file_path, emission_per_kwh=10, start=None):
    """
    Computes the CO2e emissions of the energy consumed from a csv file containing the data of power usage and time
    By default we use EDF's value of 10 CO2e/ kWh in France in 2024
    """
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    
    if start is not None:
        df = df[df['time'] >= pd.to_datetime(start)]

    # Compute the time difference between consecutive measures
    time_diffs = df['time'].diff().dt.total_seconds().fillna(0).values
    time_diffs = torch.tensor(time_diffs, dtype=torch.float32) / 3600

    # Detect larger differences to distinguish runs
    time_diff_diffs = torch.abs(time_diffs[1:] - time_diffs[:-1])
    time_diff_diffs = torch.cat([torch.tensor([0.0]), time_diff_diffs])
    threshold = torch.quantile(time_diff_diffs, 0.95)  # Use a 95% quantile threshold
    launch_ends = time_diff_diffs > threshold

    # Compute the consumed energy
    power_usage = torch.tensor(df['power_usage'].values, dtype=torch.float32)
    energy = power_usage * time_diffs
    energy[launch_ends] = 0  # Don't count launching energy
    total_energy_kwh = torch.sum(energy) / 1000  # in kWh

    # Total CO2e emissions in kg of CO2e
    total_emissions = total_energy_kwh * emission_per_kwh / 1000

    return total_energy_kwh, total_emissions.item()


"""
Dataset class and utils functions
"""     
class SequentialBatchSampler(Sampler):
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
        return math.ceil(len(self.data_source) / self.batch_size)#len(self.data_source) // self.batch_size
    
def pad_by_repeating(tensor, max_length):
    current_length = tensor.size(1)
    if current_length == 0:
        raise ValueError("Tensor length is zero, cannot pad.")
    
    if tensor.size(1) < max_length:
        repeat_times = (max_length + current_length - 1) // current_length
        repeated_tensor = tensor.repeat(1, repeat_times)[:, :max_length]
        return repeated_tensor
    return tensor
    
def collate_fn(batch):
    # Separate audio and MIDI files
    audio_files = [item[0] for item in batch if item[0].size(1) > 0]
    midi_files = [item[1] for item in batch if item[1].size(1) > 0]
    
    if not audio_files or not midi_files:
        return torch.tensor([]), torch.tensor([])
    
    max_audio_length = max(audio.size(1) for audio in audio_files)
    max_midi_length = max(midi.size(1) for midi in midi_files)

    # Pad audio and MIDI files by repeating their data
    audio_files_padded = torch.stack([pad_by_repeating(audio, max_audio_length) for audio in audio_files])
    midi_files_padded = torch.stack([pad_by_repeating(midi, max_midi_length) for midi in midi_files])

    # Return them as lists to avoid stacking
    return audio_files_padded, midi_files_padded 

class LocalDataset(Dataset):
    """
    Creates a Dataset object from localy saved CQT + midi

    Args:
        folder_path (pd.DataFrame): DataFrame containing 'file_path', 'midi_path', 'onset_path' and 'offset_path' columns.
        dev: The device on which to load the dataset (default: None)
        sr (int, optional): The sample rate of the recordings to convert the fixed_length in cqt steps (default: 22050)
        hop_length (int, optional): The hop_length to convert the fixed_length in cqt steps (default: 128)
        fixed_length (bool, optional): Whether to have a constant duration for audio/MIDI items (default: True).
        subset (float, optional): The subset of files to include in the dataset. If None, all files are used (default: None).
    """
    def __init__(self, metadata, dev=None, sr=22050, hop_length=128, fixed_length=None, subset=None, dtype=None, verbose=False):

        self.metadata    = metadata
        self.dev            = dev if dev is not None else torch.device('cpu')
        self.sr             = sr
        self.hop_length     = hop_length
        self.fixed_length   = fixed_length
        self.dtype          = dtype
        
        if subset is not None:
            num_files = int(len(self.metadata) * subset)
            self.metadata = self.metadata[:num_files]

        if self.fixed_length is not None:
            self.metadata.loc[:,'segment_indices'] = self.compute_length()
            
    def compute_length(self):
        self.fixed_length = math.ceil((self.fixed_length * self.sr) / self.hop_length)
        segment_indices = []
        for _, row in tqdm(self.metadata.iterrows()):
            M = torch.load(row.file_path, map_location=self.dev)
            segment_indices.append(-(M.shape[1] // -self.fixed_length))
        return segment_indices
    
    def __len__(self):
        if self.fixed_length is not None:
            return sum(self.metadata.segment_indices)
        else:
            return len(self.metadata)
    
    def safe_load(self, path):
        with open(path, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = torch.load(f, map_location=self.dev)
            fcntl.flock(f, fcntl.LOCK_UN)
        return data
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        
        if self.fixed_length is not None:
            cumulative_indices = np.cumsum(self.metadata.segment_indices)
            file_idx = np.searchsorted(cumulative_indices, idx, side='right')
            audio_idx = idx - cumulative_indices[file_idx - 1] if file_idx > 0 else idx
            
            row = self.metadata.iloc[file_idx]
            
            M = self.safe_load(row.file_path).to(self.dtype)
            H = self.safe_load(row.midi_path).to(self.dtype)
            onset = self.safe_load(row.onset_path).to(self.dtype)
            offset = self.safe_load(row.offset_path).to(self.dtype)
            
            start_idx = audio_idx * self.fixed_length
            end_idx = start_idx + self.fixed_length
            
            # Ensure that the end index does not exceed the length of the data
            if end_idx > M.shape[1]:
                end_idx = M.shape[1]

            M_segment = M[:, start_idx:end_idx]
            H_segment, _, _ = spec.cut_midi_segment(H, onset, offset, start_idx, end_idx)
            
            # Pad the last segment if necessary
            if M_segment.shape[1] < self.fixed_length:
                M_segment = pad_by_repeating(M_segment, self.fixed_length)
                H_segment = pad_by_repeating(H_segment, self.fixed_length)

            return M_segment, H_segment
        
        else:
            row = self.metadata.iloc[idx]
            M = self.safe_load(row.file_path).to(self.dtype)
            H = self.safe_load(row.midi_path).to(self.dtype)
            return M, H


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

    return prec, rec, f_mes, accuracy

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
   
def evaluate_model(model, file, device=None):
    y, sr = torchaudio.load(f"synth-dataset/audios/{file}")
    M, times, _ = spec.cqt_spec(y, sr, normalize_thresh=0.1)
    single_note = 'test-data/synth-single-notes'
    W, _, _, true_freqs = init.init_W(single_note, normalize_thresh=0.1)
    midi, _, _, _ = spec.midi_to_pianoroll(f"synth-dataset/midis/{file}", y, times,128,sr)

    model.eval()
    with torch.no_grad():
        W_hat, H_hat, M_hat, _ = model.forward(M, device=device)
        M_hat = M_hat.detach()  
    
    _, notes_hat = init.W_to_pitch(W_hat, true_freqs=true_freqs, use_max=True)
    midi_hat, _ = init.WH_to_MIDI(W_hat, H_hat, notes_hat, normalize=False, threshold=0.5, smoothing_window=5)
    
    return compute_metrics(midi, midi_hat) 
   
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
 
def train(model, train_loader, valid_loader, optimizer, criterion, device, epochs, W0=None, use_wandb=False):
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
    
    train_losses, valid_losses = [], []
    valid_loss_min = np.inf
    
    start_epoch = load_checkpoint("home/ids/edabier/AMT/Unrolled-NMF/models/checkpoint.pt", model, optimizer)
    
    for epoch in range(start_epoch, epochs):
        
        train_loss, valid_loss = 0, 0
        
        model.train()
        for n_item, (M, H) in enumerate(train_loader):
            
            M = torch.clamp(M, min=eps)
            M = M.to(device)
            H = H.to(device)
            
            H, norm = spec.l1_norm(H, threshold=0.01, set_min=eps)
            
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            W_hat, H_hat, M_hat, _ = model(M, device=device)
        
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            optimizer.zero_grad()
            
            if W0 is not None:
                loss_H = criterion(H_hat, H)/torch.linalg.norm(H)
                loss_W = torch.linalg.norm(W_hat)/ torch.linalg.norm(W0)
                loss = loss_H + loss_W 
            else:
                loss = criterion(H_hat, H)/torch.linalg.norm(H)
                
            if n_item % 100 == 0 and use_wandb:
                wandb.log({f"loss_H_{epoch}": loss_H, f"loss_W_{epoch}": loss_W})
            
            # Tracking gpu usage
            gpu_info = get_gpu_info()
            log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
            loss.backward()
            
            if torch.sum(torch.nonzero(torch.cat([torch.isnan(param.grad.view(-1)) if param.grad is not None else torch.nan for param in model.parameters()], 0))) != 0:
                print(f"Bacward grads nans: {torch.sum(torch.nonzero(torch.cat([torch.isnan(param.grad.view(-1)) if param.grad is not None else torch.nan for param in model.parameters()], 0)))}")
            
            optimizer.step()
            
            if torch.sum(torch.cat([torch.nonzero(torch.isnan(param.data).view(-1)) for param in model.parameters()], 0)) != 0:
                print(f"Step param nans: {torch.sum(torch.cat([torch.nonzero(torch.isnan(param.data).view(-1)) for param in model.parameters()], 0))}")
            
            train_loss += loss.item()
        
        if epoch % 5 ==0: # Display the evolution of learned NMF
            if use_wandb:
                spec.vis_cqt_spectrogram(M[0].detach().cpu(), title="original audio", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(H[0].detach().cpu(), title="original H", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(M_hat[0].detach().cpu(), title="recreated audio", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(W_hat[0].detach().cpu(), title="recreated W", use_wandb=use_wandb)
                spec.vis_cqt_spectrogram(H_hat[0].detach().cpu(), title="recreated H", use_wandb=use_wandb)
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        print(f"epoch {epoch}, loss = {train_loss:5f}")
        
        if use_wandb:
            wandb.log({"training loss": train_loss})
    
        model.eval()
        for n_item, (M, H) in enumerate(valid_loader):
            
            with torch.no_grad():
            
                M = torch.clamp(M, min=eps)
                M = M.to(device)
                H = H.to(device)
                
                # Tracking gpu usage
                gpu_info = get_gpu_info()
                log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
                W_hat, H_hat, _, _ = model(M, device=device)
                
                # Tracking gpu usage
                gpu_info = get_gpu_info()
                log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
                if W0 is not None:
                    loss_H = criterion(H_hat, H)/torch.linalg.norm(H)
                    loss_W = torch.linalg.norm(W_hat)/ torch.linalg.norm(W0)
                    loss = loss_H + loss_W 
                else:
                    loss = criterion(H_hat, H)/torch.linalg.norm(H)
                    
                valid_loss += loss.item()
                
            if n_item % 100 == 0 and use_wandb:
                wandb.log({f"valid_loss_H_{epoch}": loss_H, f"valid_loss_W_{epoch}": loss_W})
        
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        if use_wandb:
            wandb.log({"valid loss": valid_loss})
            
        save_model(model, epoch, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models")
        
        if epoch % 5 == 0: # Save the model every 5 epochs
            save_model(model, epoch, optimizer, directory="/home/ids/edabier/AMT/Unrolled-NMF/models", is_permanent=True, name="full_dataset")
            
        if valid_loss <= valid_loss_min:
          print('validation loss decreased ({:.6f} --> {:.6f})'.format(
          valid_loss_min,
          valid_loss))
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

    with torch.no_grad():
        W_hat, H_hat, M_hat, norm = model(M)
        freqs = librosa.cqt_frequencies(n_bins=288, fmin=librosa.note_to_hz('A0'), bins_per_octave=36)
        _, notes = init.W_to_pitch(W_hat, freqs, use_max=True)
        H_hat = H_hat * norm
        midi_hat, active_midi = init.WH_to_MIDI(W_hat, H_hat, notes, threshold=0.05)

    return W_hat, H_hat, M_hat, midi_hat, active_midi
