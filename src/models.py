import torch.nn as nn
import torch
import matplotlib.pyplot as plt

import os, sys
repo_path = os.path.abspath("/home/ids/edabier/AMT/onsets-and-frames")
sys.path.append(repo_path)
from onsets_and_frames import OnsetsAndFrames  # from jongwook's repo

import src.utils as utils
import src.spectrograms as spec
import src.init as init

class MU_NMF(nn.Module):
    """
    Defines the NMF solving using MU
    
    Args:
        n_iter (int): the amount of iterations of MU to do
        W_path (str, optional): the path to the folder containing the single notes recordings to initialize W, if not set, initialize syntheticly (default: None)
        beta (int, optional): the value of β for the β-divergence (default: 1)
        norm_thresh (float, optional): the normalizing threshold to initialize W (default: None)
        dtype: the data type of W, H
    """
    
    def __init__(self, n_iter, W_path=None, beta=1, norm_thresh=0.01, dtype=None):
        super().__init__()
        
        self.n_iter = n_iter
        self.W_path = W_path
        self.beta = beta
        self.norm_thresh = norm_thresh
        self.dtype = dtype
        if dtype is not None:
            self.eps = torch.finfo(type=dtype).min
        else:
            self.eps = 1e-6
    
    def forward(self, M, device=None):
        
        batch_size = None
        _, t = M.shape
        
        W, _, _, _ = init.init_W(self.W_path, normalize_thresh=self.norm_thresh, dtype=self.dtype)
        
        # Tracking gpu usage
        if device == "cuda:0":
            gpu_info = utils.get_gpu_info()
            utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            
        H = init.init_H(W, M, n_init_steps=self.n_iter, beta=self.beta, device=device, dtype=self.dtype)
        
        H = torch.clamp(H, min=self.eps)
            
        # Tracking gpu usage
        if device == "cuda:0":
            gpu_info = utils.get_gpu_info()
            utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        
        for _ in range(self.n_iter):
            # Compute WH
            Wh = W @ H
            Wh = torch.clamp(Wh, min=self.eps)

            Wh_beta_minus_2 = Wh ** (self.beta - 2)
            Wh_beta_minus_1 = Wh ** (self.beta - 1)

            # Update W
            numerator_W = (Wh_beta_minus_2 * M) @ H.T
            denominator_W = Wh_beta_minus_1 @ H.T
            denominator_W = torch.clamp(denominator_W, min=self.eps)

            W = W * (numerator_W / denominator_W)

            # Compute WH again for updating H
            Wh = W @ H
            Wh = torch.clamp(Wh, min=self.eps)

            # Update H
            numerator_H = W.T @ (Wh_beta_minus_2 * M)
            denominator_H = W.T @ Wh_beta_minus_1
            denominator_H = torch.clamp(denominator_H, min=self.eps)

            H = H * (numerator_H / denominator_H)

        M_hat = W @ H
        return W, H, M_hat


class OnsetAndFramesWrapper(nn.Module):
    """
    Placeholder class to implement the prediction of the O&F model in the same format as the other models (MU and RALMU)
    
    Remains to be done...
    """
    def __init__(self, checkpoint_path, device="cpu"):
        super().__init__()
        # Load the pretrained model
        torch.serialization.add_safe_globals([OnsetsAndFrames])
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model = OnsetsAndFrames(
            input_features=checkpoint['input_features'],
            output_features=checkpoint['output_features'],
            model_complexity=checkpoint.get('model_complexity', 1)
        )
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.device = device
        self.model.to(device)

    def forward(self, spectrogram):
        """
        spectrogram: torch.Tensor of shape (batch, freq_bins, time_steps)
                     Should be same preprocessing as the pretrained model expects.
        """
        with torch.no_grad():
            spectrogram = spectrogram.to(self.device)
            onset_pred, frame_pred, velocity_pred = self.model(spectrogram)
        return onset_pred, frame_pred, velocity_pred


class Aw_cnn(nn.Module):
    """
    Defines a 1D CNN (frequency axis) for Aw()
    Input is a single column of W (shape (288,1))
    
    Args:
        in_channels (int, optional): the size of the input, a single column by default (default: 1)
        hidden_channels (int, optional): the size of the hidden part of the model (default: 32)
    """
    def __init__(self, in_channels=1, hidden_channels=32, dtype=None):
        super(Aw_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2, dtype=dtype)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2, dtype=dtype)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus()

    # W shape: (f,l)
    def forward(self, x):
        if len(x.shape) == 2:
            batch_size, f = x.shape
        else:
            batch_size = 1
            f = x.shape[0]
        x = x.reshape(batch_size, 1, f)   # (batch, 1, f)
        y = self.relu(self.conv1(x))      # (batch, 64, f)
        y = self.relu(self.conv2(y))      # (batch, 32, f)
        y = self.softplus(self.conv3(y))  # (batch, 1, f)
        out = y.reshape(batch_size, f, 1)
        return out
    

class Aw_2d_cnn(nn.Module):
    """
    Defining a 2D CNN for Aw
    Input is 4 or 12 columns of W (shape (288,4) or (288,12))
    
    Args:
        in_channels (int, optional): the size of the input, 4 columns by default (default: 4)
        hidden_channels (int, optional): the size of the hidden part of the model (default: 2)
    """
    def __init__(self, in_channels=4, hidden_channels=2, dtype=None):
        super(Aw_2d_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels*2, kernel_size=(24*5,5), padding="same", dtype=dtype)
        self.conv2 = nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=(24*3,3), padding="same", dtype=dtype)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=(24*3,3), padding="same", dtype=dtype)
        self.relu  = nn.LeakyReLU()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        if (len(x.shape) == 3):
            batch_size, f, l = x.shape
        else:
            f, l = x.shape
            batch_size = 1
        x = x.reshape(batch_size, 1, f, l)
        y = self.relu(self.conv1(x))      # (batch, 64, f*l)
        y = self.relu(self.conv2(y))      # (batch, 32, f*l)
        y = self.softplus(self.conv3(y))  # (batch, 1, f*l)
        out = y.reshape(batch_size, f, l)
        return out

    
class Ah_cnn(nn.Module):
    """
    Defines a 1D CNN (time axis) for Ah()
    Input is a single row of H (shape (1,t))
    
    Args:
        in_channels (int, optional): the size of the input, a single row by default (default: 1)
        hidden_channels (int, optional): the size of the hidden part of the model (default: 32)
    """
    def __init__(self, in_channels=1, hidden_channels=32, dtype=None):
        super(Ah_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2, dtype=dtype)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2, dtype=dtype)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus()

    # H shape: (l,t)
    def forward(self, x):
        if (len(x.shape) == 3):
            batch_size, _, t = x.shape
        else:
            _, t = x.shape
            batch_size = 1
        x = x.reshape(batch_size, 1, t)
        y = self.relu(self.conv1(x))            # (batch, 64, t)
        y = self.relu(self.conv2(y))            # (batch, 32, t)
        y = self.relu(self.conv3(y))            # (batch, 1, t)
        out = self.softplus(y)                  # (batch, 1, t)
        out = out.view(batch_size, 1, t)        # (batch, 1, t)
        return out   
      
        
class RALMU_block(nn.Module):
    """
    A single layer/iteration of the unrolled MU algorithm with β-divergence to update W and H.
    Aw and Ah are CNNS
        
    Args:
        iter (int, optional): the iteration of MU to which the layers corresponds. We update W only every 5 iteration if this is set. (default: ``None``)
        beta (int, optional): value for the β-divergence (default: ``1`` = KL divergence)
        learnable_beta (bool, optional): whether the beta parameter should be learned or not (default: ``False``)
        hidden_channels (int, optional): controls the size of the CNNs (default: ``16``)
        shared_aw (Aw_cnn, optional): whether to use a predefined Aw acceleration model or to create one (default: ``None``)
        shared_aw (Ah_cnn, optional): whether to use a predefined Ah acceleration model or to create one (default: ``None``)
        use_ah (bool, optional): whether to use Ah in the acceleration of MU (default: ``True``)
        aw_2d (bool, optional): whether to use a 2D CNN for Aw (default: ``False``)
        clip_H (bool, optional): whether to clip the value of H after the update or not (default: ``False``)
        lambda_w (float, optional): the value of the lambda factor to add at the denominator of the update of W (default: ``None``)
        lambda_h (float, optional): the value of the lambda factor to add at the denominator of the update of H (default: ``None``)
        warmup (bool, optional): whether we warmup train (return the values of the CNNs acceleration) or not (default: ``False``)
    """
    def __init__(self, iter=None, beta=1, learnable_beta=False, hidden_channels=16, shared_aw=None, shared_ah=None, use_ah=True, aw_2d=False, clip_H=False, lambda_w=None, lambda_h=None, warmup=False, dtype=None):
        super().__init__()
        
        self.warmup = warmup
        self.use_ah = use_ah
        self.aw_2d  = aw_2d
        self.iter   = iter
        
        if dtype is not None:
            self.eps = torch.finfo(type=dtype).min
        else:
            self.eps = 1e-6
            
        self.clip_H     = clip_H 
        self.lambda_w   = lambda_w
        self.lambda_h   = lambda_h
        
        self.Aw = shared_aw if shared_aw is not None else (Aw_2d_cnn(hidden_channels=hidden_channels, dtype=dtype) if self.aw_2d else Aw_cnn(hidden_channels=hidden_channels, dtype=dtype))
        if self.use_ah:
            self.Ah = shared_ah if shared_ah is not None else Ah_cnn(hidden_channels=hidden_channels, dtype=dtype)
            
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta)) 
        else:
            self.register_buffer("beta", torch.tensor(beta))

    def forward(self, M, W, H):  
                  
        is_batched = (len(M.shape) == 3)
        
        # Add channel dimension
        if not is_batched:
            W = W.unsqueeze(0)  # Shape: (1, f, l)
            H = H.unsqueeze(0)  # Shape: (1, l, t)
        
        if self.iter is not None and (self.iter%5 ==0):
            
            wh = W @ H
            wh = torch.clamp(wh, min=self.eps)

            # Compute WH^(β - 2) * M
            wh_2 = wh.pow(self.beta - 2)
            wh_2_m = wh_2 * M
            
            # Compute WH^(β - 1)
            wh_1 = wh.pow(self.beta - 1)

            # MU for W        
            numerator_W = wh_2_m @ H.transpose(1, 2)
            denominator_W = wh_1 @ H.transpose(1, 2)
            denominator_W = torch.clamp(denominator_W, min=self.eps)
            
            if self.lambda_w is not None:
                denominator_W += self.lambda_w
                
            update_W = numerator_W / denominator_W
            
            # Use octave splitting to inject only intra-octave notes in Aw
            if self.aw_2d:
                octave_W = [W[:, :, :4]] + [W[:, :, i+4:i+16] for i in range(0, 84, 12)]
                accel_W = torch.cat([self.Aw(W_i)[0] for W_i in octave_W], 1)
                W_new = W * accel_W * update_W
            else: # Inject W column by column in Aw
                accel_W = torch.cat([self.Aw(W[:, :, i]) for i in range(88)], 2)
                W_new = W * accel_W * update_W
        else:
            accel_W = torch.ones(W.shape)
            W_new = W
            
        # Avoid going to zero by clamping    
        W_new = torch.clamp(W_new, min=self.eps)

        # Compute WH again with updated W
        wh = W_new @ H
        wh = torch.clamp(wh, min=self.eps)

        wh_2 = wh.pow(self.beta - 2)
        wh_2_m = wh_2 * M
        wh_1 = wh.pow(self.beta - 1)

        # MU for H
        numerator_H = W_new.transpose(-1, -2) @ wh_2_m
        denominator_H = W_new.transpose(-1, -2) @ wh_1
        denominator_H = torch.clamp(denominator_H, min=self.eps)
        
        if self.lambda_h is not None:
            denominator_H += self.lambda_h
            
        update_H = numerator_H / denominator_H

        # Apply learned acceleration (Ah)
        if self.use_ah:
            H_rows = [H[:, i:i+1, :] for i in range(0, 88)]
            accel_H = torch.cat([self.Ah(H_i) for H_i in H_rows], 1)
            H_new = H * accel_H * update_H
        else:
            H_new = H * update_H
            
        # Avoid going to zero by clamping
        H_new = torch.clamp(H_new, min=self.eps)

        if not is_batched:
            W_new = W_new.squeeze(0)
            H_new = H_new.squeeze(0)
            
        if self.clip_H:
            H_new = torch.clip(H_new, max=1)
        
        if self.warmup:
            return W_new, H_new, accel_W, accel_H
        else:
            return W_new, H_new

    
class RALMU(nn.Module):
    """
    Define the RALMU model as n unrolled layers of RALMU block (CNN accelerated MU iterations)
    
    Args:
        l (int, optional): the amount of distinct single notes to transcribe (default: ``88``)
        beta (int, optional): value for the β-divergence (default: ``1`` = KL divergence)
        W_path (str, optional): the path to the folder containing the recording of all the  notes. If ``None``, W is initialized with artificial data (default: ``None``)
        n_iter (int, optional): the number of unrolled iterations of MU (default: ``10``)
        n_init_steps (int, optional): the number of MU steps to do to initialize H (default: ``100``)
        hidden (int, optional): the size of the CNN filters (default: ``32``)
        use_ah (bool, optional): whether to use Ah in the acceleration of MU (default: ``True``)
        shared (bool, optional): whether Ah and Aw are shared across layers (default: ``False``)
        aw_2d (bool, optional): whether to use a 2D CNN for Aw (default: ``False``)
        clip_H (bool, optional): whether to clip the value of H after the update or not (default: ``False``)
        norm_thresh (float, optional): whether to normalize W and H (default: ``None``)
        lambda_w (float, optional): the value of the lambda factor to add at the denominator of the update of W (default: ``None``)
        lambda_h (float, optional): the value of the lambda factor to add at the denominator of the update of H (default: ``None``)
        warmup (bool, optional): whether we warmup train (return the values of the CNNs acceleration) or not (default: ``False``)
        return_layers (bool, optional): whether to return the output of each layers or just the final one (default: ``False``)
    """
    
    def __init__(self, l=88, beta=1, learnable_beta=False, W_path=None, n_iter=10, n_init_steps=100, hidden=32, use_ah=True, shared=False, aw_2d=False, clip_H=False, norm_thresh=0.01, lambda_w=None, lambda_h=None, warmup=False, return_layers=True, dtype=None, verbose=False):
        super().__init__()
        
        if dtype is not None:
            self.eps = torch.finfo(type=dtype).min
        else:
            self.eps = 1e-6
            
        self.l              = l
        self.beta           = beta
        self.W_path         = W_path
        self.n_iter         = n_iter
        self.n_init_steps   = n_init_steps
        self.shared         = shared
        self.clip           = clip_H
        self.verbose        = verbose
        self.norm_thresh    = norm_thresh
        self.warmup         = warmup 
        self.return_layers  = return_layers
        self.dtype          = dtype

        shared_aw = (Aw_2d_cnn(hidden_channels=hidden, dtype=dtype) if aw_2d else Aw_cnn(hidden_channels=hidden, dtype=dtype)) if self.shared else None
        if use_ah:
            shared_ah = Ah_cnn(hidden_channels=hidden, dtype=dtype) if self.shared else None
        else:
            shared_ah = None

        # Unrolling layers
        # self.layers = nn.ModuleList([
        #     RALMU_block(hidden_channels=hidden, shared_aw=shared_aw, shared_ah=shared_ah, use_ah=use_ah, learnable_beta=learnable_beta, aw_2d=aw_2d, clip_H=self.clip, lambda_w=lambda_w, lambda_h=lambda_h, warmup=self.warmup, dtype=dtype)
        #     for _ in range(self.n_iter)
        # ])
        self.layers = nn.ModuleList([
            RALMU_block(iter=i, hidden_channels=hidden, shared_aw=shared_aw, shared_ah=shared_ah, use_ah=use_ah, learnable_beta=learnable_beta, aw_2d=aw_2d, clip_H=self.clip, lambda_w=lambda_w, lambda_h=lambda_h, warmup=self.warmup, dtype=dtype)
            for i in range(self.n_iter)
        ])
    
    def forward(self, M, device=None):
        
        batch_size=None
        if len(M.shape) == 3: # Batched input (training phase)
            batch_size, _, t = M.shape
        elif len(M.shape) == 2: # Non-batched input (inference phase)
            _, t = M.shape
        
        W, _, _, _ = init.init_W(self.W_path, normalize_thresh=self.norm_thresh,dtype=self.dtype)
        if batch_size is not None:
            W = W.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Tracking gpu usage
        if device == "cuda:0":
            gpu_info = utils.get_gpu_info()
            utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        
        H = init.init_H(W, M, n_init_steps=self.n_init_steps, beta=self.beta, device=device, dtype=self.dtype)
        if batch_size is not None:
            H = H.unsqueeze(0).expand(batch_size, -1, -1)
            
        
        # Tracking gpu usage
        if device == "cuda:0":
            gpu_info = utils.get_gpu_info()
            utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        
        W = init.scale_W(M, W, H)
            
        W0 = W
        H0 = H
        
        if self.return_layers:
            W_layers = []
            H_layers = []
            accel_W_layers = []
            accel_H_layers = []

            for i, layer in enumerate(self.layers):
                if self.warmup:
                    W, H, accel_W, accel_H = layer(M, W, H)
                    accel_W_layers.append(accel_W)
                    accel_H_layers.append(accel_H)
                else:
                    W, H = layer(M, W, H)
                # Tracking gpu usage
                if device == "cuda:0":
                    gpu_info = utils.get_gpu_info()
                    utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
                W_layers.append(W)
                H_layers.append(H)
                if W is None or H is None:
                    print("W or H got to None")
                if torch.sum(torch.nonzero(torch.isnan(W).view(-1))) > 0 or  torch.sum(torch.nonzero(torch.isnan(H).view(-1))) > 0:
                    print("W or H got nan values")

            M_hats = [W @ H for W, H in zip(W_layers, H_layers)]

            if self.warmup:
                return W_layers, H_layers, M_hats, W0, H0, accel_W_layers, accel_H_layers
            else:
                return W_layers, H_layers, M_hats
        else:
            for l, layer in enumerate(self.layers):
                
                if self.warmup:
                    W, H, accel_W, accel_H = layer(M, W, H)
                else:
                    W, H = layer(M, W, H)
                # Tracking gpu usage
                if device == "cuda:0":
                    gpu_info = utils.get_gpu_info()
                    utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
            M_hat = W @ H
            
            if self.warmup:
                return W, H, M_hat, W0, H0, accel_W, accel_H
            else:
                return W, H, M_hat