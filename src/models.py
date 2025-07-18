import torch.nn as nn
import torchaudio
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import librosa
import time

import src.utils as utils
import src.spectrograms as spec
import src.init as init

def MU_iter(M, l, f, t, n_iter):
    # Multiplicative updates iterations
    M = torch.tensor(M)
    W = torch.rand(n_iter, f, l)
    H = torch.rand(n_iter, l, t)
    
    aw = Aw(w_size=f*l)
    ah = Ah(h_size=l*t)

    for l in range(n_iter-1):
        W[l+1] = W[l] * aw(W[l]) * (M @ H[l].T) / (W[l] @ H[l] @ H[l].T)
        H[l+1] = H[l] * ah(H[l]) * (W[l+1].T @ M) / (W[l+1].T @ W[l+1] @ H[l])
    
    M_hat = W[-1] @ H[-1]
    
    return W, H, M_hat

def MU(M, W, H, n_iter, beta=1, eps=1e-6, normalize=False):
    
    for _ in range(n_iter):
        # Compute WH
        Wh = W @ H
        Wh = torch.clamp(Wh, min=eps)

        Wh_beta_minus_2 = Wh ** (beta - 2)
        Wh_beta_minus_1 = Wh ** (beta - 1)

        # Update W
        numerator_W = (Wh_beta_minus_2 * M) @ H.T
        denominator_W = Wh_beta_minus_1 @ H.T #+ eps
        denominator_W = torch.clamp(denominator_W, min=eps)

        W = W * (numerator_W / denominator_W)
        
        if normalize:
            W = W / (W.sum(dim=1, keepdim=True) )#+ eps)

        # Compute WH again for updating H
        Wh = W @ H
        Wh = torch.clamp(Wh, min=eps)

        # Update H
        numerator_H = W.T @ (Wh_beta_minus_2 * M)
        denominator_H = W.T @ Wh_beta_minus_1 #+ eps
        denominator_H = torch.clamp(denominator_H, min=eps)

        H = H * (numerator_H / denominator_H)
        
        if normalize:
            H = H / (H.sum(dim=1, keepdim=True) )#+ eps)

    return W, H

class Aw(nn.Module):
    """
    Defining a simple MLP for Aw()
    w_size = W.shape[0]*W.shape[1] <=> l*f
    w_size -> 130 -> 75 -> w_size
    """
    def __init__(self, w_size):
        super(Aw, self).__init__()
        self.w_size = w_size
        self.fc0    = nn.Linear(w_size, 130)
        self.fc1    = nn.Linear(130, 75)
        self.fc2    = nn.Linear(75, w_size)
        self.relu   = nn.ReLU()

    # W shape: (f,l)
    def forward(self, x):
        shape   = x.shape
        x       = x.reshape(-1)
        y0      = self.relu(self.fc0(x))
        y1      = self.relu(self.fc1(y0))
        y2      = self.relu(self.fc2(y1))
        out     = y2.view(shape)
        return out
    
    
class Ah(nn.Module):
    """
    Defining a simple MLP for Ah()
    h_size = H.shape[0]*H.shape[1] <=> l*f
    h_size -> 130 -> 75 -> h_size
    """
    def __init__(self, h_size):
        super(Ah,self).__init__()
        self.h_size = h_size
        self.fc0    = nn.Linear(h_size, 130)
        self.fc1    = nn.Linear(130, 75)
        self.fc2    = nn.Linear(75, h_size)
        self.relu   = nn.ReLU()

    # H shape: (l,t)
    def forward(self, x):
        shape   = x.shape
        x       = x.reshape(-1)
        y0      = self.relu(self.fc0(x))
        y1      = self.relu(self.fc1(y0))
        y2      = self.relu(self.fc2(y1))
        out     = y2.view(shape)
        return out
    

class Aw_cnn(nn.Module):
    """
    Defining a 1D CNN (frequency axis) for Aw()
    reshapes input of shape (batch, f, l) to (batch, 1, f*l) and applies convolution to f*l
    1 channel -> 64 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1
    """
    def __init__(self, in_channels=1, hidden_channels=32, dtype=None):
        super(Aw_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2, dtype=dtype)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2, dtype=dtype)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(hidden_channels*2, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(hidden_channels, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus()

    # W shape: (f,l)
    def forward(self, x):
        # print(f"Aw in: {x.shape}")                   # (batch, f, l)
        # if (len(x.shape) == 3):
        #     batch_size, f, l = x.shape
        # else:
        #     f, l = x.shape
        #     batch_size = 1
        batch_size, f = x.shape
        # x = x.reshape(batch_size, 1, f*l)           # (batch, 1, f*l)
        # x = x.transpose(1, 2).reshape(batch_size, 1, f*l)
        x = x.reshape(batch_size, 1, f)
        # plt.imshow(x[0], cmap='Reds')
        # plt.title("Reshaped W input")
        # plt.colorbar()
        # plt.show()
        # print(f"Aw reshaped: {x.shape}")
        y = self.relu(self.bn1(self.conv1(x)))      # (batch, 64, f*l)
        y = self.relu(self.bn2(self.conv2(y)))      # (batch, 32, f*l)
        y = self.softplus(self.conv3(y))            # (batch, 1, f*l)
        # out = y.reshape(batch_size, f, l)
        out = y.reshape(batch_size, f, 1)
        # print(f"Aw out: {out.shape}")
        return out
    

class Aw_2d_cnn(nn.Module):
    """
    Defining a 2D CNN for Aw
    1 channel -> 64 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1    
    """
    def __init__(self, in_channels=1, hidden_channels=2, dtype=None):
        super(Aw_2d_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels*2, kernel_size=(24*5,5), padding="same", dtype=dtype)
        self.conv2 = nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=(24*3,3), padding="same", dtype=dtype)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=(24*3,3), padding="same", dtype=dtype)
        self.bn1 = nn.BatchNorm2d(hidden_channels*2, dtype=dtype)
        self.bn2 = nn.BatchNorm2d(hidden_channels, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        if (len(x.shape) == 3):
            batch_size, f, l = x.shape
        else:
            f, l = x.shape
            batch_size = 1
        x = x.reshape(batch_size, 1, f, l)
        y = self.relu(self.bn1(self.conv1(x)))      # (batch, 64, f*l)
        y = self.relu(self.bn2(self.conv2(y)))      # (batch, 32, f*l)
        y = self.softplus(self.conv3(y))            # (batch, 1, f*l)
        out = y.reshape(batch_size, f, l)
        return out

    
class Ah_cnn(nn.Module):
    """
    Defining a 1D CNN (time axis) for Ah()
    reshapes input of shape (batch, l, t) to (batch, 1, l*t) and applies convolution to l*t
    1 channel -> 32 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1
    """
    def __init__(self, in_channels=1, hidden_channels=32, dtype=None):
        super(Ah_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2, dtype=dtype)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2, dtype=dtype)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1, dtype=dtype)
        self.bn1 = nn.BatchNorm1d(hidden_channels*2, dtype=dtype)
        self.bn2 = nn.BatchNorm1d(hidden_channels, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus()

    # H shape: (l,t)
    def forward(self, x):
        # print(f"Ah in: {x.shape}")            # (batch, l, t)
        if (len(x.shape) == 3):
            batch_size, l, t = x.shape
        else:
            l, t = x.shape
            batch_size = 1
        # x = x.view(batch_size * l, 1, t)        # (batch, 1, t*l)
        x = x.reshape(batch_size, 1, l*t)
        # plt.imshow(x[0], cmap='Reds')
        # plt.title("Reshaped H input")
        # plt.colorbar()
        # plt.show()
        # print(f"Ah reshaped: {x.shape}")
        y = self.relu(self.bn1(self.conv1(x)))  # (batch, 64, t*l)
        y = self.relu(self.bn2(self.conv2(y)))  # (batch, 32, t*l)
        y = self.relu(self.conv3(y))            # (batch, 1, t*l)
        out = self.softplus(y)                  # (batch, 1, t*l)
        out = out.view(batch_size, l, t)        # (batch, l, t*l)
        # print(f"Ah out: {out.shape}")
        return out   
   
   
class NALMU_block(nn.Module):
    """
    A single layer/ iteration of the NALMU model
    updating W and H
    
    Args:
        f (int): number of frequency bins in the audio file to be transcribed
        l (int, optional): the amount of distinct single notes to transcribe (default: ``88``)
        beta (int, optional): value for the β-divergence (default: ``1`` = KL divergence)
        eps (int, optional): min value for MU computations (default: ``1e-6``)
        shared_w (Aw_cnn, optional): whether to use a predefined Aw acceleration matrix or to create one (default: ``None``)
        learnable_beta (bool, optional): whether to learn the value of beta (default: ``False``)
        normalize (bool, optional): whether to normalize W and H (default: ``False``)
    """
    def __init__(self, f, l=88, beta=1, eps=1e-6, shared_w=None, learnable_beta=False, normalize=False):
        super().__init__()
        
        self.eps    = eps
        self.normalize = normalize
            
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta)) 
        else:
            self.register_buffer("beta", torch.tensor(beta))
        
        self.w_accel = shared_w if shared_w is not None else nn.Parameter(torch.rand(f, l) + self.eps)

    def forward(self, M, W, H):
        
        # Add channel dimension
        W = W.unsqueeze(0)  # Shape: (1, f, l)
        H = H.unsqueeze(0)  # Shape: (1, l, t)
        
        wh = W @ H + self.eps

        # Compute WH^(β - 2) * V
        wh_2 = wh.pow(self.beta - 2)
        wh_2_m = wh_2 * M

        # Compute WH^(β - 1)
        wh_1 = wh.pow(self.beta - 1)

        # MU for W
        numerator_W = wh_2_m @ H.transpose(-1, -2)
        denominator_W = wh_1 @ H.transpose(-1, -2) + self.eps
        update_W = numerator_W / denominator_W

        # Apply learned transformation (Aw)
        # Avoid going to zero by clamping
        W_new = W * self.w_accel * update_W
        W_new = torch.clamp(W_new, min=self.eps)
        
        if self.normalize:
            W_new = W_new / (W_new.sum(dim=1, keepdim=True) + self.eps)

        # Compute WH again with updated W
        wh = W_new @ H + self.eps

        wh_2 = wh.pow(self.beta - 2)
        wh_2_m = wh_2 * M
        wh_1 = wh.pow(self.beta - 1)

        # MU for H
        numerator_H = W_new.transpose(-1, -2) @ wh_2_m
        denominator_H = W_new.transpose(-1, -2) @ wh_1 + self.eps
        update_H = numerator_H / denominator_H

        # Apply learned transformation (Ah)
        # Avoid going to zero by clamping
        H_new = H * update_H# H * self.h_accel * update_H
        H_new = torch.clamp(H_new, min=self.eps)
        
        if self.normalize:
            H_new = H_new / (H_new.sum(dim=1, keepdim=True) + self.eps)

        return W_new.squeeze(0), H_new.squeeze(0)
    

class NALMU(nn.Module) :
    """
    Unrolled NALMU model
    
    Args:
        l (int, optional): the amount of distinct single notes to transcribe (default: ``88``)
        eps (int, optional): min value for MU computations (default: ``1e-6``)
        beta (int, optional): value for the β-divergence (default: ``1`` = KL divergence)
        n_iter (int, optional): the number of unrolled iterations of MU (default: ``10``)
        W_path (str, optional): the path to the folder containing the recording of all the  notes. If ``None``, W is initialized with artificial data (default: ``None``)
        n_init_steps (int, optional): the number of MU steps to do to initialize H (default: ``100``)
        shared (bool, optional): whether Ah and Aw are shared across layers (default: ``False``)
        n_bins (int, optional): parameter for the cqt representation of W (default: ``288``)
        bins_per_octave (int, optional): parameter for the cqt representation of W (default: ``36``)
        learnable_beta (bool, optional): whether to learn the value of beta (default: ``False``)
        verbose (bool, optional): whether to display some information (default: ``False``)
        normalize (bool, optional): whether to normalize W and H (default: ``False``)
    """ 
    def __init__(self, l=88, eps=1e-6, beta=1, n_iter=10, W_path=None, n_init_steps=10, shared=False, n_bins=288, bins_per_octave=36, learnable_beta=False, verbose=False, normalize=False):
        super().__init__()
        
        self.n_bins             = n_bins
        self.bins_per_octave    = bins_per_octave
        
        self.l                  = l
        self.eps                = eps
        # self.beta               = beta
        self.W_path             = W_path
        self.eps                = eps
        self.n_iter             = n_iter
        self.n_init_steps       = n_init_steps
        self.shared             = shared
        self.verbose            = verbose
        self.normalize          = normalize
        
        shared_w = nn.Parameter(torch.rand(self.n_bins, self.l) + self.eps) if self.shared else None
        # shared_h = nn.Parameter(torch.rand(self.l, self.t) + self.eps) if self.shared else None
            
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta)) 
        else:
            self.register_buffer("beta", torch.tensor(beta))
        
        self.layers = nn.ModuleList([
            NALMU_block(self.n_bins, self.l, self.beta, self.eps, shared_w, normalize=self.normalize)
            for _ in range(self.n_iter)
        ])
        
        W0, freqs, _, _ = init.init_W(self.W_path, verbose=self.verbose)
        self.freqs = freqs
        self.register_buffer("W0", W0)
            
    def init_H(self, M):
        if len(M.shape) == 3:
            # Batched input (training phase)
            _, f, t = M.shape
        elif len(M.shape) == 2:
            # Non-batched input (inference phase)
            f, t = M.shape
            
        H0 = init.init_H(self.l, t, self.W0, M, n_init_steps=self.n_init_steps, beta=self.beta)

        self.register_buffer("H0", H0)
    
    def forward(self, M):
        
        assert hasattr(self, 'H0'), "Please run init_H, H0 and layers are not initialized"
        
        W = self.W0
        H = self.H0

        W_layers = []
        H_layers = []

        for i, layer in enumerate(self.layers):
            W, H = layer(M, W, H)
            W_layers.append(W)
            H_layers.append(H)
            if W is None or H is None:
                print("W or H got to None")

        M_hats = [W @ H for W, H in zip(W_layers, H_layers)]

        return W_layers, H_layers, M_hats
        
        
class RALMU_block(nn.Module):
    """
    A single layer/iteration of the RALMU model
    with β-divergence multiplicative updates for W and H.
    Aw and Ah are CNNS
        
    Args:
        beta (int, optional): value for the β-divergence (default: ``1`` = KL divergence)
        eps (int, optional): min value for MU computations (default: ``1e-6``)
        shared_aw (Aw_cnn, optional): whether to use a predefined Aw acceleration model or to create one (default: ``None``)
        shared_aw (Ah_cnn, optional): whether to use a predefined Ah acceleration model or to create one (default: ``None``)
        use_ah (bool, optional): whether to use Ah in the acceleration of MU (default: ``True``)
        learnable_beta (bool, optional): whether to learn the value of beta (default: ``False``)
        normalize (bool, optional): whether to normalize W and H (default: ``False``)
    """
    def __init__(self, beta=1, eps=1e-6, shared_aw=None, shared_ah=None, use_ah=True, learnable_beta=False, normalize=False, aw_2d=False, clip_H=False, dtype=None):
        super().__init__()
        
        self.use_ah = use_ah
        self.aw_2d = aw_2d
        
        self.Aw = shared_aw if shared_aw is not None else (Aw_2d_cnn(dtype=dtype) if self.aw_2d else Aw_cnn())
        if self.use_ah:
            self.Ah = shared_ah if shared_ah is not None else Ah_cnn(dtype=dtype)
            
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta)) 
        else:
            self.register_buffer("beta", torch.tensor(beta))
        self.eps = eps
        self.normalize = normalize
        self.clip_H    = clip_H 

    def forward(self, M, W, H):
        
        is_batched = (len(M.shape) == 3)
        
        # Add channel dimension
        if not is_batched:
            W = W.unsqueeze(0)  # Shape: (1, f, l)
            H = H.unsqueeze(0)  # Shape: (1, l, t)
            
        wh = W @ H + self.eps

        # Compute WH^(β - 2) * M
        wh_2 = wh.pow(self.beta - 2)
        wh_2_m = wh_2 * M

        # Compute WH^(β - 1)
        wh_1 = wh.pow(self.beta - 1)

        # MU for W
        numerator_W = wh_2_m @ H.transpose(-1, -2)
        denominator_W = wh_1 @ H.transpose(-1, -2) + self.eps
        update_W = numerator_W / denominator_W

        # Apply learned transformation (Aw)
        
        # Use octave splitting to inject only intra-octave notes in Aw
        if self.aw_2d:
            octave_W = [W[:, :, :4]] + [W[:, :, i+4:i+16] for i in range(0, 84, 12)]
            accel_W = torch.cat([self.Aw(W_i)[0] for W_i in octave_W], 1)
            W_new = W * accel_W * update_W
        else:
            accel_W = torch.cat([self.Aw(W[:, :, i]) for i in range(88)], 2)
            # print(self.Aw(W[:,:,0]).shape, accel_W.shape)
            W_new = W * accel_W * update_W
            
        # Avoid going to zero by clamping    
        W_new = torch.clamp(W_new, min=self.eps)
        
        if self.normalize:
            W_new = W_new / (W_new.sum(dim=1, keepdim=True) + self.eps)

        # Compute WH again with updated W
        wh = W_new @ H + self.eps

        wh_2 = wh.pow(self.beta - 2)
        wh_2_m = wh_2 * M
        wh_1 = wh.pow(self.beta - 1)

        # MU for H
        numerator_H = W_new.transpose(-1, -2) @ wh_2_m
        denominator_H = W_new.transpose(-1, -2) @ wh_1 + self.eps
        update_H = numerator_H / denominator_H

        # Apply learned transformation (Ah)
        if self.use_ah:
            H_rows = [H[:, i:i+1, :] for i in range(0, 88)]
            accel_H = torch.cat([self.Ah(H_i) for H_i in H_rows], 1)
            H_new = H * accel_H * update_H
        else:
            H_new = H * update_H
            
        # Avoid going to zero by clamping
        H_new = torch.clamp(H_new, min=self.eps)
        
        if self.normalize:
            H_new = H_new / (H_new.sum(dim=1, keepdim=True) + self.eps)

        if not is_batched:
            W_new = W_new.squeeze(0)
            H_new = H_new.squeeze(0)
            
        if self.clip_H:
            H_new = torch.clip(H_new, max=1)
        
        return W_new, H_new

    
class RALMU(nn.Module):
    """
    Define the RALMU model as n unrolled layers of RALMU block (CNN accelerated MU iterations)
    
    Args:
        l (int, optional): the amount of distinct single notes to transcribe (default: ``88``)
        eps (int, optional): min value for MU computations (default: ``1e-6``)
        beta (int, optional): value for the β-divergence (default: ``1`` = KL divergence)
        W_path (str, optional): the path to the folder containing the recording of all the  notes. If ``None``, W is initialized with artificial data (default: ``None``)
        n_iter (int, optional): the number of unrolled iterations of MU (default: ``10``)
        n_init_steps (int, optional): the number of MU steps to do to initialize H (default: ``100``)
        hidden (int, optional): the size of the CNN filters (default: ``32``)
        use_ah (bool, optional): whether to use Ah in the acceleration of MU (default: ``True``)
        shared (bool, optional): whether Ah and Aw are shared across layers (default: ``False``)
        n_bins (int, optional): parameter for the cqt representation of W (default: ``288``)
        bins_per_octave (int, optional): parameter for the cqt representation of W (default: ``36``)
        verbose (bool, optional): whether to display some information (default: ``False``)
        normalize (bool, optional): whether to normalize W and H (default: ``False``)
    """
    
    def __init__(self, l=88, eps=1e-6, beta=1, W_path=None, n_iter=10, n_init_steps=100, hidden=32, use_ah=True, shared=False, n_bins=288, bins_per_octave=36, downsample=False, verbose=False, normalize=False, return_layers=True, aw_2d=False, clip_H=False, dtype=None):
        super().__init__()
        
        self.n_bins          = n_bins
        self.bins_per_octave = bins_per_octave
        
        self.l              = l
        self.eps            = eps
        self.beta           = beta
        self.W_path         = W_path
        self.n_iter         = n_iter
        self.n_init_steps   = n_init_steps
        self.shared         = shared
        self.downsample     = downsample
        self.verbose        = verbose
        self.normalize      = normalize
        self.return_layers  = return_layers
        self.dtype          = dtype

        shared_aw = (Aw_2d_cnn(hidden_channels=hidden, dtype=dtype) if aw_2d else Aw_cnn(hidden_channels=hidden, dtype=dtype)) if self.shared else None
        if use_ah:
            shared_ah = Ah_cnn(hidden_channels=hidden, dtype=dtype) if self.shared else None
        else:
            shared_ah = None

        # Unrolling layers
        self.layers = nn.ModuleList([
            RALMU_block(eps=self.eps, shared_aw=shared_aw, shared_ah=shared_ah, use_ah=use_ah, normalize=self.normalize, aw_2d=aw_2d, clip_H=clip_H, dtype=dtype)
            for _ in range(self.n_iter)
        ])
    
    def forward(self, M, device=None):
        
        batch_size=None
        if len(M.shape) == 3:
            # Batched input (training phase)
            batch_size, _, t = M.shape
        elif len(M.shape) == 2:
            # Non-batched input (inference phase)
            _, t = M.shape
            
        W, _, _, _ = init.init_W(self.W_path, downsample=self.downsample, verbose=self.verbose, dtype=self.dtype)
        if batch_size is not None:
            W = W.unsqueeze(0).expand(batch_size, -1, -1)
        
        H = init.init_H(self.l, t, W, M, n_init_steps=self.n_init_steps, beta=self.beta, device=device, batch_size=batch_size, dtype=self.dtype)
        
        if self.return_layers:
            W_layers = []
            H_layers = []

            for i, layer in enumerate(self.layers):
                W, H = layer(M, W, H)
                W_layers.append(W)
                H_layers.append(H)
                if W is None or H is None:
                    print("W or H got to None")

            M_hats = [W @ H for W, H in zip(W_layers, H_layers)]

            return W_layers, H_layers, M_hats
        else:
            for layer in self.layers:
                W, H = layer(M, W, H)
            M_hat = W @ H
            return W, H, M_hat