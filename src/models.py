import torch.nn as nn
import torchaudio
import numpy as np
import torch
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
    1 channel -> 32 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1
    """
    def __init__(self, in_channels=1, hidden_channels=2):
        super(Aw_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels*2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus() 
        
        # nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        # nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='leaky_relu')

    # W shape: (f,l)
    def forward(self, x):
        # print(f"Aw in: {x.shape}") # (1, f, l)
        batch_size, f, l = x.shape
        x = x.view(batch_size * l, 1, f) # (l, 1, f)
        y = self.relu(self.bn1(self.conv1(x)))     # (l, 64, f)
        y = self.relu(self.bn2(self.conv2(y)))     # (l, 32, f)
        y = self.softplus(self.conv3(y)) # (l, 1, f)
        out = y.view(batch_size, l, f)
        out = out.permute(0,2,1)
        # print(f"Aw out: {out.shape}")
        return out
   
    
class Ah_cnn(nn.Module):
    """
    Defining a 1D CNN (time axis) for Ah()
    1 channel -> 32 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1
    """
    def __init__(self, in_channels=1, hidden_channels=32):
        super(Ah_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels*2)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus() 
        
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='leaky_relu')

    # H shape: (l,t)
    def forward(self, x):
        # print(f"Ah in: {x.shape}")      # (batch_size, l, t)
        batch_size, l, t = x.shape
        x = x.view(batch_size * l, 1, t)  # (batch_size * l, 1, t)
        y = self.relu(self.bn1(self.conv1(x)))      # (batch_size * l, 64, t)
        y = self.relu(self.bn2(self.conv2(y)))      # (batch_size * l, 32, t)
        y = self.relu(self.conv3(y))      # (batch_size * l, 1, t)
        out = self.softplus(y)            # (batch_size * l, 1, t)
        out = out.view(batch_size, l, t)  #(batch_size, l, t)
        # print(f"Ah out: {out.shape}")
        return out   
   
   
class NALMU_block(nn.Module):
    """
    A single layer/ iteration of the NALMU model
    updating W and H
    """
    def __init__(self, f, t, l=88, beta=1, eps=1e-6, shared_w=None, shared_h=None):
        super().__init__()
        
        self.eps    = eps
        self.beta   = beta
        
        if shared_w is not None:
            self.w_accel = shared_w
        else:
            self.w_accel = nn.Parameter(torch.rand(f, l) + self.eps)
        if shared_h is not None:
            self.h_accel = shared_h
        else:
            self.h_accel = nn.Parameter(torch.rand(l, t) + self.eps)
    
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
        H_new = H * self.h_accel * update_H
        H_new = torch.clamp(H_new, min=self.eps)

        return W_new.squeeze(0), H_new.squeeze(0)
    

class NALMU(nn.Module) :
    """
    Unrolled NALMU model
    """ 
    def __init__(self, t, l=88, W_path=None, n_bins=288, bins_per_octave=36, eps=1e-6, n_iter=10, n_init_steps=100, shared=False):
        super().__init__()
        
        self.t                  = t
        self.l                  = l
        self.W_path             = W_path
        self.n_bins             = n_bins
        self.bins_per_octave    = bins_per_octave
        self.eps                = eps
        self.n_iter             = n_iter
        self.n_init_steps       = n_init_steps
        self.shared             = shared
        
        shared_w = nn.Parameter(torch.rand(self.n_bins, self.l) + self.eps) if self.shared else None
        shared_h = nn.Parameter(torch.rand(self.l, self.t) + self.eps) if self.shared else None
        
        self.layers = nn.ModuleList([
            NALMU_block(self.n_bins, self.t, self.l, self.eps, shared_w, shared_h)
            for _ in range(self.n_iter)
        ])
        
        W0, freqs, sr, true_freqs = init.init_W(self.W_path, verbose=self.verbose)
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
        
        assert hasattr(self, 'W0') or hasattr(self, 'H0'), "Please run init_H, W0 or H0 are not initialized"
        
        W = self.W0
        H = self.H0

        for i, layer in enumerate(self.layers):
            W, H = layer(M, W, H)
            if W is None or H is None:
                print("W or H got to None")

        M_hat = W @ H

        return W, H, M_hat
        
        
class RALMU_block(nn.Module):
    """
    A single layer/iteration of the RALMU model
    with β-divergence multiplicative updates for W and H.
    Aw and Ah are CNNS
    """
    def __init__(self, beta=1, shared_aw=None, shared_ah=None, use_ah=True, learnable_beta=False):
        super().__init__()
        
        self.use_ah = use_ah
        self.Aw = shared_aw if shared_aw is not None else Aw_cnn()
        if self.use_ah:
            self.Ah = shared_ah if shared_ah is not None else Ah_cnn()
            
        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta)) 
        else:
            self.register_buffer("beta", torch.tensor(beta))
        self.eps = 1e-6

    def forward(self, M, W, H):
        
        # Add channel dimension
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
        # Avoid going to zero by clamping
        W_new = W * self.Aw(W) * update_W
        W_new = torch.clamp(W_new, min=self.eps)

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
        if self.use_ah:
            H_new = H * self.Ah(H) * update_H
        else:
            H_new = H * update_H
        H_new = torch.clamp(H_new, min=self.eps)

        return W_new.squeeze(0), H_new.squeeze(0)

    
class RALMU(nn.Module):
    
    def __init__(self, l=88, eps=1e-6, beta=1, W_path=None, n_iter=10, n_init_steps=100, hidden=32, use_ah=True, shared=False, verbose=False):
        super().__init__()
        
        n_bins          = 288
        bins_per_octave = 36
        
        self.l      = l
        self.eps    = eps
        self.beta   = beta
        self.W_path = W_path
        self.n_iter         = n_iter
        self.n_init_steps   = n_init_steps
        self.shared         = shared
        self.verbose        = verbose

        shared_aw = Aw_cnn(hidden_channels=hidden) if self.shared else None
        if use_ah:
            shared_ah = Ah_cnn(hidden_channels=hidden) if self.shared else None
        else:
            shared_ah = None

        # Unrolling layers
        self.layers = nn.ModuleList([
            RALMU_block(shared_aw=shared_aw, shared_ah=shared_ah, use_ah=use_ah)
            for _ in range(self.n_iter)
        ])
        
        W0, freqs, sr, true_freqs = init.init_W(self.W_path, verbose=self.verbose)
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
        
        assert hasattr(self, 'W0') or hasattr(self, 'H0'), "Please run init_WH, W0 or H0 are not initialized"
        
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
    
    def update_WH(self, W_new, H_new):
        self.W0 = W_new
        self.H0 = H_new