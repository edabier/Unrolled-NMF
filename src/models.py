import torch.nn as nn
import torchaudio
import numpy as np
import torch
import src.utils
import src.spectrograms as spec
import os
import librosa

def init_W(folder_path, hop_length=512, bins_per_octave=36, n_bins=252):
    templates = []
    file_list = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.wav'))])

    for fname in file_list:
        path = os.path.join(folder_path, fname)
        y, sr = torchaudio.load(path)
        spec_db, _, _ = spec.cqt_spec(y, sample_rate=sr, hop_length=hop_length,
                                 bins_per_octave=bins_per_octave, n_bins=n_bins)
        
        # Choose frame with max energy (sum across frequencies)
        energy_per_frame = np.sum(spec_db, axis=0)
        best_frame_idx = np.argmax(energy_per_frame)
        template = spec_db[:, best_frame_idx]

        # Convert from dB to linear for multiplication use
        template_lin = librosa.db_to_amplitude(template)

        # Normalize
        template_lin /= np.linalg.norm(template_lin) + 1e-8
        
        templates.append(template_lin)

    # W of shape f * (88*4)
    W = np.stack(templates, axis=1)
    
    return torch.tensor(W, dtype=torch.float32)

def init_H(l, t, W, M, n_init_steps):
    eps = 1e-8
    H = torch.rand(l, t) + eps
    # create H with n iterations of MU
    
    for _ in range(n_init_steps):
        H = H * (W.T @ M) / (W.T @ (W @ H) + eps)
    
    return H

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
    def __init__(self, in_channels=1, hidden_channels=32):
        super(Aw_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=0)
        self.relu  = nn.ReLU()
        # We use a sigmoid activation to limit values between 0 and 1
        # To avoid too big updates that could lead to exploding gradients
        self.sigmoid = nn.Sigmoid() 

    # W shape: (f,l)
    def forward(self, x):
        print(f"Aw forward: {x.shape}")
        # x = x.permute(0, 2, 1)#.unsqueeze(1) # (l, 1, f)
        y = self.relu(self.conv1(x))     # (l, 64, f)
        y = self.relu(self.conv2(y))     # (l, 32, f)
        y = self.sigmoid(self.conv3(y))  # (l, 1, f)
        out = y
        # out = y.squeeze(1)#.permute(1, 0) # (f, l)
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
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=1, padding=0)
        self.relu  = nn.ReLU()
        # We use a sigmoid activation to limit values between 0 and 1
        # To avoid too big updates that could lead to exploding gradients
        self.sigmoid = nn.Sigmoid() 

    # H shape: (l,t)
    def forward(self, x):
        print(f"Ah forward: {x.shape}")
        x = x.unsqueeze(1)              # (l, 1, t)
        y = self.relu(self.conv1(x))    # (l, 64, t)
        y = self.relu(self.conv2(y))    # (l, 32, t)
        y = self.relu(self.conv3(y))    # (l, 1, t)
        y = self.sigmoid(y)             # (l, 1, t)
        out = y.squeeze(1)              # (l, t)
        return out   
    
    
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
    
    
class RALMU_block(nn.Module):
    """
    A single layer/iteration of the RALMU model
    updating both W and H
    """
    def __init__(self, f, t, l, shared_aw=None, shared_ah=None):
        super().__init__()
        if shared_aw is not None:
            self.Aw = shared_aw
        else:
            self.Aw = Aw(w_size=f*l)
        if shared_ah is not None:
            self.Ah = shared_ah
        else:
            self.Ah = Ah(h_size=l*t)
            
    def forward(self, M, W, H):
        eps = 1e-4
        
        # Update W W_l+1 = W_l * Aw(W_l) * M.W_l^T/ W_l.H_l.H_l^T
        Aw_out  = self.Aw(W)
        numer_W = M @ H.transpose(-1, -2)
        denom_W = W @ (H @ H.transpose(-1, -2)) + eps
        W_new   = W * Aw_out * (numer_W / denom_W)
        

        # Update H H_l+1 = H_l * Ah(H_l) * W_l+1^T.M/ W_l+1^T.W_l+1.H_l
        Ah_out  = self.Ah(H)
        numer_H = W_new.transpose(-1, -2) @ M
        denom_H = (W_new.transpose(-1, -2) @ W_new) @ H + eps
        
        H_new   = H * Ah_out * (numer_H / denom_H)

        return W_new, H_new
        
        
class RALMU_block2(nn.Module):
    """
    A single layer/iteration of the RALMU model
    with β-divergence multiplicative updates for W and H.
    Aw and Ah are CNNS
    """
    def __init__(self, beta=1, shared_aw=None, shared_ah=None, learnable_beta=False):
        super().__init__()
        if shared_aw is not None:
            self.Aw = shared_aw
        else:
            self.Aw = Aw_cnn()

        if shared_ah is not None:
            self.Ah = shared_ah
        else:
            self.Ah = Ah_cnn()

        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

        self.eps = 1e-6

    def forward(self, M, W, H):
        
        # Add channel dimension
        W = W.unsqueeze(0)  # Shape: (1, f, l)
        H = H.unsqueeze(0)  # Shape: (1, l, t)
        
        print("M: ", M.shape)
        print("W: ", W.shape)
        print("H: ", H.shape)
        
        wh = W @ H + self.eps

        # Compute WH^(β - 2) * V
        wh_pow = wh.pow(self.beta - 2)
        wh_2_m = wh_pow * M

        # Compute WH^(β - 1)
        wh_1 = wh.pow(self.beta - 1)

        # MU for W
        numerator_W = wh_2_m @ H.transpose(-1, -2)
        denominator_W = wh_1 @ H.transpose(-1, -2) + self.eps
        update_W = numerator_W / denominator_W

        # Apply learned transformation (Aw), element-wise multiplication
        # Avoid going to zero by clamping
        W_new = W * self.Aw(W) * update_W
        W_new = torch.clamp(W_new, min=self.eps)

        # Compute WH again with updated W
        wh = W_new @ H + self.eps

        wh_pow = wh.pow(self.beta - 2)
        wh_2_m = wh_pow * M
        wh_1 = wh.pow(self.beta - 1)

        # MU for H
        numerator_H = W_new.transpose(-1, -2) @ wh_2_m
        denominator_H = W_new.transpose(-1, -2) @ wh_1 + self.eps
        update_H = numerator_H / denominator_H

        # Apply learned transformation (Ah), element-wise multiplication
        # Avoid going to zero by clamping
        H_new = H * self.Ah(H) * update_H
        H_new = torch.clamp(H_new, min=self.eps)

        return W_new, H_new


class RALMU(nn.Module):
    
    def __init__(self, f, t, l=88, W=None, H=None, n_iter=10, n_init_steps=5, shared=False, eps=1e-5):
        super().__init__()
        self.f = f
        self.t = t
        self.l = l
        self.n_iter         = n_iter
        self.n_init_steps   = n_init_steps
        self.shared         = shared

        # Optional shared MLPs
        shared_aw = Aw_cnn() if shared else None
        shared_ah = Ah_cnn() if shared else None

        # Unrolling layers
        self.layers = nn.ModuleList([
            RALMU_block2(shared_aw=shared_aw, shared_ah=shared_ah)
            for _ in range(n_iter)
        ])
        
        if W is not None and H is not None:
            self.W0 = self.register_buffer("W0", W.clone())
            self.H0 = self.register_buffer("H0", H.clone())
            print("copied W and H")
        else:
            W0 = torch.rand(f, l) + eps
            H0 = torch.rand(l, t) + eps
            self.register_buffer("W0", W0)
            self.register_buffer("H0", H0)
            print("initialized W and H")

    def forward(self, M):
        W = self.W0
        H = self.H0

        for layer in self.layers:
            W, H = layer(M, W, H)
            
            #     if torch.isnan(W).any() or torch.isinf(W).any():
            #         print("NaNs or Infs detected in W!")
            #     if torch.isnan(H).any() or torch.isinf(H).any():
            #         print("NaNs or Infs detected in H!")

            
            # W = torch.nan_to_num(W, nan=1e-6)
            # H = torch.nan_to_num(H, nan=1e-6)

        M_hat = W @ H

        return W, H, M_hat