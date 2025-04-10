import torch.nn as nn
import numpy as np
import torch
import utils

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

    def forward(self, x):
        shape       = x.shape
        x           = x.reshape(-1)
        y0          = self.fc0(x)
        y0_relu     = self.relu(y0)
        y1          = self.fc1(y0_relu)
        y1_relu     = self.relu(y1)
        y2          = self.fc2(y1_relu)
        y2_relu     = self.relu(y2)
        out         = y2_relu.view(shape)
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

    def forward(self, x):
        shape       = x.shape
        x           = x.reshape(-1)
        y0          = self.fc0(x)
        y0_relu     = self.relu(y0)
        y1          = self.fc1(y0_relu)
        y1_relu     = self.relu(y1)
        y2          = self.fc2(y1_relu)
        y2_relu     = self.relu(y2)
        out         = y2_relu.view(shape)
        return out
    
    
def MU_iter(M, l, f, t, n_iter):
    # Multiplicative updates iterations
    epsilon = 1e-3
    W = torch.rand(n_iter, f, l) + epsilon
    H = torch.rand(n_iter, l, t) + epsilon

    for l in range(n_iter-1):
        W[l+1] = W[l] * Aw(W[l]) * (M @ H[l].T) / (W[l] @ H[l] @ H[l].T)
        H[l+1] = H[l] * Ah(H[l]) * (W[l+1].T @ M) / (W[l+1].T @ W[l+1] @ H[l])
    
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
    """
    def __init__(self, f, t, l, beta=1, shared_aw=None, shared_ah=None, learnable_beta=False):
        super().__init__()
        if shared_aw is not None:
            self.Aw = shared_aw
        else:
            self.Aw = Aw(w_size=f*l)

        if shared_ah is not None:
            self.Ah = shared_ah
        else:
            self.Ah = Ah(h_size=l*t)

        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.register_buffer("beta", torch.tensor(beta))

        self.eps = 1e-6

    def forward(self, V, W, H):
        
        wh = W @ H + self.eps

        # Compute WH^(β - 2) * V
        wh_pow = wh.pow(self.beta - 2)
        wh_2_v = wh_pow * V

        # Compute WH^(β - 1)
        wh_1 = wh.pow(self.beta - 1)

        # MU for W
        numerator_W = wh_2_v @ H.transpose(-1, -2)
        denominator_W = wh_1 @ H.transpose(-1, -2) + self.eps
        update_W = numerator_W / denominator_W

        # Apply learned transformation (Aw), element-wise multiplication
        # Avoid going to zero by clamping
        W_new = W * self.Aw(W) * update_W
        W_new = torch.clamp(W_new, min=self.eps)

        # Compute WH again with updated W
        wh = W_new @ H + self.eps

        wh_pow = wh.pow(self.beta - 2)
        wh_2_v = wh_pow * V
        wh_1 = wh.pow(self.beta - 1)

        # MU for H
        numerator_H = W_new.transpose(-1, -2) @ wh_2_v
        denominator_H = W_new.transpose(-1, -2) @ wh_1 + self.eps
        update_H = numerator_H / denominator_H

        # Apply learned transformation (Ah), element-wise multiplication
        # Avoid going to zero by clamping
        H_new = H * self.Ah(H) * update_H
        H_new = torch.clamp(H_new, min=self.eps)

        return W_new, H_new


class RALMU(nn.Module):
    
    def __init__(self, f, t, l=88, n_iter=10, shared=False, eps=1e-5, alpha=0.1):
        super().__init__()
        self.f = f
        self.t = t
        self.l = l
        self.n_iter = n_iter
        self.shared = shared

        # Optional shared MLPs
        shared_aw = Aw(w_size=f*l) if shared else None
        shared_ah = Ah(h_size=l*t) if shared else None

        # Unrolling layers
        self.layers = nn.ModuleList([
            RALMU_block2(f, t, l, shared_aw=shared_aw, shared_ah=shared_ah)
            for _ in range(n_iter)
        ])

        self.W0 = nn.Parameter(torch.rand(f, l) * alpha + eps)
        self.H0 = nn.Parameter(torch.rand(l, t) * alpha + eps)
        
        # print("W0 after initialization:")
        # print(self.W0.min(), self.W0.max())
        # print("H0 after initialization:")
        # print(self.H0.min(), self.H0.max())

    def forward(self, M):
        W = self.W0
        H = self.H0

        for layer in self.layers:
            W, H = layer(M, W, H)
            
            if torch.isnan(W).any() or torch.isinf(W).any():
                print("NaNs or Infs detected in W!")
            if torch.isnan(H).any() or torch.isinf(H).any():
                print("NaNs or Infs detected in H!")
            
            W.retain_grad()
            H.retain_grad()
            
            # W = torch.nan_to_num(W, nan=1e-6)
            # H = torch.nan_to_num(H, nan=1e-6)

        M_hat = W @ H

        return W, H, M_hat