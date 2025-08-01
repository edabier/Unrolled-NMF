import torch.nn as nn
import torch
import matplotlib.pyplot as plt

import src.utils as utils
import src.spectrograms as spec
import src.init as init

class Aw_cnn(nn.Module):
    """
    Defining a 1D CNN (frequency axis) for Aw()
    Input is a single column of W (shape (288,1))
    1 channel -> 64 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1
    """
    def __init__(self, in_channels=1, hidden_channels=32, dtype=None):
        super(Aw_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2, dtype=dtype)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2, dtype=dtype)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1, dtype=dtype)
        # self.bn1 = nn.BatchNorm1d(hidden_channels*2, dtype=dtype)
        # self.bn2 = nn.BatchNorm1d(hidden_channels, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus()

    # W shape: (f,l)
    def forward(self, x):
        # print(f"Aw in: {x.shape}")
        batch_size, f = x.shape
        x = x.reshape(batch_size, 1, f)             # (batch, 1, f)
        y = self.relu(self.conv1(x))      # (batch, 64, f)
        y = self.relu(self.conv2(y))      # (batch, 32, f)
        y = self.softplus(self.conv3(y))            # (batch, 1, f)
        out = y.reshape(batch_size, f, 1)
        # print(f"Aw out: {out.shape}")
        return out
    

class Aw_2d_cnn(nn.Module):
    """
    Defining a 2D CNN for Aw
    Input is 4 or 12 columns of W (shape (288,4) or (288,12))
    1 channel -> 64 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1    
    """
    def __init__(self, in_channels=1, hidden_channels=2, dtype=None):
        super(Aw_2d_cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels*2, kernel_size=(24*5,5), padding="same", dtype=dtype)
        self.conv2 = nn.Conv2d(hidden_channels*2, hidden_channels, kernel_size=(24*3,3), padding="same", dtype=dtype)
        self.conv3 = nn.Conv2d(hidden_channels, 1, kernel_size=(24*3,3), padding="same", dtype=dtype)
        # self.bn1 = nn.BatchNorm2d(hidden_channels*2, dtype=dtype)
        # self.bn2 = nn.BatchNorm2d(hidden_channels, dtype=dtype)
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
        y = self.softplus(self.conv3(y))            # (batch, 1, f*l)
        out = y.reshape(batch_size, f, l)
        return out

    
class Ah_cnn(nn.Module):
    """
    Defining a 1D CNN (time axis) for Ah()
    Input is a single row of H (shape (1,t))
    1 channel -> 32 ch kernel=5 pad=2 -> 1 ch kernel=3 pad=1
    """
    def __init__(self, in_channels=1, hidden_channels=32, dtype=None):
        super(Ah_cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels*2, kernel_size=5, padding=5 // 2, dtype=dtype)
        self.conv2 = nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=3 // 2, dtype=dtype)
        self.conv3 = nn.Conv1d(hidden_channels, 1, kernel_size=3, padding=1, dtype=dtype)
        # self.bn1 = nn.BatchNorm1d(hidden_channels*2, dtype=dtype)
        # self.bn2 = nn.BatchNorm1d(hidden_channels, dtype=dtype)
        self.relu  = nn.LeakyReLU()
        # We use a softplus activation to force > 0 output
        # and to avoid too big updates that could lead to exploding gradients
        self.softplus = nn.Softplus()

    # H shape: (l,t)
    def forward(self, x):
        # print(f"Ah in: {x.shape}")            # (batch, 1, t)
        if (len(x.shape) == 3):
            batch_size, _, t = x.shape
        else:
            _, t = x.shape
            batch_size = 1
        x = x.reshape(batch_size, 1, t)
        y = self.relu(self.conv1(x))  # (batch, 64, t)
        y = self.relu(self.conv2(y))  # (batch, 32, t)
        y = self.relu(self.conv3(y))            # (batch, 1, t)
        out = self.softplus(y)                  # (batch, 1, t)
        out = out.view(batch_size, 1, t)        # (batch, 1, t)
        # print(f"Ah out: {out.shape}")
        return out   
      
        
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
    """
    def __init__(self, beta=1, learnable_beta=False, shared_aw=None, shared_ah=None, use_ah=True, aw_2d=False, clip_H=False, lambda_w=None, lambda_h=None, dtype=None):
        super().__init__()
        
        self.use_ah     = use_ah
        self.aw_2d      = aw_2d
        
        if dtype is not None:
            self.eps = torch.finfo(type=dtype).min
        else:
            self.eps = torch.finfo().min
            
        self.clip_H     = clip_H 
        self.lambda_w   = lambda_w
        self.lambda_h   = lambda_h
        
        self.Aw = shared_aw if shared_aw is not None else (Aw_2d_cnn(dtype=dtype) if self.aw_2d else Aw_cnn())
        if self.use_ah:
            self.Ah = shared_ah if shared_ah is not None else Ah_cnn(dtype=dtype)
            
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
        
        if len(torch.nonzero(torch.isnan(W[0].view(-1)))) > 0:
            print("Nan in input W")
            spec.vis_cqt_spectrogram(W[0].detach().cpu(), title=f"W with NaNs")
        if len(torch.nonzero(torch.isnan(H[0].view(-1)))) > 0:
            print("Nan in input H")
            spec.vis_cqt_spectrogram(H[0].detach().cpu(), title=f"H with NaNs")
        
        # Use octave splitting to inject only intra-octave notes in Aw
        if self.aw_2d:
            octave_W = [W[:, :, :4]] + [W[:, :, i+4:i+16] for i in range(0, 84, 12)]
            accel_W = torch.cat([self.Aw(W_i)[0] for W_i in octave_W], 1)
            W_new = W * accel_W * update_W
        else: # Inject W column by column in Aw
            accel_W = torch.cat([self.Aw(W[:, :, i]) for i in range(88)], 2)
            W_new = W * accel_W * update_W
            
        if len(torch.nonzero(torch.isnan(accel_W[0].view(-1)))) > 0:
            print("Nan in accel_W")
        
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
        verbose (bool, optional): whether to display some information (default: ``False``)
        norm_thresh (float, optional): whether to normalize W and H (default: ``None``)
    """
    
    def __init__(self, l=88, beta=1, learnable_beta=False, W_path=None, n_iter=10, n_init_steps=100, hidden=32, use_ah=True, shared=False, aw_2d=False, clip_H=False, norm_thresh=0.01, lambda_w=None, lambda_h=None, downsample=False, return_layers=True, dtype=None, verbose=False):
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
        self.downsample     = downsample
        self.verbose        = verbose
        self.norm_thresh    = norm_thresh
        self.return_layers  = return_layers
        self.dtype          = dtype

        shared_aw = (Aw_2d_cnn(hidden_channels=hidden, dtype=dtype) if aw_2d else Aw_cnn(hidden_channels=hidden, dtype=dtype)) if self.shared else None
        if use_ah:
            shared_ah = Ah_cnn(hidden_channels=hidden, dtype=dtype) if self.shared else None
        else:
            shared_ah = None

        # Unrolling layers
        self.layers = nn.ModuleList([
            RALMU_block(shared_aw=shared_aw, shared_ah=shared_ah, use_ah=use_ah, learnable_beta=learnable_beta, aw_2d=aw_2d, clip_H=clip_H, lambda_w=lambda_w, lambda_h=lambda_h, dtype=dtype)
            for _ in range(self.n_iter)
        ])
    
    # @profile
    def forward(self, M, device=None):
        
        batch_size=None
        if len(M.shape) == 3: # Batched input (training phase)
            batch_size, _, t = M.shape
        elif len(M.shape) == 2: # Non-batched input (inference phase)
            _, t = M.shape
        
        normalize = self.norm_thresh is not None
        W, _, _, _ = init.init_W(self.W_path, downsample=self.downsample, normalize_thresh=self.norm_thresh, verbose=self.verbose, dtype=self.dtype)
        if batch_size is not None:
            W = W.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Tracking gpu usage
        gpu_info = utils.get_gpu_info()
        utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        
        H = init.init_H(self.l, t, W, M, n_init_steps=self.n_init_steps, beta=self.beta, device=device, batch_size=batch_size, dtype=self.dtype)

        self.norm = None
        if normalize:
            H, self.norm = spec.l1_norm(H, threshold=self.norm_thresh, set_min=self.eps)
        
        H = torch.clamp(H, min=self.eps)
            
        # Tracking gpu usage
        gpu_info = utils.get_gpu_info()
        utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
        
        if self.return_layers:
            W_layers = []
            H_layers = []

            for i, layer in enumerate(self.layers):
                W, H = layer(M, W, H)
                # Tracking gpu usage
                gpu_info = utils.get_gpu_info()
                utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
                
                W_layers.append(W)
                H_layers.append(H)
                if W is None or H is None:
                    print("W or H got to None")
                if torch.sum(torch.nonzero(torch.isnan(W).view(-1))) > 0 or  torch.sum(torch.nonzero(torch.isnan(H).view(-1))) > 0:
                    print("W or H got nan values")

            M_hats = [W @ H for W, H in zip(W_layers, H_layers)]

            return W_layers, H_layers, M_hats, self.norm
        else:
            for layer in self.layers:
                W, H = layer(M, W, H)
                
                # Tracking gpu usage
                gpu_info = utils.get_gpu_info()
                utils.log_gpu_info(gpu_info, filename="/home/ids/edabier/AMT/Unrolled-NMF/logs/gpu_info_log.csv")
            M_hat = W @ H
            return W, H, M_hat, self.norm