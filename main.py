import sys
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import src.utils as utils
import src.models as models
import src.spectrograms as spec
import src.init as init

if __name__ == '__main__':
    lr, epochs, batch_size = sys.argv[0], sys.argv[1],sys.argv[2]
    
    """
    Loading the dataset
    """
    dataset_name    = "piano-dataset"
    dataset         = utils.MaestroNMFDataset(f"{dataset_name}/audios", f"{dataset_name}/midis", fixed_length=False, use_H=False, num_files=None)
    train_size      = 0.8

    train_set, valid_set = torch.utils.data.random_split(dataset, [train_size, 1-train_size])

    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
    
    
    """
    Loading the model
    """
    W_path = 'test-data/synth-single-notes'
    ralmu = models.RALMU(l=88, beta=1, W_path=W_path, n_iter=5, n_init_steps=1, hidden=8, shared=True, return_layers=False)
    utils.model_infos(ralmu, names=False)
    
    
    """
    Running the training
    """
    optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    criterion   = nn.MSELoss()
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ralmu.to(device)
    
    losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, optimizer, criterion, device, epochs, valid_loader)