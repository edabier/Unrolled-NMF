import sys
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from adasp_data_management import music
import wandb
import plotly.graph_objects as go
import pynvml
import argparse

import src.utils as utils
import src.models as models
import src.spectrograms as spec
import src.init as init


if __name__ == '__main__':
    
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        dev = "cpu"
    
    print(f"Start of the script, device = {dev}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--length", default=None, type=float)
    parser.add_argument("--subset", default=1, type=float)
    parser.add_argument("--split", default=0.8, type=float)
    args = parser.parse_args()

    n_iter = args.iter
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    fixed_length = args.length
    subset = args.subset
    split = args.split
    
    #########################################################
    print("Loading the dataset...")
    maps = music.Maps("/tsi/mir/maps")
    metadata = maps.pdf_metadata
    dataset = utils.MapsDataset(metadata, fixed_length=False, subset=0.01, verbose=True, mean_filter=True)
    
    sampler = utils.SequentialBatchSampler(dataset, batch_size)
    data_loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=utils.collate_fn)
    train_loader = []
    valid_loader = []

    for i, batch in enumerate(data_loader):
        if i % 5 == 0:  # Every 5th batch goes to validation
            valid_loader.append(batch)
        else:
            train_loader.append(batch)
    print(f"Train dataset: {len(train_loader)}, valid dataset: {len(valid_loader)}")
    
    ##########################################################
    print("Loading the model...")
    W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    ralmu = models.RALMU(l=88, beta=1, W_path=W_path, n_iter=10, n_init_steps=1, hidden=8, shared=True, return_layers=False, batch_size=batch_size, smaller_A=True)
    
    
    ##########################################################
    print("Preparing the training...")
    optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    criterion   = nn.MSELoss()
    ralmu = ralmu.to(dev)
    
    print("Starting training...")
    
    losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, optimizer, criterion, dev, epochs, valid_loader)
    
    # if np.abs(losses[0]-losses[1]) > 1e2:
    #     losses = losses[1:]
    #     valid_losses = valid_losses[1:]
        
    # plt.plot(losses, label='train loss')
    # plt.plot(valid_losses, label='valid loss')
    # plt.ylabel("MSE")
    # plt.xlabel("epochs")
    # plt.legend()
    
    print("Training complete!")
    pynvml.nvmlShutdown()