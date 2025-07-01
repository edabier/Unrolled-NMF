import sys
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from adasp_data_management import music
import wandb
import argparse

import src.utils as utils
import src.models as models
import src.spectrograms as spec
import src.init as init

if __name__ == '__main__':
    print("Start of the script")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--subset", default=1, type=float)
    parser.add_argument("--split", default=0.8, type=float)
    args = parser.parse_args()

    n_iter = args.iter
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    subset = args.subset
    split = args.split
    
    #########################################################
    print("Loading the dataset...")
    maps = music.Maps("/tsi/mir/maps")
    metadata = maps.pdf_metadata
    dataset = utils.MapsDataset(metadata, fixed_length=True, subset=subset)
    
    train_set, valid_set = torch.utils.data.random_split(dataset, [split, 1-split])   
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    
    train_lengths = [train_set[i][0].shape[1] for i in range(len(train_set))]
    valid_lengths = [valid_set[i][0].shape[1] for i in range(len(valid_set))]

    # Plot the distributions
    plt.figure(figsize=(12, 6))
    plt.hist(train_lengths, bins=30, alpha=0.7, label='Training Set', color='blue')
    plt.hist(valid_lengths, bins=30, alpha=0.7, label='Validation Set', color='orange')
    plt.title('Distribution of Data Lengths')
    plt.xlabel('Length of spec_db_segment')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ##########################################################
    # print("Loading the model...")
    # W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    # ralmu = models.RALMU(l=88, beta=1, W_path=W_path, n_iter=n_iter, n_init_steps=1, hidden=8, shared=True, return_layers=False)
    # utils.model_infos(ralmu, names=False)
    
    
    ##########################################################
    print("Preparing the training...")
    # optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    criterion   = nn.MSELoss()
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # ralmu = ralmu.to(device)
    
    print("Starting training...")
    
    # losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, optimizer, criterion, device, epochs, valid_loader)
    
    # if np.abs(losses[0]-losses[1]) > 1e2:
    #     losses = losses[1:]
    #     valid_losses = valid_losses[1:]
        
    # plt.plot(losses, label='train loss')
    # plt.plot(valid_losses, label='valid loss')
    # plt.ylabel("MSE")
    # plt.xlabel("epochs")
    # plt.legend()
    
    print("Training complete!")