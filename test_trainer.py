import sys
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from adasp_data_management import music
import wandb
import plotly.graph_objects as go
import pynvml
import argparse
import tracemalloc
import memory_profiler

import src.utils as utils
import src.models as models
import src.spectrograms as spec
import src.init as init


@profile
def main(num_workers, n_iter, lr, epochs, batch, length, filter, subset, split):
    
    dtype = torch.float16
    shared = True
    clip = True
    aw2d = False

    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        dev = "cpu"

    print(f"Start of the script, device = {dev}")

    #########################################################
    print("Loading the dataset...")
    # maps = music.Maps("/tsi/mir/maps")
    # metadata = maps.pdf_metadata
    path = "/home/ids/edabier/AMT/Unrolled-NMF/MAPS/metadata.csv"
    metadata = pd.read_csv(path)

    train_data, test_data   = train_test_split(metadata, train_size=split, random_state=1)
    train_data, valid_data  = train_test_split(train_data, train_size=split, random_state=1)
    print("Split the dataset - done ✓")
    
    train_set   = utils.LocalMapsDataset(train_data, dev=dev, fixed_length=length, subset=subset, verbose=False, dtype=dtype)
    print("Loaded the train dataset - done ✓")
    test_set    = utils.LocalMapsDataset(test_data, dev=dev, fixed_length=length, subset=subset, verbose=False, dtype=dtype)
    print("Loaded the test dataset - done ✓")
    valid_set   = utils.LocalMapsDataset(valid_data, dev=dev, fixed_length=length, subset=subset, verbose=False, dtype=dtype)
    print("Loaded the valid dataset - done ✓")

    # train_set   = utils.MapsDataset(train_data, use_H=True, fixed_length=length, subset=subset, verbose=False, sort=filter, filter=filter, dtype=dtype)
    # print("Loaded the train dataset - done ✓")
    # test_set    = utils.MapsDataset(test_data, use_H=True, fixed_length=length, subset=subset, verbose=False, sort=filter, filter=filter, dtype=dtype)
    # print("Loaded the test dataset - done ✓")
    # valid_set   = utils.MapsDataset(valid_data, use_H=True, fixed_length=length, subset=subset, verbose=False, sort=filter, filter=filter, dtype=dtype)
    # print("Loaded the valid dataset - done ✓")

    train_sampler   = utils.SequentialBatchSampler(train_set, batch_size=batch)
    train_loader    = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=utils.collate_fn, num_workers=num_workers)
    print(f"Created the train dataloader with {num_workers} workers- done ✓")

    test_sampler   = utils.SequentialBatchSampler(test_set, batch_size=batch)
    test_loader     = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=utils.collate_fn, num_workers=num_workers)
    print("Created the test dataloader - done ✓")

    valid_sampler   = utils.SequentialBatchSampler(valid_set, batch_size=batch)
    valid_loader    = DataLoader(valid_set, batch_sampler=valid_sampler, collate_fn=utils.collate_fn, num_workers=num_workers)
    print("Created the valid dataloader - done ✓")

    print(f"Train dataset: {len(train_loader)}, valid dataset: {len(valid_loader)}")

    print("Loading the model...")
    W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    ralmu = models.RALMU(l=88, beta=1, W_path=W_path, n_iter=n_iter, n_init_steps=1, shared=shared, return_layers=False, aw_2d=aw2d, clip_H=clip, dtype=dtype)
    ralmu = ralmu.cuda(device=dev)

    ##########################################################
    print("Preparing the training...")
    optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    criterion   = nn.MSELoss()

    print("Starting training...")

    losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, valid_loader, optimizer, criterion, dev, epochs)

    print("Training complete!")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--length", default=None, type=float)
    parser.add_argument("--filter", default=False, type=bool)
    parser.add_argument("--subset", default=1, type=float)
    parser.add_argument("--split", default=0.8, type=float)
    args = parser.parse_args()

    num_workers = args.num_workers
    n_iter = args.iter
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    fixed_length = args.length
    filter = args.filter
    subset = args.subset
    split = args.split
    
    main(num_workers, n_iter, lr, epochs, batch_size, fixed_length, filter, subset, split)
    
    # #########################################################
    # print("Loading the dataset...")
    # maps = music.Maps("/tsi/mir/maps")
    # metadata = maps.pdf_metadata

    # train_data, test_data   = train_test_split(metadata, train_size=split, random_state=1)
    # train_data, valid_data  = train_test_split(train_data, train_size=split, random_state=1)

    # dtype = torch.float16

    # train_set   = utils.MapsDataset(train_data, fixed_length=fixed_length, subset=subset, verbose=True, sort=filter, filter=filter, dtype=dtype)
    # test_set    = utils.MapsDataset(test_data, fixed_length=fixed_length, subset=subset, verbose=True, sort=filter, filter=filter, dtype=dtype)
    # valid_set   = utils.MapsDataset(valid_data, fixed_length=fixed_length, subset=subset, verbose=True, sort=filter, filter=filter, dtype=dtype)

    # train_sampler   = utils.SequentialBatchSampler(train_set, batch_size=batch_size)
    # train_loader    = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=utils.collate_fn)

    # test_sampler   = utils.SequentialBatchSampler(test_set, batch_size=batch_size)
    # test_loader     = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=utils.collate_fn)

    # valid_sampler   = utils.SequentialBatchSampler(valid_set, batch_size=batch_size)
    # valid_loader    = DataLoader(valid_set, batch_sampler=valid_sampler, collate_fn=utils.collate_fn)
    # print(f"Train dataset: {len(train_loader)}, valid dataset: {len(valid_loader)}")
    
    # ##########################################################
    # print("Loading the model...")
    # W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    # ralmu = models.RALMU(l=88, beta=1, W_path=W_path, n_iter=10, n_init_steps=1, hidden=8, shared=True, return_layers=False, smaller_A=True, dtype=dtype)
    
    # ##########################################################
    # print("Preparing the training...")
    # optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    # criterion   = nn.MSELoss()
    # ralmu = ralmu.to(dev)
    
    # print("Starting training...")
    
    # losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, valid_loader, optimizer, criterion, dev, epochs)
    
    # # if np.abs(losses[0]-losses[1]) > 1e2:
    # #     losses = losses[1:]
    # #     valid_losses = valid_losses[1:]
        
    # # plt.plot(losses, label='train loss')
    # # plt.plot(valid_losses, label='valid loss')
    # # plt.ylabel("MSE")
    # # plt.xlabel("epochs")
    # # plt.legend()
    
    # print("Training complete!")
    # pynvml.nvmlShutdown()