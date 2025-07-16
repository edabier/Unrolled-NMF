import sys
import matplotlib.pyplot as plt
import torchaudio
import numpy as np
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

import src.utils as utils
import src.models as models
import src.spectrograms as spec
import src.init as init


if __name__ == '__main__':
    
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        print(f"{torch.cuda.is_available()}")
        dev = "cpu"
    
    print(f"Start of the script, device = {dev}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--length", default=None, type=float)
    parser.add_argument("--filter", default=False, type=bool)
    parser.add_argument("--subset", default=1, type=float)
    parser.add_argument("--split", default=0.8, type=float)
    args = parser.parse_args()

    n_iter = args.iter
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    fixed_length = args.length
    filter = args.filter
    subset = args.subset
    split = args.split
    
    snap2 = tracemalloc.take_snapshot()
    
    #########################################################
    print("Loading the dataset...")
    maps = music.Maps("/tsi/mir/maps")
    metadata = maps.pdf_metadata

    train_data, test_data   = train_test_split(metadata, train_size=split, random_state=1)
    train_data, valid_data  = train_test_split(train_data, train_size=split, random_state=1)
    
    snap3 = tracemalloc.take_snapshot()

    dtype = torch.float16

    train_set   = utils.MapsDataset(train_data, fixed_length=fixed_length, subset=subset, verbose=True, sort=filter, filter=filter, dtype=dtype)
    test_set    = utils.MapsDataset(test_data, fixed_length=fixed_length, subset=subset, verbose=True, sort=filter, filter=filter, dtype=dtype)
    valid_set   = utils.MapsDataset(valid_data, fixed_length=fixed_length, subset=subset, verbose=True, sort=filter, filter=filter, dtype=dtype)
    
    snap4 = tracemalloc.take_snapshot()

    train_sampler   = utils.SequentialBatchSampler(train_set, batch_size=batch_size)
    train_loader    = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=utils.collate_fn)
    
    snap5 = tracemalloc.take_snapshot()

    test_sampler   = utils.SequentialBatchSampler(test_set, batch_size=batch_size)
    test_loader     = DataLoader(test_set, batch_sampler=test_sampler, collate_fn=utils.collate_fn)
    
    snap6 = tracemalloc.take_snapshot()

    valid_sampler   = utils.SequentialBatchSampler(valid_set, batch_size=batch_size)
    valid_loader    = DataLoader(valid_set, batch_sampler=valid_sampler, collate_fn=utils.collate_fn)
    
    snap7 = tracemalloc.take_snapshot()
    print(f"Train dataset: {len(train_loader)}, valid dataset: {len(valid_loader)}")
    
    ##########################################################
    print("Loading the model...")
    W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    ralmu = models.RALMU(l=88, beta=1, W_path=W_path, n_iter=10, n_init_steps=1, hidden=8, shared=True, return_layers=False, batch_size=batch_size, smaller_A=True, dtype=dtype)
    
    snap8 = tracemalloc.take_snapshot()
    
    ##########################################################
    print("Preparing the training...")
    optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    criterion   = nn.MSELoss()
    ralmu = ralmu.to(dev)
    
    snap9 = tracemalloc.take_snapshot()
    
    print("Starting training...")
    
    losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, optimizer, criterion, dev, epochs, valid_loader)
    
    snap10 = tracemalloc.take_snapshot()
    tracemalloc.stop()
    
    stats1_2 = snap2.compare_to(snap1, 'lineno')
    stats2_3 = snap3.compare_to(snap2, 'lineno')
    stats3_4 = snap4.compare_to(snap3, 'lineno')
    stats4_5 = snap5.compare_to(snap4, 'lineno')
    stats5_6 = snap6.compare_to(snap5, 'lineno')
    stats6_7 = snap7.compare_to(snap6, 'lineno')
    stats7_8 = snap8.compare_to(snap7, 'lineno')
    stats8_9 = snap9.compare_to(snap8, 'lineno')
    stats9_10 = snap10.compare_to(snap9, 'lineno')

    # Save the comparison results to a text file
    with open('/home/ids/edabier/AMT/Unrolled-NMF/models/memory_leak_analysis.txt', 'w') as f:
        f.write("[ Memory usage increase from snapshot 1 to snapshot 2 ]\n")
        for stat in stats1_2[:10]:
            f.write(f"{stat}\n")

        f.write("\n[ Memory usage increase from snapshot 2 to snapshot 3 ]\n")
        for stat in stats2_3[:10]:
            f.write(f"{stat}\n")
            
        f.write("[ Memory usage increase from snapshot 3 to snapshot 4 ]\n")
        for stat in stats3_4[:10]:
            f.write(f"{stat}\n")

        f.write("\n[ Memory usage increase from snapshot 4 to snapshot 5 ]\n")
        for stat in stats4_5[:10]:
            f.write(f"{stat}\n")
            
        f.write("[ Memory usage increase from snapshot 5 to snapshot 6 ]\n")
        for stat in stats5_6[:10]:
            f.write(f"{stat}\n")

        f.write("\n[ Memory usage increase from snapshot 6 to snapshot 7 ]\n")
        for stat in stats6_7[:10]:
            f.write(f"{stat}\n")
            
        f.write("[ Memory usage increase from snapshot 7 to snapshot 8 ]\n")
        for stat in stats7_8[:10]:
            f.write(f"{stat}\n")

        f.write("\n[ Memory usage increase from snapshot 8 to snapshot 9 ]\n")
        for stat in stats8_9[:10]:
            f.write(f"{stat}\n")

        f.write("\n[ Memory usage increase from snapshot 9 to snapshot 10 ]\n")
        for stat in stats9_10[:10]:
            f.write(f"{stat}\n")

        # Detailed traceback for the top memory consumers
        f.write("\n[ Detailed traceback for the top memory consumers ]\n")
        for stat in stats9_10[:1]:
            f.write('\n'.join(stat.traceback.format()) + '\n')

    print("Memory leak analysis saved to 'memory_leak_analysis.txt'")
    
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