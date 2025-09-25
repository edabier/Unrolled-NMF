import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split
from adasp_data_management import music
import wandb
import argparse

import src.utils as utils
import src.models as models
import src.spectrograms as spec
import src.init as init

"""
This code runs the training of the RALMU model on a dataset based on the metadata.csv file provided
The training can be done from scratch or from a warmup trained model by loading a state_dictionnary

We can choose the structure of the model to train:  
    - n_iter: the amount of unrolled iterations
    - shared: whether the CNNs are shared between unrolled layers or not
    - aw_2d: whether to use a 2D or 1D CNN for Aw
"""

def main(num_workers, n_iter, lr, epochs, batch, length, subset, split, clip):
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    dtype = None
    shared = False
    aw2d = False

    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        dev = "cpu"

    print(f"Start of the script, device = {dev}")

    #########################################################
    print("Loading the dataset...")
    path = "/home/ids/edabier/AMT/Unrolled-NMF/MAPS/metadata.csv"
    metadata = pd.read_csv(path)

    train_data, test_data   = train_test_split(metadata, train_size=split, random_state=1)
    train_data, valid_data  = train_test_split(train_data, train_size=split, random_state=1)
    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    print("Split the dataset - done ✓")
    
    train_set   = utils.LocalDataset(train_data, use_midi=True, fixed_length=length, subset=subset, dtype=dtype)
    valid_set   = utils.LocalDataset(valid_data, use_midi=True, fixed_length=length, subset=subset, dtype=dtype)
    print("Loaded the datasets - done ✓")

    train_sampler   = utils.SequentialBatchSampler(train_set, batch_size=batch)
    collate_fn = utils.create_collate_fn(use_midi=True)
    train_loader    = DataLoader(train_set, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=num_workers)

    valid_sampler   = utils.SequentialBatchSampler(valid_set, batch_size=batch)
    collate_fn = utils.create_collate_fn(use_midi=True)
    valid_loader    = DataLoader(valid_set, batch_sampler=valid_sampler, collate_fn=collate_fn, num_workers=num_workers)
    print("Created the dataloaders - done ✓")

    print(f"Train dataset: {len(train_loader)}, valid dataset: {len(valid_loader)}")

    print("Loading the model...")
    W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    print(f"Clip = {clip}")
    ralmu = models.RALMU(l=88, beta=1, W_path=W_path, hidden=16, n_iter=n_iter, n_init_steps=10, shared=shared, return_layers=False, aw_2d=aw2d, clip_H=clip, dtype=dtype)
    ralmu = ralmu.to(dev)
    
    state_dict = torch.load('/home/ids/edabier/AMT/Unrolled-NMF/models/non_shared_layers_warmup.pt', map_location=dev)
    ralmu.load_state_dict(state_dict['model_state_dict'])
    
    # W0, _, _, _ = init.init_W(ralmu.W_path, normalize_thresh=ralmu.norm_thresh, verbose=ralmu.verbose, dtype=ralmu.dtype)

    ##########################################################
    print("Preparing the training...")
    optimizer   = torch.optim.AdamW(ralmu.parameters(), lr=lr)
    criterion   = nn.MSELoss(reduction='sum')
    
    print("Starting training...")
    
    losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, valid_loader, optimizer, criterion, dev, epochs, save_name="1e-2lr_intermitent_W_clip_10%", use_wandb=True)
    # losses, valid_losses, W_hat, H_hat = utils.train(ralmu, train_loader, valid_loader, optimizer, criterion, dev, epochs, load_check="/home/ids/edabier/AMT/Unrolled-NMF/models/checkpoint.pt", save_name="clip_10%", use_wandb=True)

    print("Training complete!")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--length", default=None, type=float)
    parser.add_argument("--subset", default=1, type=float)
    parser.add_argument("--split", default=0.8, type=float)
    parser.add_argument("--clip", default=0, type=int)
    args = parser.parse_args()

    num_workers = args.num_workers
    n_iter = args.iter
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch
    fixed_length = args.length
    subset = args.subset
    split = args.split
    clip_arg = args.clip
    clip = True if clip_arg == 1 else False
    
    print(f"Starting test_trainer.py with arguments: num_workers={num_workers}, n_iter={n_iter}, lr={lr}, epochs={epochs}, batch_size={batch_size}, fixed_length={fixed_length}, subset={subset} and split={split}")
    
    main(num_workers, n_iter, lr, epochs, batch_size, fixed_length, subset, split, clip)