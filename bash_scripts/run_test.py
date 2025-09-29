import pandas as pd
import torch
import torch.nn as nn
import numpy as np
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
This code runs tests of the RALMU model as well as 10 and 10000 iterations of the classic MU algorithm
The tests are carried out on the MAPS and Guitarset dataset from locally saved version (obtained from the `maps_to_cqt.py` and `guitarset_to_cqt.py` files)

For every point of the dataset and for the 3 models, we compute:
    - recall
    - accuracy
    - precision
    - f_mesure
    - inference time
"""

def main(num_workers, n_iter, length, subset, split):
    
    # Set the multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    if torch.cuda.is_available():
        dev = "cuda:0"
        torch.set_default_device(dev)
    else:
        dev = "cpu"
        
    run = wandb.init(
        project=f"Testing_models",
        config={
            "subset": subset,
            "split": split
        },
    )
        
    length=10
    dtype=None
    batch=1
    
    # print("Loading MAPS...")
    # path_maps = "/home/ids/edabier/AMT/Unrolled-NMF/MAPS/metadata.csv"
    # metadata_maps = pd.read_csv(path_maps)
    
    print("Loading Guitarset...")
    path_guitarset = "/home/ids/edabier/AMT/Unrolled-NMF/Guitarset/metadata_full.csv"    
    metadata_guitarset = pd.read_csv(path_guitarset)

    # train_data_maps, test_data_maps   = train_test_split(metadata_maps, train_size=split, random_state=1)
    # test_data_maps = test_data_maps.reset_index(drop=True)
    
    train_data_guitarset, test_data_guitarset   = train_test_split(metadata_guitarset, train_size=split, random_state=1)
    test_data_guitarset = test_data_guitarset.reset_index(drop=True)
    test_set_guitarset   = utils.LocalDataset(test_data_guitarset, use_midi=True, fixed_length=length, subset=subset, dtype=dtype)
    print("Split the datasets - done ✓")

    # test_set_maps   = utils.LocalDataset(test_data_maps, use_midi=True, fixed_length=length, subset=subset, dtype=dtype)
    # print("Loaded the datasets - done ✓")
    
    collate_fn     = utils.create_collate_fn(use_midi=True) 

    # test_sampler_maps   = utils.SequentialBatchSampler(test_set_maps, batch_size=batch)
    # test_loader_maps    = DataLoader(test_set_maps, batch_sampler=test_sampler_maps, collate_fn=collate_fn, num_workers=num_workers)
    
    test_sampler_guitarset   = utils.SequentialBatchSampler(test_set_guitarset, batch_size=batch)
    test_loader_guitarset    = DataLoader(test_set_guitarset, batch_sampler=test_sampler_guitarset, collate_fn=collate_fn, num_workers=num_workers)
    print("Created the dataloaders - done ✓")

    # print("Loading the model...")
    W_path = 'AMT/Unrolled-NMF/test-data/synth-single-notes'
    # ralmu = models.RALMU(l=88, beta=1, W_path=W_path, hidden=16, n_iter=n_iter, n_init_steps=1, shared=True, return_layers=False, aw_2d=False, clip_H=False, dtype=dtype)
    # ralmu = ralmu.to(dev)
    # state_dict = torch.load('/home/ids/edabier/AMT/Unrolled-NMF/models/checkpoint.pt')
    # ralmu.load_state_dict(state_dict['model_state_dict'])
    
    # print("Loading the MU model...")
    mu_10 = models.MU_NMF(n_iter=10, W_path=W_path, beta=1, norm_thresh=0.01)
    mu_10 = mu_10.to(dev)
    
    mu_10k = models.MU_NMF(n_iter=10000, W_path=W_path, beta=1, norm_thresh=0.01)
    mu_10k = mu_10k.to(dev)
    print("Model loaded - done ✓")
    
    criterion = nn.MSELoss()
    
    ############### MAPS 
    
    # metrics_ralmu_maps = utils.test_model(ralmu, test_loader_maps, dev, criterion)
    # df_ralmu_maps = pd.DataFrame.from_dict(metrics_ralmu_maps)
    # df_ralmu_maps.to_csv("/home/ids/edabier/AMT/Unrolled-NMF/metrics_ralmu_maps.csv")
    # # wandb.log({"ralmu_maps_loss": np.mean(df_ralmu_maps["loss"]), "ralmu_maps_inf": np.mean(df_ralmu_maps["inference_time"])})
    # wandb.log({"ralmu_maps_prec": np.mean(df_ralmu_maps["precision"]), "ralmu_maps_acc": np.mean(df_ralmu_maps["accuracy"]), "ralmu_maps_rec": np.mean(df_ralmu_maps["recall"]), "ralmu_maps_f": np.mean(df_ralmu_maps["f_mesure"]), "ralmu_maps_inf": np.mean(df_ralmu_maps["inference_time"])})
    # print("RALMU tested on MAPS - done ✓")
    
    # metrics_mu_10_maps = utils.test_model(mu_10, test_loader_maps, dev, criterion)
    # df_mu_maps = pd.DataFrame.from_dict(metrics_mu_10_maps)
    # df_mu_maps.to_csv("/home/ids/edabier/AMT/Unrolled-NMF/metrics_mu_10_maps.csv")
    # # wandb.log({"mu_10_maps_loss": np.mean(metrics_mu_10_maps["loss"]), "mu_10_maps_inf": np.mean(metrics_mu_10_maps["inference_time"])})
    # wandb.log({"mu_10_maps_prec": np.mean(metrics_mu_10_maps["precision"]), "mu_10_maps_acc": np.mean(metrics_mu_10_maps["accuracy"]), "mu_10_maps_rec": np.mean(metrics_mu_10_maps["recall"]), "mu_10_maps_f": np.mean(metrics_mu_10_maps["f_mesure"]), "mu_10_maps_inf": np.mean(metrics_mu_10_maps["inference_time"])})
    # print("MU 10 tested on MAPS - done ✓")
    
    # metrics_mu_10k_maps = utils.test_model(mu_10k, test_loader_maps, dev, criterion)
    # df_mu_maps = pd.DataFrame.from_dict(metrics_mu_10k_maps)
    # df_mu_maps.to_csv("/home/ids/edabier/AMT/Unrolled-NMF/metrics_mu_10k_maps.csv")
    # # wandb.log({"mu_10k_maps_loss": np.mean(metrics_mu_10k_maps["loss"]), "mu_10k_maps_inf": np.mean(metrics_mu_10k_maps["inference_time"])})
    # wandb.log({"mu_10k_maps_prec": np.mean(metrics_mu_10k_maps["precision"]), "mu_10k_maps_acc": np.mean(metrics_mu_10k_maps["accuracy"]), "mu_10k_maps_rec": np.mean(metrics_mu_10k_maps["recall"]), "mu_10k_maps_f": np.mean(metrics_mu_10k_maps["f_mesure"]), "mu_10k_maps_inf": np.mean(metrics_mu_10k_maps["inference_time"])})
    # print("MU 100 tested on MAPS - done ✓")
    
    ################ GUITARSET #################
    
    # metrics_ralmu_guitarset = utils.test_model(ralmu, test_loader_guitarset, dev)
    # df_ralmu_guitarset = pd.DataFrame.from_dict(metrics_ralmu_guitarset)
    # df_ralmu_guitarset.to_csv("/home/ids/edabier/AMT/Unrolled-NMF/metrics_ralmu_guitarset.csv")
    # # wandb.log({"ralmu_guitarset_loss": np.mean(df_ralmu_guitarset["loss"]), "ralmu_guitarset_inf": np.mean(df_ralmu_guitarset["inference_time"])})
    # wandb.log({"ralmu_guitarset_prec": np.mean(df_ralmu_guitarset["precision"]), "ralmu_guitarset_acc": np.mean(df_ralmu_guitarset["accuracy"]), "ralmu_guitarset_rec": np.mean(df_ralmu_guitarset["recall"]), "ralmu_guitarset_f": np.mean(df_ralmu_guitarset["f_mesure"]), "ralmu_guitarset_inf": np.mean(df_ralmu_guitarset["inference_time"])})
    # print("RALMU tested on guitarset - done ✓")
    
    metrics_mu_10_guitarset = utils.test_model(mu_10, test_loader_guitarset, dev)
    df_mu_guitarset = pd.DataFrame.from_dict(metrics_mu_10_guitarset)
    df_mu_guitarset.to_csv("/home/ids/edabier/AMT/Unrolled-NMF/metrics_mu_10_guitarset.csv")
    # wandb.log({"mu_10_guitarset_loss": np.mean(metrics_mu_10_guitarset["loss"]), "mu_10_guitarset_inf": np.mean(metrics_mu_10_guitarset["inference_time"])})
    wandb.log({"mu_guitarset_10_prec": np.mean(metrics_mu_10_guitarset["precision"]), "mu_10_guitarset_acc": np.mean(metrics_mu_10_guitarset["accuracy"]), "mu_10_guitarset_rec": np.mean(metrics_mu_10_guitarset["recall"]), "mu_10_guitarset_f": np.mean(metrics_mu_10_guitarset["f_mesure"]), "mu_10_guitarset_inf": np.mean(metrics_mu_10_guitarset["inference_time"])})
    print("MU 10 tested on guitarset - done ✓")
    
    metrics_mu_10k_guitarset = utils.test_model(mu_10k, test_loader_guitarset, dev)
    df_mu_guitarset = pd.DataFrame.from_dict(metrics_mu_10k_guitarset)
    df_mu_guitarset.to_csv("/home/ids/edabier/AMT/Unrolled-NMF/metrics_mu_10k_guitarset.csv")
    # wandb.log({"mu_10k_guitarset_loss": np.mean(metrics_mu_10k_guitarset["loss"]), "mu_10k_guitarset_inf": np.mean(metrics_mu_10k_guitarset["inference_time"])})
    wandb.log({"mu_10k_guitarset_prec": np.mean(metrics_mu_10k_guitarset["precision"]), "mu_10k_guitarset_acc": np.mean(metrics_mu_10k_guitarset["accuracy"]), "mu_10k_guitarset_rec": np.mean(metrics_mu_10k_guitarset["recall"]), "mu_10k_guitarset_f": np.mean(metrics_mu_10k_guitarset["f_mesure"]), "mu_10k_guitarset_inf": np.mean(metrics_mu_10k_guitarset["inference_time"])})
    print("MU 10k tested on guitarset - done ✓")
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--iter", default=10, type=int)
    parser.add_argument("--length", default=None, type=float)
    parser.add_argument("--subset", default=1, type=float)
    parser.add_argument("--split", default=0.8, type=float)
    args = parser.parse_args()

    num_workers = args.num_workers
    n_iter = args.iter
    fixed_length = args.length
    subset = args.subset
    split = args.split

    main(num_workers, n_iter, fixed_length, subset, split)
