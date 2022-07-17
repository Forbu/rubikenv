import pandas as pd
import numpy as np

from rubikenv.rubikgym import rubik_cube
from rubikenv.generate_dataset import generate_full_dataset_history
from rubikenv.models import RubikTransformer
from rubikenv.models import RubikDense

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import torch

from prettytable import PrettyTable

import wandb
import pytorch_lightning as pl

# from pytorch ligntning import wandb logger
from pytorch_lightning.loggers import WandbLogger

import os

import argparse

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def generate_and_create_dataloader(batch_size, nb_epoch, batch_size_dataloader=32):
    """
    1. Generate the data with generate_full_dataset_history 
    2. Create the Tensordataset
    3. Create the Dataloader
    """
    dataset, reward, reverse_action = generate_full_dataset_history(batch_size, nb_epoch)
    dataset = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(reward), torch.from_numpy(reverse_action))
    dataloader = DataLoader(dataset, batch_size=batch_size_dataloader, shuffle=True )
    return dataloader

def train_full(model, nb_epoch_train, batch_size_gen, nb_epoch_generate, batch_size_dataloader=32):
    """
    Here we train the model with the dataloader
    """

    # init wandb logger
    # we read the key string in the scripts/key.txt file
    try:
        with open('/app/scripts/key.txt', 'r') as f:
            key = f.read()
        
        wandb.login(key=key)
        wandb.init(project='rubik_perf', entity='forbu14', settings=wandb.Settings(start_method="thread"))

        wandb_logger = WandbLogger(project="rubik_perf")
    except Exception as e:
        print("error loading the key for wandb : we will not use any logger for the run")
        wandb_logger = None
        print(e)
        print("key :")
        print(key)

    # we train the model
    for epoch_idx in range(nb_epoch_train):
        
        # we init the dataloader
        dataloader = generate_and_create_dataloader(batch_size=batch_size_gen, nb_epoch=nb_epoch_generate, batch_size_dataloader=batch_size_dataloader)

        trainer = pl.Trainer(gpus=1, max_epochs=1, logger=wandb_logger)
        
        trainer.fit(model, dataloader)
        trainer.save_checkpoint("/app/data/checkopoint.ckpt")

    return model

if __name__ == '__main__':

    # here we retrieve the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_epoch_train', type=int, default=100)
    parser.add_argument('--batch_size_gen', type=int, default=12)
    parser.add_argument('--nb_epoch_generate', type=int, default=10000)
    parser.add_argument('--batch_size_dataloader', type=int, default=128)

    args = parser.parse_args()

    model = RubikTransformer(hidden_size=256, num_layers=4, num_heads=8, dropout=0.1, spatial_embedding_size=128, color_embedding_size=128, output_size=12)
    #model = RubikDense(hidden_size=1024, color_embedding_size=5, output_size=12)

    model = train_full(model=model, nb_epoch_train=100, batch_size_gen=15, nb_epoch_generate=10000, batch_size_dataloader=128)

