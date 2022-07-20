"""
Code to reproduce the paper :
Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

https://arxiv.org/pdf/1911.08265v2.pdf

Title of the paper : 
Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

We apply the algorithm to the rubik cube environment
"""

from turtle import back
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
from rubikenv.generate_dataset import generate_full_dataset_history_trainOptim_format
from rubikenv.rubikgym import rubik_cube
from rubikenv.models_searchOptim import RubikTransformer_search
from rubikenv.utils import check_solvable_for_random_shuffle, estimate_solvability_rate_searchOptim

import wandb
import pytorch_lightning as pl

# from pytorch ligntning import wandb logger
from pytorch_lightning.loggers import WandbLogger

from einops import rearrange

import argparse

def generate_and_create_dataloader_searchOptim(batch_size, nb_epoch, batch_size_dataloader=32):
    """
    1. Generate the data with generate_full_dataset_history 
    2. Create the Tensordataset
    3. Create the Dataloader
    """
    dataset, reward = generate_full_dataset_history_trainOptim_format(batch_size, nb_epoch)
    dataset = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(reward))
    dataloader = DataLoader(dataset, batch_size=batch_size_dataloader, shuffle=True)
    return dataloader

def train_init_model(model, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader, wandb_logger):

    # the first training session is done with the "randomize" dataset
    # we train the model
    for _ in range(nb_epoch_train_randomize):
        
        # we init the dataloader
        dataloader = generate_and_create_dataloader_searchOptim(batch_size=batch_size_gen, nb_epoch=nb_epoch_generate,
                                                     batch_size_dataloader=batch_size_dataloader)

        trainer = pl.Trainer(gpus=1, max_epochs=1, logger=wandb_logger)
        
        trainer.fit(model, dataloader)

    return model

def train_model(model, nb_epoch_train, batch_size_gen, nb_epoch_generate, batch_size_dataloader, wandb_logger):
    # TODO later
    pass

def training_model_full(model, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader):
    """
    We train the model using the dataloader and model
    """

    # wandb logger
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


    model_init = train_init_model(model, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader, wandb_logger)

    return model_init

if __name__ == '__main__':

    # here we retrieve the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size_gen', type=int, default=15)
    parser.add_argument('--nb_epoch_generate', type=int, default=10000)
    parser.add_argument('--batch_size_dataloader', type=int, default=16)
    parser.add_argument('--nb_epoch_train_randomize', type=int, default=2)

    # parse arguments to get the value of nb_try and nb_shuffle for evaluation
    parser.add_argument('--nb_try', type=int, default=2)
    parser.add_argument('--nb_shuffle', type=int, default=10)

    args = parser.parse_args()
    

    batch_size_gen = args.batch_size_gen
    nb_epoch_generate = args.nb_epoch_generate
    batch_size_dataloader = args.batch_size_dataloader
    nb_epoch_train_randomize = args.nb_epoch_train_randomize

    model = RubikTransformer_search()

    # model training
    model = training_model_full(model, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader)

    # we evaluate the model
    nb_try = args.nb_try
    nb_shuffle = args.nb_shuffle

    performance = estimate_solvability_rate_searchOptim(model, nb_try=nb_try, nb_shuffle=nb_shuffle)
    print("performance :")
    print(performance)

