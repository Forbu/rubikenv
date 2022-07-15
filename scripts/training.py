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


def generate_and_create_dataloader(batch_size, nb_epoch, batch_size_dataloader=32):
    """
    1. Generate the data with generate_full_dataset_history 
    2. Create the Tensordataset
    3. Create the Dataloader
    """
    dataset, reward, reverse_action = generate_full_dataset_history(batch_size, nb_epoch)
    dataset = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(reward), torch.from_numpy(reverse_action))
    dataloader = DataLoader(dataset, batch_size=batch_size_dataloader, shuffle=True)
    return dataloader

def train_full(nb_epoch_train, batch_size_gen, nb_epoch_generate, batch_size_dataloader=32):
    """
    Here we train the model with the dataloader
    """
    # we init the model
    model = RubikTransformer()

    # init wandb logger
    wandb.login(key="6a69a3d412547caebfe5d3893eac34070d764fe2")
    wandb.init(project='rubik_perf', entity='forbu14', settings=wandb.Settings(start_method="thread"))

    
    # we train the model
    for _ in range(nb_epoch_train):
        
        # we init the dataloader
        dataloader = generate_and_create_dataloader(batch_size=batch_size_gen, nb_epoch=nb_epoch_generate, batch_size_dataloader=batch_size_dataloader)

        trainer = pl.Trainer(gpus=1, max_epochs=1, resume_from_checkpoint="/app/data/checkopoint.ckpt", logger=None)
        trainer.fit(model, dataloader)

        trainer.save_checkpoint("/app/data/checkopoint.ckpt")

    return model

if __name__ == '__main__':

    model = RubikTransformer()

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

    count_parameters(model)

    model = train_full(nb_epoch_train=100, batch_size_gen=15, nb_epoch_generate=10000, batch_size_dataloader=64)
    print(model)
