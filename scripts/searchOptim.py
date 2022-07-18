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

import wandb
import pytorch_lightning as pl

# from pytorch ligntning import wandb logger
from pytorch_lightning.loggers import WandbLogger

from einops import rearrange

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

def train_init_model(model, temporal_step, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader, wandb_logger):

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

def inference_model(model, rubik_cube):
    """
    We generate the dataset with an inference model (using the searchOptim procedure)
    """

    # we state from the rubik_cube the state of the environment that we convert into a tensor
    state_init = rubik_cube.state

    # Init the list of states to explore
    states_to_explore = [state_init]

    # init the iterative limit and increment it each time we add a new state to explore
    iter_limit = 100
    iter_increment = 0

    while iter_increment < iter_limit:

        # we evaluate all the state in the state to explore list with the model
        states_to_explore_tensor = torch.tensor(states_to_explore, dtype=torch.float32)
        
        # the states_to_explore_tensor is a tensor of shape (nb_state, 3, 3, 6) of int type we convert it into a tensor of shape (1, nb_state, 3 * 3 * 6)
        states_to_explore_tensor = rearrange(states_to_explore_tensor, "nb_state a b c -> batch nb_state (a b c)", batch=1)

        # we evaluate the model
        values_nodes = model(states_to_explore_tensor)

        # now we can greedly explore the best action
        # we get the index of the best state to explore
        # TODO add randomize element (softmax) to explore randomly sometimes
        index_best_state = torch.argmax(values_nodes[0, :, 0])

        # TODO : cut the list of states to explore to limit the size of the list to 100 (later)
        # if the state have already been explored we do not add it to the list of states to explore

        # check if the resulting state is a terminal state
        if rubik_cube.is_terminal(states_to_explore[index_best_state]):
            return 1

        # now we retrieve the corresponding state to explore
        state_to_explore = states_to_explore[index_best_state]

        # first we add the state to the list of states to explore
        node_toexplore = generate_all_next_states(state_to_explore)
        states_to_explore.append(node_toexplore)

        iter_increment += 1

    return 0


def generate_all_next_states(init_state):
    """
    We generate all possible state that we can reach with one action coming from the init_state
    """
    # 1. new possible states
    states_new = []

    # 2. We generate all the possible actions
    actions = np.aranges(12)

    # 3. We apply the actions to the rubik's cube
    for action in actions:

        rubik_cube = rubik_cube(init_state)
        rubik_cube.move(action)

        states_new.append(rubik_cube.state)

    return states_new

def training_model_full(model, temporal_step, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader):
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


    model_init = train_init_model(model, temporal_step, nb_epoch_train_randomize, batch_size_gen, nb_epoch_generate, batch_size_dataloader, wandb_logger)

