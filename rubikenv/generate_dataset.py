import numpy as np
import pandas as pd
import sys
import os

# testing library :=
import unittest
import rubikenv.rubikgym as rb
import argparse
from tqdm import tqdm

def compute_inverse_action(action):
    pos = (action // 4) 
    inv_pos = (action - pos*4)
    inv_pos_new = (inv_pos  + 2) % 4
    
    return pos*4 + inv_pos_new 

def generate_batch_action(batch_size):
    """
    Here we select a random list of action for the batch (batch_size number of action)
    and then we apply those actions to the newly initialize rubik_cube
    We save into a dataset the state of the rubik_cube after each action, the reward, the done flag and reverse action
    """
    # we init the rubik
    rubik_ = rb.rubik_cube()

    # we init the batch of action
    batch_action = np.random.randint(0, 12, batch_size)

    reward = np.zeros(batch_size)
    batch_reverse_action = np.zeros(batch_size)
    batch_state = np.zeros((batch_size, 3, 3, 6))

    # we apply the batch of action to the rubik_cube
    for idx, i in enumerate(batch_action):
        rubik_.move(i)

        batch_reverse_action[idx] = compute_inverse_action(i)
        batch_state[idx] = rubik_.state
        reward[idx] = -idx/10.

    return batch_state, reward, batch_reverse_action

def generate_full_dataset_history(batch_size, nb_epoch):
    """
    Here we generate a dataset of size batch_size * nb_epoch
    """
    dataset = np.zeros((batch_size * nb_epoch, 3, 3, 6))
    reward = np.zeros((batch_size * nb_epoch))
    reverse_action = np.zeros((batch_size * nb_epoch))

    for i in tqdm(range(nb_epoch)):
        batch_state, batch_reward, batch_reverse_action = generate_batch_action(batch_size)
        dataset[i * batch_size:(i + 1) * batch_size] = batch_state
        reward[i * batch_size:(i + 1) * batch_size] = batch_reward
        reverse_action[i * batch_size:(i + 1) * batch_size] = batch_reverse_action

    # here we should make a sampling with a preferance for later action (and low reward)

    # reshape the dataset for (batch_size, 3, 3, 6) to (batch_size, 9, 6)
    dataset = dataset.reshape((batch_size * nb_epoch, 9, 6))

    return dataset, reward, reverse_action

def generate_and_save_dataset(batch_size, nb_epoch, dataset_name):
    """
    Here we generate a dataset of size batch_size * nb_epoch
    and then we save the 3 arrays into a .npz file
    """
    dataset, reward, reverse_action = generate_full_dataset_history(batch_size, nb_epoch)
    np.savez(dataset_name, dataset=dataset, reward=reward, reverse_action=reverse_action)


def generate_full_dataset_history_trainOptim_format(nb_shuffle, nb_epoch):
    
    """
    Very specific procedure to generate a training dataset
    
    1. We generate multiple shuffle of the rubik
    2. In each suffle create a dataset of size (1, 12, 3, 3, 6) (state value) and (1, 12, 1) (reward value)
    """
    
    dataset = np.zeros((nb_epoch, nb_shuffle, 3, 3, 6))
    reward = np.zeros((nb_epoch, nb_shuffle))
    
    for i in tqdm(range(nb_epoch)):
        batch_state, batch_reward, batch_reverse_action = generate_batch_action(nb_shuffle)
        dataset[i, :, :, :, :] = batch_state
        reward[i, :] = batch_reward

    # reshape the dataset for (batch_size, 3, 3, 6) to (batch_size, 9, 6)
    dataset = dataset.reshape((nb_epoch, nb_shuffle, 3*3*6))
    reward = reward.reshape((nb_epoch, nb_shuffle, 1))
    
    return dataset, reward
    
