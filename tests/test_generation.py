from rubikenv.generate_dataset import generate_and_save_dataset, generate_full_dataset_history
from rubikenv.generate_dataset import generate_full_dataset_history_trainOptim_format
import unittest

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import torch

import numpy as np

def test_generation():
    """
    Here we generate a dataset of size batch_size * nb_epoch
    """
    batch_size = 10
    nb_epoch = 10
    dataset_name = "/app/data/test_dataset.npz"

    dataloader = generate_and_save_dataset(batch_size, nb_epoch, dataset_name)

def test_dataloader():
    dataset, reward, reverse_action = generate_full_dataset_history(batch_size=15, nb_epoch=3)
    dataset = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(reward), torch.from_numpy(reverse_action))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for i, (state, reward, reverse_action) in enumerate(dataloader):
        break

    assert state.shape == torch.Size([8, 9, 6])
    assert reward.shape == torch.Size([8])
    assert reverse_action.shape == torch.Size([8])

def test_dataloader_searchOptim():
    """
    test for the dataloader of the search optim model
    """

    nb_shuffle = 10
    batch_size = 8
    nb_epoch = 30

    dataset, reward = generate_full_dataset_history_trainOptim_format(nb_shuffle=nb_shuffle, nb_epoch=nb_epoch)

    dataset = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(reward))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, (state, reward) in enumerate(dataloader):
        break

    assert state.shape == torch.Size([batch_size, nb_shuffle, 3*3*6])
    assert reward.shape == torch.Size([batch_size, nb_shuffle, 1])


# launching the test
if __name__ == '__main__':
    unittest.main()