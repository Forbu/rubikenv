import pandas as pd
import numpy as np

from rubikenv.rubikgym import rubik_cube
from rubikenv.generate_dataset import generate_full_dataset_history
from rubikenv.models import RubikTransformer
from rubikenv.models import RubikDense

from torch.utils.data import DataLoader
from torch.utils.data import Dataset, TensorDataset
import torch

import pytorch_lightning as pl

import unittest

def test_generate_and_create_dataloader():
    """
    1. Generate the data with generate_full_dataset_history 
    2. Create the Tensordataset
    3. Create the Dataloader
    """
    dataset, reward, reverse_action = generate_full_dataset_history(batch_size=1, nb_epoch=1)
    dataset = TensorDataset(torch.from_numpy(dataset), torch.from_numpy(reward), torch.from_numpy(reverse_action))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
def test_model():
    """
    we test a forward pass on the two rubik's model with a state input of shape (8, 3, 3, 6) for RubikTransformer and (8, 3, 3, 6) for RubikDense
    """
    # we init the model
    model_transformer = RubikTransformer()
    model_transformer.eval()
    
    # we init the state tensor
    state = torch.randint(0, 6, (8, 9, 6))

    # we forward pass the state tensor
    action_logit, value = model_transformer(state)

    assert action_logit.shape == torch.Size([8, 12])
    assert value.shape == torch.Size([8, 1])

    # we init the model
    model_dense = RubikDense()
    model_dense.eval()

    # we forward pass the state tensor
    action_logit, value = model_dense(state)

    assert action_logit.shape == torch.Size([8, 12])
    assert value.shape == torch.Size([8, 1])

# launching the test
if __name__ == '__main__':
    unittest.main()