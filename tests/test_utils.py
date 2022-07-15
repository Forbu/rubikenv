from rubikenv.rubikgym import rubik_cube
from rubikenv.generate_dataset import generate_full_dataset_history
from rubikenv.models import RubikTransformer
from rubikenv.models import RubikDense

from rubikenv.utils import estimate_solvability_rate
from rubikenv.utils import check_solvable_for_random_shuffle

import unittest

import torch

def test_estimate_solvability_rate():
    """
    we test the estimate_solvability_rate function
    """
    # we init the model
    model_transformer = RubikTransformer()
    model_transformer.eval()

    # we forward pass the state tensor
    estimate_solvability_rate(model_transformer, nb_try=10, batch_size=5)

# launching the test
if __name__ == '__main__':
    unittest.main()