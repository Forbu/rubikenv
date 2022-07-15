import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import rubikenv.rubikgym as rb


def check_solvable_for_random_shuffle(batch_size=12, model=None):
    """
    In this method we check if the model can solve the Rubik's cube.
    :return:
    """
    # we init the rubik
    rubik_ = rb.rubik_cube()

    # we init the batch of action
    batch_action = np.random.randint(0, 12, batch_size)

    # we apply the batch of action to the rubik_cube
    for idx, i in enumerate(batch_action):
        rubik_.move(i)

    # we get the state of the rubik_cube
    state_gym = rubik_.state

    # now we try to solve the rubik_cube
    # first we convert the state_gym (numpy array) into a tensor
    state_ = torch.from_numpy(state_gym).long()

    # we reshape the state_ to (1, 9, 6)
    state_ = state_.view(1, 9, 6)

    # we pass in inference mode
    with torch.no_grad():

        iteration = 0
        # we recurrently apply the model to the state_ until the state_ is solved
        while not rubik_.is_solved() and iteration <= 30:

            # we get the action logits and value
            action_logits, value = model.forward(state_)

            # we get the action with the highest probability
            action = torch.argmax(action_logits).item()

            # change format of action : simple tensor to float
            action = float(action)

            # we apply the action to the rubik_cube
            rubik_.move(action)

            # we get the state of the rubik_cube
            state_gym = rubik_.state

            # here we check if the state_ is solved
            if rubik_.is_solved():
                return 1

            # now we try to solve the rubik_cube
            # first we convert the state_gym (numpy array) into a tensor
            state_ = torch.from_numpy(state_gym).long()

            # we reshape the state_ to (1, 9, 6)
            state_ = state_.view(1, 9, 6)

            iteration += 1

    return 0

def estimate_solvability_rate(model, nb_try, batch_size):
    """
    In this function we estimate success rate of the model
    :param model:
    :param nb_try:
    :param batch_size:
    :return:
    """
    success = 0
    for i in range(nb_try):
        success += check_solvable_for_random_shuffle(batch_size, model)
    return success / nb_try

