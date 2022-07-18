import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

import rubikenv.rubikgym as rb
from einops import rearrange


def check_solvable_for_random_shuffle(batch_size=12, model=None, device=torch.device("cpu")):
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

    # conversion
    model.to(device)

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

def estimate_solvability_rate(model, nb_try, batch_size, device=torch.device("cpu")):
    """
    In this function we estimate success rate of the model
    :param model:
    :param nb_try:
    :param batch_size:
    :return:
    """
    success = 0
    for i in range(nb_try):
        success += check_solvable_for_random_shuffle(batch_size, model, device)
    return success / nb_try



def inference_model(model, rubik_cube):
    """
    We generate the dataset with an inference model (using the searchOptim procedure)
    """

    # we state from the rubik_cube the state of the environment that we convert into a tensor
    state_init = rubik_cube.state

    # Init the list of states to explore
    states_to_explore = [state_init]

    # init the iterative limit and increment it each time we add a new state to explore
    iter_limit = 50
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
        state_selected = states_to_explore[index_best_state]
        
        rubik_cube_selected = rubik_cube(state_selected)
        if rubik_cube_selected.is_solved():
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

def randomize_rubik_cube(nb_shuffle):
    """
    We return a randomize rubik's cube
    """
    # we init the rubik
    rubik_ = rb.rubik_cube()
    # we init the batch of action
    batch_action = np.random.randint(0, 12, nb_shuffle)
    # we apply the batch of action to the rubik_cube
    for idx, i in enumerate(batch_action):
        rubik_.move(i)
    return rubik_

def estimate_solvability_rate_searchOptim(model, nb_try, device=torch.device("cpu"), nb_shuffle=12):
    """
    In this function we estimate success rate of the model
    :param model:
    :param nb_try:
    :param nb_shuffle:
    :return:
    """
    model = model.to(device)
    success = 0
    for i in range(nb_try):
        rubik_cube = randomize_rubik_cube(nb_shuffle=nb_shuffle)
        success += inference_model(model, rubik_cube)
    return success / nb_try