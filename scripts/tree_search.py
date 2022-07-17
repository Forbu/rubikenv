"""
Script for solving rubik's cube with simple tree search
"""

import pandas as pd
import numpy as np
from rubikenv.rubikgym import rubik_cube
from copy import deepcopy, copy


def create_random_rubik_state(nb_shuffle=10):
    """
    Create a random rubik's cube state
    """
    # init the rubik's cube
    rubik = rubik_cube()

    # choose a list of random action
    action_list = np.random.randint(0, 12, size=nb_shuffle)

    # apply the action to the rubik's cube
    for action in action_list:
        rubik.move(action)


    return rubik

def solve_rubik(nb_shuffle=10, depth_limit=10):
    """
    Function that initialize a random rubik's cube
    then proceed to solve it with a simple tree search (BFS)
    """

    # init the rubik's cube
    rubik = create_random_rubik_state(nb_shuffle)
    rubik.actions_list = [] # we init the list of action

    # we list all the possible action
    action_list = np.arange(12)

    # now we explore the graph of possibilities
    # we init a simple queue of node to explore until we find the solution
    queue = []
    queue.append(rubik)
    while len(queue) > 0:

        # we pop the first element of the queue
        rubik = queue.pop(0)

        # we check if the rubik action list are not too long
        if len(rubik.actions_list) > depth_limit:
            return "not solved"
        # we check if the rubik's cube is solved
        if rubik.is_solved():
            # if it is solved we return the list of action to solve the rubik's cube
            return rubik.actions_list
        # if it is not solved we add the possible action to the queue
        for action in action_list:
            # we create a copy of the rubik's cube
            new_rubik = deepcopy(rubik)
            new_rubik.move(action)

            queue.append(new_rubik)

def ratio_solving(nb_try=100, nb_shuffle=10, depth_limit=10):
    """
    Count the average number of situation where we can solve the rubik's cube using tree search
    """
    nb_solved = 0
    nb_not_solved = 0
    for i in range(nb_try):
        print("try {}".format(i))
        if solve_rubik(nb_shuffle=nb_shuffle, depth_limit=depth_limit) != "not solved":
            nb_solved += 1
        else:
            nb_not_solved += 1
    return nb_solved / (nb_solved + nb_not_solved)


if __name__ == "__main__":
    """
    We run the test to retrieve the average number of situation where we can solve the rubik's cube using tree search
    """
    print(ratio_solving(nb_try=10, nb_shuffle=4, depth_limit=5))



