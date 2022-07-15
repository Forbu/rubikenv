#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:51:06 2018
@author: adrien
"""
import numpy as np
import pandas as pd

import gym
from gym import spaces

class rubik_cube:
    
    """
    This is a rubik's cube class simulator
    
    Attributes :
        - state : a 3x3x6 array of value between 1 and 6
    
    """
    
    number_of_face = 6
    number_of_element_in_sideface = 3
    
    def __init__(self, init_state=None):
        """
        Initialisation of the rubik
        
        """
        
        # init state initialisation
        if init_state is not None:
            
            init_state = init_state.astype(int)
            self.state = init_state
            self.init_state = np.copy(init_state)
        else:
            # perfect cube
            init_state = np.zeros((self.number_of_element_in_sideface, 
                                   self.number_of_element_in_sideface,self.number_of_face))
            
            for i in range(self.number_of_face):
                init_state[:,:,i] = i
            
            init_state = init_state.astype(int)
                
            self.state = init_state
            self.init_state = np.copy(init_state)
        # other ?
        
    def setInit(self):
        # perfect cube
        init_state = np.zeros((self.number_of_element_in_sideface, 
                               self.number_of_element_in_sideface,self.number_of_face))

        for i in range(self.number_of_face):
            init_state[:, :, i] = i
            
        init_state = init_state.astype(int)
        self.state = init_state
        self.init_state = np.copy(init_state)
        
        
    def move(self,index_move):
        """
        For the convention there is exactly 12 possible moves
        the move are indexed between 0 and 11
        
        the index is in 
        [X Y Z] with
        X : 0 1 2 3  
        Y : 4 5 6 7
        Z : 8 9 10 11
        
        The first two number here are the move corresponding the a certain 
        position on the face.
        
        The two other number at the end are the inverse of those move (the two first)
        
        X Y and Z corresponding to the rotation angle
        """
        value_side = index_move % 2 # entre 0 et 1 the position of the rotation on the face
        value_side_rotation = index_move // 4 # entre 0 et 2 the rotation index of the array
        value_side_inverse = (index_move % 4)//2 # entre 0 et 1 if inverse or not
        
        #print("value_side= ", str(value_side))
        #print("value_side_rotation= ", str(value_side_rotation))
        #print("value_side_inverse= ", str(value_side_inverse))
        
        if value_side == 1:
            value_side = 2 # correction to simplify the calculation 
        
        if value_side_rotation == 0:
            
            # inversion value
            if value_side_inverse == 0:
                self.state[:,value_side,[5,1,4,3]] = self.state[:,value_side,[1,4,3,5]]
                
                if value_side == 0: 
                    self.state[:,:,0] = np.rot90(self.state[:,:,0],k=3)
                else:
                    self.state[:,:,2] = np.rot90(self.state[:,:,2])
                    
            else:
                self.state[:,value_side,[5,1,4,3]] = self.state[:,value_side,[3,5,1,4]]
                
                if value_side == 0: 
                    self.state[:,:,0] = np.rot90(self.state[:,:,0])
                else:
                    self.state[:,:,2] = np.rot90(self.state[:,:,2], k=3)
                

        elif value_side_rotation == 1:
            
            # inversion value
            if value_side_inverse == 0:
                self.state[:,value_side,[5,0,4,2]] = self.state[:,value_side,[0,4,2,5]]
                
                if value_side == 0: 
                    self.state[:,:,1] = np.rot90(self.state[:,:,1],k=3)
                else:
                    self.state[:,:,3] = np.rot90(self.state[:,:,3])
                    
            else:
                self.state[:,value_side,[5,0,4,2]] = self.state[:,value_side,[2,5,0,4]]
                
                if value_side == 0: 
                    self.state[:,:,1] = np.rot90(self.state[:,:,1])
                else:
                    self.state[:,:,3] = np.rot90(self.state[:,:,3], k=3)
                
                
        # TODO again
        elif value_side_rotation == 2:
            
            tmp_state = np.copy(self.state)
            # inversion value
            if value_side_inverse == 0:
                # TODO more complex
                self.state[:,value_side,0] = tmp_state[value_side,:,1][::-1]
                self.state[2-value_side,:,3] = tmp_state[:,value_side,0]
                self.state[:,2-value_side,2] = tmp_state[2-value_side,:,3][::-1]
                self.state[value_side,:,1] = tmp_state[:,2-value_side,2]                
                
                if value_side == 0: 
                    self.state[:,:,4] = np.rot90(self.state[:,:,4],k=3)
                else:
                    self.state[:,:,5] = np.rot90(self.state[:,:,5])
                    
            else:
                
                self.state[value_side,:,1] = tmp_state[:,value_side,0][::-1]
                self.state[:,value_side,0] = tmp_state[2-value_side,:,3]
                self.state[2-value_side,:,3] = tmp_state[:,2-value_side,2][::-1]
                self.state[:,2-value_side,2] = tmp_state[value_side,:,1]
                
                if value_side == 0: 
                    self.state[:,:,4] = np.rot90(self.state[:,:,4])
                else:
                    self.state[:,:,5] = np.rot90(self.state[:,:,5], k=3)        
        
        
    def move_cube(self, index_move,state):
        """
        For the convention there is exactly 12 possible moves
        the move are indexed between 0 and 11
        
        the index is in 
        [X Y Z] with
        X : 0 1 2 3  
        Y : 4 5 6 7
        Z : 8 9 10 11
        
        The first two number here are the move corresponding the a certain 
        position on the face.
        
        The two other number at the end are the inverse of those move (the two first)
        
        X Y and Z corresponding to the rotation angle
        """
        value_side = index_move % 2 # entre 0 et 1 the position of the rotation on the face
        value_side_rotation = index_move // 4 # entre 0 et 2 the rotation index of the array
        value_side_inverse = (index_move % 4)//2 # entre 0 et 1 if inverse or not
        
        #print("value_side= ", str(value_side))
        #print("value_side_rotation= ", str(value_side_rotation))
        #print("value_side_inverse= ", str(value_side_inverse))
        
        if value_side == 1:
            value_side = 2 # correction to simplify the calculation 
        
        if value_side_rotation == 0:
            
            # inversion value
            if value_side_inverse == 0:
                state[:,value_side,[5,1,4,3]] = state[:,value_side,[1,4,3,5]]
                
                if value_side == 0: 
                    state[:,:,0] = np.rot90(state[:,:,0],k=3)
                else:
                    state[:,:,2] = np.rot90(state[:,:,2])
                    
            else:
                state[:,value_side,[5,1,4,3]] = state[:,value_side,[3,5,1,4]]
                
                if value_side == 0: 
                    state[:,:,0] = np.rot90(state[:,:,0])
                else:
                    state[:,:,2] = np.rot90(state[:,:,2], k=3)
                

        elif value_side_rotation == 1:
            
            # inversion value
            if value_side_inverse == 0:
                state[:,value_side,[5,0,4,2]] = state[:,value_side,[0,4,2,5]]
                
                if value_side == 0: 
                    state[:,:,1] = np.rot90(state[:,:,1],k=3)
                else:
                    state[:,:,3] = np.rot90(state[:,:,3])
                    
            else:
                state[:,value_side,[5,0,4,2]] = state[:,value_side,[2,5,0,4]]
                
                if value_side == 0: 
                    state[:,:,1] = np.rot90(state[:,:,1])
                else:
                    state[:,:,3] = np.rot90(state[:,:,3], k=3)
                
                
        # TODO again
        elif value_side_rotation == 2:
            
            tmp_state = np.copy(state)
            # inversion value
            if value_side_inverse == 0:
                # TODO more complex
                state[:,value_side,0] = tmp_state[value_side,:,1][::-1]
                state[2-value_side,:,3] = tmp_state[:,value_side,0]
                state[:,2-value_side,2] = tmp_state[2-value_side,:,3][::-1]
                state[value_side,:,1] = tmp_state[:,2-value_side,2]                
                
                if value_side == 0: 
                    state[:,:,4] = np.rot90(state[:,:,4],k=3)
                else:
                    state[:,:,5] = np.rot90(state[:,:,5])
                    
            else:
                
                state[value_side,:,1] = tmp_state[:,value_side,0][::-1]
                state[:,value_side,0] = tmp_state[2-value_side,:,3]
                state[2-value_side,:,3] = tmp_state[:,2-value_side,2][::-1]
                state[:,2-value_side,2] = tmp_state[value_side,:,1]
                
                if value_side == 0: 
                    state[:,:,4] = np.rot90(state[:,:,4])
                else:
                    state[:,:,5] = np.rot90(state[:,:,5], k=3)
                    
        return state

class rubikgym(gym.Env, rubik_cube):
    reward_range = (-1, 1)
    spec = None

    # Set these in ALL subclasses
    action_space = spaces.Discrete(12)
    
    # flatten discret space
    observation_space = spaces.MultiDiscrete([6 for _ in range(3*3*6)])
    
    def __init__(self):
        gym.Env.__init__(self)
        rubik_cube.__init__(self)
        
    def step(self, action):
        self.move(action)
        return self.state, 0, 0, 0
        
    def reset(self):
        self.setInit(), 0
        
    def render(self, mode='human'):
        print(self.state)
        
    def set_init(self, state):
        self.init_state = state
        self.state = state

    def is_solved(self):
        return np.all(self.state == self.init_state)
        