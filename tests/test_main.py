import numpy as np
import pandas as pd
import sys
import os

# testing library :=
import unittest
import rubikenv.rubikgym as rb

class TestRubik(unittest.TestCase):

    def setUp(self):
        
        # init the rubik
        self.rubik_ = rb.rubik_cube()

    
    def test_init(self):
        pass
        
    def test_move_all(self):
        self.rubik_ = rb.rubik_cube()
        print("testing all the move")
        
        for i in range(12):
            self.rubik_.move(i)
            print(i)
            unique, counts = np.unique(self.rubik_.state, return_counts=True)
            print(dict(zip(unique, counts)))
        print(self.rubik_.state)

    def test_move(self):
        self.rubik_ = rb.rubik_cube()
        i = 8
        self.rubik_.move(i)
        print('move : ' + str(i))
        unique, counts = np.unique(self.rubik_.state, return_counts=True)
        print(dict(zip(unique, counts)))
        print(self.rubik_.state)

    def test_inverse_ok(self):
        
        # we test for the 6 move that there inverse is corresponding with there real move !
        for i in [0,1,4,5,8,9]:
            self.rubik_ = rb.rubik_cube()
            self.rubik_.move(i)
            self.rubik_.move(i+2)            
            np.testing.assert_array_equal(self.rubik_.state, self.rubik_.init_state)
        
    def test_random(self):
        self.rubik_ = rb.rubik_cube()
        
        for i in np.random.randint(0,12,1000):
            self.rubik_.move(i)
         
        # we check that there is 9 values for each color [0,1,2,3,4,5,6]
        unique, counts = np.unique(self.rubik_.state, return_counts=True)
        info_dict = dict(zip(unique, counts))
        
        for key in info_dict.keys():
            self.assertEqual(info_dict[key], 9)
            
        #print(self.rubik_.state)
        
    def test_init(self):
        self.rubik_ = rb.rubik_cube()
        print(self.rubik_.init_state)
        
    def test_gym(self):
        
        self.rubik = rb.rubikgym()
        self.rubik.step(0)
        
        
        
        
    
    
# launching the test
if __name__ == '__main__':
    unittest.main()