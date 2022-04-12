"""
no policy at the moment. Just random walk.
"""

import numpy as np

class action_random_walk:
    def __init__(self):
        self.rng = np.random.default_rng()

    def new_action(self):
        rints = self.rng.integers(low=-1, high=2, size=2)
        while np.all(rints == 0): # to avoid [0,0] case
            rints = self.rng.integers(low=-1, high=2, size=2)

        return rints
        
"""
for i in range (10):
    an = action_random_walk().new_action()
    print(an)
"""
