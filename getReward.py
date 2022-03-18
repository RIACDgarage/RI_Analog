"""
 Reward function
 calculate reward based on merit values from Spice
"""
import numpy as np

class getReward:
    def __init__(self, oldValue):
        self.oldValue = oldValue # could be initialized from zero
        self.slope1 = 5.0
        self.slope2 = 1.0/3.0
        self.cutoff = 3.0 # accept value for merit1 (smaller)

    def newReward(self, merit):
        if merit[0] <= self.cutoff:
            value = merit[0]*(-self.slope2) + (self.slope2*self.cutoff-1)
        else:
            value = merit[0]*(-self.slope1) + (self.slope1*self.cutoff-1) 
        value = value + merit[1]
        reward = value - self.oldValue
        self.oldValue = value
        
        return [reward, value]
        # reward for the action policy. value for Q function
