import numpy as np
class getReward:
    def __init__(self, oldR, newR):
        self.oldR = oldR
        self.newR = newR

    def newReward(self):
        if self.newR[0] <= 3: # note newR[0] and oldR[0] are between 0~100
            tdR = 1000
        elif self.oldR[0] > self.newR[0]:
            tdR = self.oldR[0] - self.newR[0]
        else: # unfavour direction to be punished
            tdR = (self.oldR[0] - self.newR[0])*10

        spR = (self.newR[1] - self.oldR[1]) * 1e7
        
        return (tdR, spR)

"""
r0 = getReward([10.0, 1e-6], [12.0, 10e-6])
print(r0)
"""
