import numpy as np

class ActionDesignConverter:
    def __init__(self, act, oldDesign):
        self.act = act
        # act is 2x1 vector will elements of value (-1,0,1)
        self.oldDesign = oldDesign
        # oldDesign is 2x1 vector with int elements range from 3 to 1000

    def new_design(self):
        aRound = np.ceil(self.oldDesign / 10) #if smaller than 1, do 1
        # toDo: need to deal with the state not explored when oldDesign gets 
        #       large
        aRound = aRound.astype(int)
        newDsn = aRound * self.act + self.oldDesign

        # control newDsn with 3 to 1000
        st3 = newDsn < 3
        lt1k = newDsn > 1000
        if np.any(st3):
            newDsn = newDsn * (~st3) + 3 * st3
        if np.any(lt1k):
            newDsn = newDsn * (~lt1k) + 1000 * lt1k

        return newDsn
        
"""
width = np.array([3,1000], dtype=np.int32)
act = np.array([1, -1], dtype=np.int32)
an = ActionDesignConverter(act, width).new_design()
print(an)
"""
