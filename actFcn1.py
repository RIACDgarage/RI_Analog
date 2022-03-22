"""
 action function one
 this action is limited to (-1, 0, +1) of the current state
 return all the possible next states
"""
import numpy as np

def actFcn1(sT, dmin, d1max, d2max): # sT is current state at t
    st1 = np.zeros([8,2], dtype=np.int32) # state array of t+1
    k = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            if (i != 0 or j != 0):
                st1[k] = np.minimum(np.maximum(np.add(sT, [i,j]), 
                                    [dmin, dmin]), [d1max, d2max])
                k = k + 1
    return st1
