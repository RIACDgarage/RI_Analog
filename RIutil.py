"""===================================
 collection of various small functions
==================================="""
import numpy as np
import pandas as pd

"""===============================================
 Reward function for myopic process
 calculate reward based on merit values from Spice
==============================================="""
class calcReward:
    def __init__(self):
        self.slope1 = 5.0
        self.slope2 = 1.0/3.0
        self.cutoff = 3.0 # acceptance value for merit1 (smaller)

    def calc(self, merit): # translate Spice merits to reward
        if merit[0] <= self.cutoff:
            reward = merit[1]*5
        else:
            reward = merit[0]*(-self.slope1) + (self.slope1*self.cutoff) 
        
        return [reward]

"""=================================
 Do state value normalization for NN
================================="""
class normStv:
    def __init__(self, d1max, d2max, rbias, rweight):
        self.d1max, self.d2max = d1max, d2max
        self.rb, self.rw = rbias, rweight

    def norm2NNinit(self, optstv):
        state, result = optstv.readNP()
        state[:,0] = np.log(state[:,0])/np.log(self.d1max)
        state[:,1] = np.log(state[:,1])/np.log(self.d2max)
        result = self.rw*result + self.rb
        return state, result

    def norm2NN(self, state):
        st0 = np.array(state, dtype=np.float32)
        st0[:,0] = np.log(st0[:,0])/np.log(self.d1max)
        st0[:,1] = np.log(st0[:,1])/np.log(self.d2max)
        return st0

    def norm2Data(self, state):
        state[:,0] = np.rint(np.exp(np.multiply(state[:,0],np.log(self.d1max))))
        state[:,1] = np.rint(np.exp(np.multiply(state[:,1],np.log(self.d2max))))
        return state

"""==============================================================
 generate a Monte Carlo state with equal probability in log space
=============================================================="""
def ranDsn(dmin, d1max, d2max):
    d0 = np.random.rand()*(np.log(d1max) - np.log(dmin)) + np.log(dmin)
    d0 = int(np.exp(d0))
    d1 = np.random.rand()*(np.log(d2max) - np.log(dmin)) + np.log(dmin)
    d1 = int(np.exp(d1))
    return [d0, d1]

"""======================================================
 At state s, all actions of a, that leads to states s' 
 actFcn1 is limited to (-1, 0, +1) of the current state s
======================================================"""
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

"""==============================================
 building mesh or Monte Carlo states -> get value
 for NN initial training
=============================================="""
def stvInit(style, dmin, d1max, d2max, optstv, spiceIF):
    sample_1d = 20
    samp2d = sample_1d * sample_1d
    if style == "mesh":
        x0 = np.linspace(np.log(dmin), np.log(d1max), num=sample_1d)
        y0 = np.linspace(np.log(dmin), np.log(d2max), num=sample_1d)
        x, y = np.meshgrid(x0 ,y0)
        design = np.exp([x, y]).astype(int).T.reshape(samp2d,2)
    else:
        x = np.random.uniform(low=np.log(dmin), high=np.log(d1max), size=samp2d)
        y = np.random.uniform(low=np.log(dmin), high=np.log(d2max), size=samp2d)
        design = np.exp([x, y]).astype(int).T
    for i in range (samp2d): spiceIF.runSpice(design[i])
