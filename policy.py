"""
 iterate policy evaluation first
 for two variables design, there is 8 directions for next action
 (more direction is possible, but let's keep it at 8)
 for each direction, we will check N steps farther
 so there will 8N estimations to be done
 best value among the 8N will have (1-eps) chance to be picked
 there is eps chance to pick action at random
"""

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from getReward import getReward
from ranDsn import ranDsn
from getReward import getReward
from spiceIF import spiceIF
from vpnet import vpnet

class policy:
    def __init__(self, eps, dmin, d1max, d2max):
        self.rng = np.random.default_rng()
        self.eps = eps # epsilon greedy
        self.dmin = dmin # minimun size to be taken, was 3
        self.d1max = d1max # maximun size for design1
        self.d2max = d2max # maximum size for design2
        self.r0 = getReward(0.0) # initial 0.0 can be ignored, care value only
        self.greedyFlag = True
        self.e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt")
        self.vp = vpnet("vpnet_model")


    def action(self, sT):
        if self.rng.random() < self.eps:
            self.greedyFlag = False
        else:
            self.greedyFlag = True

        if self.greedyFlag == True:
            # actions to move to 8 adjacent states at t+1
            st1 = np.zeros([8,2], dtype=np.int32)
            merit = np.zeros([8,2], dtype=np.float32)
            rdummy = 0.0
            value = np.zeros(8, dtype=np.float32)
            k = 0
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if (i != 0 or j != 0):
                        st1[k] = np.minimum(np.maximum(np.add(sT, [i,j]), 
                                    [self.dmin, self.dmin]),
                                    [self.d1max, self.d2max])
                        # do real. Switch to approximate latter
                        merit[k] = self.e0.runSpice(st1[k])
                        rdummy, value[k] = self.r0.newReward(merit[k])
                        k = k + 1
            # do policy value function iteration, new value passed in runSpice
            self.vp.vpIterate()
            # greedy action
            approx_value = self.vp.vpnet.predict(st1)
            imax = np.argmax(approx_value)
            return st1[imax]

        if self.greedyFlag == False:
            newDsn = ranDsn(self.dmin, self.d1max, self.d2max)
            return newDsn







