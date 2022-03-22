"""
 iterate policy evaluation
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
from actFcn1 import actFcn1

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
        self.vp = vpnet("vpnet_model", d1max, d2max)


    def action(self, sT):
        if self.rng.random() < self.eps:
            self.greedyFlag = False
        else:
            self.greedyFlag = True

        if self.greedyFlag == True:
            # get all possible s' of t+1
            st1 = actFcn1(sT, self.dmin, self.d1max, self.d2max)
            # get q(s,a)
            approx_value = self.vp.vpnet.predict(st1)
            imax = np.argmax(approx_value)
            # return with the optimal action
            return st1[imax]

        if self.greedyFlag == False:
            newDsn = ranDsn(self.dmin, self.d1max, self.d2max)
            return newDsn







