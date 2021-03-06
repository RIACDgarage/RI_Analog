"""
 for two variables design, there is 8 directions for next action
 (more direction is possible, but let's keep it at 8)
 for each direction, we will check N steps farther
 so there will 8N estimations to be done
 best value among the 8N will have (1-eps) chance to be picked
 there is eps chance to pick action at random
"""

from typing import Any

import numpy as np
import tensorflow as tf
import pandas as pd
from reward_calculator import RewardCalculator
from random_design import random_design_function

from model_q1net import Q1Net
from model_q2net import Q2Net

from utility import Utility

class ActionRandomWalkWithTwoNets:
    def __init__(self, N, eps, dmin, d1max, d2max):
        self.rng = np.random.default_rng()
        self.q1net: Any = Q1Net.load_q1net_model()
        self.q2net: Any = Q2Net.load_q2net_model()
        self.N = N # the probing distance
        self.eps = eps # epsilon greedy
        self.dmin = dmin # minimun size to be taken, was 3
        self.d1max = d1max # maximun size for design1
        self.d2max = d2max # maximum size for design2
        self.r0 = RewardCalculator(0.0) # initial 0.0 can be ignored, care value only
        self.greedyFlag = True

    def new_action(self, oldDesign):
        # 2 cases for random walk: 1st random < eps, 2nd value seems flat
        if self.rng.random() < self.eps: # let's explore
            self.greedyFlag = False
        else:
            self.greedyFlag = True

        if self.greedyFlag == True: # let's use argmax from Q function
            # explore 5 steps from oldDesign
            explr = pd.DataFrame([oldDesign],columns=['design1','design2'])
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    if (i != 0 or j != 0):
                        for k in range (self.N):
                            d0 = oldDesign + np.multiply([i,j], (k+1))
                            df0 = pd.DataFrame([d0],columns=['design1',
                                                             'design2'])
                            explr = pd.concat([explr, df0], ignore_index=True)
            # clean up the potential exceptions
            explr = explr.loc[(explr['design1'] <= self.d1max) &
                              (explr['design2'] <= self.d2max) ]
            explr = explr.loc[(explr['design1'] >= self.dmin) &
                              (explr['design2'] >= self.dmin) ]
            # checking their Q value
            # normalize input first, check scale agree with NN
            input = explr.to_numpy(dtype=np.float32)
            input[:,0] = np.log(input[:,0])/np.log(self.d1max)
            input[:,1] = np.log(input[:,1])/np.log(self.d2max)
            # get prediction from NN
            pred1 = self.q1net.predict(input) * 100
            pred2 = self.q2net.predict(input) * 100
            # calculate reward function
            ilen = len(pred1)
            value = np.zeros(ilen)
            for i in range (ilen):
                rtemp, vtemp = self.r0.new_reward((pred1[i], pred2[i]))
                value[i] = vtemp
            imax = np.argmax(value)
            if imax == 0:
                print("local minimum or flat, do random walk")
                self.greedyFlag = False
            else:
                e0 = explr.to_numpy()
                print("new maximum=", value[imax])
                print("at design ", e0[imax])
                return (e0[imax], self.greedyFlag)

        if self.greedyFlag == False: # let's do random walk
            """
            rints = self.rng.integers(low=-1, high=2, size=2)
            while np.all(rints == 0): # to avoid [0,0] case
                rints = self.rng.integers(low=-1, high=2, size=2)
            aRound = np.divide(oldDesign, 5) 
            if aRound[0] < 10: #if smaller than 10, do 10
                aRound[0] = 10
            if aRound[1] < 10:
                aRound[1] = 10
            aRound = aRound.astype(int)
            newDsn = aRound * rints + oldDesign
            # control newDsn within dmin and d?max
            st3 = newDsn < self.dmin
            if np.any(st3):
                newDsn = newDsn * (~st3) + 3 * st3
            if newDsn[0] > self.d1max: newDsn[0] = self.d1max
            if newDsn[1] > self.d2max: newDsn[1] = self.d2max
            """
            # let's just wrap to another universe
            newDsn = random_design_function(self.dmin, self.d1max, self.d2max)

            return (newDsn, self.greedyFlag)

"""
a0 = ActionRandomWalkWithTwoNets(5, 0.2, 3, 300, 1000)
oldDesign = [50, 50]
for i in range (100):
    new_design = a0.new_action(oldDesign)
    print("new_design=",new_design)
    oldDesign = new_design
"""
