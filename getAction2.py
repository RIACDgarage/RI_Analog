"""
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

class getAction:
    def __init__(self, N, eps, dmin, d1max, d2max):
        self.rng = np.random.default_rng()
        self.q1net = tf.keras.models.load_model("q1net_model")
        self.q2net = tf.keras.models.load_model("q2net_model")
        self.N = N # the probing distance
        self.eps = eps # epsilon greedy
        self.dmin = dmin # minimun size to be taken, was 3
        self.d1max = d1max # maximun size for design1
        self.d2max = d2max # maximum size for design2

    def newAction(self, oldDesign):

        if self.rng.random() < self.eps: # let's explore
            print("explore")
            rints = self.rng.integers(low=-1, high=2, size=2)
            while np.all(rints == 0): # to avoid [0,0] case
                rints = self.rng.integers(low=-1, high=2, size=2)
            aRound = np.divide(oldDesign, 10) 
            if aRound[0] < 1: #if smaller than 1, do 1
                aRound[0] = 1
            if aRound[1] < 1:
                aRound[1] = 1
            aRound = aRound.astype(int)
            newDsn = aRound * rints + oldDesign
            # control newDsn with 3 to 1000
            st3 = newDsn < 3
            lt1k = newDsn > 1000
            if np.any(st3):
                newDsn = newDsn * (~st3) + 3 * st3
            if np.any(lt1k):
                newDsn = newDsn * (~lt1k) + 1000 * lt1k
            if newDsn[0] > 300:
                newDsn[0] = 300
            return newDsn

        else: # let's use argmax from Q function

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
            # normalize input in log scale, property of MOS size
            input = explr.to_numpy(dtype=np.float32)
            input[:,0] = np.log(input[:,0])/np.log(self.d1max)
            input[:,1] = np.log(input[:,1])/np.log(self.d2max)

            pred1 = self.q1net.predict(input) * 100
            imin = np.argmin(pred1)
            e0 = explr.to_numpy()
            if imin == 0:
                print("local minimum arrived, wrap to the other side of universe")
                return [self.rng.integers(low=self.dmin, high=self.d1max),
                        self.rng.integers(low=self.dmin, high=self.d2max)]
            else:
                print("new minimum=", pred1[imin])
                print("at design ", e0[imin])
                return e0[imin]

"""
a0 = getAction(5, 0.2, 3, 300, 1000)
oldDesign = [50, 50]
for i in range (100):
    newDesign = a0.newAction(oldDesign)
    print("newDesign=",newDesign)
    oldDesign = newDesign
"""
