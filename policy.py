"""
 iterate policy evaluation
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from ri_utility import RewardCalcutorMyopic
from ri_utility import random_design_function
from ri_utility import action_function_1
from ri_utility import StateValueNormalizer

class Policy:
    def __init__(self, eps, dmin, d1max, d2max, vp, norm):
        self.rng = np.random.default_rng()
        self.eps = eps # epsilon greedy
        self.dmin = dmin # minimun size to be taken, was 3
        self.d1max = d1max # maximun size for design1
        self.d2max = d2max # maximum size for design2
        self.r0 = RewardCalcutorMyopic()
        self.n0 = norm
        self.vp = vp # pytorch model for prediction


    def action(self, sT):
        if self.rng.random() < self.eps: greedyFlag = False
        else:                            greedyFlag = True

        if greedyFlag == True:
            st1 = action_function_1(sT, self.dmin, self.d1max, self.d2max)
            # get q(s')
            stnn1 = torch.tensor(self.n0.normalize_to_neural_network(st1))
            self.vp.eval() # evaluation mode for inferencing
            approx_value = self.vp(stnn1).detach().numpy()
            imax = np.argmax(approx_value)
            # return with the optimal action
            return st1[imax]

        else:
            newDsn = random_design_function(self.dmin, self.d1max, self.d2max)
            return newDsn
