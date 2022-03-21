"""
  neural network for policy value function prediction
"""

import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from normSpice import normSpice
import pathlib
from getReward import getReward

class vpnet:
    def __init__(self, vpmodeldir):
        self.vpmodeldir = vpmodeldir
        if pathlib.Path(self.vpmodeldir).is_dir(): # model found, use it
            self.vpnet = tf.keras.models.load_model(self.vpmodeldir)
        else:
            self.vpnet = tf.keras.models.Sequential()
            self.vpnet.add(tf.keras.layers.Dense(32, activation='relu', 
                                                 input_shape=(2,)))
            self.vpnet.add(tf.keras.layers.Dense(32, activation='relu'))
            self.vpnet.add(tf.keras.layers.Dense(32, activation='relu'))
            self.vpnet.add(tf.keras.layers.Dense(1))
            #lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            #    initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.9)
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95,
                staircase=True)
            opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
            self.vpnet.compile(optimizer = opt, loss = 'mse')
        self.r0 = getReward(0.0)

    def vpIterate(self):
        # still store updated value in spiceHist, to be revised latter
        spiceData = "spiceHist.csv"
        eval_ratio = 5 # ratio of training data for evaluation
        d1max = 300 # limit training within certain size
        d2max = 1000
        d0 = normSpice(spiceData, eval_ratio, d1max, d2max).runNorm()
        lent = len(d0[1])
        value = np.zeros(lent)
        for i in range (lent): dummy, value[i] = self.r0.newReward(d0[1][i,:])
        lenv = len(d0[3])
        vval = np.zeros(lenv)
        for i in range (lenv): dummy, vval[i] = self.r0.newReward(d0[3][i,:])
        h0 = self.vpnet.fit( [d0[0]], [value], epochs = 100, 
                             batch_size = 1024, validation_data = 
                             ([d0[2]], [vval]) )
        self.vpnet.save(self.vpmodeldir)
