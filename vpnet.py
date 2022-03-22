"""
  neural network for policy value function prediction
"""

import tensorflow as tf
import numpy as np
from normStv import normStv
import pathlib
from sklearn.utils import shuffle

class vpnet:
    def __init__(self, vpmodeldir, d1max, d2max):
        self.vpmodeldir = vpmodeldir
        if pathlib.Path(self.vpmodeldir).is_dir(): # model found, use it
            self.vpnet = tf.keras.models.load_model(self.vpmodeldir)
        else:
            self.vpnet = tf.keras.models.Sequential()
            self.vpnet.add(tf.keras.layers.Dense(64, activation='relu', 
                                                 input_shape=(2,)))
            self.vpnet.add(tf.keras.layers.Dense(64, activation='relu'))
            self.vpnet.add(tf.keras.layers.Dense(64, activation='relu'))
            self.vpnet.add(tf.keras.layers.Dense(1))
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.95,
            staircase=True)
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        self.vpnet.compile(optimizer = opt, loss = 'mse')

        self.ns0 = normStv(d1max, d2max)

    def vpIterate(self, optstvFile, epochs):
        input, result = self.ns0.norm2NN(optstvFile)
        input, result = shuffle(input, result)
        h0 = self.vpnet.fit( [input], [result], epochs = epochs, 
                             batch_size = 1024)
        self.vpnet.save(self.vpmodeldir)
        return h0
