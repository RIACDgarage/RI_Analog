import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normSpice import normSpice

q1net = tf.keras.models.load_model("q1net_model")
"""
q1net = tf.keras.models.Sequential()
q1net.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)))
q1net.add(tf.keras.layers.Dense(32, activation='relu'))
q1net.add(tf.keras.layers.Dense(32, activation='relu'))
q1net.add(tf.keras.layers.Dense(1))
"""
#lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#    initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.9)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95,
    staircase=True)
opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
q1net.compile(optimizer = opt, loss = 'mse')

spiceData = "spiceHist.csv"
eval_ratio = 5 # ratio of training data for evaluation
d1max = 300 # limit training within certain size
d2max = 1000
d0 = normSpice(spiceData, eval_ratio, d1max, d2max).runNorm()

h0 = q1net.fit( [d0[0]], [d0[1][:,0]], epochs = 2000, batch_size = 1024,
                   validation_data = ([d0[2]], [d0[3][:,0]]) )

plt.figure() # new plot for every episode
#plt.ylim([0.0, 0.4])
plt.grid(axis='y')
plt.plot(h0.history['loss'], label="training loss")
plt.plot(h0.history['val_loss'], label="eval loss")
plt.legend()
plotname = "loss.png"
plt.savefig(plotname)

q1net.save("q1net_model")

