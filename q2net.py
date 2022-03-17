import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from normSpice import normSpice

q2net = tf.keras.models.load_model("q2net_model")
"""
q2net = tf.keras.models.Sequential()
q2net.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))
q2net.add(tf.keras.layers.Dense(16, activation='relu'))
q2net.add(tf.keras.layers.Dense(1))
"""
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95)
#lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#    initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95,
#    staircase=True)
opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
q2net.compile(optimizer = opt, loss = 'mse')

spiceData = "spiceHist.csv"
eval_ratio = 5 # ratio of training data for evaluation
d1max = 300 # limit training within certain size
d2max = 1000
d0 = normSpice(spiceData, eval_ratio, d1max, d2max).runNorm()

h0 = q2net.fit( [d0[0]], [d0[1][:,1]], epochs = 10000, batch_size = 1024,
                   validation_data = ([d0[2]], [d0[3][:,1]]) )

plt.figure() # new plot for every episode
#plt.ylim([0.0, 0.4])
plt.grid(axis='y')
plt.plot(h0.history['loss'], label="training loss")
plt.plot(h0.history['val_loss'], label="eval loss")
plt.legend()
plotname = "q2loss.png"
plt.savefig(plotname)

q2net.save("q2net_model")

