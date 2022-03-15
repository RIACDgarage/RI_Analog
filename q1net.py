import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#q1net = tf.keras.models.load_model("q1net_model")

q1net = tf.keras.models.Sequential()
q1net.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)))
q1net.add(tf.keras.layers.Dense(8, activation='relu'))
q1net.add(tf.keras.layers.Dense(1))

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    initial_learning_rate=1e-2, decay_steps=400, decay_rate=0.9)
opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
q1net.compile(optimizer = opt, loss = 'mse')

tinput = np.random.random((1,2))
print("before fit=",q1net.predict(tinput))
# see what's inside the un-trained NN
print("weights: ",q1net.get_weights())

# training data from previous spice runs
spiceHist = pd.read_csv(r'spiceHist.csv')
input = np.array(spiceHist[['design1', 'design2']],dtype=np.float32)
result = np.array(spiceHist[['merit1']],dtype=np.float32)
# normalize input
input = np.log(input)/np.log(1000)
result = result/100
dlen = len(input)
print("data length=", dlen)

loss = np.ones(dlen)
step = 0 # for learning rate decay
# shuffle data for each episode
episode = 5
for j in range (episode):
    input, result = shuffle(input, result)
    for i in range (dlen):
        islice = np.array([input[i,:]])
        rslice = np.array([result[i]])
        h0 = q1net.fit([islice], [rslice], epochs = 1)
        lr_schedule(step) # update learning rate
        step = step + 1
        loss[i] = np.array(h0.history['loss'])

    plt.figure() # new plot for every episode
    #plt.ylim([0.0, 0.4])
    plt.grid(axis='y')
    plt.plot(loss)
    plotname = "loss" + str(j) + ".png"
    plt.savefig(plotname)

q1net.save("q1net_model")

