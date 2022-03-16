import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

q1net = tf.keras.models.load_model("q1net_model")
"""
q1net = tf.keras.models.Sequential()
q1net.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)))
q1net.add(tf.keras.layers.Dense(64, activation='relu'))
q1net.add(tf.keras.layers.Dense(1))
"""
#lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#    initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.9)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=10000, decay_rate=0.96,
    staircase=True)
opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
q1net.compile(optimizer = opt, loss = 'mse')

tinput = np.random.random((1,2))
print("before fit=",q1net.predict(tinput))
# see what's inside the un-trained NN
print("weights: ",q1net.get_weights())

# training data from previous spice runs
df = pd.read_csv(r'spiceHist.csv')
# limit largest design, original is 1000
dmax = 200
spiceHist = df.loc[(df['design1'] <= dmax) & (df['design2'] <= dmax)]
input = np.array(spiceHist[['design1', 'design2']],dtype=np.float32)
result = np.array(spiceHist[['merit1']],dtype=np.float32)
# normalize input
input = np.log(input)/np.log(dmax)
result = result/100
dlen = len(input)
len_eval = int(dlen / 5) # how many to separate from training for eval
print("data length=", dlen, "for evaluation=", len_eval)

# shuffle data for each episode
episode = 1
for j in range (episode):
    input, result = shuffle(input, result)
    
    i_train = input[len_eval:]
    i_eval = input[:len_eval]
    r_train = result[len_eval:]
    r_eval = result[:len_eval]

    h0 = q1net.fit([i_train], [r_train], epochs = 2000, batch_size = 512,
                   validation_data = ([i_eval], [r_eval]))

    plt.figure() # new plot for every episode
    #plt.ylim([0.0, 0.4])
    plt.grid(axis='y')
    plt.plot(h0.history['loss'], label="training loss")
    plt.plot(h0.history['val_loss'], label="eval loss")
    plt.legend()
    plotname = "loss" + str(j) + ".png"
    plt.savefig(plotname)

q1net.save("q1net_model")

