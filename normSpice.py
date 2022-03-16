"""
 Do spice data normalization
 Prepare for training
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
#import tensorflow as tf

class normSpice:
    def __init__(self, spiceDataFile, evalRatio, dmax):
        self.spData = spiceDataFile
        self.evalRatio = evalRatio
        self.scaler = MinMaxScaler()
        # limit largest design, original is 1000
        self.dmax = dmax

    def runNorm(self):
        df = pd.read_csv(self.spData)
        spiceHist = df.loc[(df['design1'] <= self.dmax) & 
                           (df['design2'] <= self.dmax)]
        input = np.array(spiceHist[['design1', 'design2']],dtype=np.float32)
        result = np.array(spiceHist[['merit1']],dtype=np.float32)
        # normalize input in log scale, property of MOS size
        input = np.log(input)/np.log(self.dmax)
        self.scaler.fit(result)
        result = self.scaler.transform(result)
        input, result = shuffle(input, result)
        
        # separate data for training and evaluation
        dlen = len(input)
        len_eval = int(dlen / self.evalRatio)
        print("data length=", dlen, "for evaluation=", len_eval)

        i_train = input[len_eval:]
        i_eval = input[:len_eval]
        r_train = result[len_eval:]
        r_eval = result[:len_eval]

        return (i_train, r_train, i_eval, r_eval)

    def revNorm(self, result):
        return self.scaler.reverse_transform(result)
"""
spiceDataFile = "spiceHist.csv"
d0 = normSpice(spiceDataFile, 5, 200).runNorm()

q1net = tf.keras.models.Sequential()
q1net.add(tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)))
q1net.add(tf.keras.layers.Dense(1))
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
q1net.compile(optimizer = opt, loss = 'mse')

h0 = q1net.fit( [d0[0]], [d0[1]], epochs=1, batch_size=512,
                validation_data = ([d0[2]], [d0[3]]) )
"""
