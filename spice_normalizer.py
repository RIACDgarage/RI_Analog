"""
 Do spice data normalization
 Prepare for training
"""
import numpy as np
import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
#import tensorflow as tf

class SpiceNormalizer:
    def __init__(self, spiceDataFile, evalRatio, d1max, d2max):
        self.spData = spiceDataFile
        self.evalRatio = evalRatio
        #self.scaler = MinMaxScaler()
        # limit largest design
        self.d1max = d1max
        self.d2max = d2max

    def normalize(self):
        df = pd.read_csv(self.spData)
        spiceHist = df.loc[(df['design1'] <= self.d1max) & 
                           (df['design2'] <= self.d2max)]
        input = np.array(spiceHist[['design1', 'design2']],dtype=np.float32)
        result = np.array(spiceHist[['merit1', 'merit2']],dtype=np.float32)
        # normalize input in log scale, property of MOS size
        input[:,0] = np.log(input[:,0])/np.log(self.d1max)
        input[:,1] = np.log(input[:,1])/np.log(self.d2max)
        #self.scaler.fit(result)
        #result = self.scaler.transform(result)
        result[:,0] = result[:,0]/100.0
        result[:,1] = result[:,1]/100.0
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

    #def revNorm(self, result):
        #return self.scaler.reverse_transform(result)
"""
spiceDataFile = Utility.get_output_filepath('spiceHist.csv')
d0 = SpiceNormalizer(spiceDataFile, 5, 200).normalize()

q1net = tf.keras.models.Sequential()
q1net.add(tf.keras.layers.Dense(4, activation='relu', input_shape=(2,)))
q1net.add(tf.keras.layers.Dense(1))
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
q1net.compile(optimizer = opt, loss = 'mse')

h0 = q1net.fit( [d0[0]], [d0[1]], epochs=1, batch_size=512,
                validation_data = ([d0[2]], [d0[3]]) )
"""
