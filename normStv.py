"""
 Do optimal state value normalization for NN
"""
import numpy as np
import pandas as pd

class normStv:
    def __init__(self, d1max, d2max):
        self.d1max = d1max
        self.d2max = d2max

    def norm2NN(self, dataFile):
        df = pd.read_csv(dataFile)
        input = np.array(df[['state0', 'state1']], dtype=np.float32)
        result = np.array(df['reward'], dtype=np.float32)
        # normalize input in log scale
        input[:,0] = np.log(input[:,0])/np.log(self.d1max)
        input[:,1] = np.log(input[:,1])/np.log(self.d2max)
        result = (result+500)/600
        
        return input, result

    def norm2Data(self, state):
        st0 = np.ones((len(state),2), dtype=np.int32)
        st0[:,0] = np.rint(np.exp(np.multiply(state[:,0],np.log(self.d1max))))
        st0[:,1] = np.rint(np.exp(np.multiply(state[:,1],np.log(self.d2max))))

        return st0
