"""
 handle read/write of optimal state value file
 optstv.csv
"""
import pandas as pd
import numpy as np
import pathlib

class optstv:
    def __init__(self, optstvFile):
        self.optstvFile = optstvFile

        if pathlib.Path(self.optstvFile).is_file():
            self.optstv = pd.read_csv(optstvFile)
        else:
            self.optstv = pd.DataFrame(columns=['state0','state1',
                                                   'reward'])
            self.optstv.to_csv(self.optstvFile)

    def writeStv(self, state, reward):
        dfno = self.optstv.query('(state0==@state[0]) & (state1==@state[1])')
        if dfno.empty:
            nd0 = np.concatenate((state,reward), axis=None)
            df0 = pd.DataFrame([nd0],columns=['state0','state1','reward'])
            df0 = df0.astype({'state0':'int32', 'state1':'int32'})
            self.optstv = pd.concat([self.optstv, df0], ignore_index=True)
            self.optstv.to_csv(self.optstvFile, index=False)
            updateFlag = True
        else: updateFlag = False
        return updateFlag

    def readStv(self, state):
        dfno = self.optstv.query('(state0==@state[0]) & (state1==@state[1])')
        if dfno.empty:
            foundFlag = False
            r0 = 0.0
        else:
            foundFlag = True
            r0 = dfno.to_numpy().flatten()[2]
        return r0, foundFlag        
    
