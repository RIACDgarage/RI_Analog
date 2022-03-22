"""
 Run preliminary training for state vaule function NN,
 to improve convergency in the policy
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from getReward import getReward
from optstv import optstv
from vpnet import vpnet
import pathlib

r0 = getReward(0.0)
dmin = 3
d1max = 300
d2max = 1000

optstv = optstv("optstv.csv")
vp = vpnet("vpnet_model", d1max, d2max)

# read existing simulation result
spiceHist = pd.read_csv('spiceHist.csv')
nr0 = spiceHist.to_numpy()
dlen = len(nr0)
reward = np.zeros(dlen)
for i in range (dlen):
    dummy, reward[i] = r0.newReward(nr0[i][2:4])

# convert to optstv data
# tedious file IO. Let's do this in bulk on a new optstv.csv file next time
"""
count = 0
for i in range (dlen):
    if optstv.writeStv(nr0[i,0:2], reward[i]): # tedious file IO
        count = count + 1
print("add data to optstv =", count)
"""
# do vp iteration
epochs = 10000
h0 = vp.vpIterate(optstv.optstvFile, epochs)
