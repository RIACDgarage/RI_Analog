"""
 run policy
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from getReward import getReward
from ranDsn import ranDsn
from spiceIF import spiceIF
from policy import policy
from optstv import optstv
from vpnet import vpnet
from plotPolicyTraj import plotPolicyTraj

# some process parameters
epsilon_greedy = 0.1
dmin = 3
d1max = 300
d2max = 1000

# initial state
st0 = [50, 50]
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt") # spice interface
r0 = getReward(0.0)
merit = e0.runSpice(st0)
dummy, reward = r0.newReward(merit)
bestDesign = [st0, reward]
optstv = optstv("optstv.csv")
vp = vpnet("vpnet_model", d1max, d2max)

p0 = policy(epsilon_greedy, dmin, d1max, d2max)
episode = 200
dplot = np.ones((episode+1,2),dtype=np.int32) # for trajectory plot
dplot[0] = st0
for i in range (episode):    
    st1 = p0.action(st0) # agent action
    merit = e0.runSpice(st1) # environment response
    dummy, reward = r0.newReward(merit) # translate merit to reward
    updateFlag = optstv.writeStv(st1, reward) # store optimal state value

    # do policy value function iteration
    epochs = 100
    if updateFlag: h0 = vp.vpIterate(optstv.optstvFile, epochs)

    st0 = st1 # update state for t+1
    dplot[i+1] = st1

    # define MDP termination criterion
    if reward > bestDesign[1]: bestDesign = [st1, reward]
    if reward > 90:
        print("Design target reached, existing")
        break
    elif i == episode-1:
        print("Episode expired, best design is", bestDesign[0])
        print("best reward is", bestDesign[1])

plotPolicyTraj(dplot)

