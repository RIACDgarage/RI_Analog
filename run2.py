"""
 run policy with state value function approx vpnet
 regression of vpnet to optimal state value optstv
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import pathlib
import pandas as pd
import RIutil as util
from spiceIF2 import spiceIF
from policy import policy
from optstv import optstv
import vptorch
from plotPolicyTraj import plotPolicyTraj

# parameter definition
epsilon_greedy = 0.1
dmin, d1max, d2max, rbias, rweight = 3, 300, 1000, 0.8, 0.0016

# initialize 
vp = vptorch.vpnet().to("cpu") # policy state-value NN
vpModelFile = "vpmodel.pth"
if pathlib.Path(vpModelFile).is_file():
    vp.load_state_dict(torch.load(vpModelFile))
opt0 = optstv("optstv.csv") # prepare the optimal state value function
st0 = [50, 50] # or st0 = util.ranDsn(), for initial state
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt", opt0)
if opt0.lenStv() < 100: util.stvInit("mesh", dmin, d1max, d2max, opt0, e0)
r0 = util.calcReward() # initial reward function
reward = e0.runSpice(st0) # put initial state to environment
n0 = util.normStv(d1max, d2max, rbias, rweight) # init normalization function
bestDesign = [st0, reward]

p0 = policy(epsilon_greedy, dmin, d1max, d2max, vp, n0)

# pyTorch training settings
vp = vptorch.vpnet().to("cpu")
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(vp.parameters(), lr=1e-3)

episode = 200
dplot = np.ones((episode+1,2),dtype=np.int32) # for trajectory plot
dplot[0] = st0
# Do MDP in #episode times
for i in range (episode):
    st1 = p0.action(st0) # agent action
    reward = e0.runSpice(st1) # environment response
    updateFlag = opt0.writeStv(st1, reward) # store optimal state value

    if updateFlag: # new state vaule, do back annotation
        sNN, rNN = n0.norm2NNinit(optstv)
        srDataset = vptorch.stateRewardDataset(sNN, np.array([rNN]).T)
        data_loader = DataLoader(srDataset, batch_size=50, shuffle=True)
        vp.train() # model in training mode
        epochs = 100
        for t in range (epochs):
            print(f"Epoch {t+1}\n--------------------------------")
            vptorch.train(data_loader, vp, loss_fn, optimizer)
        torch.save(vp.state_dict(), vpModelFile)

    st0 = st1 # update state for t+1
    dplot[i+1] = st1

    # define MDP termination criterion
    if isinstance(reward, list): reward = reward[0]
    if reward > bestDesign[1]: bestDesign = [st1, reward]
    if reward > 90:
        print("Design target reached, existing")
        break
    elif i == episode-1:
        print("Episode expired, best design is", bestDesign[0])
        print("best reward is", bestDesign[1])

plotPolicyTraj(dplot)

