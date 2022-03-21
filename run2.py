"""
 run policy
"""
import numpy as np
#import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from getReward import getReward
from ranDsn import ranDsn
from getReward import getReward
from spiceIF import spiceIF
from policy import policy

# some process parameters
epsilon_greedy = 0.1
dmin = 3
d1max = 300
d2max = 1000
episode = 200

# initial state
st0 = [50, 50]
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt") # spice interface
r0 = getReward(0.0)
merit = e0.runSpice(st0)
reward, value = r0.newReward(merit)
bestDesign = [st0, value]

# for ploting policy trajactory
plt.title("Action Trajectory")
xmax = st0[0]
xmin = st0[0]
ymax = st0[1]
ymin = st0[1]

p0 = policy(epsilon_greedy, dmin, d1max, d2max)
for i in range (episode):
    st1 = p0.action(st0)
    merit = e0.runSpice(st1) # merit at t
    reward, value = r0.newReward(merit)

    # update t-1 state
    plt.arrow(st0[0], st0[1], st1[0]-st0[0], st1[1]-st0[1], head_width=2)
    st0 = st1

    # record the best design so far
    print("design=", st1)
    print("merit=", merit)
    print("value=", value)
    if value > bestDesign[1]:
        bestDesign = [st1, value]

    # record the space design been reached for plot
    if st1[0] > xmax: xmax = st1[0]
    if st1[0] < xmin: xmin = st1[0]
    if st1[1] > ymax: ymax = st1[1]
    if st1[1] < ymin: ymin = st1[1]

    # termination of process
    if value > 90:
        print("Design target reached, existing")
        break
    elif i == episode-1:
        print("Episode expired, best design is", bestDesign[0])
        print("best value is", bestDesign[1])

# plot spice result as background
df = pd.read_csv('spiceHist.csv')
df1 = df
#df1 = df.loc[(df['design1'] <= xmax) & (df['design1'] >= xmin) &
#             (df['design2'] <= ymax) & (df['design2'] >= ymin)]
xs = df1["design1"].to_numpy()
ys = df1["design2"].to_numpy()
m1 = df1["merit1"].to_numpy()
m2 = df1["merit2"].to_numpy()
mlen = len(m1)
value = np.zeros(mlen)
r1 = getReward(0.0)
for i in range (mlen):
    treward, value[i] = r1.newReward((m1[i], m2[i]))
plt.scatter(xs, ys, c=value, cmap='inferno')

plt.xscale("log")
plt.yscale("log")
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
#plt.xlim([dmin,d1max])
#plt.ylim([dmin,d2max])
plt.grid()
plt.colorbar()
plotname = "policyTraj.png"
plt.savefig(plotname)
