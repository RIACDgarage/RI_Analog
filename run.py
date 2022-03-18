
"""
example file to interact with enviroment of SPICE

The procedure is to set an Action (a design with device parameters), feed into
the SPICE simulator, and get the Reward back from SPICE output.

Action will be store in a file "action.txt", which will be read by SPICE for
its simulation.

Reward will be parsed from SPICE output file of "spiceout.txt".
"""
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from getAction2 import getAction
from getReward import getReward
from spiceIF import spiceIF
from ranDsn import ranDsn

# action variables
N = 5 # range of exploit
eps = 0.1 # epsilon greedy, 0 is most greedy

# design range
dmin = 3
d1max = 300
d2max = 1000

# initial run
#d0 = [3, 9] # initial guess
d0 = ranDsn(dmin, d1max, d2max)
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt") # spice interface
r0 = getReward(0.0)
a0 = getAction(N, eps, dmin, d1max, d2max)
merit_tm1 = e0.runSpice(d0) # merit of t-1
reward, value = r0.newReward(merit_tm1)
bestDesign = [d0, value]
plt.title("Action Trajectory")
xmax = d0[0]
xmin = d0[0]
ymax = d0[1]
ymin = d0[1]

episode = 200
for i in range (episode):
    design, greedy = a0.newAction(d0)
    merit_t = e0.runSpice(design) # merit at t
    reward, value = r0.newReward(merit_t)

    # update t-1 state
    merit_tm1 = merit_t
    if greedy:
        color = 'k'
    else:
        color = 'r'
    plt.arrow(d0[0], d0[1], design[0]-d0[0], design[1]-d0[1], 
              color = color, head_width=2)
    d0 = design

    # record the best design so far
    print("design=", design)
    print("merit=", merit_t)
    print("value=", value)
    if value > bestDesign[1]:
        bestDesign = [design, value]

    # record the space design been reached
    if design[0] > xmax: xmax = design[0]
    if design[0] < xmin: xmin = design[0]
    if design[1] > ymax: ymax = design[1]
    if design[1] < ymin: ymin = design[1]

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
#plt.xlim([xmin,xmax])
#plt.ylim([ymin,ymax])
plt.xlim([dmin,d1max])
plt.ylim([dmin,d2max])
plt.grid()
plt.colorbar()
plotname = "actionTraj.png"
plt.savefig(plotname)
