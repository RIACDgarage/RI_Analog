
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
from getAction2 import getAction
from getReward import getReward
from spiceIF import spiceIF

# action variables
N = 5 # range of exploit
eps = 0.0 # epsilon greedy, 0 is most greedy

# design range
dmin = 3
d1max = 300
d2max = 1000

# initial run
d0 = [60, 60] # initial guess
#d0 = [np.random.randint(low=dmin, high=d1max),
#      np.random.randint(low=dmin, high=d2max)]
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt") # spice interface
r0 = getReward(0.0)
a0 = getAction(N, eps, dmin, d1max, d2max)
merit_tm1 = e0.runSpice(d0) # merit of t-1
reward, value = r0.newReward(merit_tm1)
plt.title("Action Trajectory")

for episode in range (100):
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
              color = color, head_width=0.2)
    d0 = design

    print("design=", design)
    print("merit=", merit_t)
    print("reward=", reward)

    if value > 18:
        print("Design target reached, existing")
        break

#plt.xscale("log")
#plt.yscale("log")
#plt.xlim([3,300])
#plt.ylim([3,1000])
plt.grid()
plotname = "actionTraj.png"
plt.savefig(plotname)
