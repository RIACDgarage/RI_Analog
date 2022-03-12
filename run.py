
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
from getAction import getAction
from act2design import act2design
from getReward import getReward
from spiceIF import spiceIF

# define action to be performed
d0 = np.array((50,100),dtype=np.int32) # initial guess
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt") # spice interface
merit0 = e0.runSpice(d0)

for episode in range (10):
    action = getAction().newAction()
    design = act2design(action, d0).newDesign()
    merit1 = e0.runSpice(design)
    reward = getReward(merit0, merit1).newReward()
    # update t-1 state
    merit0 = merit1
    d0 = design

    print("action=", action)
    subprocess.run(["cat", "action.txt"])
    print("merid=", merit0)
    print("reward=", reward)
