
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

# define action to be performed
action0 = 1e-5
action1 = 2e-5
action = np.array((action0,action1))

paramFile = "action.txt"
actionFile = open(paramFile, "w")
actionFile.writelines([".param"," w0=",str(action[0])," w1=",str(action[1]),\
                       "\n"])
actionFile.close()

# interact with environment, the SPICE simulator
spiceFile = "inverter.sp"
spiceOutput = "spiceout.txt"
subprocess.run(["ngspice", "-b", "-o", spiceOutput, spiceFile])

# read the reward from environment output
rewardRawFile = open(spiceOutput, "rt")
rewardString1 = "tdiffpercent"
rewardString2 = "speedperpower"
for line in rewardRawFile:
    if rewardString1 in line:
        rtemp = line.split("=")
        reward0 = float(rtemp[1].strip())
        print("reward0=", reward0, ", reward0 must smaller than 3")
    elif rewardString2 in line:
        rtemp = line.split("=")
        reward1 = float(rtemp[1].strip())
        print("reward1=",reward1, ", reward1 larger the better")
rewardRawFile.close()
