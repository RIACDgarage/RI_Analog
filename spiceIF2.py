"""
 spiceIF2.py
 dealing with interface and interaction with ngSpice
"""
import subprocess
import numpy as np
from RIutil import calcReward
from optstv import optstv

class spiceIF:
    def __init__(self, actionFile, spiceFile, spiceOut, optstv):
        self.faction = actionFile
        self.fspice = spiceFile
        self.fspout = spiceOut
        self.opt = optstv
        self.mStr0 = "tdiffpercent" # for parsing spiceOut
        self.mStr1 = "speedperpower"
        self.r0 = calcReward()

    def runSpice(self, design):
        rwd, doSpiceFlag = self.opt.readStv(design)
        if doSpiceFlag:
            print("design not find in History, will do spice simulation")
        else:
            print("existing case in History, obtain result without spice")
            return rwd

        #--- if decided to run spice
        if doSpiceFlag:
            designFile = open(self.faction, "w")
            # write design parameter to file
            # design parameter are converted from action, with unit change
            designFile.writelines([".param", " w0=", str(design[0]*1.8e-7),\
                                   " w1=", str(design[1]*1.8e-7), "\n"])
            designFile.close()

            # interact with environment, the SPICE simulator
            subprocess.run(["ngspice", "-b", "-o", self.fspout, self.fspice])

            # get merit from Spice output
            meritRawFile = open(self.fspout, "rt")
            notfndM0, notfndM1 = True, True
            for line in meritRawFile:
                if self.mStr0 in line:
                    rtemp = line.split("=")
                    mrfs0, notfndM0 = rtemp[1].strip(), False
                elif self.mStr1 in line:
                    rtemp = line.split("=")
                    mrfs1, notfndM1 = rtemp[1].strip(), False
            meritRawFile.close()
            if (notfndM0 or notfndM1):
                print(f"{self.mStr0} or {self.mStr1} not found in \
                {self.fspout}, please check your file")
            else:
                if mrfs0 == 'failed': m0, m1 = 100.0, 0.0
                elif mrfs1 == 'failed': m0, m1 = float(mrfs0), 0.0
                else: m0, m1 = float(mrfs0), float(mrfs1)
            reward = self.r0.calc([m0, m1])
            #--- end of spice operation

            # save new spice run to history file
            self.opt.writeStv(design, reward)
            return reward

