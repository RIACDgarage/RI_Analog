"""
 spiceIF.py
 dealing with interface and interaction with ngSpice
 2022/Mar/13 add feature to check and store existing simulation data
"""

import subprocess
import numpy as np
import pandas as pd

class spiceIF:
    def __init__(self, paramFile, spiceFile, spiceOut):
        self.fparam = paramFile
        self.fspice = spiceFile
        self.fspout = spiceOut
        self.fdata = "spiceHist.csv"
        self.meritString1 = "tdiffpercent"
        self.meritString2 = "speedperpower"
        self.m0 = 0.0
        self.m1 = 0.0
        self.runSpiceFlag = True

    def runSpice(self, design): # design link to action, merit to reward
        # Read history data of spice run. If the state match, grab result
        spiceHist = pd.read_csv(r'spiceHist.csv')
        d0 = spiceHist.query('(design1==@design[0]) & (design2==@design[1])')
        if d0.empty:
            print("design not find in History, will do spice simulation")
            self.runSpiceFlag = True
        else:
            print("existing case in History, obtain result without spice")
            self.runSpiceFlag = False
            self.m0, self.m1 = d0.to_numpy().flatten()[2:4]

        #--- if decided to run spice
        if self.runSpiceFlag:
            designFile = open(self.fparam, "w")
            # write design parameter to file
            # design parameter are converted from action, with unit change
            designFile.writelines([".param", " w0=", str(design[0]*1.8e-7),\
                                   " w1=", str(design[1]*1.8e-7), "\n"])
            designFile.close()

            # interact with environment, the SPICE simulator
            subprocess.run(["ngspice", "-b", "-o", self.fspout, self.fspice])

            # get merit from Spice output
            meritRawFile = open(self.fspout, "rt")
            for line in meritRawFile:
                if self.meritString1 in line:
                    rtemp = line.split("=")
                    s0 = rtemp[1].strip()
                    if s0 == 'failed':
                        self.m0 = 100.0
                    else:
                        self.m0 = float(s0)
                    # m0 range from 0 to 100. Target is to have m0 <= 3
                elif self.meritString2 in line:
                    rtemp = line.split("=")
                    s0 = rtemp[1].strip()
                    if s0 == 'failed':
                        self.m1 = 0
                    else:
                        self.m1 = float(s0)
                    # Target is to have m1 as large as possible
            meritRawFile.close()
            #--- end of spice operation

            # save new spice run to history file
            nd = np.concatenate((design,(self.m0, self.m1))) #dtype changed
            d1 = pd.DataFrame([nd],columns=['design1','design2',\
                                            'merit1','merit2'])
            d1 = d1.astype({'design1':'int32', 'design2':'int32'})
            spiceHist = pd.concat([spiceHist,d1], ignore_index=True)
            spiceHist.to_csv(self.fdata, index=False)

        return (self.m0, self.m1)

"""
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt")
merit = e0.runSpice([50, 100])
subprocess.run(["cat", "action.txt"])
print("merit=", merit)
"""
