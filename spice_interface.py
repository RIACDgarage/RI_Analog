"""
 spice_interface.py
 dealing with interface and interaction with ngSpice
 2022/Mar/13 add feature to check and store existing simulation data
"""

import subprocess
import numpy as np
import pandas as pd

from utility import Utility

class SpiceInterface:
    def __init__(self, paramFile, spiceFile, spiceOut):
        self.fparam = paramFile
        self.fspice = spiceFile
        self.fspout = spiceOut
        self.fdata = Utility.get_output_filepath('spiceHist.csv')
        self.meritString1 = "tdiffpercent"
        self.meritString2 = "speedperpower"
        self.m0 = 0.0
        self.m1 = 0.0
        self.s0 = 'failed'
        self.s1 = 'failed'
        self.run_spiceFlag = True

    def run_spice(self, design): # design link to action, merit to reward
        # Read history data of spice run. If the state match, grab result
        spiceHist = pd.read_csv(self.fdata)
        d0 = spiceHist.query('(design1==@design[0]) & (design2==@design[1])')
        if d0.empty:
            print("design not find in History, will do spice simulation")
            self.run_spice_flag = True
        else:
            print("existing case in History, obtain result without spice")
            self.run_spice_flag = False
            self.m0, self.m1 = d0.to_numpy().flatten()[2:4]

        #--- if decided to run spice
        if self.run_spice_flag:
            designFile = open(self.fparam, "w")
            # write design parameter to file
            # design parameter are converted from action, with unit change
            designFile.writelines([".param", " w0=", str(design[0]*1.8e-7),\
                                   " w1=", str(design[1]*1.8e-7), "\n"])
            designFile.close()

            # interact with environment, the SPICE simulator
            subprocess.run(["\\Spice64\\bin\\ngspice", "-b", "-o", self.fspout, self.fspice])

            # get merit from Spice output
            meritRawFile = open(self.fspout, "rt")
            for line in meritRawFile:
                if self.meritString1 in line:
                    rtemp = line.split("=")
                    self.s0 = rtemp[1].strip()
                elif self.meritString2 in line:
                    rtemp = line.split("=")
                    self.s1 = rtemp[1].strip()
            meritRawFile.close()

            if self.s0 == 'failed':
                self.m0 = 100.0
                self.m1 = 0.0
            elif self.s1 == 'failed':
                self.m0 = float(self.s0)
                self.m1 = 0.0
            else:
                self.m0 = float(self.s0)
                self.m1 = float(self.s1)
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
e0 = SpiceInterface(
    Utility.get_intermediate_input_filepath('action.txt'),
    Utility.get_input_filepath('inverter.sp'),
    Utility.get_output_filepath('spiceout.txt'))
merit = e0.run_spice([50, 100])
subprocess.run(["cat", "action.txt"])
print("merit=", merit)
"""
