"""
 spiceIF.py
 dealing with interface and interaction with ngSpice
"""

import subprocess

class spiceIF:
    def __init__(self, paramFile, spiceFile, spiceOut):
        self.fparam = paramFile
        self.fspice = spiceFile
        self.fspout = spiceOut
        self.meritString1 = "tdiffpercent"
        self.meritString2 = "speedperpower"
        self.m0 = 0.0
        self.m1 = 0.0

    def runSpice(self, design): # design link to action, merit to reward
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
                self.m0 = float(rtemp[1].strip())
                # m0 range from 0 to 100. Target is to have m0 <= 3
            elif self.meritString2 in line:
                rtemp = line.split("=")
                self.m1 = float(rtemp[1].strip())
                # Target is to have m1 as large as possible
        meritRawFile.close()
        return (self.m0, self.m1)

"""
e0 = spiceIF("action.txt", "inverter.sp", "spiceout.txt")
merit = e0.runSpice([20, 40])
subprocess.run(["cat", "action.txt"])
print("merit=", merit)
"""
