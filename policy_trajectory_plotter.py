"""
 plot policy trajectory
"""

import matplotlib.pyplot as plt

import numpy as np
from numpy import ndarray

import pandas as pd

from utility import Utility

class PolicyTrajectoryPlotter:
    """
    plot policy trajectory
    """
    @staticmethod
    def plot(dplot: ndarray, plot_filename: str = Utility.get_output_filepath('policyTraj.png')):
        plt.title('Policy Trajectory')
        xmax = np.max(dplot[:,0])
        xmin = np.min(dplot[:,0])
        ymax = np.max(dplot[:,0])
        ymin = np.min(dplot[:,0])

        for i in range (len(dplot)-1):
            dx = dplot[i+1, 0] - dplot[i, 0]
            dy = dplot[i+1, 1] - dplot[i, 1]
            plt.arrow(dplot[i,0], dplot[i,1], dx, dy, head_width=2)

        # plot optimal vp as background
        df = pd.read_csv(Utility.get_output_filepath('optstv.csv'))
        df1 = df.loc[(df['state0'] <= xmax) & (df['state0'] >= xmin) &
                     (df['state1'] <= ymax) & (df['state1'] >= ymin)]
        xs = df1['state0'].to_numpy()
        ys = df1['state1'].to_numpy()
        rw = df1['reward'].to_numpy()
        dlen = len(rw)
        plt.scatter(xs, ys, c=rw, cmap='inferno')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])
        plt.grid()
        plt.colorbar()
        plt.savefig(plot_filename)
