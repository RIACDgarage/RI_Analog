
"""
example file to interact with enviroment of SPICE

The procedure is to set an Action (a design with device parameters), feed into
the SPICE simulator, and get the Reward back from SPICE output.

Action will be store in a file "action.txt", which will be read by SPICE for
its simulation.

Reward will be parsed from SPICE output file of "spiceout.txt".
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from action_random_walk_with_two_nets import ActionRandomWalkWithTwoNets
from reward_calculator import RewardCalculator
from spice_interface import SpiceInterface
from random_design import random_design_function
from utility import Utility

def process_app_ri_analog_with_reward_arguments(parser):
    """
    Process arguments.
    """
    if parser is None:
        raise 'input argument, parser, is None'
    parser.add_argument(
        '--episodes',
        type=int,
        required=False,
        default=200,
        help='reinforcement learning episodes.')
    parser.add_argument(
        '--epsilon',
        type=float,
        required=False,
        default=0.1,
        help='epsilon.')
    parser.add_argument(
        '--dmin',
        type=int,
        required=False,
        default=3,
        help='policy dmin.')
    parser.add_argument(
        '--d1max',
        type=int,
        required=False,
        default=300,
        help='policy d1max.')
    parser.add_argument(
        '--d2max',
        type=int,
        required=False,
        default=1000,
        help='policy d2max.')

def app_ri_analog_with_reward():
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_app_ri_analog_with_reward_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    # ------------------------------------------------------------------------
    # action variables
    N = 5 # range of exploit
    eps = args.epsilon # epsilon greedy, 0 is most greedy
    # design range
    episodes = args.episodes
    dmin = args.dmin
    d1max = args.d1max
    d2max = args.d2max
    # ------------------------------------------------------------------------
    # initial run
    #d0 = [3, 9] # initial guess
    d0 = random_design_function(dmin, d1max, d2max)
    e0 = SpiceInterface(
        Utility.get_intermediate_input_filepath('action.txt'),
        Utility.get_input_filepath('inverter.sp'),
        Utility.get_output_filepath('spiceout.txt')) # spice interface
    r0 = RewardCalculator(0.0)
    a0 = ActionRandomWalkWithTwoNets(N, eps, dmin, d1max, d2max)
    merit_tm1 = e0.run_spice(d0) # merit of t-1
    reward, value = r0.new_reward(merit_tm1)
    bestDesign = [d0, value]
    plt.title("Action Trajectory")
    xmax = d0[0]
    xmin = d0[0]
    ymax = d0[1]
    ymin = d0[1]

    for i in range (episodes):
        design, greedy = a0.new_action(d0)
        merit_t = e0.run_spice(design) # merit at t
        reward, value = r0.new_reward(merit_t)

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
        elif i == episodes-1:
            print("Episode expired, best design is", bestDesign[0])
            print("best value is", bestDesign[1])

    # plot spice result as background
    df = pd.read_csv(Utility.get_output_filepath('spiceHist.csv'))
    df1 = df
    #df1 = df.loc[(df['design1'] <= xmax) & (df['design1'] >= xmin) &
    #             (df['design2'] <= ymax) & (df['design2'] >= ymin)]
    xs = df1["design1"].to_numpy()
    ys = df1["design2"].to_numpy()
    m1 = df1["merit1"].to_numpy()
    m2 = df1["merit2"].to_numpy()
    mlen = len(m1)
    value = np.zeros(mlen)
    r1 = RewardCalculator(0.0)
    for i in range (mlen):
        treward, value[i] = r1.new_reward((m1[i], m2[i]))
    plt.scatter(xs, ys, c=value, cmap='inferno')

    plt.xscale("log")
    plt.yscale("log")
    #plt.xlim([xmin,xmax])
    #plt.ylim([ymin,ymax])
    plt.xlim([dmin,d1max])
    plt.ylim([dmin,d2max])
    plt.grid()
    plt.colorbar()
    plot_filename: str = Utility.get_output_filepath('actionTraj.png')
    plt.savefig(plot_filename)

def main():
    """
    The main() function.
    """
    app_ri_analog_with_reward()

if __name__ == '__main__':
    main()
