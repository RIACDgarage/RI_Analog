"""
 Run preliminary training for state vaule function NN,
 to improve convergency in the policy
"""

import argparse

import numpy as np
import tensorflow as tf
import pandas as pd
from reward_calculator import RewardCalculator
from optimal_state_value import OptimalStateValue
from model_vpnet_tensorflow import VpnetModelTensorflow

from utility import Utility

def process_app_train_vpnet_model_arguments(parser):
    """
    Process arguments.
    """
    if parser is None:
        raise 'input argument, parser, is None'
    parser.add_argument(
        '--epochs',
        type=int,
        required=False,
        default=10000,
        help='reinforcement learning epochs.')
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

def app_train_vpnet_model():
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_app_train_vpnet_model_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    # ------------------------------------------------------------------------
    # parameter definition
    epochs = args.epochs
    dmin = args.dmin
    d1max = args.d1max
    d2max = args.d2max
    r0 = RewardCalculator(0.0)

    optstv = OptimalStateValue(Utility.get_output_filepath('optstv.csv'))
    vp = VpnetModelTensorflow(d1max, d2max, vpmodeldir = Utility.get_model_filepath("vpnet_model_tensorflow"))

    # read existing simulation result
    spiceHist = pd.read_csv(Utility.get_output_filepath('spiceHist.csv'))
    nr0 = spiceHist.to_numpy()
    dlen = len(nr0)
    reward = np.zeros(dlen)
    for i in range (dlen):
        dummy, reward[i] = r0.new_reward(nr0[i][2:4])

    # convert to optstv data
    # tedious file IO. Let's do this in bulk on a new optstv.csv file next time
    """
    count = 0
    for i in range (dlen):
        if optstv.write_optimal_state_value(nr0[i,0:2], reward[i]): # tedious file IO
            count = count + 1
    print("add data to optstv =", count)
    """
    # do vp iteration
    h0 = vp.iterate(optstv.optimal_state_value_file, epochs)

def main():
    """
    The main() function.
    """
    app_train_vpnet_model()

if __name__ == '__main__':
    main()
