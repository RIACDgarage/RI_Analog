
from typing import Any

import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_q1net import Q1Net
from model_q2net import Q2Net

from utility import Utility

def process_app_keras_model_evalator_arguments(parser):
    """
    Process arguments.
    """
    if parser is None:
        raise 'input argument, parser, is None'
    parser.add_argument(
        '--samples',
        type=int,
        required=False,
        default=10000,
        help='reinforcement learning samples.')
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

def app_keras_model_evalator():
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_app_keras_model_evalator_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    # ------------------------------------------------------------------------
    # parameter definition
    samples = args.samples
    # limit the max design, original is 1000
    d1max = args.d1max
    d2max = args.d2max

    q1net_model: Any = Q1Net.load_q1net_model()
    q2net_model: Any = Q2Net.load_q2net_model()

    # generate uniform random design in log space
    design1 = np.random.uniform(low=np.log(3), high=np.log(d1max), size=samples)
    design2 = np.random.uniform(low=np.log(3), high=np.log(d2max), size=samples)
    # scale design to value 0 to 1 
    design1 = design1/np.log(d1max)
    design2 = design2/np.log(d2max)
    design = np.array([design1, design2])
    design = design.T

    # make prediction from trained Keras models
    pred1 = q1net_model.predict(design) * 100
    pred2 = q2net_model.predict(design) * 100

    design[:,0] = np.exp(design[:,0]*np.log(d1max))
    design[:,1] = np.exp(design[:,1]*np.log(d2max))
    x = design[:,0]
    y = design[:,1]

    # plot predict
    #plt.figure()
    plt.subplot(2, 2, 2)
    plt.title("q1net model predict")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(x, y, c=pred1, cmap='inferno')
    plt.colorbar()
    plt.grid()
    #plot_filename: str = Utility.get_output_filepath('q1net.png')
    #plt.savefig(plot_filename)

    #plt.figure()
    plt.subplot(2, 2, 4)
    plt.title("q2net model predict")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(x, y, c=pred2, cmap='inferno')
    plt.colorbar()
    plt.grid()
    #plot_filename: str = Utility.get_output_filepath('q2net.png')
    #plt.savefig(plot_filename)

    # plot spice
    df = pd.read_csv(Utility.get_output_filepath('spiceHist.csv'))
    df0 = df.loc[(df['design1'] <= d1max) & (df['design2'] <= d2max)]
    xs = df0["design1"].to_numpy()
    ys = df0["design2"].to_numpy()
    m1 = df0["merit1"].to_numpy()
    m2 = df0["merit2"].to_numpy()

    #plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("spice merit1")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(xs, ys, c=m1, cmap='inferno')
    plt.colorbar()
    plt.grid()
    #plot_filename: str = Utility.get_output_filepath('merit1.png')
    #plt.savefig(plot_filename)

    #plt.figure()
    plt.subplot(2, 2, 3)
    plt.title("spice merit2")
    plt.xscale("log")
    plt.yscale("log")
    plt.scatter(xs, ys, c=m2, cmap='inferno')
    plt.colorbar()
    plt.grid()
    #plot_filename: str = Utility.get_output_filepath('merit2.png')
    #plt.savefig(plot_filename)

    plt.suptitle("Keras Model Prediction vs. Spice")
    plot_filename: str = Utility.get_output_filepath('kerasModelEval.png')
    plt.savefig(plot_filename)

def main():
    """
    The main() function.
    """
    app_keras_model_evalator()

if __name__ == '__main__':
    main()
