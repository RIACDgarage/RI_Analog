
from typing import Any

import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spice_normalizer import SpiceNormalizer

from utility import Utility

def process_model_q1net_arguments(parser):
    """
    Process arguments.
    """
    if parser is None:
        raise 'input argument, parser, is None'
    parser.add_argument(
        '--operation',
        type=int,
        required=False,
        default=0,
        help='operation 0 for loading, 1 for creating.')

class Q1Net:
    """
    Q1Net
    """
    @staticmethod
    def load_q1net_model() -> Any:
        """
        Load a q1net model from file.
        """
        q1net_model: Any = tf.keras.models.load_model(Utility.get_model_filepath("q1net_model"))
        return q1net_model
    @staticmethod
    def initialize_q1net_model() -> Any:
        """
        Initialize a q1net model
        """
        q1net_model: Any = tf.keras.models.Sequential()
        q1net_model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)))
        q1net_model.add(tf.keras.layers.Dense(32, activation='relu'))
        q1net_model.add(tf.keras.layers.Dense(32, activation='relu'))
        q1net_model.add(tf.keras.layers.Dense(1))
        return q1net_model

    @staticmethod
    def model_q1net():
        """
        Load or create a new q1net, train it, and save it back.
        """
        # --------------------------------------------------------------------
        parser = argparse.ArgumentParser()
        # --------------------------------------------------------------------
        process_model_q1net_arguments(parser)
        # --------------------------------------------------------------------
        args: argparse.Namespace = parser.parse_args()
        # --------------------------------------------------------------------
        # parameter definition
        operation: int = args.operation
        q1net_model: Any = Q1Net.load_q1net_model() if operation == 0 else Q1Net.initialize_q1net_model()
        # --------------------------------------------------------------------
        #lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        #    initial_learning_rate=1e-2, decay_steps=100, decay_rate=0.9)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95,
            staircase=True)
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        q1net_model.compile(optimizer = opt, loss = 'mse')
        # --------------------------------------------------------------------
        spiceData = Utility.get_output_filepath('spiceHist.csv')
        eval_ratio = 5 # ratio of training data for evaluation
        d1max = 300 # limit training within certain size
        d2max = 1000
        d0 = SpiceNormalizer(spiceData, eval_ratio, d1max, d2max).normalize()
        h0 = q1net_model.fit(
            [d0[0]],
            [d0[1][:,0]],
            epochs = 2000,
            batch_size = 1024,
            validation_data = ([d0[2]], [d0[3][:,0]]))
        # --------------------------------------------------------------------
        plt.figure() # new plot for every episode
        #plt.ylim([0.0, 0.4])
        plt.grid(axis='y')
        plt.plot(h0.history['loss'], label="training loss")
        plt.plot(h0.history['val_loss'], label="eval loss")
        plt.legend()
        plot_filename: str = Utility.get_output_filepath('q1net_model_loss.png')
        plt.savefig(plot_filename)
        # --------------------------------------------------------------------
        q1net_model.save(Utility.get_model_filepath("q1net_model"))
        # --------------------------------------------------------------------

def main():
    """
    The main() function.
    """
    Q1Net.model_q1net()

if __name__ == '__main__':
    main()
