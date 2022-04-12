
from typing import Any

import argparse

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spice_normalizer import SpiceNormalizer

from utility import Utility

def process_model_q2net_arguments(parser):
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

class Q2Net:
    """
    Q2Net
    """
    @staticmethod
    def load_q2net_model() -> Any:
        """
        Load a q2net model from file.
        """
        q2net_model: Any = tf.keras.models.load_model(Utility.get_model_filepath("q2net_model"))
        return q2net_model
    @staticmethod
    def initialize_q2net_model() -> Any:
        """
        Initialize a q2net model
        """
        q2net_model: Any = tf.keras.models.Sequential()
        q2net_model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)))
        q2net_model.add(tf.keras.layers.Dense(16, activation='relu'))
        q2net_model.add(tf.keras.layers.Dense(1))
        return q2net_model

    @staticmethod
    def model_q2net():
        """
        Load or create a new q2net, train it, and save it back.
        """
        # --------------------------------------------------------------------
        parser = argparse.ArgumentParser()
        # --------------------------------------------------------------------
        process_model_q2net_arguments(parser)
        # --------------------------------------------------------------------
        args: argparse.Namespace = parser.parse_args()
        # --------------------------------------------------------------------
        # parameter definition
        operation: int = args.operation
        q2net_model: Any = Q2Net.load_q2net_model() if operation == 0 else Q2Net.initialize_q2net_model()
        # --------------------------------------------------------------------
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95)
        #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #    initial_learning_rate=1e-2, decay_steps=5000, decay_rate=0.95,
        #    staircase=True)
        opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        q2net_model.compile(optimizer = opt, loss = 'mse')
        # --------------------------------------------------------------------
        spiceData = Utility.get_output_filepath('spiceHist.csv')
        eval_ratio = 5 # ratio of training data for evaluation
        d1max = 300 # limit training within certain size
        d2max = 1000
        d0 = SpiceNormalizer(spiceData, eval_ratio, d1max, d2max).normalize()
        h0 = q2net_model.fit(
            [d0[0]],
            [d0[1][:,1]],
            epochs = 10000,
            batch_size = 1024,
            validation_data = ([d0[2]], [d0[3][:,1]]))
        # --------------------------------------------------------------------
        plt.figure() # new plot for every episode
        #plt.ylim([0.0, 0.4])
        plt.grid(axis='y')
        plt.plot(h0.history['loss'], label="training loss")
        plt.plot(h0.history['val_loss'], label="eval loss")
        plt.legend()
        plot_filename: str = Utility.get_output_filepath('q2net_model_loss.png')
        plt.savefig(plot_filename)
        # --------------------------------------------------------------------
        q2net_model.save(Utility.get_model_filepath("q2net_model"))
        # --------------------------------------------------------------------

def main():
    """
    The main() function.
    """
    Q2Net.model_q2net()

if __name__ == '__main__':
    main()
