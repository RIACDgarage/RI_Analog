"""
 run policy with state value function approx vpnet
 regression of vpnet to optimal state value optstv
"""

import argparse

# ---- NOTE ---- A temporary solution to avoid the error: OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
# ---- NOTE ---- libiomp5md.dll can be installed in a Conda environment and by a package, e.g., Pythorh
# ---- NOTE ---- REFERENCE: https://stackoverflow.com/questions/64209238/error-15-initializing-libiomp5md-dll-but-found-libiomp5md-dll-already-initial
import os

from utility import Utility
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
from numpy import ndarray

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

import pathlib
import pandas as pd

import ri_utility
from spice_interface_advanced import SpiceInterfaceAdvanced
from policy import Policy
from optimal_state_value import OptimalStateValue

import model_vpnet_pytorch
from model_vpnet_pytorch import VpnetModelPytorch

from policy_trajectory_plotter import PolicyTrajectoryPlotter

def process_app_ri_analog_with_mdp_arguments(parser):
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
    parser.add_argument(
        '--rbias',
        type=float,
        required=False,
        default=0.8,
        help='normalization rbias.')
    parser.add_argument(
        '--rweight',
        type=float,
        required=False,
        default=0.0016,
        help='normalization rweight.')

def app_ri_analog_with_mdp():
    # ------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    # ------------------------------------------------------------------------
    process_app_ri_analog_with_mdp_arguments(parser)
    # ------------------------------------------------------------------------
    args: argparse.Namespace = parser.parse_args()
    # ------------------------------------------------------------------------
    # parameter definition
    episodes = args.episodes
    epsilon_greedy = args.epsilon
    dmin = args.dmin
    d1max = args.d1max
    d2max = args.d2max
    rbias = args.rbias
    rweight = args.rweight
    # ------------------------------------------------------------------------
    # initialize 
    vp = VpnetModelPytorch().to("cpu") # policy state-value NN
    vpModelFile = "vpmodel.pth"
    if pathlib.Path(vpModelFile).is_file():
        vp.load_state_dict(torch.load(vpModelFile))
    opt0 = OptimalStateValue(Utility.get_output_filepath('optstv.csv')) # prepare the optimal state value function
    st0 = ri_utility.random_design_function(dmin, d1max, d2max) # or st0 = [20,20], for initial state
    e0 = SpiceInterfaceAdvanced(
        Utility.get_intermediate_input_filepath('action.txt'),
        Utility.get_input_filepath('inverter.sp'),
        Utility.get_output_filepath('spiceout.txt'),
        opt0)
    if opt0.number_optimal_state_values() < 100: ri_utility.init_state_value("mesh", dmin, d1max, d2max, opt0, e0)
    r0 = ri_utility.RewardCalcutorMyopic() # initial reward function
    reward = e0.run_spice(st0) # put initial state to environment
    n0 = ri_utility.StateValueNormalizer(d1max, d2max, rbias, rweight) # init normalization function
    bestDesign = [st0, reward]

    p0 = Policy(epsilon_greedy, dmin, d1max, d2max, vp, n0)

    # pyTorch training settings
    vp = VpnetModelPytorch().to("cpu")
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(vp.parameters(), lr=1e-3)

    dplot: ndarray = np.ones((episodes+1,2),dtype=np.int32) # for trajectory plot
    dplot[0] = st0
    # Do MDP in #episodes times
    for i in range (episodes):
        st1 = p0.action(st0) # agent action
        reward = e0.run_spice(st1) # environment response
        updateFlag = opt0.write_optimal_state_value(st1, reward) # store optimal state value

        if updateFlag: # new state vaule, do back annotation
            sNN, rNN = n0.normalize_to_neural_network_init(opt0)
            srDataset = model_vpnet_pytorch.VpnetModelateRewardDataset(sNN, np.array([rNN]).T)
            data_loader = DataLoader(srDataset, batch_size=50, shuffle=True)
            vp.train() # model in training mode
            epochs = 100
            for t in range (epochs):
                print(f"Epoch {t+1}\n--------------------------------")
                model_vpnet_pytorch.train(data_loader, vp, loss_fn, optimizer)
            torch.save(vp.state_dict(), vpModelFile)

        st0 = st1 # update state for t+1
        dplot[i+1] = st1

        # define MDP termination criterion
        if isinstance(reward, list): reward = reward[0]
        if isinstance(bestDesign[1], list): bestDesign[1] = bestDesign[1][0]
        if reward > bestDesign[1]: bestDesign = [st1, reward]
        if reward > 80:
            print("Design target reached, existing")
            break
        elif i == episodes-1:
            print("Episode expired, best design is", bestDesign[0])
            print("best reward is", bestDesign[1])

    PolicyTrajectoryPlotter.plot(dplot)

def main():
    """
    The main() function.
    """
    app_ri_analog_with_mdp()

if __name__ == '__main__':
    main()
