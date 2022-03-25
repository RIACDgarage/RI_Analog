"""
  neural network for policy value function prediction
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import numpy as np
from optstv import optstv
from RIutil import normStv
from spiceIF2 import spiceIF
from RIutil import stvInit
import pathlib

class stateRewardDataset(Dataset):
    def __init__(self, tinput, tresult):
        self.tinput = tinput
        self.tresult = tresult
    def __len__(self):
        return len(self.tinput)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tinput[idx], self.tresult[idx]

class vpnet(nn.Module): # a child class of nn.Module
    def __init__(self):
        super(vpnet, self).__init__() # inherite properties from nn.Module
        self.vpStack = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        reward = self.vpStack(x)
        return reward

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (st, r) in enumerate(dataloader):
        st, r = st.to("cpu"), r.to("cpu")
        pred = model(st)
        loss = loss_fn(pred, r)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(st)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
