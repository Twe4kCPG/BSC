
# %%
import time
import models.fft_model
import models.inception
import torch
import torch.nn as nn
import torch.utils.data
import sys
import os

import torchsummary

from tqdm import tqdm
import pandas as pd
import copy

import plotly.express as ep


import torchsummary


from torch.utils.flop_counter import FlopCounterMode




class Network(nn.Module):
    def __init__(self, number_of_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=12, out_channels=128,
                      kernel_size=80, stride=64),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256,
                      kernel_size=11, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128,
                      kernel_size=7, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=number_of_classes),
        )

    def forward(self, x):
        return self.layers(x)

#%%

input = torch.randn(2,12, 5000)

# %%
net = Network(2)
torchsummary.summary(net, (12, 5000), device='cpu')

net.eval()
flop_counter = FlopCounterMode(net, depth=1)
with flop_counter:
    net(input).sum().backward()

#%%
net = models.inception.InceptionTime(2)
net.eval()
torchsummary.summary(net, (12, 5000), device='cpu')

flop_counter = FlopCounterMode(net, depth=1)
with flop_counter:
    net(input).sum().backward()

#%%
net = models.fft_model.FFTModel(2)
net.eval()
torchsummary.summary(models.fft_model.FFTModel(2), (12, 5000), device='cpu')

flop_counter = FlopCounterMode(net, depth=1)
with flop_counter:
    net(input).sum().backward()
# %%
