"""
    Timing the networks both fully and layer-wise. The time is reported as the mean and standard deviation of SAMPLE_SIZE
    number of pushes. Before this a BURN_IN number of pushes is carried out and discarded.
"""

HPC = False

import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying THETIS/" if HPC else \
    "/home/tenra/PycharmProjects/Master-Thesis-2020/Classifying THETIS/"
os.chdir(path)

import torch as tc
from tools.decomp import compressNet
from tools.models import numFLOPsPerPush, numParamsByLayer, Specs, Net2, conv_dims, numParams
from tools.trainer import get_variable
from torch.autograd import Variable
from torch.nn import Conv3d, Linear, MaxPool3d, Dropout, Dropout3d
from torch.nn.functional import relu, softmax
import torch.nn as nn
import numpy as np
from time import time

SAMPLE_SIZE = 1000
BURN_IN = SAMPLE_SIZE // 10
test = get_variable(Variable(tc.rand((1, 4, 28, 120, 160))))


# %% Define the networks
net = Net2(4, 28, 120, 160)
if HPC:
    net.load_state_dict(tc.load("/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/THETIS_new.pt"))
else:
    net.load_state_dict(
        tc.load("/home/tenra/PycharmProjects/Master-Thesis-2020/Trained networks/THETIS_new.pt"))
netDec = compressNet(net)
netDec.eval()
net.eval()

# %% Timing the compressed network using the pytorch profiler
with tc.autograd.profiler.profile() as prof:
    for _ in range(SAMPLE_SIZE + BURN_IN):
        netDec(test)

# Taking out the individual times in order to make statistics:
dcmp_times = tc.empty((SAMPLE_SIZE + BURN_IN, 12))

i, j = 0, 0
for event in prof.function_events:
    if event.name == "aten::conv3d" or event.name == "aten::linear":
        dcmp_times[i, j] = event.cpu_time_total
        if j == 11:
            j = 0
            i += 1
        else:
            j += 1

import matplotlib.pyplot as plt
plt.figure()
plt.plot(dcmp_times[0, 3:] / 1000)
plt.show()

dcmp_times = dcmp_times[BURN_IN:]
dcmp_means = tc.mean(dcmp_times, dim=0) / 1000
print(tc.mean(dcmp_times, dim=0) / 1000)

# %% Timing the original network using the pytorch profiler
with tc.autograd.profiler.profile() as prof:
    for _ in range(SAMPLE_SIZE + BURN_IN):
        net(test)

# Taking out the individual times in order to make statistics:
orig_times = tc.empty((SAMPLE_SIZE + BURN_IN, 5))

i, j = 0, 0
for event in prof.function_events:
    if event.name == "aten::conv3d" or event.name == "aten::linear":
        orig_times[i, j] = event.cpu_time_total
        if j == 4:
            j = 0
            i += 1
        else:
            j += 1

import matplotlib.pyplot as plt
plt.figure()
plt.plot(orig_times[:, 1:] / 1000)
plt.show()

orig_times = orig_times[BURN_IN:]
orig_means = tc.mean(orig_times, dim=0) / 1000
print(tc.mean(orig_times, dim=0) / 1000)

# %% Printing the whole thing
dcmp_times = dcmp_times[400:750]
orig_times = orig_times[400:750]

layer_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11]]
layer_names = ["Conv 1", "Conv 2", "Linear 1", "Linear 2", "Linear 3"]
print(f"{'':-^100}\n{'  Timing the Networks  ':-^100s}\n{'':-^100}")
print(f"\n{'Layer': ^20s}{'Original': ^20s}{'Compressed': ^40s}{'Speed-up': ^20s}\n{'':-^100s}")
for i, orig in enumerate(orig_means):
    dcmp_comp = tc.sum(dcmp_means[layer_groups[i]])
    sub = 0
    if len(layer_groups[i]) == 3:
        print(f"{'': ^60s}{dcmp_means[layer_groups[i][sub]]: ^20.4f}")
        sub += 1
    print(f"{layer_names[i]: ^20s}{orig: ^20.4f}{dcmp_comp: ^20.4f}"
          f"{dcmp_means[layer_groups[i][sub]]: ^20.4f}{orig / dcmp_comp: ^20.4f}")
    sub += 1
    if len(layer_groups[i]) > 1:
        print(f"{'': ^60s}{dcmp_means[layer_groups[i][sub]]: ^20.4f}")
    print(f"{'':-^100s}")