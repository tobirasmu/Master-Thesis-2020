import timeit
code = """
import os
path = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

import numpy as np
import tensorly as tl
tl.set_backend('pytorch')
from tensorly.decomposition import partial_tucker
import matplotlib.pyplot as plt
import torch as tc
from nnFunctions import training, loadMNIST, Data, showImage, showWrong, plotMany, get_slice
# Packages related to pytorch nn framework

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d
from copy import deepcopy
from VBMF import EVBMF

# Importing the data
# Here the data should be storred as pictures (matrices) and not vectors.
fullData = loadMNIST(normalize=True)

# Some of the pictures are hard to recognize even for humans.
plotMany(fullData.x_train, 30, 20)

data = fullData.subset(1000, 1000, 1000)
# %% Making a network with a convolution
_, ch, height, width = data.x_train.shape


def compute_conv_dim(dim_size, kernel_size, stride_size, padding_size):
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.c1 = Conv2d(in_channels=ch, out_channels=10, kernel_size=5, stride=1, padding=2)
        self.c2 = Conv2d(in_channels=10, out_channels=40, kernel_size=5, stride=1, padding=0)

        self.l1 = Linear(in_features=(compute_conv_dim(28, 5, 1, 0) ** 2) * 40, out_features=10)

    def forward(self, x):
        x = relu(self.c1(x))
        x = relu(self.c2(x))

        x = tc.flatten(x, 1)
        return softmax(self.l1(x), dim=1)


net = Net()


# %% Making the network with the decomposed kernel for the second layer
def conv_to_tucker2(layer, ranks=None):
    """
    Takes a pretrained convolutional layer and decomposes is using partial
    tucker with the given ranks.
    """
    # Making the decomposition of the weights
    weights = layer.weight.data
    # (Estimating the ranks using VBMF)
    ranks = estimate_ranks(weights, [0, 1]) if ranks is None else ranks
    # Decomposing
    core, [last, first] = partial_tucker(weights, modes=[0, 1], ranks=ranks)

    # Making the layer into 3 sequential layers using the decomposition
    first_layer = Conv2d(in_channels=first.shape[0], out_channels=first.shape[1],
                         kernel_size=1, stride=1, padding=0, bias=False)

    core_layer = Conv2d(in_channels=core.shape[1], out_channels=core.shape[0],
                        kernel_size=layer.kernel_size, stride=layer.stride,
                        padding=layer.padding, bias=False)

    last_layer = Conv2d(in_channels=last.shape[1], out_channels=last.shape[0],
                        kernel_size=1, stride=1, padding=0, bias=True)

    # The decomposition is chosen as weights in the network (output, input, height, width)
    first_layer.weight.data = tc.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core  # no reshaping needed
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    # The bias from the original layer is added to the last convolution
    last_layer.bias.data = layer.bias.data

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def estimate_ranks(weight_tensor, dimensions):
    """
    Estimates the sufficient ranks for a given tensor
    """
    ranks = []
    for dim in dimensions:
        _, diag, _, _ = EVBMF(tl.unfold(weight_tensor, dim))
        ranks.append(diag.shape[dim])
    return ranks


# %% Decomposing
net2 = deepcopy(net)
net2.c2 = conv_to_tucker2(net.c2, ranks=[2, 2])


class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()

        self.c1 = Conv2d(in_channels=ch, out_channels=10, kernel_size=5, stride=1, padding=2)

        self.c21 = Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 10), stride=1, bias=False)
        self.c22 = Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=1, bias=False)
        self.c23 = Conv2d(in_channels=2, out_channels=40, kernel_size=1, stride=1, bias=True)
        #self.c2 = nn.Sequential(c21, c22, c23)

        self.l1 = Linear(in_features=(compute_conv_dim(28, 5, 1, 0) ** 2) * 40, out_features=10, bias=True)

    def forward(self, x):
        x = relu(self.c1(x))
        x = tc.reshape(x, (-1, 1, 784, 10))
        #print(x.shape)
        x = self.c21(x)
        x = tc.reshape(x, (-1, 2, 28, 28))
        #print(x.shape)
        x = self.c22(x)
        #print(x.shape)
        x = relu(self.c23(x))
        #print(x.shape)
        x = tc.flatten(x, 1)
        return softmax(self.l1(x), dim=1)


net3 = Net3()
net3.c1.weight.data = net2.c1.weight.data
net3.c21.weight.data = tc.reshape(net2.c2[0].weight.data, (2, 1, 1, 10))
net3.c22.weight.data = net2.c2[1].weight.data
net3.c23.weight.data = net2.c2[2].weight.data
net3.c23.bias.data = net2.c2[2].bias.data
net3.l1.weight.data = net2.l1.weight.data
net3.l1.bias.data = net2.l1.bias.data

# %%
from time import process_time
testX = Variable(tc.from_numpy(data.x_train[0:1000]))
"""
# %%
t1 = timeit.timeit('net2(testX)', setup=code, number=10)
t2 = timeit.repeat('net3(testX)', setup=code, number=10)

t = process_time()
net2(testX)
time1 = process_time() - t
t = process_time()
net3(testX)
time2 = process_time() - t
print(time1, time2, "Ratio: ", time2 / time1)

def numParams(net):
    """
    Returns the number of parameters in the entire network.
    """
    return sum(np.prod(p.size()) for p in net.parameters())

