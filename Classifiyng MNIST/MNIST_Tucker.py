#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:41:30 2020

@author: Tobias
"""
# %% Loading the data and all the libraries

from nnFunctions import training, loadMNIST, Data, showImage, showWrong, plotMany, get_slice
import numpy as np
from numpy.linalg import pinv, inv
import tensorly as tl
import torch as tc
from tensorly.decomposition import parafac, tucker, non_negative_parafac, non_negative_tucker, matrix_product_state
from tensorly.tenalg import kronecker, multi_mode_dot, mode_dot
import matplotlib.pyplot as plt

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d

fullData = loadMNIST()

# %% Defining the full network from the LeNet-5 architecture.
_, channels, width, height = data.x_train.shape

def conv_dim(dim, kernel, stride, padding):
    return int((dim - kernel + 2 * padding) / stride + 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # The convolutions
        self.conv1 = Conv2d(in_channels = channels, out_channels=6, kernel_size=5, padding=2, stride=1)
        dim1 = conv_dim(height, kernel = 5, padding = 2, stride= 1)
        dim1P = conv_dim(dim1, kernel = 2, padding = 0, stride = 2)
        self.conv2 = Conv2d(in_channels = 6, out_channels = 16, kernel_size=5, padding=0, stride=1)
        dim2 = conv_dim(dim1P, kernel = 5, padding = 0, stride=1)
        dim2P = conv_dim(dim2, kernel = 2, padding = 0, stride = 2)
        
        # The average pooling
        self.pool = AvgPool2d(kernel_size = 2, stride = 2, padding=0)
        
        self.lin_in_feats = 16 * (dim2P **2)
        # The linear layers
        self.l1 = Linear(in_features = self.lin_in_feats, out_features=120, bias = True)
        self.l2 = Linear(in_features = 120, out_features=84, bias = True)
        self.l_out = Linear(in_features = 84, out_features=10, bias = True)
    
    def forward(self,x):
        x = relu(self.conv1(x))
        x = self.pool(x)
            
        x = relu(self.conv2(x))
        x = self.pool(x)
            
        x = x.reshape((-1, self.lin_in_feats))
            
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        return softmax(relu(self.l_out(x)), dim = 1)
        
net = Net()
x = np.random.normal(0,1,(5,1,28,28)).astype('float32')
out = net(Variable(tc.from_numpy(x)))
print(out)

# %% Training the network
data = fullData.subset(10000, 2000, 10000)

optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
training(net, data, batch_size = 100, num_epochs = 20, optimizer = optimizer)

# %% Printing the filters that still look random to me
names_and_vars = {x[0]: x[1] for x in net.named_parameters()}
print(names_and_vars.keys())

np_W = names_and_vars['conv1.weight'].data.numpy()

channels_out, channels_in, filter_size, _ = np_W.shape
n = int(channels_out**0.5)

np_W_res = np_W.reshape(filter_size, filter_size, channels_in, 3, 2)
fig, ax = plt.subplots(3,2)
print("learned filter values")
for i in range(3):
    for j in range(2):
        ax[i,j].imshow(np_W_res[:,:,0,i,j], cmap='gray',interpolation='none')
        ax[i,j].xaxis.set_major_formatter(plt.NullFormatter())
        ax[i,j].yaxis.set_major_formatter(plt.NullFormatter())