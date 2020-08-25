#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:42:30 2020

@author: Tobias
"""

"""
In this file a CNN for the MNIST dataset will be implemented and tested.
"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import torch as tc
from nnFunctions import training, Data, showImage, plotMany, get_slice
# Packages related to pytorch nn framework

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d

# Importing the data
# Here the data should be storred as pictures (matrices) and not vectors.
mndata = MNIST()

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('int32')
x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('int32')

# Making it into a stack of matrices
channels, rows, cols = 1, 28, 28  # 1 channel since BW and 28x28 pics
x_train = x_train.reshape((-1,1,28,28))
x_test = x_test.reshape((-1,1,28,28))

# Some of the pictures are hard to recognize even for humans.
plotMany(x_train,30,20)

# %% Defining the CNN

# Hyper parameters
num_classes = 10
channels, height, width = x_train.shape[1:]
num_filters_conv1 = 16
kernel_size_conv1 = 5 # [height, width]
stride_conv1 = 1       # [height, width]
padding_conv1 = 2
num_l1 = 100

def compute_conv_dim(dim_size):
    return int((dim_size - kernel_size_conv1 + 2 * padding_conv1) / stride_conv1 + 1)

# The class
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # Convolutional layer
        self.conv1 = Conv2d(in_channels = channels, out_channels= num_filters_conv1, kernel_size = kernel_size_conv1,
                            stride= stride_conv1, padding=padding_conv1)
        
        # Dimension after convolution
        self.conv_out_height = compute_conv_dim(height)
        self.conv_out_width = compute_conv_dim(width)
        
        # The dropout function
        self.dropout = Dropout2d(p = 0.5)
        
        # Number of features into the dense layer
        # self.l1_in_features = channels * height * width
        self.l1_in_features = num_filters_conv1 * self.conv_out_height * self.conv_out_width
        # Dense layer and output
        self.l1 = Linear(in_features = self.l1_in_features, out_features = num_l1, bias = True)
        self.l_out = Linear(in_features = num_l1, out_features = num_classes, bias = False)
        
    def forward(self, x):
        # First a convolution
        x = relu(self.conv1(x))
        # Reshaping to prepare for dense layer
        x = x.view(-1,self.l1_in_features)
        # Dropout
        self.dropout(relu(self.l1(x)))
        # Dense layer
        x = relu(self.l1(x))
        # Output
        return softmax(self.l_out(x), dim = 1)
    
net = Net()
print(net)

# %% Testing forward pass
x = np.random.normal(0,1,(5,1,28,28)).astype('float32')
out = net(Variable(tc.from_numpy(x)))

# %%
data = Data(x_train[0:10000], y_train[0:10000], x_train[10000:11000], y_train[10000:11000], x_test[0:10000], y_test[0:10000])
optimizer = optim.Adam(net.parameters(), lr = 0.0001)
training(net, data, batch_size = 100, num_epochs=20, optimizer = optimizer)

# %%
names_and_vars = {x[0]: x[1] for x in net.named_parameters()}
print(names_and_vars.keys())

np_W = names_and_vars['conv1.weight'].data.numpy()

channels_out, channels_in, filter_size, _ = np_W.shape
n = int(channels_out**0.5)

np_W_res = np_W.reshape(filter_size, filter_size, channels_in, n, n)
fig, ax = plt.subplots(n,n)
print("learned filter values")
for i in range(n):
    for j in range(n):
        ax[i,j].imshow(np_W_res[:,:,0,i,j], cmap='gray',interpolation='none')
        ax[i,j].xaxis.set_major_formatter(plt.NullFormatter())
        ax[i,j].yaxis.set_major_formatter(plt.NullFormatter())

