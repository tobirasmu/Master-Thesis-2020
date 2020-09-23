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

# Importing the data
# Here the data should be storred as pictures (matrices) and not vectors.
fullData = loadMNIST(normalize=True)


# Some of the pictures are hard to recognize even for humans.
plotMany(fullData.x_train,30,20)

y# %% 2 hidden dense layers

num_classes = 10
channels, height, width = fullData.x_train.shape[1:]
num_l1 = 512
num_l2 = 512
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.l1 = Linear(in_features=height*width, out_features=num_l1, bias = True)
        self.l2 = Linear(in_features=num_l1, out_features=num_l2, bias=True)
        self.l_out = Linear(in_features=num_l2, out_features=num_classes, bias=False)
        #self.norm = BatchNorm1d(num_l1)
        
        self.dropout = Dropout(p = 0.4)
        
    def forward(self, x):
        x = x.view(-1,height*width)
        x = self.dropout(relu(self.l1(x)))
        
        x = self.dropout(relu(self.l2(x)))
        
        return softmax(self.l_out(x), dim = 1)

net = Net()
print(net)

# %% Training the network with two hidden layers

data = fullData.subset(10000,5000,10000)

optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
training(net, data, batch_size = 100, num_epochs=100, optimizer = optimizer)

# %% Defining the CNN first simple with one convolutional layer

# Hyper parameters
num_classes = 10
channels, height, width = fullData.x_train.shape[1:]
num_filters_conv1 = 32
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
        self.dropout2 = Dropout2d(p = 0.4)
        self.dropout = Dropout(p = 0.4)
        
        # Number of features into the dense layer
        # self.l1_in_features = channels * height * width
        self.l1_in_features = num_filters_conv1 * self.conv_out_height * self.conv_out_width
        # Dense layer and output
        self.l1 = Linear(in_features = self.l1_in_features, out_features = num_l1, bias = True)
        self.l_out = Linear(in_features = num_l1, out_features = num_classes, bias = False)
        
    def forward(self, x):
        # First a convolution
        x = self.dropout2(relu(self.conv1(x)))
        # Reshaping to prepare for dense layer
        x = x.view(-1,self.l1_in_features)
        # Dropout
        x = self.dropout(relu(self.l1(x)))
        # Dense layer
        #x = relu(self.l1(x))
        # Output
        return softmax(self.l_out(x), dim = 1)
    
net = Net()
print(net)

# %% Training the network with 1 convolutional layer

data = fullData.subset(10000,5000,10000)
optimizer = optim.Adam(net.parameters(), lr = 0.0001, weight_decay = 0.1)

#optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
training(net, data, batch_size = 100, num_epochs=30, optimizer = optimizer)


# %% Trying a high-performance one
# Hyper parameters
num_classes = 10
channels, height, width = fullData.x_train.shape[1:]

num_filters = (4,16)
kernel_size = (5,5)
stride_size = (1,1)
padding_size = (2,0)

num_l1 = 120

def compute_conv_dim(dim_size, kernel_size, stride_size, padding_size):
    return int((dim_size - kernel_size + 2 * padding_size) / stride_size + 1)
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1st layer
        self.conv1 = Conv2d(in_channels= channels, out_channels=num_filters[0], 
                            kernel_size=kernel_size[0], stride=stride_size[0], padding= padding_size[0])
        self.size_c1 = compute_conv_dim(height, kernel_size[0], stride_size[0], padding_size[0]) // 2
        # 2nd layer
        self.conv2 = Conv2d(in_channels = num_filters[0], kernel_size=5, 
                             out_channels=num_filters[1], stride=stride_size[1], padding = padding_size[1])
        self.size_c2 = compute_conv_dim(self.size_c1, kernel_size[1], stride_size[1], padding_size[1]) // 2
        
        # Pooling
        self.pool = MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
        
        # Batch normalization
        self.normC1 = BatchNorm2d(num_filters[0])
        self.normC2 = BatchNorm2d(num_filters[1])
        self.norm = BatchNorm1d(num_l1)
        
        # Dropout
        self.dropout2 = Dropout2d(p = 0.4)
        self.dropout = Dropout(p = 0.4)
        
        # Number of features going into dense layer
        self.l1_in_features = num_filters[1] * self.size_c2 ** 2
        self.l1 = Linear(self.l1_in_features, num_l1, bias = True)
        self.l_out = Linear(in_features = num_l1, out_features=num_classes, bias=False)
        
        
    def forward(self, x):
        # First convolution including dropout and batchnorm
        x = self.normC1(relu(self.conv1(x)))
        # Pooling
        x = self.pool(self.dropout2(x))
        # 2nd convolution
        x = self.normC2(relu(self.conv2(x)))
        # Pooling
        x = self.pool(self.dropout2(x))
        # Reshaping before dense layer
        x = x.view(-1,self.l1_in_features)
        # Doing the dense layer
        x = self.dropout(relu(self.l1(x)))
        # Output layer
        return softmax(self.l_out(x), dim = 1)
        
net = Net()
print(net) 

# %% Testing forward pass
x = np.random.normal(0,1,(5,1,28,28)).astype('float32')
out = net(Variable(tc.from_numpy(x)))
print(out)

# %% Training the big network

data = fullData.subset(40000,20000,10000)
optimizer = optim.Adam(net.parameters(), lr = 0.001)

#optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
training(net, data, batch_size = 100, num_epochs=30, optimizer = optimizer)

# %% Showing wrong images
net.eval()
predictions = tc.max(net(tc.from_numpy(data.x_test[:1000])),1)[1].numpy()
showWrong(data.x_test[:1000], predictions, data.y_test[:1000])

# %% Printing the filters that still look random to me
names_and_vars = {x[0]: x[1] for x in net.named_parameters()}
print(names_and_vars.keys())

np_W = names_and_vars['conv1.weight'].data.numpy()

channels_out, channels_in, filter_size, _ = np_W.shape
n = int(channels_out**0.5)

np_W_res = np_W.reshape(filter_size, filter_size, channels_in, 2, 2)
fig, ax = plt.subplots(2,2)
print("learned filter values")
for i in range(2):
    for j in range(2):
        ax[i,j].imshow(np_W_res[:,:,0,i,j], cmap='gray',interpolation='none')
        ax[i,j].xaxis.set_major_formatter(plt.NullFormatter())
        ax[i,j].yaxis.set_major_formatter(plt.NullFormatter())

# %%
        
idx = 1
plt.figure()
plt.imshow(data.x_train[idx,0],cmap='gray',interpolation='none')
plt.title('Inut Image')
plt.show()

#visalize the filters convolved with an input image
from scipy.signal import convolve2d
np_W_res = np_W.reshape(filter_size, filter_size, channels_in, 2, 2)
fig, ax = plt.subplots(2,2,figsize=(9,9))
print("Response from input image convolved with the filters")
for i in range(2):
    for j in range(2):
        ax[i,j].imshow(convolve2d(data.x_train[idx,0],np_W_res[:,:,0,i,j],mode='same'),
                       cmap='gray',interpolation='none')
        ax[i,j].xaxis.set_major_formatter(plt.NullFormatter())
        ax[i,j].yaxis.set_major_formatter(plt.NullFormatter())
