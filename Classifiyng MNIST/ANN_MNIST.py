#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:59:51 2020

@author: Tobias

In the following an initial ANN will be made for the MNIST dataset. It
will be done using the pytorch-framework.

"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import torch as tc
from nnFunctions import *
# Packages related to pytorch nn framework

import torch.nn as nn
import torch.nn.functional as Fun
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable

# Importing the data
mndata = MNIST()

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('int32')
x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('int32')

# Some of the pictures are hard to recognize even for humans.
plotMany(x_train,30,20)

# %%
# Example of plotting
plt.subplot(1,2,1)
showImage(x_train[201],y_train[201])
plt.subplot(1,2,2)
showImage(x_train[100],y_train[100])

#%% ANN using the pytorch framework


# Hyper parameters from this data
num_classes = 10
num_features = x_train.shape[1]
num_hidden = 512
    
# Defining the inital network
class Net(nn.Module):
    
    def __init__(self, num_features, num_hidden, num_output):
        super(Net,self).__init__() # Initializing and inheriting from the nn.module class
        
        # Input layer
        self.W_1 = Parameter(init.kaiming_normal_(tc.Tensor(num_hidden,num_features)))
        self.b_1 = Parameter(init.constant_(tc.Tensor(num_hidden),0))
        
        # Hidden layer
        self.W_2 = Parameter(init.kaiming_normal_(tc.Tensor(num_output,num_hidden)))
        self.b_2 = Parameter(init.constant_(tc.Tensor(num_output),0))
        
        # Activation
        self.activation = tc.nn.ReLU()
        
    def forward(self,x):
        x = Fun.linear(x,self.W_1, self.b_1)
        x = self.activation(x)
        x = Fun.linear(x,self.W_2, self.b_2)
        return x
        
# Initializing an instance of the network

net = Net(num_features, num_hidden, num_classes)

#%% How to forward-pass some dummy data

x = np.random.normal(0, 1, (45, 28*28)).astype('float32')
B = net(Variable(tc.from_numpy(x))) # same as net.forward

#%% 
# Defining the inital network
class Net2(nn.Module):
    
    def __init__(self, num_features, num_hidden, num_hidden2, num_output):
        super(Net2,self).__init__() # Initializing and inheriting from the nn.module class
        
        # Input layer
        self.W_1 = Parameter(init.xavier_normal_(tc.Tensor(num_hidden,num_features)))
        self.b_1 = Parameter(init.constant_(tc.Tensor(num_hidden),0))
        
        # Hidden layer
        self.W_2 = Parameter(init.xavier_normal_(tc.Tensor(num_hidden2,num_hidden)))
        self.b_2 = Parameter(init.constant_(tc.Tensor(num_hidden2),0))
        
        # Hidden layer 2
        self.W_3 = Parameter(init.xavier_normal_(tc.Tensor(num_output,num_hidden2)))
        self.b_3 = Parameter(init.constant_(tc.Tensor(num_output),0))
        
        # Activation
        self.activation = tc.nn.ReLU()
        
    def forward(self,x):
        x = Fun.linear(x,self.W_1, self.b_1)
        x = self.activation(x)
        x = Fun.linear(x,self.W_2, self.b_2)
        x = self.activation(x)
        x = Fun.linear(x,self.W_3, self.b_3)
        return x
        

# %% Testing with some data
        # Using 1000 samples and 500 validation samples
##
net = Net(784, 512, 10)
data = Data(x_train[:2000,:],y_train[:2000],x_train[2000:3000,:],y_train[2000:3000])

optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.9)
training(net,data, 100, 100, optimizer, every = 2)

#%% 2 hidden layers  
    
net2 = Net2(784, 512, 512, 10)
optimizer = optim.Adam(net2.parameters(), lr = 0.001, weight_decay=0.1)
training(net2,data, 100,100,optimizer, every = 5)
    

