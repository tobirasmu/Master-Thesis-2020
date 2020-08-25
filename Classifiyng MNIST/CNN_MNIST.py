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
from nnFunctions import *
# Packages related to pytorch nn framework

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
import torch.nn.init as init
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d

# Importing the data
mndata = MNIST()

x_train, y_train = mndata.load_training()
x_test, y_test = mndata.load_testing()

x_train, y_train = np.array(x_train).astype('float32'), np.array(y_train).astype('int32')
x_test, y_test = np.array(x_test).astype('float32'), np.array(y_test).astype('int32')

# Some of the pictures are hard to recognize even for humans.
plotMany(x_train,30,20)

# %% Defining the CNN

# Hyper parameters
num_classes = 10

# The class
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        