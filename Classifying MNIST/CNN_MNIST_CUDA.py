#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:12:15 2020

@author: Tobias

In this file, a CNN for MNIST will be implemented by help from CUDA (GPU). The 
network architecture from https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
will be used. 

"""

from pic_functions import loadMNIST, Data

# Loading the data and picking out a subset
fullData = loadMNIST()

data = fullData.subset(10000,2000,5000)
print(data)

# %% Defining the CNN

