#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:38:38 2020

@author: Tobias

In this file, it is attempted to decompose some 3s and 4s from the MNIST data set.
"""

from nnFunctions import training, loadMNIST, Data, showImage, showWrong, plotMany, get_slice
import numpy as np

data = loadMNIST()

# %% Making a tensor of only 3s and 4s. 

X_all = data.x_train

indices4 = np.where(((data.y_train == 4) + (data.y_train == 3)))

X = X_all[indices4]

plotMany(X, 30, 20)

""" 
Now X is a tensor of only 3s and 4s
"""

# %% 
