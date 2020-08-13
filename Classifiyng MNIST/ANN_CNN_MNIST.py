#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 14:59:51 2020

@author: Tobias

In the following an initial ANN and CNN will be made for the MNIST dataset. It
will be done using the pytorch-framework.

"""

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import torch as tc

mndata = MNIST()

img_train, labs_train = mndata.load_training()
img_test, labs_test = mndata.load_testing()

img_train, labs_train = np.array(img_train), np.array(labs_train)
img_test, labs_test = np.array(img_test), np.array(labs_test)

def showImage(img, label=""):
    plt.imshow(np.reshape(np.array(img),newshape=(28,28)),vmin=0,vmax=255,cmap="gray")
    plt.title(label)
    
# Example of plotting
plt.subplot(1,2,1)
showImage(img_train[201],labs_train[201])
plt.subplot(1,2,2)
showImage(img_train[100],labs_train[100])

# Some of the pictures are hard to recognize even for humans.

#%% ANN using the pytorch framework
    
    