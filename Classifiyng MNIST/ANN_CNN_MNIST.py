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

#%% For being able to plot the handwritten digits. Either one by one, or a matrix of them
def showImage(img, label=""):
    plt.imshow(np.reshape(img,newshape=(28,28)),vmin=0,vmax=255,cmap="gray")
    plt.title(label)
    
def plotMany(img_L,B=10,H=10):
    # B is how many pictures on the x-axis, and H is the y-axis
    nr = 0
    canvas = np.zeros((1,28*B))
    for i in range(H):
        temp = img_L[nr].reshape((28,28))
        nr+= 1
        for j in range(B-1):
            temp = np.concatenate((temp,img_L[nr].reshape((28,28))),axis = 1)
            nr+= 1
        canvas = np.concatenate((canvas, temp),axis = 0)
    plt.imshow(canvas[1:,:],vmin=0,vmax=255,cmap="gray")
    plt.axis('off')

    #%%
    
# Example of plotting
plt.subplot(1,2,1)
showImage(img_train[201],labs_train[201])
plt.subplot(1,2,2)
showImage(img_train[100],labs_train[100])
#%%
# Some of the pictures are hard to recognize even for humans.
plotMany(img_train,20,30)
#%% ANN using the pytorch framework
    
    