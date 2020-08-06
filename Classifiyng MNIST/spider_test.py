#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:03:11 2020

@author: Tobias Engelhardt Rasmussen
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

#%% Example of plotting
    plt.subplot(1,2,1)
    showImage(img_train[201],labs_train[201])
    plt.subplot(1,2,2)
    showImage(img_train[100],labs_train[100])

#%% Preparing data and defining functions

Nnu = 20000
small_train = tc.tensor(img_train[0:Nnu]).float()
small_train_y = np.zeros((Nnu,10))
small_train_y[np.arange(0,Nnu),labs_train[0:Nnu]] = 1
small_train_y = tc.tensor(small_train_y)
    
def sigmoid(x): # Returns the sigmoid function of the value x
    return(1/(1 + tc.exp(-x)))

def difSigmoid(x):
    xexp = tc.exp(-x)
    return(xexp/((1+xexp)**2))

def softMax(X):
    X = tc.exp(X)
    return(X.T*1/tc.sum(X,axis=1)).T
    

#%%
# Initializing the weights and the shape of the network
input_shape = 784
output_shape = 10
network_shape = (100,100)
alpha = 0.001

w_in = alpha*tc.randn((input_shape,network_shape[0]))
b_in = alpha*tc.randn((1,network_shape[0]))

w_hidden = alpha*tc.randn(network_shape)
b_hidden = alpha*tc.randn((1,network_shape[1]))

w_out = alpha*tc.randn((network_shape[-1],output_shape))
b_out = alpha*tc.randn((1,output_shape))

### Training loop
learning_rate = 0.1
for it in range(1500):
    # Allocating space for the individual updates per observation
    w_in_update = tc.zeros((input_shape,network_shape[0]))
    b_in_update = tc.zeros((1,network_shape[0]))
    w_hidden_update = tc.zeros(network_shape)
    b_hidden_update = tc.zeros((1,network_shape[1]))
    w_out_update = tc.zeros((network_shape[-1],output_shape))
    b_out_update = tc.zeros((1,output_shape))
    
    z0 = tc.mm(small_train,w_in) + b_in
    a0 = sigmoid(z0)
    z1 = tc.mm(a0,w_hidden) + b_hidden
    a1 = sigmoid(z1)
    z2 = tc.mm(a1,w_out) + b_out
    a2 = sigmoid(z2)
    
    # Doing back-propagation for each observation
    N = small_train_y.shape[0]
    for i in range(N):
        # Output weights
        temp = ((2*(a2[i] - small_train_y[i])*difSigmoid(z2[i])).view(1,10)).float()
        dC0_dW2 = tc.mm(a1[i].view(100,1),temp)
        w_out_update += (1/N)*dC0_dW2
        b_out_update += (1/N)*temp
        
        # Hidden layer weights
        temp2 = tc.mm(temp,w_out.T)*difSigmoid(z1[i])
        dC0_dW1 = tc.mm(a0[i].view(100,1),temp2)
        w_hidden_update += (1/N)*dC0_dW1
        b_hidden_update += (1/N)*temp2
        
        # Input layer weights
        temp3 = tc.mm(temp2,w_hidden.T)*difSigmoid(z0[i])
        dC0_dW0 = tc.mm(small_train[i].view(784,1),temp3)
        w_in_update += (1/N)*dC0_dW0
        b_in_update += (1/N)*temp3
    # Updating the weights        
    w_in += -learning_rate*w_in_update
    b_in += -learning_rate*b_in_update
    
    w_hidden += -learning_rate*w_hidden_update
    b_hidden += -learning_rate*b_hidden_update
    
    w_out += -learning_rate*w_out_update
    b_out += -learning_rate*b_out_update
    #print(w_in_update, w_hidden_update, w_out_update)
    print("At iteration ",it, " the error is: ",sum(sum((a2-small_train_y)**2)))
    print("------------||-------------")
   
#%%
    z0 = tc.mm(small_train,w_in) + b_in
    a0 = sigmoid(z0)
    z1 = tc.mm(a0,w_hidden) + b_hidden
    a1 = sigmoid(z1)
    z2 = tc.mm(a1,w_out) + b_out
    a2 = sigmoid(z2)
    print(a2.argmax(dim=1))
