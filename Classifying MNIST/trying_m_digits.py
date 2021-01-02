#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:38:38 2020

@author: Tobias

In this file, it is attempted to decompose some set of digits from the MNIST data set.
"""

import os
path = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)
from pic_functions import training, loadMNIST, Data, showImage, showWrong, plotMany, get_slice
import numpy as np
from numpy.linalg import pinv, inv
import tensorly as tl
tl.set_backend('pytorch')
import torch as tc
from tensorly.decomposition import parafac, tucker, non_negative_parafac, non_negative_tucker, matrix_product_state, partial_tucker
from tensorly.tenalg import kronecker, multi_mode_dot, mode_dot
import matplotlib.pyplot as plt
from time import time, process_time_ns

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d


data = loadMNIST()

# %% Making a tensor of a subset of the digits

digits = (3, 4)

X_all = data.x_train
Y_all = data.y_train

whereTot = np.zeros(len(data.y_train))
for i in range(len(digits)):
    whereTot += (data.y_train == digits[i])
indicesAll = np.where(whereTot)    

X_sub = X_all[indicesAll].reshape((-1, 28, 28))

# Making sure Ys are classes 0...k and not the original ones
Y_subIni = Y_all[indicesAll]
Y_sub = tc.zeros(Y_subIni.shape[0])
for i in range(1, len(digits)):
    Y_sub += (Y_subIni == digits[i])*i

plotMany(X_sub, 30, 20)

N = len(Y_sub)
print("There is a total of %d digits of the types:" % N, digits)

# %% Tucker decomposition

nTrain = int(0.7 * N)
nVal = int(0.8 * N)

# The decomposition
K = partial_tucker(tl.tensor(X_sub[:nTrain]), modes = [0], ranks=[2])
core, [A] = K

X_hat = tl.tucker_to_tensor(K)

plotMany(X_sub, 10, 10)
plotMany(X_hat, 10, 10)

plt.figure()
A3s = A[tc.where(Y_sub[:nTrain] == 0)[0]]
A4s = A[tc.where(Y_sub[:nTrain] == 1)[0]]
plt.scatter(A4s[:,0], A4s[:,1], facecolor = 'Blue', edgecolor='blue', marker = "x",s = 10)   
plt.scatter(A3s[:,0], A3s[:,1], facecolor = 'None', edgecolor='red', s = 10)
m4s = tc.mean(A4s, 0)
m3s = tc.mean(A3s, 0)
mAll = tc.mean(A, 0)
plt.scatter(m4s[0],m4s[1], facecolor="Black", s = 50, marker = "s")
plt.scatter(m3s[0],m3s[1], facecolor="Black", s = 50, marker = "^")
plt.scatter(mAll[0],mAll[1], facecolor="Black", s = 50)
plt.legend(labels = ('4','3','Mean 4','Mean 3', "Overall"))
plt.xlabel('1. loading in A')         
plt.ylabel('2. loading in A')
plt.title('Loadings of A for all the training examples')

general4 = mode_dot(core, m4s, mode = 0)
general3 = mode_dot(core, m3s, mode = 0)
general = mode_dot(core, mAll, mode = 0)
showImage(general4)
showImage(general3)
showImage(general)
# %%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = A[:,0].numpy(), A[:,1].numpy()
hist, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[0, 4], [0, 4]])
#%% Partial Tucker decomposition

A2 = A2.numpy()
A2_new = tl.unfold(tl.tensor(X_sub[nTrain:]), mode = 0) @ pinv(tl.unfold(core2, mode = 0))
X2_new = mode_dot(core2, A2_new, mode = 0)
A2_new = A2_new.numpy()

# %% Trying just a dense neural network

data = Data(X_sub[:nTrain], Y_sub[:nTrain], X_sub[nTrain:nVal], Y_sub[nTrain:nVal], X_sub[nVal:], Y_sub[nVal:], normalize = False)

num_classes = len(digits)
height, width = data.x_train.shape[1:]
num_l1 = 20

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.l1 = Linear(in_features=height*width, out_features=num_l1, bias = True)
        #self.l2 = Linear(in_features=num_l1, out_features=num_l2, bias=True)
        self.l_out = Linear(in_features=num_l1, out_features=num_classes, bias=True)
        #self.norm = BatchNorm1d(num_l1)
        
        self.dropout = Dropout(p = 0.2)
        
    def forward(self, x):
        x = x.view(-1,height*width)
        x = self.dropout(relu(self.l1(x)))
        
        #x = self.dropout(relu(self.l2(x)))
        
        return softmax(self.l_out(x), dim = 1)

net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum=0.5)

training(net, data, 100, 300, optimizer, every = 5)

# %% Trying with the loadings from A

A_new = tl.unfold(multi_mode_dot(X_sub[nTrain:],[pinv(B), pinv(C)], modes = [1,2]),mode = 0) @ pinv(tl.unfold(core, mode = 0))
X_new = multi_mode_dot(core, [A_new, B, C], modes = [0,1,2])

plotMany(X_new, 10,10)

# %%
num_classes = len(digits)
num_features = A2_new.shape[1]
num_l1 = 20

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.l1 = Linear(in_features = num_features, out_features = num_l1, bias = True)
        
        self.l_out = Linear(in_features = num_l1, out_features=num_classes, bias = True)
        
    def forward(self, x):
        x = relu(self.l1(x))
        return softmax(self.l_out(x), dim = 1)
    
net = Net()
print(net)
dataDecomp = Data(A2, Y_sub[:nTrain], A2_new[:(nVal-nTrain)], Y_sub[nTrain:nVal], A2_new[(nVal-nTrain):], Y_sub[nVal:], normalize = False)

# %% Training the decomposed network
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.7)
training(net, dataDecomp, 100, 1000, optimizer, every = 5)
