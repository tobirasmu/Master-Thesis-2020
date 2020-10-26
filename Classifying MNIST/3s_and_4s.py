#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:38:38 2020

@author: Tobias

In this file, it is attempted to decompose some 3s and 4s from the MNIST data set.
"""

from nnFunctions import training, loadMNIST, Data, showImage, showWrong, plotMany, get_slice
import numpy as np
from numpy.linalg import pinv, inv
import tensorly as tl
import torch as tc
from tensorly.decomposition import parafac, tucker, non_negative_parafac, non_negative_tucker, matrix_product_state
from tensorly.tenalg import kronecker
import matplotlib.pyplot as plt

data = loadMNIST()

# %% Making a tensor of only 3s and 4s. 

X_all = data.x_train
Y_all = data.y_train

indices4_and_3 = np.where(((data.y_train == 4) + (data.y_train == 3)))

X34 = X_all[indices4_and_3]
Y34 = Y_all[indices4_and_3]
Y34 = (Y34 == 4)*1

plotMany(X34, 30, 20)

N = len(Y34)
print("There is a total of %d 3s and 4s" % N)
""" 
Now X is a tensor of only 3s and 4s
"""

# %% Decomposing using TUCKER
from tensorly.tenalg import multi_mode_dot, mode_dot
nTrain = 5000

K = tucker(tl.tensor(X34[:nTrain].reshape((-1,28,28))), ranks=[2,28,28])
A, B, C = K[1]
core = K[0]

X_hat = tl.tucker_to_tensor(K)
# K = (core, [A, B, C])
#X_hat = tl.tucker_to_tensor((core,[A, B, C]))

plt.close('all')
plotMany(X34, 10, 10)
plotMany(X_hat, 10, 10)

plt.figure()
plt.hist(A)
"""
Looks as though the decomposition is able to find "archetypes" of 3s and 4s !!
"""
# The least squares approximation of A, given B, C and the data. 
A_realish = tl.unfold(multi_mode_dot(X34[:nTrain].reshape((-1,28,28)), [pinv(B), pinv(C)], modes = [1,2]), mode = 0) @ pinv(tl.unfold(core, mode = 0))

"""
It is the same! How exciting! 
"""
# %% Showing the "archetypes" of 3 and 4. 

arche3 = multi_mode_dot(core, [np.array([[-0.01, -0.015]]),B ,C ], modes = [0,1,2])
arche4 = multi_mode_dot(core, [np.array([[-0.01, 0.015]]),B ,C ], modes = [0,1,2])

showImage(arche3)
showImage(arche4)

# %% Using the matrix-product-state function from tensorly

K2 = matrix_product_state(X34[:nTrain].reshape(-1,28,28), [1,2,5,1])
X_hat2 = tl.mps_to_tensor(K2)

plt.close('all')
plotMany(X34, 20, 10)
plotMany(X_hat2, 20, 10)

# %% Loading NN packages

from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as Fun
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d, Dropout, BatchNorm1d

# %%
data = Data(X34[:nTrain], Y34[:nTrain], X34[nTrain:6000], Y34[nTrain:6000], X34[6000:], Y34[6000:], normalize = False)

num_classes = 2
channels, height, width = data.x_train.shape[1:]
num_l1 = 1

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.l1 = Linear(in_features=height*width, out_features=num_l1, bias = True)
        #self.l2 = Linear(in_features=num_l1, out_features=num_l2, bias=True)
        self.l_out = Linear(in_features=num_l1, out_features=num_classes, bias=False)
        #self.norm = BatchNorm1d(num_l1)
        
        self.dropout = Dropout(p = 0.4)
        
    def forward(self, x):
        x = x.view(-1,height*width)
        x = self.dropout(relu(self.l1(x)))
        
        #x = self.dropout(relu(self.l2(x)))
        
        return softmax(self.l_out(x), dim = 1)

net = Net()
print(net)

# %% Training the net

optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.5)

training(net, data, 100, 300, optimizer)

# %% Now trying with the loadings
""" 
Before trying with the loadings, we need to decompose the testing and validation samples in the
same way as the training set. 
"""
A_new = tl.unfold(multi_mode_dot(X34[nTrain:].reshape((-1,28,28)),[pinv(B), pinv(C)], modes = [1,2]),mode = 0) @ pinv(tl.unfold(core, mode = 0))
X_new = multi_mode_dot(core, [A_new, B, C], modes = [0,1,2])
"""
Now A_hat is the loading matrix of the 1st dimension of the test-set decomposed
w.r.t the training set. 
"""
plotMany(X_new, 20,20)
plotMany(X34[5000:],20,20)

# %% Net with loadings
num_classes = 2
num_features = 2
num_l1 = 1

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.l1 = Linear(in_features = 2, out_features = num_l1, bias = True)
        
        self.l_out = Linear(in_features = num_l1, out_features=2, bias = False)
        
    def forward(self, x):
        x = relu(self.l1(x))
        return softmax(self.l_out(x), dim = 1)
    
net = Net()
print(net)


# %% Optimizing
optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum=0.7)
dataDecomp = Data(A, Y34[:nTrain], A_new[:1000], Y34[nTrain:6000], A_new[1000:], Y34[6000:], normalize = False)
training(net, dataDecomp, 100, 300, optimizer, every = 10)


# %% TT-decomp first looks
# Calculation of one value from the approximated tensor

G1 = tl.tensor([[1,4,5,6,1,3,5]])
G2 = tl.tensor([[1,2,3],[3,2,5],[4,5,6],[8,6,3],[1,4,2],[1,2,3],[2,3,4]])
G3 = tl.tensor([[8,3,1]]).reshape((3,1))
hej = np.dot(G1, np.dot(G2, G3))

summ = 0
# So a1 and a2 are the auxilary indices! 
for a1 in range(7):
    for a2 in range(3):
        summ += G1[0,a1]*G2[a1,a2]*G3[a2,0]
        
print("If %d is the same as %d, then I figured it out!!" % (summ, hej))
