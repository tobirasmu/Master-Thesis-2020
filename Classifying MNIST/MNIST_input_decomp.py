#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 16:38:38 2020

@author: Tobias

In this file, it is attempted to decompose some set of digits from the MNIST data set.
"""
HPC = True
import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying MNIST/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)
from pic_functions import training, loadMNIST, Data, showImage, plotMany
import numpy as np
from numpy.linalg import pinv
import tensorly as tl

tl.set_backend('pytorch')
import torch as tc
from tensorly.decomposition import partial_tucker
from tensorly.tenalg import multi_mode_dot, mode_dot
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, softmax
from torch.nn import Linear, Dropout

data = loadMNIST()

# %% Making a tensor of a subset of the digits
"""
    Digits is a tuple with the specific digits that is to be considered. 3 and 4 seems to be fairly easy to distinguish, 
    while 4 and 9 seems similar hence harder for the decomposition algorithm to distinguish.
"""
digits = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

X_all = data.x_train
Y_all = data.y_train

whereTot = np.zeros(len(data.y_train))
for i in range(len(digits)):
    whereTot += (data.y_train == digits[i])
indicesAll = np.where(whereTot)

X_sub = X_all[indicesAll].reshape((-1, 28, 28))

# Making sure Ys are classes 0...k and not the original ones
Y_subIni = Y_all[indicesAll]
Y_sub = np.zeros(Y_subIni.shape[0])
for i in range(1, len(digits)):
    Y_sub += (Y_subIni == digits[i]) * i

# Plotting the first 30x20 pictures from the data set
plotMany(X_sub, 30, 20)

N = len(Y_sub)
print("There is a total of {:d} digits of the types: ".format(N), *digits)

# %% Tucker decomposition
"""
    Here the data set is split into a training and a testing set. The training set will be decomposed along the 
    observation dimension with the given rank (at least the number of digits)
"""
nTrain = int(0.7 * N)
nVal = int(0.8 * N)

# Specify the rank
rank = 2

# The decomposition
K = partial_tucker(tl.tensor(X_sub[:nTrain]), modes=[0], ranks=[rank])
core, [A] = K

X_hat = tl.tucker_to_tensor(K)

plotMany(X_sub, 10, 10)
plotMany(X_hat, 10, 10)

# %% Plotting a scatter-plot if there are 2 digits
if len(digits) == 2:
    plt.figure()
    A3s = A[tc.where(tc.from_numpy(Y_sub)[:nTrain] == 0)[0]]
    A4s = A[tc.where(tc.from_numpy(Y_sub)[:nTrain] == 1)[0]]
    plt.scatter(A3s[:, 0], A3s[:, 1], facecolor='None', edgecolor='red', s=10)
    plt.scatter(A4s[:, 0], A4s[:, 1], facecolor='Blue', edgecolor='blue', marker="x", s=10)
    m4s = tc.mean(A4s, 0)
    m3s = tc.mean(A3s, 0)
    mAll = tc.mean(A, 0)
    plt.scatter(m3s[0], m3s[1], facecolor="Black", s=50, marker="^")
    plt.scatter(m4s[0], m4s[1], facecolor="Black", s=50, marker="s")
    plt.scatter(mAll[0], mAll[1], facecolor="Black", s=50)
    plt.legend(labels=(str(digits[0]), str(digits[1]), 'Mean ' + str(digits[0]), 'Mean ' + str(digits[1]), "Overall"))
    plt.xlabel('1. loading in A')
    plt.ylabel('2. loading in A')
    plt.title('Loadings of A for all the training examples')
    plt.show()

for i in range(len(digits)):
    this_mean = tc.mean(A[tc.where(tc.from_numpy(Y_sub)[:nTrain] == i)[0]], 0)
    this_general = mode_dot(core, this_mean, mode=0)
    showImage(this_general)
all_general = tc.mean(A, 0)
general = mode_dot(core, all_general, mode=0)
showImage(general)

# %% Trying just a dense neural network
"""
    The very simple ANN used in this experiment consists of just one hidden layer with a given number of hidden neurons
    (num_l1). 
"""
data = Data(X_sub[:nTrain], Y_sub[:nTrain], X_sub[nTrain:nVal], Y_sub[nTrain:nVal], X_sub[nVal:], Y_sub[nVal:],
            normalize=False)

num_classes = len(digits)
_, height, width = data.x_train.shape
# Number of hidden units
num_l1 = 20


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.l1 = Linear(in_features=height * width, out_features=num_l1, bias=True)
        # self.l2 = Linear(in_features=num_l1, out_features=num_l2, bias=True)
        self.l_out = Linear(in_features=num_l1, out_features=num_classes, bias=True)
        # self.norm = BatchNorm1d(num_l1)

        #self.dropout = Dropout(p=0.2)

    def forward(self, x):
        x = x.view(-1, height * width)
        x = relu(self.l1(x))

        # x = self.dropout(relu(self.l2(x)))

        return softmax(self.l_out(x), dim=1)


net = Net()
if tc.cuda.is_available():
    print("Cuda enabled \n")
    net = net.cuda()

print(net)

optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.5)
this_save = "/zhome/2a/c/108156/Outputs/MNIST_results/original.png" if HPC else "/Users/Tobias/Desktop/MNIST_test" \
                                                                                "/original.png "
training(net, data, 100, 500, optimizer, every=5, saveAt=this_save)

# %% Trying with the loadings from A

A_new = tl.unfold(tc.from_numpy(X_sub)[nTrain:], mode=0) @ pinv(tl.unfold(core, mode=0))
X_new = multi_mode_dot(core, [A_new], modes=[0, 1, 2])

plotMany(X_new, 10, 10)

# %% Defining the decomposed network

num_classes = len(digits)
# Uses the same number of hidden units as the other


class Net(nn.Module):

    def __init__(self, num_features):
        super(Net, self).__init__()

        self.l1 = Linear(in_features=num_features, out_features=num_l1, bias=True)

        self.l_out = Linear(in_features=num_l1, out_features=num_classes, bias=True)

    def forward(self, x):
        x = relu(self.l1(x))
        return softmax(self.l_out(x), dim=1)


net = Net(rank)
print(net)
dataDecomp = Data(A.numpy(), Y_sub[:nTrain], A_new[:(nVal - nTrain)].numpy(), Y_sub[nTrain:nVal], A_new[(nVal - nTrain):].numpy(),
                  Y_sub[nVal:], normalize=False)

# %% Training the decomposed network
# optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.5)
# training(net, dataDecomp, 100, 500, optimizer, every=5)

# %% Looping over and training the different networks
ranks = (2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 100, 150, 300)
test_accuracies = []

for rank in ranks:
    print("{:-^60s}\n{:-^60s}\n{:-^60s}".format("", " Rank {:3d} ".format(rank), ""))
    # Decomposing the input kernel
    K = partial_tucker(tl.tensor(X_sub[:nTrain]), modes=[0], ranks=[rank])
    core, [A] = K
    # Estimating A_new for the testing data
    A_new = tl.unfold(tc.from_numpy(X_sub)[nTrain:], mode=0) @ pinv(tl.unfold(core, mode=0))
    # Making a new network to fit the rank
    net = Net(rank)
    if tc.cuda.is_available():
        net = net.cuda()
    this_data = Data(A.numpy(), Y_sub[:nTrain], A_new[:(nVal - nTrain)].numpy(),
                     Y_sub[nTrain:nVal], A_new[(nVal - nTrain):].numpy(), Y_sub[nVal:], normalize=False)
    optimizer = optim.SGD(net.parameters(), lr=0.5, momentum=0.5)
    saveAt = "/zhome/2a/c/108156/Outputs/MNIST_results/" if HPC else "/Users/Tobias/Desktop/MNIST_test/"
    saveAt = saveAt + "rank_" + str(rank) + ".png"
    test_accuracies.append((rank, training(net, this_data, 100, 1000, optimizer, every=5,
                                    saveAt=saveAt)))

print("The accuracies are:")
print(test_accuracies)