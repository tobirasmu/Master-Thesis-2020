#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:41:30 2020

@author: Tobias
"""
# %% Loading the data and all the libraries
import os

path = "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

from pic_functions import train_epoch, eval_epoch, loadMNIST, conv_to_tucker1, conv_to_tucker2, lin_to_tucker2, \
    lin_to_tucker1, numParams
from time import time, process_time, process_time_ns
import numpy as np
import tensorly as tl
from copy import deepcopy
import torch as tc
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, softmax
from torch.nn import Linear, Conv2d, MaxPool2d, Dropout2d, Dropout
from VBMF import EVBMF

tl.set_backend('pytorch')

# %% Loading the data
fullData = loadMNIST()

# %% Defining the full network from the LeNet-5 architecture.
_, channels, width, height = fullData.x_train.shape


def conv_dim(dim, kernel, stride, padding):
    return int((dim - kernel + 2 * padding) / stride + 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # The convolutions
        self.conv1 = Conv2d(in_channels=channels, out_channels=6, kernel_size=5, padding=2, stride=1)
        dim1 = conv_dim(height, kernel=5, padding=2, stride=1)
        dim1P = conv_dim(dim1, kernel=2, padding=0, stride=2)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0, stride=1)
        dim2 = conv_dim(dim1P, kernel=5, padding=0, stride=1)
        dim2P = conv_dim(dim2, kernel=2, padding=0, stride=2)

        # The average pooling
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)

        # The dropout
        self.dropout1 = Dropout(0.2)
        self.dropout2 = Dropout2d(0.2)

        self.lin_in_feats = 16 * (dim2P ** 2)
        # The linear layers
        self.l1 = Linear(in_features=self.lin_in_feats, out_features=120, bias=True)
        self.l2 = Linear(in_features=120, out_features=84, bias=True)
        self.l_out = Linear(in_features=84, out_features=10, bias=True)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool(x)

        x = relu(self.conv2(x))
        x = self.pool(x)
        # x = self.dropout2(x)

        x = tc.flatten(x, 1)

        x = relu(self.l1(x))
        # x = self.dropout1(x)
        x = relu(self.l2(x))
        return softmax(relu(self.l_out(x)), dim=1)


net = Net()
print(net)
# Trying one forward push
x_test = np.random.normal(0, 1, (5, 1, 28, 28)).astype('float32')
out = net(Variable(tc.from_numpy(x_test)))
print(out)

# %% Training the network
BATCH_SIZE = 128
NUM_EPOCHS = 60

data = fullData.subset(10000, 10000, 10000)


def train(thisNet, in_data, lr=0.1, momentum=0.5, factor=1.1):
    train_accs, valid_accs, test_accs = [], [], []
    m_inc = (0.9 - momentum) / 8
    inc = NUM_EPOCHS // 8  # We want 8 updates to the momentum and the learning rate
    epoch = 0
    while epoch < NUM_EPOCHS:
        epoch += 1
        try:
            if epoch % inc == 0:
                momentum += m_inc
                momentum = momentum if momentum <= 0.9 else 0.9
                lr /= factor
            optimizer = optim.SGD(thisNet.parameters(), lr=lr, momentum=momentum)
            print("Epoch %d: (mom: %f, lr: %f)" % (epoch, momentum, lr))
            print("Train: ", end='')
            loss, train_acc = train_epoch(thisNet, in_data.x_train, in_data.y_train, optimizer, BATCH_SIZE)
            print(" Validation: ", end='')
            valid_acc = eval_epoch(thisNet, in_data.x_val, in_data.y_val, BATCH_SIZE)
            print(" Testing: ", end='')
            test_acc = eval_epoch(thisNet, in_data.x_test, in_data.y_test, BATCH_SIZE)
            train_accs += [train_acc]
            valid_accs += [valid_acc]
            test_accs += [test_acc]
            print("\nCost {0:.7}, train acc: {1:.4}, val acc: {2:.4}, test acc: {3:.4}".format(
                loss, train_acc, valid_acc, test_acc))
        except KeyboardInterrupt:
            print('\n KeyboardInterrupt')
            break
    epochs = np.arange(len(train_accs))
    plt.figure()
    plt.plot(epochs, train_accs, 'r', epochs, valid_accs, 'b')
    plt.legend(['Train accuracy', 'Validation accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')


train(net, data)


# %% Trying to decompose the learned network
# Making a copy
netDec = deepcopy(net)

netDec.conv1 = conv_to_tucker1(netDec.conv1)
netDec.conv2 = conv_to_tucker2(netDec.conv2)
netDec.l1 = lin_to_tucker2(netDec.l1)
netDec.l2 = lin_to_tucker1(netDec.l2)

# %% The change in number of parameters
print("Before: {}, after: {}, which is {:.3}".format(numParams(net), numParams(netDec),
                                                     numParams(netDec) / numParams(net)))
# Not the biggest decomp... How about accuracy?
t = time()
acc2 = eval_epoch(netDec, data.x_test, data.y_test)
print(time() - t)
t = time()
acc1 = eval_epoch(net, data.x_test, data.y_test)
print(time() - t)
print("\nAccuracy before: ", acc1)
print("\nAccuracy after: ", acc2)
# So the accuracy have decreased by a bit. Maybe it can be finetuned?

# %% Timing just one forward pass
x_test = Variable(tc.from_numpy(data.x_train[100]).unsqueeze(0))
allTimeNet = 0
allTimeDec = 0
for i in range(1000):
    t = process_time_ns()
    net(x_test)
    allTimeNet += process_time_ns() - t
    t = process_time_ns()
    netDec(x_test)
    allTimeDec += process_time_ns() - t
print(allTimeNet / 1000, " ", allTimeDec / 1000, "  ", (allTimeDec / 1000) / (allTimeNet / 1000))

# %% Fine-tuning the decomposed network
train(netDec, data, lr=0.01, factor=2)

# %% 
print("Before: {}, after: {}, which is {:.3}".format(numParams(net), numParams(netDec),
                                                     numParams(netDec) / numParams(net)))
# Not the biggest decomp... How about accuracy?
times1 = []
for i in range(10):
    t = process_time()
    acc1 = eval_epoch(netDec, data.x_test, data.y_test)
    times1.append(process_time() - t)
print(np.mean(times1[1:]))
times2 = []
for i in range(10):
    t = process_time()
    acc2 = eval_epoch(net, data.x_test, data.y_test)
    times2.append(process_time() - t)
print(np.mean(times2[1:]))
print("\nAccuracy before: ", acc2)
print("\nAccuracy after: ", acc1)
