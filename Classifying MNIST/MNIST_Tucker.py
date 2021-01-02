#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:41:30 2020

@author: Tobias
"""
HPC = True
import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying MNIST/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

from pic_functions import train_epoch, eval_epoch, loadMNIST, conv_to_tucker1, conv_to_tucker2, lin_to_tucker2, \
    lin_to_tucker1, numParams, get_data, get_variable, plotAccs
import numpy as np
import tensorly as tl
from copy import deepcopy
import torch as tc
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, softmax
from torch.nn import Linear, Conv2d, MaxPool2d, Dropout2d, Dropout

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
if tc.cuda.is_available():
    print("----  Network converted to CUDA  ----\n")
    net = net.cuda()
print(net)
# Trying one forward push
x_test = np.random.normal(0, 1, (5, 1, 28, 28)).astype('float32')
out = net(get_variable(Variable(tc.from_numpy(x_test))))

# %% Training the network
BATCH_SIZE = 128
NUM_EPOCHS = 100
LR_UPDs = 8

data = fullData


def train(thisNet, in_data, lr=0.1, momentum=0.5, factor=1.1, num_epochs=NUM_EPOCHS):
    train_accs, valid_accs, test_accs = [], [], []
    m_inc = (0.9 - momentum) / LR_UPDs
    inc = NUM_EPOCHS // LR_UPDs  # Making sure we make the correct number of updates to the learning rate and momentum
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
    saveAt = "/zhome/2a/c/108156/Outputs/accuracies_MNIST.png" if HPC else "/Users/Tobias/Desktop/accuracies_MNIST.png"
    plotAccs(train_accs, valid_accs, saveName=saveAt)


print("{:-^60s}\n{:-^60s}\n{:-^60s}".format("", "  Learning the full network  ", ""))
train(net, data)

# %% Trying to decompose the learned network
# Making a copy
print("\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format("", "  Decomposing the learned network  ", ""))
if tc.cuda.is_available():
    net = net.cpu()
netDec = deepcopy(net)

netDec.conv1 = conv_to_tucker1(netDec.conv1)
netDec.conv2 = conv_to_tucker2(netDec.conv2)
netDec.l1 = lin_to_tucker2(netDec.l1)
netDec.l2 = lin_to_tucker1(netDec.l2)

if tc.cuda.is_available():
    net = net.cuda()
    netDec = netDec.cuda()

# %% The change in number of parameters
print("The number of parameters:\nBefore: {}, after: {}, which is {:.3}".format(numParams(net), numParams(netDec),
                                                                                numParams(netDec) / numParams(net)))
# Not the biggest decomp... How about accuracy?
acc_dec = eval_epoch(netDec, data.x_test, data.y_test, BATCH_SIZE)
acc_ori = eval_epoch(net, data.x_test, data.y_test, BATCH_SIZE)
print("\nAccuracy before: {}   Accuracy after: {}".format(acc_ori, acc_dec))

# %% Fine-tuning the decomposed network
print("\n{:-^60s}\n{:-^60s}\n{:-^60s}\n".format("", "  Fine-tuning the decomposed network  ", ""))
train(netDec, data, lr=0.01, factor=2, num_epochs=50)

acc_dec = eval_epoch(netDec, data.x_test, data.y_test, BATCH_SIZE)
acc_ori = eval_epoch(net, data.x_test, data.y_test, BATCH_SIZE)
print("-- Final accuracy differences --")
print("\n Accuracy before: {}   Accuracy after: {}".format(acc_ori, acc_dec))