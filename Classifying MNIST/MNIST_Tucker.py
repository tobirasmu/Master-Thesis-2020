#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:41:30 2020

@author: Tobias
"""
HPC = False
import os

path = "/zhome/2a/c/108156/Master-Thesis-2020/Classifying MNIST/" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Classifying MNIST/"
os.chdir(path)

from pic_functions import train_epoch, eval_epoch, loadMNIST, numParams, get_variable, plotAccs
from pic_networks import Net, compressNetwork
import numpy as np
import tensorly as tl
import torch as tc
from torch.autograd import Variable
import torch.optim as optim

tl.set_backend('pytorch')

# %% Loading the data
fullData = loadMNIST()

# %% Defining the full network from the LeNet-5 architecture.
_, channels, width, height = fullData.x_train.shape

net = Net(channels, height)

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
    inc = num_epochs // LR_UPDs  # Making sure we make the correct number of updates to the learning rate and momentum
    epoch = 0
    while epoch < num_epochs:
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

# Saving the network in order to time it
save_at = "/zhome/2a/c/108156/Master-Thesis-2020/Trained networks/MNIST_network.pt" if HPC else \
    "/Users/Tobias/Google Drev/UNI/Master-Thesis-Fall-2020/Trained networks/MNIST_network.pt"
tc.save(net.state_dict(), save_at)

# Compressing the network
netDec = compressNetwork(net)


if tc.cuda.is_available():
    net = net.cuda()
    netDec = netDec.cuda()
print(netDec)

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